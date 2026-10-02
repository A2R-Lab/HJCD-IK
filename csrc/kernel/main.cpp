#include "kernel/hjcd_kernel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct Args {
    std::string mode = "single";   // "single" | "sweep" | "from_csv"
    int batch_size = 2000;
    int num_solutions = 1;
    int num_targets = 100;
    std::string yaml_out = "results.yml";
    
    std::string csv_in  = "panda_solutions_multi_targets.csv";
    std::string csv_out = "hjcd_mmd_q.csv";
};

static int parse_integer(const std::string& value, const std::string& name, bool positive = false) {
    std::size_t end = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &end);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be " + (positive ? "a positive" : "an") + " integer");
    }
    if (end != value.size() || (positive && parsed <= 0))
        throw std::invalid_argument(name + " must be " + (positive ? "a positive" : "an") + " integer");
    return parsed;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            a.mode = std::string(argv[i] + 7);
        } else if (std::strncmp(argv[i], "--batch_size=", 13) == 0) {
            a.batch_size = parse_integer(argv[i] + 13, "--batch_size", true);
        } else if (std::strncmp(argv[i], "--num_solutions=", 16) == 0) {
            a.num_solutions = parse_integer(argv[i] + 16, "--num_solutions", true);
        } else if (std::strncmp(argv[i], "--num_targets=", 14) == 0) {
            a.num_targets = parse_integer(argv[i] + 14, "--num_targets", true);
        } else if (std::strncmp(argv[i], "--yaml_out=", 11) == 0) {
            a.yaml_out = std::string(argv[i] + 11);
        } else if (std::strncmp(argv[i], "--csv_in=", 9) == 0) {
            a.csv_in = std::string(argv[i] + 9);
        } else if (std::strncmp(argv[i], "--csv_out=", 10) == 0) {
            a.csv_out = std::string(argv[i] + 10);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::cout <<
                "Usage: ./app "
                "[--mode=single|sweep|from_csv] "
                "[--batch_size=2000] [--num_solutions=1] "
                "[--num_targets=100] "
                "[--yaml_out=results.yml] "
                "[--csv_in=panda_solutions_multi_targets.csv] "
                "[--csv_out=hjcd_mmd_q.csv]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(argv[i]));
        }
    }
    if (a.mode != "single" && a.mode != "sweep" && a.mode != "from_csv")
        throw std::invalid_argument("Unknown --mode=" + a.mode + " (use 'single', 'sweep', or 'from_csv')");
    return a;
}

static void finish_output(std::ofstream& stream, const std::string& path) {
    // close() flushes buffered writes too; checking only is_open() misses full disks.
    stream.close();
    if (!stream) throw std::runtime_error("Failed to write output: " + path);
}

static void write_yaml_flat(
    const std::string& path,
    const std::vector<int>& batch_sizes,
    const std::vector<double>& time_ms,
    const std::vector<double>& pos_err_mm,
    const std::vector<double>& ori_err_rad)
{
    std::ofstream y(path);
    if (!y) throw std::runtime_error("Failed to open output: " + path);
    y << std::setprecision(17);

    auto write_list = [&](const char* key, auto&& vec) {
        y << key << ":\n";
        for (const auto& v : vec) {
            y << "  - " << v << "\n";
        }
    };

    write_list("Batch-Size", batch_sizes);
    write_list("IK-time(ms)", time_ms);
    write_list("Pos-Error", pos_err_mm);
    write_list("Ori-Error", ori_err_rad);
    finish_output(y, path);
}

static std::string trim(const std::string& value) {
    const auto space = [](unsigned char ch) { return std::isspace(ch); };
    const auto first = std::find_if_not(value.begin(), value.end(), space);
    const auto last = std::find_if_not(value.rbegin(), value.rend(), space).base();
    return first < last ? std::string(first, last) : std::string();
}

static double parse_finite_real(const std::string& value, const std::string& name) {
    std::size_t end = 0;
    double parsed = 0;
    try {
        parsed = std::stod(value, &end);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be a finite number");
    }
    if (end != value.size() || !std::isfinite(parsed))
        throw std::invalid_argument(name + " must be a finite number");
    return parsed;
}

// This numeric interchange format has unquoted, comma-separated fields.
static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(line.size());
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else { cur.push_back(c); }
    }
    out.push_back(cur);
    for (auto& field : out) field = trim(field);
    return out;
}

struct PoseRow {
    int target_id;
    // target pose in [x,y,z,qw,qx,qy,qz]
    std::array<double,7> wxyz_pose;
};

static std::vector<PoseRow> load_unique_targets_from_tracik_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open csv_in: " + path);
    }
    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Empty CSV: " + path);
    }
    auto cols = split_csv_line(header);

    auto find_col = [&](const std::string& name)->int{
        for (int i = 0; i < (int)cols.size(); ++i) {
            if (cols[i] == name) return i;
        }
        return -1;
    };

    const int idx_tid = find_col("target_id");
    const int idx_px  = find_col("target_px");
    const int idx_py  = find_col("target_py");
    const int idx_pz  = find_col("target_pz");
    const int idx_qx  = find_col("target_qx");
    const int idx_qy  = find_col("target_qy");
    const int idx_qz  = find_col("target_qz");
    const int idx_qw  = find_col("target_qw");

    if (idx_tid < 0 || idx_px < 0 || idx_py < 0 || idx_pz < 0 ||
        idx_qx < 0 || idx_qy < 0 || idx_qz < 0 || idx_qw < 0) {
        throw std::runtime_error("CSV missing required target_* columns.");
    }

    std::unordered_set<int> seen;
    std::vector<PoseRow> out;
    std::string line;
    std::size_t line_number = 1;
    while (std::getline(in, line)) {
        ++line_number;
        if (trim(line).empty()) continue;
        try {
            const auto f = split_csv_line(line);
            if (f.size() != cols.size())
                throw std::invalid_argument("wrong number of CSV fields");
            PoseRow row;
            row.target_id = parse_integer(f[idx_tid], "target_id");
            row.wxyz_pose = {
                parse_finite_real(f[idx_px], "target_px"),
                parse_finite_real(f[idx_py], "target_py"),
                parse_finite_real(f[idx_pz], "target_pz"),
                parse_finite_real(f[idx_qw], "target_qw"),
                parse_finite_real(f[idx_qx], "target_qx"),
                parse_finite_real(f[idx_qy], "target_qy"),
                parse_finite_real(f[idx_qz], "target_qz"),
            };
            const auto& pose = row.wxyz_pose;
            if (std::max({std::abs(pose[3]), std::abs(pose[4]),
                          std::abs(pose[5]), std::abs(pose[6])}) == 0.0)
                throw std::invalid_argument("target quaternion must be nonzero");
            // TRAC-IK exports multiple solutions per target. Validate every row before
            // deduplicating, so corrupt repeated rows are not silently hidden.
            if (seen.insert(row.target_id).second) out.push_back(row);
        } catch (const std::invalid_argument& error) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) + ": " + error.what());
        }
    }
    if (in.bad()) throw std::runtime_error("Failed to read csv_in: " + path);
    return out;
}

static void append_solution_vectors(
    int S,
    int B,
    double elapsed_ms_total,
    const double* pos_err,
    const double* ori_err,
    std::vector<int>& y_batch,
    std::vector<double>& y_time_ms,
    std::vector<double>& y_pos,
    std::vector<double>& y_ori)
{
    const double per_sample_ms = elapsed_ms_total / std::max(1, S);
    for (int r = 0; r < S; ++r) {
        y_batch.push_back(B);
        y_time_ms.push_back(per_sample_ms);
        y_pos.push_back(pos_err[r]);
        y_ori.push_back(ori_err[r]);
    }
}

int main(int argc, char** argv) try {
    const Args args = parse_args(argc, argv);

    // The native API lazily initializes and shares the model on the current CUDA device.
    const int N = grid_num_joints();
    const int B = args.batch_size;
    int S = args.num_solutions;

    using clock = std::chrono::steady_clock;

    if (args.mode == "single" || args.mode == "sweep") {
        // "single" is a one-target sweep.
        const int T = args.mode == "single" ? 1 : args.num_targets;
        auto targets = sample_random_target_poses<double>(nullptr, T, /*seed=*/0ull);
        if ((int)targets.size() < T) {
            std::cerr << "Failed to sample " << T << " target poses.\n";
            return 1;
        }

        std::vector<int>    y_batch;
        std::vector<double> y_time, y_pos, y_ori;
        y_batch.reserve((size_t)T * S);
        y_time.reserve((size_t)T * S);
        y_pos.reserve((size_t)T * S);
        y_ori.reserve((size_t)T * S);

        for (int t = 0; t < T; ++t) {
            const auto t0 = clock::now();
            auto res = generate_ik_solutions<double>(targets[t].data(), B, S);
            const auto t1 = clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            append_solution_vectors(res.count, B, elapsed_ms, res.pos_errors, res.ori_errors,
                                    y_batch, y_time, y_pos, y_ori);
            if (((t + 1) % 50) == 0)
                std::cout << "[sweep] processed " << (t + 1) << " / " << T << " targets...\n";
        }

        write_yaml_flat(args.yaml_out, y_batch, y_time, y_pos, y_ori);
        std::cout << "[OK] wrote " << args.yaml_out
                  << " (" << y_batch.size() << " entries from " << T << " targets).\n";
        return 0;
    }

    {
        if (S != 50) {
            std::cerr << "[from_csv] INFO: overriding --num_solutions=" << S
                      << " → 50 for MMD sampling.\n";
            S = 50;
        }

        std::vector<PoseRow> targets;
        try {
            targets = load_unique_targets_from_tracik_csv(args.csv_in);
        } catch (const std::exception& e) {
            std::cerr << "[from_csv] " << e.what() << "\n";
            return 2;
        }
        if (targets.empty()) {
            std::cerr << "[from_csv] No targets found in " << args.csv_in << "\n";
            return 1;
        }

        std::ofstream out(args.csv_out);
        if (!out.is_open()) {
            std::cerr << "[from_csv] Cannot open csv_out for write: " << args.csv_out << "\n";
            return 2;
        }
        out << std::setprecision(9) << std::fixed;
        out << "target_id,sample_id";
        for (int j = 1; j <= N; ++j) out << ",q" << j;
        out << "\n";

        std::size_t processed = 0;
        std::size_t rows_written = 0;
        for (const auto& t : targets) {
            // Pose: [x,y,z,qw,qx,qy,qz]
            double target_pose[7];
            for (int i = 0; i < 7; ++i) target_pose[i] = t.wxyz_pose[i];

            auto res = generate_ik_solutions<double>(target_pose, B, S);

            for (int r = 0; r < res.count; ++r) {
                const double* qrow = res.joint_config + (size_t)r * N;
                out << t.target_id << "," << r;
                for (int j = 0; j < N; ++j) out << "," << qrow[j];
                out << "\n";
            }
            rows_written += static_cast<std::size_t>(res.count);

            processed++;
            if ((processed % 50) == 0) {
                std::cout << "[from_csv] processed " << processed << " / " << targets.size() << " targets...\n";
            }
        }

        finish_output(out, args.csv_out);
        std::cout << "[from_csv] Wrote " << args.csv_out
                  << " with " << rows_written << " samples from " << targets.size()
                  << " targets (q only).\n";
        return 0;
    }
} catch (const std::exception& error) {
    std::cerr << "HJCD-IK: " << error.what() << '\n';
    return 1;
}
