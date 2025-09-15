// main.cpp
#include "include/hjcd_kernel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" int grid_num_joints();
void init_joint_limits_from_grid();

struct Args {
    std::string mode = "single";   // "single" | "sweep" | "from_csv"
    int batch_size = 2000;
    int num_solutions = 1;
    int num_targets = 100;
    std::string yaml_out = "results.yml";
    
    std::string csv_in  = "panda_solutions_multi_targets.csv";
    std::string csv_out = "hjcd_mmd_q.csv";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            a.mode = std::string(argv[i] + 7);
        } else if (std::strncmp(argv[i], "--batch_size=", 13) == 0) {
            a.batch_size = std::max(1, std::atoi(argv[i] + 13));
        } else if (std::strncmp(argv[i], "--num_solutions=", 16) == 0) {
            a.num_solutions = std::max(1, std::atoi(argv[i] + 16));
        } else if (std::strncmp(argv[i], "--num_targets=", 14) == 0) {
            a.num_targets = std::max(1, std::atoi(argv[i] + 14));
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
        }
    }
    return a;
}

static void write_yaml_flat(
    const std::string& path,
    const std::vector<int>& batch_sizes,
    const std::vector<double>& time_ms,
    const std::vector<double>& pos_err_m,
    const std::vector<double>& ori_err_rad)
{
    std::ofstream y(path);
    y << std::setprecision(17);

    auto write_list = [&](const char* key, auto&& vec) {
        y << key << ":\n";
        for (const auto& v : vec) {
            y << "  - " << v << "\n";
        }
    };

    write_list("Batch-Size", batch_sizes);
    write_list("IK-time(ms)", time_ms);
    write_list("Pos-Error", pos_err_m);
    write_list("Ori-Error", ori_err_rad);
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(line.size());
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else { cur.push_back(c); }
    }
    out.push_back(cur);
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
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_csv_line(line);
        if ((int)f.size() <= std::max({idx_tid, idx_px, idx_py, idx_pz, idx_qx, idx_qy, idx_qz, idx_qw})) {
            continue;
        }
        int tid = 0;
        try { tid = std::stoi(f[idx_tid]); } catch (...) { continue; }
        if (seen.count(tid)) continue;
        seen.insert(tid);

        PoseRow row;
        row.target_id = tid;

        double x=0,y=0,z=0,qx=0,qy=0,qz=0,qw=1;
        try {
            x  = std::stod(f[idx_px]);
            y  = std::stod(f[idx_py]);
            z  = std::stod(f[idx_pz]);
            qx = std::stod(f[idx_qx]);
            qy = std::stod(f[idx_qy]);
            qz = std::stod(f[idx_qz]);
            qw = std::stod(f[idx_qw]);
        } catch (...) { continue; }

        if (qw < 0.0) { qw = -qw; qx = -qx; qy = -qy; qz = -qz; }
        row.wxyz_pose = { x, y, z, qw, qx, qy, qz };
        out.push_back(row);
    }
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

int main(int argc, char** argv) {
    const Args args_in = parse_args(argc, argv);
    Args args = args_in;

    auto* d_robotModel = grid::init_robotModel<double>();
    init_joint_limits_from_grid();

    const int N = grid_num_joints();
    const int B = args.batch_size;
    int S = args.num_solutions;

    using clock = std::chrono::steady_clock;

    std::vector<int>    y_batch;
    std::vector<double> y_time;
    std::vector<double> y_pos;
    std::vector<double> y_ori;

    if (args.mode == "single") {
        uint64_t seed = 0ull;
        auto targets = sample_random_target_poses<double>(d_robotModel, 1, seed);
        if (targets.empty()) {
            std::cerr << "Failed to sample target pose.\n";
            return 1;
        }
        double target_pose[7];
        for (int j = 0; j < 7; ++j) target_pose[j] = targets[0][j];

        const auto t0 = clock::now();
        auto res = generate_ik_solutions<double>(target_pose, d_robotModel, B, S);
        const auto t1 = clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        append_solution_vectors(S, B, elapsed_ms, res.pos_errors, res.ori_errors,
                                y_batch, y_time, y_pos, y_ori);

        write_yaml_flat(args.yaml_out, y_batch, y_time, y_pos, y_ori);
        std::cout << "[OK] wrote " << args.yaml_out
                  << " with " << S << " solutions (single target).\n";

        delete[] res.joint_config;
        delete[] res.pose;
        delete[] res.pos_errors;
        delete[] res.ori_errors;
        return 0;
    }

    if (args.mode == "sweep") {
        const int T = args.num_targets;
        uint64_t seed = 0ull;
        auto targets = sample_random_target_poses<double>(d_robotModel, T, seed);
        if ((int)targets.size() < T) {
            std::cerr << "Failed to sample " << T << " target poses.\n";
            return 1;
        }

        y_batch.reserve((size_t)T * S);
        y_time.reserve((size_t)T * S);
        y_pos.reserve((size_t)T * S);
        y_ori.reserve((size_t)T * S);

        std::size_t processed = 0;
        for (int t = 0; t < T; ++t) {
            double target_pose[7];
            for (int j = 0; j < 7; ++j) target_pose[j] = targets[t][j];

            const auto t0 = clock::now();
            auto res = generate_ik_solutions<double>(target_pose, d_robotModel, B, /*NUM_SOLUTIONS=*/S);
            const auto t1 = clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            append_solution_vectors(S, B, elapsed_ms, res.pos_errors, res.ori_errors,
                                    y_batch, y_time, y_pos, y_ori);

            delete[] res.joint_config;
            delete[] res.pose;
            delete[] res.pos_errors;
            delete[] res.ori_errors;

            processed++;
            if ((processed % 50) == 0) {
                std::cout << "[sweep] processed " << processed << " / " << T << " targets...\n";
            }
        }

        write_yaml_flat(args.yaml_out, y_batch, y_time, y_pos, y_ori);
        std::cout << "[OK] wrote sweep results to " << args.yaml_out
                  << " (" << (T * S) << " entries; " << T << " targets x " << S << " solutions each).\n";
        return 0;
    }

    if (args.mode == "from_csv") {
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
        for (const auto& t : targets) {
            // Pose: [x,y,z,qw,qx,qy,qz]
            double target_pose[7];
            for (int i = 0; i < 7; ++i) target_pose[i] = t.wxyz_pose[i];

            auto res = generate_ik_solutions<double>(target_pose, d_robotModel, B, S);

            for (int r = 0; r < S; ++r) {
                const double* qrow = res.joint_config + (size_t)r * N;
                out << t.target_id << "," << r;
                for (int j = 0; j < N; ++j) out << "," << qrow[j];
                out << "\n";
            }

            delete[] res.joint_config;
            delete[] res.pose;
            delete[] res.pos_errors;
            delete[] res.ori_errors;

            processed++;
            if ((processed % 50) == 0) {
                std::cout << "[from_csv] processed " << processed << " / " << targets.size() << " targets...\n";
            }
        }

        std::cout << "[from_csv] Wrote " << args.csv_out
                  << " with " << targets.size() << " targets x 50 samples each (q only).\n";
        return 0;
    }

    std::cerr << "Unknown --mode=" << args.mode << " (use 'single', 'sweep', or 'from_csv').\n";
    return 2;
}
