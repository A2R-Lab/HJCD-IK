#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace grid {
    template<typename T> struct robotModel;
    template<typename T> robotModel<T>* init_robotModel();
}

#define PI 3.14159265358979323846

/**
 * @brief Move-only owner of a native solve's row-major host buffers.
 *
 * Read only the first count rows. A returned candidate is not a success certificate:
 * check both error arrays against your tolerances. Collision filtering can yield count == 0.
 * Buffers remain valid until this object is reset, destroyed, or moved from.
 * Do not delete the individual pointers; ownership follows the Result object.
 */
template<typename T>
struct Result {
    T* joint_config = nullptr;  ///< count x grid_num_joints() joint angles, radians.
    T* pose = nullptr;          ///< count x 7 poses: [x,y,z,qw,qx,qy,qz], positions in meters.
    T* pos_errors = nullptr;    ///< count position errors, millimeters.
    T* ori_errors = nullptr;    ///< count orientation errors, radians.
    T elapsed_time{};          ///< Host-observed solve duration, milliseconds (not per candidate).
    int count = 0;              ///< Number of returned candidates, possibly less than requested.

    Result() = default;
    ~Result() { reset(); }
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    Result(Result&& other) noexcept { *this = std::move(other); }
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            reset();
            joint_config = std::exchange(other.joint_config, nullptr);
            pose = std::exchange(other.pose, nullptr);
            pos_errors = std::exchange(other.pos_errors, nullptr);
            ori_errors = std::exchange(other.ori_errors, nullptr);
            elapsed_time = other.elapsed_time;
            count = other.count;
            other.elapsed_time = T{};
            other.count = 0;
        }
        return *this;
    }

    /// Release all owned buffers and restore the empty state; safe to call repeatedly.
    void reset() noexcept {
        delete[] joint_config;
        delete[] pose;
        delete[] pos_errors;
        delete[] ori_errors;
        joint_config = pose = pos_errors = ori_errors = nullptr;
        elapsed_time = T{};
        count = 0;
    }
};

// Validate before touching CUDA; scale first so finite extreme quaternions normalize safely.
template<typename T>
std::array<T, 7> normalized_target_pose(const T* pose) {
    std::array<T, 7> result;
    for (int i = 0; i < 7; ++i) {
        if (!std::isfinite(pose[i])) throw std::invalid_argument("target_pose values must be finite");
        result[i] = pose[i];
    }
    for (int i = 0; i < 3; ++i)
        if (std::abs(result[i]) > std::numeric_limits<float>::max())
            throw std::invalid_argument("target_pose position exceeds fp32 range");
    T scale = 0;
    for (int i = 3; i < 7; ++i) scale = std::max(scale, std::abs(result[i]));
    if (!(scale > 0)) throw std::invalid_argument("target_pose quaternion must be non-zero");
    T sum = 0;
    for (int i = 3; i < 7; ++i) { result[i] /= scale; sum += result[i] * result[i]; }
    const T norm = std::sqrt(sum);
    for (int i = 3; i < 7; ++i) result[i] /= norm;
    return result;
}

/**
 * @brief Solve one end-effector target with a GPU batch of candidate configurations.
 *
 * Compiled instantiations are <double,double>, <double,float>, and <float,double>.
 * T controls I/O; RT controls LM compute precision; coarse search always uses float.
 * Calls serialize shared native state on the calling thread's current CUDA device.
 * Cached models require that context to remain alive; cudaDeviceReset is unsupported.
 *
 * @param target_pose Nonnull seven-vector [x,y,z,qw,qx,qy,qz], meters and scalar-first
 * quaternion. Finite nonzero quaternions are normalized without mutating this input.
 * @param d_robotModel Retained for source compatibility; ignored. Precision-specific
 * internally cached generated models are used by the solver.
 * @param b_size Positive candidate count (not a target count), bounded by CUDA indexing.
 * @param num_solutions Positive maximum number of distinct candidates requested.
 * @param collision_free Enable post-solve collision processing; requires a collision build.
 * @param problems_json_text MotionBenchMaker-style JSON, required with collision_free.
 * @param problem_set_name Key under the JSON "problems" object.
 * @param problem_idx Nonnegative index within the selected problem set.
 * @param write_stats Append diagnostics to ik_stats.csv; write failures throw.
 * @param collision_mode 0 = environment-only soft ranking, 1 = hard self/environment
 * filtering, 2 = both; -1 reads HJCD_CC_MODE (defaults to hard). Ignored in open-world solves.
 * @return Owning result. Candidates may be approximate, fewer than requested, or empty.
 * Collision checks describe configurations, never paths.
 * @throws std::invalid_argument For invalid scalar/pose values.
 * @throws std::runtime_error For checked CUDA or output failures. Scene errors also throw.
 * GRiD-generated model initialization retains its upstream abort-on-CUDA-error policy.
 */
template<typename T, typename RT = double>
Result<T> generate_ik_solutions(
    T* target_pose,
    const grid::robotModel<T>* d_robotModel,
    int b_size,
    int num_solutions = 1,
    bool collision_free = false,
    const char* problems_json_text = nullptr,
    const char* problem_set_name = nullptr,
    int problem_idx = 0,
    bool write_stats = false,
    int collision_mode = -1
);

/**
 * @brief Sample reachable world-frame poses from a seeded Halton joint sequence.
 *
 * T may be float or double. The result uses [x,y,z,qw,qx,qy,qz], meters and
 * scalar-first quaternions. Sampling respects joint limits but does not check collision.
 * Calls share the solver lock and use the calling thread's current CUDA device.
 * @param d_robotModel Device model matching the compiled robot and T, or nullptr for the cache.
 * @param num_configs Positive target count bounded by CUDA launch indexing.
 * @param seed Unsigned seed used to shift the Halton sequence.
 */
template<typename T>
std::vector<std::array<T, 7>> sample_random_target_poses(
    const grid::robotModel<T>* d_robotModel,
    int num_configs,
    std::uint64_t seed
);

/// Lazily initialize generated joint limits on the current device; normally called internally.
void init_joint_limits_from_grid();

/// Return the compiled actuated joint count without initializing CUDA.
extern "C" int grid_num_joints();
/// Return whether the compiled header includes collision geometry, without initializing CUDA.
extern "C" bool grid_has_collision();
