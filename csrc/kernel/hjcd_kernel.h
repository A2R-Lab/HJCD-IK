#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace grid {
    template<typename T> struct robotModel;
}

#if defined(HJCDIK_DIAGNOSTICS)
struct CollisionWorld;
// Private test build only: rows contain pose7, row-major 6xN Jacobian, free verdict.
std::vector<double> diagnostic_model(const std::vector<double>& q, bool fp64,
                                    const CollisionWorld* world, float margin, bool check_self);
#endif

/// Collision policy of one solve. `none` is an open-world solve. `hard` filters self and environment
/// collisions against the compiled sphere model; `soft` ranks candidates by environment penetration
/// only (no self-collision check, no filtering) and requires a non-empty world; `both` ranks and filters.
enum class CollisionMode : int { none = 0, hard = 1, soft = 2, both = 3 };

/**
 * @brief Per-solve diagnostics, collected only when requested. Unmeasured fields stay -1.
 *
 * Accuracy tallies use position error < 5 mm and orientation error < 1e-3 rad. In hard/both mode the
 * collision tallies include self and environment checks; in soft mode they count environment
 * penetration only. env_cost_* are refined-candidate penetration summaries in soft/both mode.
 */
struct SolveStats {
    int batch_size = -1;             ///< Coarse candidates (B).
    int n_refined = -1;              ///< LM-refined candidates (K x repeats).
    int n_ik_accurate = -1;          ///< Refined candidates inside the accuracy band.
    int n_coll_free_refined = -1;    ///< Refined candidates passing the selected collision check.
    int n_feasible = -1;             ///< Accurate and collision-free refined candidates.
    int n_ik_lost = -1;              ///< Accurate refined candidates rejected by the collision check.
    int n_coll_in_refined = -1;      ///< Refined candidates rejected by the collision check.
    int n_coll_in_coarse = -1;       ///< Coarse candidates rejected by the collision check.
    double env_cost_min_mm = -1;     ///< Minimum refined environment penetration cost (soft/both).
    double env_cost_max_mm = -1;     ///< Maximum refined environment penetration cost (soft/both).
    double env_cost_mean_mm = -1;    ///< Mean refined environment penetration cost (soft/both).
    int n_returned = -1;             ///< Returned candidates.
    int n_returned_ik_accurate = -1; ///< Returned candidates inside the accuracy band.
    int n_returned_coll_free = -1;   ///< Returned candidates passing the selected collision check.
    double pct_returned_coll_free = -1; ///< 100 * n_returned_coll_free / n_returned (0 when none returned).
    int n_returned_feasible = -1;    ///< Returned candidates accurate and collision-free.
};

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
    bool has_stats = false;     ///< True when stats was collected for this solve.
    SolveStats stats{};         ///< Diagnostics; meaningful only when has_stats.

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
            has_stats = other.has_stats;
            stats = other.stats;
            other.elapsed_time = T{};
            other.count = 0;
            other.has_stats = false;
            other.stats = SolveStats{};
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
        has_stats = false;
        stats = SolveStats{};
    }
};

/**
 * @brief Host description of an obstacle world in the robot base frame, metres, scalar-first quaternions.
 *
 * Row-major flat arrays: spheres rows are [cx cy cz r]; capsules rows are [x0 y0 z0 x1 y1 z1 r]
 * (segment endpoints and radius); boxes rows are [cx cy cz qw qx qy qz sx sy sz] with full side
 * lengths. Cylinders are not a native primitive: callers convert them to capsules first.
 */
struct WorldSpec {
    std::vector<float> spheres;   ///< 4 floats per sphere.
    std::vector<float> capsules;  ///< 7 floats per capsule.
    std::vector<float> boxes;     ///< 10 floats per box.
    int num_spheres() const { return static_cast<int>(spheres.size() / 4); }
    int num_capsules() const { return static_cast<int>(capsules.size() / 7); }
    int num_boxes() const { return static_cast<int>(boxes.size() / 10); }
    bool empty() const { return spheres.empty() && capsules.empty() && boxes.empty(); }
};

/**
 * @brief Per-solve knobs beyond the collision policy. Defaults reproduce the historical solver.
 *
 * `margin_m` is the clearance (metres) below which an environment pair counts as a hit;
 * self pairs retain zero margin when `check_self` is true. Zero margin is
 * today's contact-free rule. Both apply to `hard` and `both` only (invalid otherwise). `seed`
 * shifts the coarse search's random initial configurations (it changes which valid candidates are
 * found, never their validity). `q_init` (row-major `num_init` x N, radians/metres, clamped to joint
 * limits on the device) replaces the initial configurations of the first `num_init` coarse blocks;
 * `num_init` must not exceed the batch size.
 */
struct SolveOptions {
    float margin_m = 0.0f;
    bool check_self = true;
    std::uint32_t seed = 1337u;
    const double* q_init = nullptr;
    int num_init = 0;
};

/// Owning device copy of a WorldSpec, bound to the CUDA device current when it was made. Opaque.
struct CollisionWorld;

/**
 * @brief Validate and upload a world to the calling thread's current CUDA device.
 *
 * Requires a collision-enabled build. Values must be finite and within fp32 range; radii and side
 * lengths positive; quaternions non-zero (they are normalised). The returned object frees its device
 * memory on destruction, on the device it was made on, and may be shared between solves.
 * @throws std::invalid_argument For invalid geometry. @throws std::runtime_error Without collision
 * geometry in the build, or on CUDA failures.
 */
std::shared_ptr<CollisionWorld> make_collision_world(const WorldSpec& spec);
/// CUDA device the world lives on.
int collision_world_device(const CollisionWorld& world);
/// True when the world holds no primitive.
bool collision_world_empty(const CollisionWorld& world);
/// Primitive counts: spheres, capsules, boxes.
std::array<int, 3> collision_world_counts(const CollisionWorld& world);

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
 * Calls serialize shared native state on the calling thread's current CUDA device and use
 * internally cached, precision-specific generated models. Cached models require that context
 * to remain alive; cudaDeviceReset is unsupported.
 *
 * @param target_pose Nonnull seven-vector [x,y,z,qw,qx,qy,qz], meters and scalar-first
 * quaternion. Finite nonzero quaternions are normalized without mutating this input.
 * @param b_size Positive candidate count (not a target count), bounded by CUDA indexing.
 * @param num_solutions Positive maximum number of distinct candidates requested.
 * @param collision Collision policy; anything but none requires a collision-enabled build.
 * @param world Uploaded obstacle world on the current device, or nullptr for an empty environment.
 * hard/both with an empty world check self-collision only; soft with an empty world is rejected.
 * @param collect_stats Fill Result::stats (sets Result::has_stats). Never changes the solution.
 * @param options Margin, self-collision toggle, seed and warm starts; see SolveOptions.
 * @return Owning result. Candidates may be approximate, fewer than requested, or empty.
 * Collision checks describe configurations, never paths.
 * @throws std::invalid_argument For invalid scalar/pose values, a soft solve without obstacles, or
 * a world made on another device.
 * @throws std::runtime_error For checked CUDA failures or a collision request on an open-world build.
 * Generated model/limit initialization uses GRiD's checked APIs and raises on failure.
 */
template<typename T, typename RT = double>
Result<T> generate_ik_solutions(
    const T* target_pose,
    int b_size,
    int num_solutions = 1,
    CollisionMode collision = CollisionMode::none,
    const CollisionWorld* world = nullptr,
    bool collect_stats = false,
    const SolveOptions& options = SolveOptions{}
);

/**
 * @brief Sample reachable world-frame poses from a seeded Halton joint sequence.
 *
 * T may be float or double. The result uses [x,y,z,qw,qx,qy,qz], meters and
 * scalar-first quaternions. Sampling respects joint limits but does not check collision.
 * Calls share the solver lock and use the calling thread's current CUDA device.
 * @param num_configs Positive target count bounded by CUDA launch indexing.
 * @param seed Unsigned seed used to shift the Halton sequence.
 */
template<typename T>
std::vector<std::array<T, 7>> sample_random_target_poses(
    int num_configs,
    std::uint64_t seed
);

/// Lazily initialize generated joint limits on the current device; normally called internally.
void init_joint_limits_from_grid();

/// Return the compiled actuated joint count without initializing CUDA.
extern "C" int grid_num_joints();
/// Return whether the compiled header includes collision geometry, without initializing CUDA.
extern "C" bool grid_has_collision();
