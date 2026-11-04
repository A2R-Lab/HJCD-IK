// Python binding for a fixed-base TREE model artifact (humanoid-v1 M2b): metadata + batched target FK.
// Built only by hjcdik/_tree_build.py through grid_rbd.build_artifact; never part of the packaged module.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "tree/hjcd_tree_fk.h"
#include "tree/hjcd_tree_lm.h"
#include <cmath>
#include <stdexcept>
#include "hjcd_targets.cuh"

#ifndef HJCDIK_TREE_MODULE_NAME
#error "HJCDIK_TREE_MODULE_NAME must name the per-artifact module"
#endif
#ifndef HJCDIK_TREE_ABI
#define HJCDIK_TREE_ABI 7
#endif
#ifndef HJCDIK_ROBOT_NAME
#define HJCDIK_ROBOT_NAME ""
#endif
#ifndef HJCDIK_TARGETS_SHA256
#define HJCDIK_TARGETS_SHA256 ""
#endif

namespace py = pybind11;
using namespace hjcd_targets;

namespace {

int source_code(const std::string& source) {
    if (source == "metadata") return 0;
    if (source == "grid") return 1;
    throw std::invalid_argument("source must be 'metadata' or 'grid'");
}

template<typename T>
py::tuple poses(py::array_t<T, py::array::c_style | py::array::forcecast> q, int warps, const std::string& source,
                int repeats) {
    if (q.ndim() != 2 || q.shape(1) != NUM_JOINTS)
        throw std::invalid_argument("q must have shape (B, " + std::to_string(NUM_JOINTS) + ")");
    const int batch = int(q.shape(0));
    py::array_t<T> out({py::ssize_t(batch), py::ssize_t(NUM_TARGETS), py::ssize_t(4), py::ssize_t(4)});
    float ms = 0.f;
    {
        py::gil_scoped_release release;
        hjcd_tree::target_poses<T>(q.data(), batch, out.mutable_data(), warps, source_code(source), repeats, &ms);
    }
    return py::make_tuple(out, repeats > 0 ? py::object(py::float_(ms)) : py::object(py::none()));
}

template<typename T>
py::dict info(int warps) {
    const auto k = hjcd_tree::kernel_info<T>(warps);
    py::dict d;
    d["num_regs"] = k.num_regs; d["local_bytes"] = k.local_bytes; d["static_shared_bytes"] = k.static_shared_bytes;
    d["dynamic_shared_bytes"] = k.dynamic_shared_bytes; d["max_threads_per_block"] = k.max_threads_per_block;
    d["ptx_version"] = k.ptx_version; d["binary_version"] = k.binary_version;
    return d;
}

template<typename X, size_t M>
std::vector<X> vec(const X (&a)[M]) { return std::vector<X>(a, a + M); }


template<typename T>
using carr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<typename T>
struct ProblemArrays {
    carr<T> target, wp, wo, axis, twist, tool; carr<unsigned int> active; carr<int> mode;
    std::vector<T> tool_cm;
    hjcd_tree::ProblemIn<T> in{};
};

void need(bool ok, const std::string& what) { if (!ok) throw std::invalid_argument(what); }

// Problem arrays per configuration (B rows). tool: optional (K,4,4) ROW-major rigid transforms that
// REPLACE the model tools for this call; converted here to the device's column-major layout.
template<typename T>
ProblemArrays<T> problem(py::ssize_t B, carr<T> target, carr<unsigned int> active, carr<T> wp, carr<T> wo,
                         carr<int> mode, carr<T> axis, py::object twist, py::object tool) {
    ProblemArrays<T> P;
    const py::ssize_t K = NUM_TARGETS;
    need(target.ndim() == 3 && target.shape(0) == B && target.shape(1) == K && target.shape(2) == 7, "target must be (B,K,7)");
    need(active.ndim() == 1 && active.shape(0) == B, "active must be (B,)");
    need(wp.ndim() == 2 && wp.shape(0) == B && wp.shape(1) == K, "w_pos must be (B,K)");
    need(wo.ndim() == 2 && wo.shape(0) == B && wo.shape(1) == K, "w_ori must be (B,K)");
    need(mode.ndim() == 2 && mode.shape(0) == B && mode.shape(1) == K, "mode must be (B,K)");
    need(axis.ndim() == 3 && axis.shape(0) == B && axis.shape(1) == K && axis.shape(2) == 3, "axis must be (B,K,3)");
    P.target = target; P.active = active; P.wp = wp; P.wo = wo; P.mode = mode; P.axis = axis;
    if (twist.is_none()) P.twist = carr<T>({B, K, py::ssize_t(5)}), std::fill(P.twist.mutable_data(), P.twist.mutable_data() + B * K * 5, T(0));
    else P.twist = twist.cast<carr<T>>();
    need(P.twist.ndim() == 3 && P.twist.shape(0) == B && P.twist.shape(1) == K && P.twist.shape(2) == 5, "twist must be (B,K,5)");
    for (py::ssize_t i = 0; i < B * K; ++i) {
        need(mode.data()[i] >= 0 && mode.data()[i] <= 3, "mode must be 0..3");
        need(std::isfinite(double(wp.data()[i])) && wp.data()[i] >= 0 && std::isfinite(double(wo.data()[i])) && wo.data()[i] >= 0,
             "weights must be finite and >= 0");
    }
    for (py::ssize_t i = 0; i < B * K * 7; ++i) need(std::isfinite(double(target.data()[i])), "target must be finite");
    for (py::ssize_t i = 0; i < B * K * 3; ++i) need(std::isfinite(double(axis.data()[i])), "axis must be finite");
    P.in.target = P.target.data(); P.in.active = P.active.data(); P.in.w_pos = P.wp.data(); P.in.w_ori = P.wo.data();
    P.in.mode = P.mode.data(); P.in.axis = P.axis.data(); P.in.twist = P.twist.data(); P.in.tool = nullptr;
    if (!tool.is_none()) {
        P.tool = tool.cast<carr<T>>();
        const bool per_problem = P.tool.ndim() == 4;
        need((P.tool.ndim() == 3 && P.tool.shape(0) == K && P.tool.shape(1) == 4 && P.tool.shape(2) == 4)
             || (per_problem && P.tool.shape(0) == B && P.tool.shape(1) == K && P.tool.shape(2) == 4 && P.tool.shape(3) == 4),
             "tool must be (K,4,4) or (P,K,4,4) row-major");
        const py::ssize_t KT = per_problem ? B * K : K;
        P.tool_cm.resize(KT * 16);
        P.in.tool_per_problem = per_problem ? 1 : 0;
        for (py::ssize_t k = 0; k < KT; ++k) {
            const T* M = P.tool.data() + 16 * k;
            for (int i = 0; i < 16; ++i) need(std::isfinite(double(M[i])), "tool must be finite");
            need(M[12] == T(0) && M[13] == T(0) && M[14] == T(0) && M[15] == T(1), "tool must be homogeneous (last row 0 0 0 1)");
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                double d = 0; for (int r = 0; r < 3; ++r) d += double(M[4*r + i]) * double(M[4*r + j]);
                need(std::fabs(d - (i == j)) < (sizeof(T) == 4 ? 1e-5 : 1e-9), "tool rotation must be orthonormal");
            }
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) P.tool_cm[16 * k + 4 * c + r] = M[4 * r + c];
        }
        P.in.tool = P.tool_cm.data();
    }
    return P;
}

// ---- floating base (M4b) ----
// base poses: (B,7) or (P,S,7) [x,y,z, qw,qx,qy,qz], WORLD frame, WXYZ. Copied and normalised here (the input is
// never modified); non-finite values or a quaternion norm <= 1e-12 are rejected.
template<typename T>
std::vector<T> base_copy(py::object base, py::ssize_t B) {
    auto a = base.cast<carr<T>>();
    need(a.size() == B * 7 && a.shape(a.ndim() - 1) == 7, "base_poses must be (B,7) / (P,S,7) [x,y,z,qw,qx,qy,qz]");
    std::vector<T> out(a.data(), a.data() + B * 7);
    for (py::ssize_t c = 0; c < B; ++c) {
        T* r = out.data() + 7 * c;
        for (int i = 0; i < 7; ++i) need(std::isfinite(double(r[i])), "base_poses must be finite");
        const double n = std::sqrt(double(r[3])*r[3] + double(r[4])*r[4] + double(r[5])*r[5] + double(r[6])*r[6]);
        need(n > 1e-12, "base quaternion must be non-zero");
        for (int i = 3; i < 7; ++i) r[i] = T(double(r[i]) / n);
    }
    return out;
}

// base_update: None | dict with the bbc53a9 keys (unknown keys rejected); base_bounds: None | (lower[3], upper[3])
// call-wide box. Per-problem / per-candidate bounds are NOT supported and are rejected.
void apply_base_config(hjcd_tree::LMConfig& cfg, py::object base_update, py::object base_bounds) {
    if (!base_update.is_none()) {
        auto d = base_update.cast<py::dict>();
        static const char* keys[] = {"enabled", "interval", "damping", "scale_p", "scale_R", "step_scale",
                                     "max_translation_step", "max_rotation_step"};
        for (auto item : d) {
            const std::string k = py::str(item.first);
            bool ok = false; for (auto kk : keys) ok |= (k == kk);
            need(ok, "unknown base_update key '" + k + "'");
        }
        auto num = [&](const char* k, double def) { return d.contains(k) ? d[k].cast<double>() : def; };
        auto pos = [&](const char* k, double def, bool allow_none) {
            if (allow_none && d.contains(k) && d[k].is_none()) return 0.0;
            const double v = num(k, def);
            need(std::isfinite(v) && v > 0, std::string("base_update['") + k + "'] must be > 0 and finite");
            return v;
        };
        cfg.base_enabled = d.contains("enabled") ? int(d["enabled"].cast<bool>()) : 0;
        cfg.base_interval = d.contains("interval") ? d["interval"].cast<int>() : 1;
        need(cfg.base_interval >= 1, "base_update['interval'] must be >= 1");
        cfg.base_damping = num("damping", 1e-3);
        need(std::isfinite(cfg.base_damping) && cfg.base_damping >= 0, "base_update['damping'] must be >= 0 and finite");
        cfg.base_scale_p = pos("scale_p", 1.0, false); cfg.base_scale_R = pos("scale_R", 1.0, false);
        cfg.base_step_scale = pos("step_scale", 1.0, false);
        cfg.base_max_translation = pos("max_translation_step", 0.05, true);
        cfg.base_max_rotation = pos("max_rotation_step", 0.10, true);
    }
    if (!base_bounds.is_none()) {
        auto b = base_bounds.cast<carr<double>>();
        need(b.ndim() == 2 && b.shape(0) == 2 && b.shape(1) == 3,
             "base_bounds must be one call-wide (lower[3], upper[3]) pair; per-problem bounds are not supported");
        for (int i = 0; i < 3; ++i) {
            const double lo = b.data()[i], hi = b.data()[3 + i];
            need(std::isfinite(lo) && std::isfinite(hi) && lo <= hi, "base_bounds must be finite with lower <= upper");
            cfg.base_lo[i] = lo; cfg.base_hi[i] = hi;
        }
    }
}

template<typename T>
py::dict evaluate_py(carr<T> q, carr<T> target, carr<unsigned int> active, carr<T> wp, carr<T> wo, carr<int> mode,
                     carr<T> axis, py::object twist, py::object tool, double lambda, py::object base) {
    need(q.ndim() == 2 && q.shape(1) == NUM_JOINTS, "q must be (B,N)");
    const py::ssize_t B = q.shape(0), K = NUM_TARGETS, N = NUM_JOINTS;
    auto P = problem<T>(B, target, active, wp, wo, mode, axis, twist, tool);
    std::vector<T> bcopy;
    if (!base.is_none()) { bcopy = base_copy<T>(base, B); P.in.base = bcopy.data(); }
    carr<T> pose({B, K, py::ssize_t(4), py::ssize_t(4)}), e({B, K, py::ssize_t(6)}), pn({B, K}), on({B, K}),
            J({B, K * 6, N}), scale({B, K * 6}), A_raw({B, N, N}), b_raw({B, N}), A({B, N, N}), b({B, N}),
            colnorm2({B, N}), Ap({B, N, N}), bp({B, N}), dq({B, N});
    carr<unsigned int> pinned({B}); carr<int> fail({B});
    hjcd_tree::EvalOut<T> o{pose.mutable_data(), e.mutable_data(), pn.mutable_data(), on.mutable_data(), J.mutable_data(),
                            scale.mutable_data(), A_raw.mutable_data(), b_raw.mutable_data(), A.mutable_data(), b.mutable_data(),
                            colnorm2.mutable_data(), Ap.mutable_data(), bp.mutable_data(), dq.mutable_data(),
                            pinned.mutable_data(), fail.mutable_data()};
    { py::gil_scoped_release r; hjcd_tree::evaluate<T>(q.data(), int(B), P.in, T(lambda), o); }
    py::dict d;
    d["pose"] = pose; d["e"] = e; d["pn"] = pn; d["on"] = on; d["J"] = J; d["scale"] = scale; d["A_raw"] = A_raw;
    d["b_raw"] = b_raw; d["A"] = A; d["b"] = b; d["colnorm2"] = colnorm2; d["Ap"] = Ap; d["bp"] = bp; d["dq"] = dq;
    d["pinned"] = pinned; d["fail"] = fail;
    return d;
}

template<typename T>
py::dict base_step_py(carr<T> q, carr<T> target, carr<unsigned int> active, carr<T> wp, carr<T> wo, carr<int> mode,
                      carr<T> axis, py::object twist, py::object tool, py::object base, py::object base_update,
                      py::object base_bounds, double eps_pos, double eps_ori) {
    need(q.ndim() == 2 && q.shape(1) == NUM_JOINTS, "q must be (B,N)");
    const py::ssize_t B = q.shape(0);
    auto P = problem<T>(B, target, active, wp, wo, mode, axis, twist, tool);
    std::vector<T> bcopy = base_copy<T>(base, B); P.in.base = bcopy.data();
    hjcd_tree::LMConfig cfg{eps_pos, eps_ori, 5e-3, 1e-3, 0, 0, 1, 0, 0};
    apply_base_config(cfg, base_update, base_bounds);
    carr<T> out({B, py::ssize_t(65)});
    { py::gil_scoped_release r; hjcd_tree::base_step<T>(q.data(), int(B), P.in, cfg, out.mutable_data()); }
    py::dict d; d["raw"] = out; d["base_in"] = carr<T>({B, py::ssize_t(7)}, bcopy.data());
    return d;
}

template<typename T>
py::array_t<T> proposals_py(carr<T> q, carr<T> target, carr<unsigned int> active, carr<T> wp, carr<T> wo, carr<int> mode,
                            carr<T> axis, py::object twist, py::object tool, py::object base) {
    need(q.ndim() == 2 && q.shape(1) == NUM_JOINTS, "q must be (B,N)");
    const py::ssize_t B = q.shape(0);
    auto P = problem<T>(B, target, active, wp, wo, mode, axis, twist, tool);
    std::vector<T> bcopy;
    if (!base.is_none()) { bcopy = base_copy<T>(base, B); P.in.base = bcopy.data(); }
    carr<T> out({B, py::ssize_t(NUM_JOINTS), py::ssize_t(6)});
    hjcd_tree::CoarseConfig cc;
    { py::gil_scoped_release r; hjcd_tree::proposals<T>(q.data(), int(B), P.in, cc, out.mutable_data()); }
    return out;
}

template<typename T>
py::dict lm_py(carr<T> q0, carr<T> target, carr<unsigned int> active, carr<T> wp, carr<T> wo, carr<int> mode,
               carr<T> axis, py::object twist, py::object tool, double eps_pos, double eps_ori, double lambda_init,
               int k_max, int stag_patience, double stag_rel, int warps, int repeats, bool uniform_lambda) {
    need(q0.ndim() == 2 && q0.shape(1) == NUM_JOINTS, "q0 must be (B,N)");
    need(eps_pos > 0 && eps_ori > 0 && lambda_init > 0 && k_max >= 0, "tolerances, lambda_init must be > 0; k_max >= 0");
    const py::ssize_t B = q0.shape(0), K = NUM_TARGETS, N = NUM_JOINTS;
    auto P = problem<T>(B, target, active, wp, wo, mode, axis, twist, tool);
    carr<T> x({B, N}), pn0({B, K}), on0({B, K}), pn({B, K}), on({B, K}), cost({B}), ephys0({B}), ephys({B});
    py::array_t<unsigned char> success({B}); carr<int> iters({B}), acc({B}), sfail({B}), kicks({B});
    hjcd_tree::LMOut<T> o{x.mutable_data(), pn0.mutable_data(), on0.mutable_data(), pn.mutable_data(), on.mutable_data(),
                          cost.mutable_data(), ephys0.mutable_data(), ephys.mutable_data(), success.mutable_data(),
                          iters.mutable_data(), acc.mutable_data(), sfail.mutable_data(), kicks.mutable_data()};
    hjcd_tree::LMConfig cfg{eps_pos, eps_ori, lambda_init, stag_rel, k_max, stag_patience, warps, repeats, uniform_lambda ? 1 : 0};
    float ms = 0.f;
    { py::gil_scoped_release r; hjcd_tree::lm<T>(q0.data(), int(B), P.in, cfg, o, &ms); }
    py::dict d;
    d["x"] = x; d["pn0"] = pn0; d["on0"] = on0; d["pn"] = pn; d["on"] = on; d["cost"] = cost; d["ephys0"] = ephys0;
    d["ephys"] = ephys; d["success"] = success.attr("astype")("bool"); d["iters"] = iters; d["accepted"] = acc;
    d["solve_fail"] = sfail; d["kicks"] = kicks;
    if (repeats > 0) d["kernel_ms"] = ms;
    return d;
}

template<typename T>
py::dict lm_info(int warps) {
    const auto k = hjcd_tree::lm_kernel_info<T>(warps);
    py::dict d;
    d["num_regs"] = k.num_regs; d["local_bytes"] = k.local_bytes; d["static_shared_bytes"] = k.static_shared_bytes;
    d["dynamic_shared_bytes"] = k.dynamic_shared_bytes; d["max_threads_per_block"] = k.max_threads_per_block;
    return d;
}

template<typename T>
py::dict select_dict(hjcd_tree::SelectOut<T>& o, carr<T>& q, carr<T>& pe, carr<T>& oe, carr<T>& cost,
                     py::array_t<double>& ephys, carr<int>& seed, py::array_t<unsigned char>& exact,
                     py::array_t<unsigned char>& valid, carr<int>& ns, carr<int>& nv, carr<int>& nc) {
    py::dict d;
    d["joint_config"] = q; d["position_errors"] = pe; d["orientation_errors"] = oe; d["cost_lm"] = cost;
    d["cost_physical"] = ephys; d["selected_seed_ids"] = seed; d["success"] = exact.attr("astype")("bool");
    d["valid"] = valid.attr("astype")("bool"); d["num_solved"] = ns; d["num_valid"] = nv; d["num_converged"] = nc;
    d["problem_success"] = ns.attr("__gt__")(0);
    return d;
}

#define HJCD_SEL_ARRAYS(P_, M_) \
    carr<T> sq({P_, M_, py::ssize_t(NUM_JOINTS)}), spe({P_, M_, py::ssize_t(NUM_TARGETS)}), soe({P_, M_, py::ssize_t(NUM_TARGETS)}), \
            scost({P_, M_}); py::array_t<double> seph({P_, M_}); carr<int> sseed({P_, M_}); \
    py::array_t<unsigned char> sexact({P_, M_}), svalid({P_, M_}); carr<int> ns({P_}), nv({P_}), nc({P_}); \
    hjcd_tree::SelectOut<T> so{sq.mutable_data(), spe.mutable_data(), soe.mutable_data(), scost.mutable_data(), \
        seph.mutable_data(), sseed.mutable_data(), sexact.mutable_data(), svalid.mutable_data(), \
        ns.mutable_data(), nv.mutable_data(), nc.mutable_data()};

// ---- environment collision (M6a) ----
// Host copies that own the rows EnvCollision points at. environment: dict(spheres=(M,4), capsules=(M,7),
// cuboids=(M,10), planes=(M,4)) in the WORLD frame; robot_spheres (NS,5) = (anchor, ox, oy, oz, r).
template<typename T>
struct EnvHost {
    std::vector<T> robot, sph, cap, cub, pln;
    hjcd_tree::EnvCollision<T> ec;
};
inline int env_selection_code(const std::string& sel) {
    if (sel == "gate") return hjcd_tree::ENV_SELECT_GATE;
    if (sel == "rank") return hjcd_tree::ENV_SELECT_RANK;
    if (sel == "annotate") return hjcd_tree::ENV_SELECT_ANNOTATE;
    throw std::invalid_argument("environment_selection must be gate|rank|annotate");
}
template<typename T>
std::vector<T> env_rows(const py::dict& env, const char* key, py::ssize_t w) {
    if (!env.contains(key)) return {};
    auto a = env[key].cast<carr<T>>();
    if (a.size() == 0) return {};
    need(a.ndim() == 2 && a.shape(1) == w, std::string("environment['") + key + "'] must be (M," + std::to_string(w) + ")");
    return std::vector<T>(a.data(), a.data() + a.size());
}
template<typename T>
void build_env(EnvHost<T>& h, py::object robot_spheres, py::object environment, double margin, const std::string& selection) {
    need(!robot_spheres.is_none(), "environment collision needs robot_spheres (NS,5) = (anchor, ox, oy, oz, r)");
    auto r = robot_spheres.cast<carr<T>>();
    need(r.ndim() == 2 && r.shape(1) == 5 && r.shape(0) >= 1, "robot_spheres must be (NS,5), NS >= 1");
    h.robot.assign(r.data(), r.data() + r.size());
    py::dict env = environment.is_none() ? py::dict() : environment.cast<py::dict>();
    for (auto item : env) {
        const std::string k = py::str(item.first);
        need(k == "spheres" || k == "capsules" || k == "cuboids" || k == "planes",
             "environment keys must be spheres|capsules|cuboids|planes, got '" + k + "'");
    }
    h.sph = env_rows<T>(env, "spheres", 4); h.cap = env_rows<T>(env, "capsules", 7);
    h.cub = env_rows<T>(env, "cuboids", 10); h.pln = env_rows<T>(env, "planes", 4);
    h.ec.spheres_robot = h.robot.data(); h.ec.n_robot = int(r.shape(0));
    h.ec.spheres = h.sph.data(); h.ec.n_spheres = int(h.sph.size() / 4);
    h.ec.capsules = h.cap.data(); h.ec.n_capsules = int(h.cap.size() / 7);
    h.ec.cuboids = h.cub.data(); h.ec.n_cuboids = int(h.cub.size() / 10);
    h.ec.planes = h.pln.data(); h.ec.n_planes = int(h.pln.size() / 4);
    need(std::isfinite(margin) && margin >= 0.0, "environment_margin must be finite and >= 0");
    h.ec.margin = margin; h.ec.selection = env_selection_code(selection);
}

template<typename T>
py::dict env_collision_py(carr<T> q, py::object robot_spheres, py::object environment, py::object base, double margin,
                          bool per_sphere) {
    need(q.ndim() == 2 && q.shape(1) == NUM_JOINTS, "q must be (B,N)");
    const py::ssize_t B = q.shape(0);
    need(B >= 1, "need B >= 1");
    EnvHost<T> h; build_env<T>(h, robot_spheres, environment, margin, "gate");
    std::vector<T> bcopy;
    if (!base.is_none()) {   // validated (finite, non-zero quaternion) but NOT normalised here: the kernel normalises,
        auto a = base.cast<carr<T>>();   // exactly as on the solve_problems path, so both see bit-identical bases
        need(a.size() == B * 7 && a.shape(a.ndim() - 1) == 7, "base must be (B,7) [x,y,z,qw,qx,qy,qz]");
        bcopy.assign(a.data(), a.data() + B * 7);
        for (py::ssize_t c = 0; c < B; ++c) {
            const T* r = bcopy.data() + 7 * c;
            for (int i = 0; i < 7; ++i) need(std::isfinite(double(r[i])), "base must be finite");
            need(double(r[3])*r[3] + double(r[4])*r[4] + double(r[5])*r[5] + double(r[6])*r[6] > 1e-24, "base quaternion must be non-zero");
        }
    }
    const py::ssize_t NS = h.ec.n_robot;
    py::array_t<unsigned char> st({B}); carr<T> mc({B}); carr<int> am({B});
    carr<T> sw({per_sphere ? B : py::ssize_t(1), NS, py::ssize_t(3)}), sc({per_sphere ? B : py::ssize_t(1), NS});
    hjcd_tree::EnvOut<T> o{};
    o.status = st.mutable_data(); o.min_clearance = mc.mutable_data(); o.argmin_sphere = am.mutable_data();
    if (per_sphere) { o.sphere_world = sw.mutable_data(); o.sphere_clearance = sc.mutable_data(); }
    { py::gil_scoped_release g; hjcd_tree::env_collision<T>(q.data(), bcopy.empty() ? nullptr : bcopy.data(), int(B), h.ec, o); }
    py::dict d;
    d["status"] = st; d["min_clearance"] = mc; d["argmin_sphere"] = am;
    d["collision_free"] = st.attr("__eq__")(0); d["colliding"] = st.attr("__eq__")(1); d["evaluated"] = st.attr("__ne__")(2);
    if (per_sphere) { d["sphere_world"] = sw; d["sphere_clearance"] = sc; }
    d["margin"] = margin; d["num_robot_spheres"] = NS;
    d["num_obstacles"] = h.ec.n_spheres + h.ec.n_capsules + h.ec.n_cuboids + h.ec.n_planes;
    py::dict t; t["prep_ms"] = o.prep_ms; t["upload_ms"] = o.upload_ms; t["kernel_ms"] = o.kernel_ms; d["timing"] = t;
    return d;
}

// ---- self-collision (M6b) ----
template<typename T>
struct SelfHost { std::vector<T> robot; std::vector<int> pairs; hjcd_tree::SelfCollision<T> sc; };
template<typename T>
void build_self(SelfHost<T>& h, py::object robot_spheres, py::object pairs, double margin, const std::string& selection) {
    need(!robot_spheres.is_none(), "self-collision needs robot_spheres (NS,5) = (anchor, ox, oy, oz, r)");
    need(!pairs.is_none(), "self-collision needs self_pairs (NP,2): checked body pairs (hjcdik._tree_collision.self_collision_pairs)");
    auto r = robot_spheres.cast<carr<T>>();
    need(r.ndim() == 2 && r.shape(1) == 5 && r.shape(0) >= 1, "robot_spheres must be (NS,5), NS >= 1");
    h.robot.assign(r.data(), r.data() + r.size());
    auto pr = pairs.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    need(pr.size() == 0 || (pr.ndim() == 2 && pr.shape(1) == 2), "self_pairs must be (NP,2) body anchors");
    h.pairs.assign(pr.data(), pr.data() + pr.size());
    need(std::isfinite(margin) && margin >= 0.0, "self_margin must be finite and >= 0");
    h.sc.spheres_robot = h.robot.data(); h.sc.n_robot = int(r.shape(0));
    h.sc.pairs = h.pairs.data(); h.sc.n_pairs = int(h.pairs.size() / 2);
    h.sc.margin = margin; h.sc.selection = env_selection_code(selection);
}
template<typename T>
carr<int> pair_bodies(const SelfHost<T>& h, const int* pi, const int* pj, py::ssize_t B) {   // closest pair -> anchors
    carr<int> o({B, py::ssize_t(2)});
    for (py::ssize_t b = 0; b < B; ++b) {
        o.mutable_data()[2*b] = pi[b] < 0 ? -2 : int(h.robot[5 * size_t(pi[b])]);
        o.mutable_data()[2*b + 1] = pj[b] < 0 ? -2 : int(h.robot[5 * size_t(pj[b])]);
    }
    return o;
}
template<typename T>
py::dict self_collision_py(carr<T> q, py::object robot_spheres, py::object pairs, double margin, py::object base) {
    need(q.ndim() == 2 && q.shape(1) == NUM_JOINTS, "q must be (B,N)");
    const py::ssize_t B = q.shape(0);
    need(B >= 1, "need B >= 1");
    if (!base.is_none()) {   // accepted for API symmetry; validated, never used (a rigid base cannot change self-collision)
        auto a = base.cast<carr<T>>();
        need(a.size() == B * 7 && a.shape(a.ndim() - 1) == 7, "base must be (B,7)");
    }
    SelfHost<T> h; build_self<T>(h, robot_spheres, pairs, margin, "annotate");
    py::array_t<unsigned char> st({B}); carr<T> mc({B}); carr<int> pi({B}), pj({B}); py::array_t<long long> ne({B});
    hjcd_tree::SelfOut<T> o{};
    o.status = st.mutable_data(); o.min_clearance = mc.mutable_data(); o.pair_i = pi.mutable_data(); o.pair_j = pj.mutable_data();
    o.evaluated_pairs = ne.mutable_data();
    { py::gil_scoped_release g; hjcd_tree::self_collision<T>(q.data(), int(B), h.sc, o); }
    py::dict d;
    d["status"] = st; d["min_clearance"] = mc; d["pair_i"] = pi; d["pair_j"] = pj;
    d["body_pair"] = pair_bodies<T>(h, pi.data(), pj.data(), B); d["evaluated_pairs"] = ne;
    d["collision_free"] = st.attr("__eq__")(0); d["evaluated"] = st.attr("__ne__")(2);
    d["eligible_pairs"] = o.eligible_pairs; d["num_bodies"] = o.n_bodies; d["num_body_pairs"] = o.n_body_pairs; d["margin"] = margin;
    py::dict t; t["prep_ms"] = o.prep_ms; t["upload_ms"] = o.upload_ms; t["kernel_ms"] = o.kernel_ms; d["timing"] = t;
    return d;
}

// Batched fixed-base solve: seeds (P,S,N); problem arrays (P,...); tool (K,4,4) shared or (P,K,4,4).
template<typename T>
py::dict solve_problems_py(carr<T> seeds, carr<T> target, carr<unsigned int> active, carr<T> wp, carr<T> wo,
                           carr<int> mode, carr<T> axis, py::object twist, py::object tool, int num_solutions,
                           py::object candidate_ids, double eps_pos, double eps_ori, double lambda_init, int k_max,
                           int stag_patience, double stag_rel, int warps, bool uniform_lambda, bool return_all,
                           py::object base_poses, py::object base_update, py::object base_bounds,
                           const std::string& coarse_mode, int coarse_iters, unsigned long long seed, py::object problem_seeds,
                           const std::string& environment_collision, py::object environment, py::object robot_spheres,
                           double environment_margin, const std::string& environment_selection,
                           const std::string& self_collision, py::object self_pairs, double self_margin,
                           const std::string& self_selection) {
    need(seeds.ndim() == 3 && seeds.shape(2) == NUM_JOINTS, "seed_configs must be (P,S,N)");
    const py::ssize_t P_ = seeds.shape(0), S_ = seeds.shape(1), M_ = num_solutions, K_ = NUM_TARGETS, N_ = NUM_JOINTS;
    need(P_ >= 1 && S_ >= 1, "need P >= 1 and S >= 1");
    need(M_ >= 1 && M_ <= S_, "num_solutions must be in [1, S]");
    need(eps_pos > 0 && eps_ori > 0 && lambda_init > 0 && k_max >= 0, "tolerances, lambda_init must be > 0; k_max >= 0");
    for (py::ssize_t i = 0; i < P_ * S_ * N_; ++i) need(std::isfinite(double(seeds.data()[i])), "seed_configs must be finite");
    for (py::ssize_t p = 0; p < P_; ++p) need(active.ndim() == 1 && active.shape(0) == P_ && active.data()[p] != 0u
                                              && (active.data()[p] >> K_) == 0u, "active masks must be (P,), non-empty, < 2^K");
    auto Pa = problem<T>(P_, target, active, wp, wo, mode, axis, twist, tool);
    std::vector<T> bcopy;
    const bool floating = !base_poses.is_none();
    need(floating || (base_update.is_none() && base_bounds.is_none()), "base_update / base_bounds require base_poses");
    if (floating) {
        auto bp = base_poses.cast<carr<T>>();
        need(bp.ndim() == 3 && bp.shape(0) == P_ && bp.shape(1) == S_ && bp.shape(2) == 7, "base_poses must be (P,S,7)");
        bcopy = base_copy<T>(base_poses, P_ * S_); Pa.in.base = bcopy.data();
    }
    carr<long long> cid;
    if (!candidate_ids.is_none()) {
        cid = candidate_ids.cast<carr<long long>>();
        need(cid.ndim() == 2 && cid.shape(0) == P_ && cid.shape(1) == S_, "candidate_ids must be (P,S)");
    }
    const py::ssize_t B = P_ * S_;
    carr<T> x({P_, S_, N_}), pn0({P_, S_, K_}), on0({P_, S_, K_}), pn({P_, S_, K_}), on({P_, S_, K_}), cost({P_, S_}),
            eph0({P_, S_}), eph({P_, S_});
    py::array_t<unsigned char> succ({P_, S_}); carr<int> iters({P_, S_}), acc({P_, S_}), sfail({P_, S_}), kicks({P_, S_});
    hjcd_tree::LMOut<T> raw{x.mutable_data(), pn0.mutable_data(), on0.mutable_data(), pn.mutable_data(), on.mutable_data(),
                            cost.mutable_data(), eph0.mutable_data(), eph.mutable_data(), succ.mutable_data(),
                            iters.mutable_data(), acc.mutable_data(), sfail.mutable_data(), kicks.mutable_data()};
    HJCD_SEL_ARRAYS(P_, M_)
    hjcd_tree::LMConfig cfg{eps_pos, eps_ori, lambda_init, stag_rel, k_max, stag_patience, warps, 0, uniform_lambda ? 1 : 0};
    apply_base_config(cfg, base_update, base_bounds);
    carr<T> all_base({P_, S_, py::ssize_t(7)}), sel_base({P_, M_, py::ssize_t(7)}); carr<int> bdiag({P_, S_, py::ssize_t(4)});
    if (floating) { raw.base = all_base.mutable_data(); raw.base_diag = bdiag.mutable_data(); so.base = sel_base.mutable_data(); }
    // coarse (M5): "none" (default here) | "auto" (bbc53a9 public default: popcount(mask) >= 2) | "multi_target"
    need(coarse_mode == "none" || coarse_mode == "auto" || coarse_mode == "multi_target",
         "coarse_mode must be none|auto|multi_target");
    const bool run_coarse = coarse_mode != "none" && coarse_iters > 0;
    std::vector<unsigned char> use_c(P_ * S_, 0);
    std::vector<unsigned int> cid32, pseeds;
    carr<T> cx({P_, S_, N_}), c0({P_, S_}), c1({P_, S_}), cpn({P_, S_, K_}), con({P_, S_, K_});
    carr<int> cit({P_, S_}), ckick({P_, S_}); py::array_t<unsigned char> csucc({P_, S_});
    hjcd_tree::CoarseConfig ccfg; ccfg.k_max = coarse_iters; ccfg.seed = seed;
    hjcd_tree::CoarseIO<T> cio{};
    if (run_coarse) {
        for (py::ssize_t p = 0; p < P_; ++p) {
            const int pc = __builtin_popcount(active.data()[p]);
            for (py::ssize_t s = 0; s < S_; ++s) use_c[p * S_ + s] = (coarse_mode == "multi_target" || pc >= 2) ? 1 : 0;
        }
        if (cid.size()) { cid32.resize(P_ * S_); for (py::ssize_t i = 0; i < P_ * S_; ++i) cid32[i] = (unsigned int)cid.data()[i]; }
        if (!problem_seeds.is_none()) {
            auto ps = problem_seeds.cast<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>>();
            need(ps.ndim() == 1 && ps.shape(0) == P_, "problem_seeds must be (P,)");
            pseeds.assign(ps.data(), ps.data() + P_);
        }
        cio.use_coarse = use_c.data(); cio.problem_seeds = pseeds.empty() ? nullptr : pseeds.data();
        cio.candidate_ids = cid32.empty() ? nullptr : cid32.data();
        cio.x = cx.mutable_data(); cio.cost0 = c0.mutable_data(); cio.cost1 = c1.mutable_data(); cio.pn = cpn.mutable_data();
        cio.on = con.mutable_data(); cio.iters = cit.mutable_data(); cio.kicks = ckick.mutable_data(); cio.success = csucc.mutable_data();
    } else {
        need(problem_seeds.is_none(), "problem_seeds only affect the coarse stage; coarse_mode is none");
    }
    // environment collision (M6a): "disabled" (default; M5 path, no env arguments accepted) | "final" (post-LM check
    // of the final (joints, base) pair, then environment-aware selection). Nothing else is implemented.
    need(environment_collision == "disabled" || environment_collision == "final",
         "environment_collision must be 'disabled' or 'final' (in-loop / hard / coarse-gated modes are not implemented)");
    const bool env_on = environment_collision == "final";
    if (!env_on) need(environment.is_none() && environment_margin == 0.0 && environment_selection == "gate",
                      "environment / environment_margin / environment_selection require environment_collision='final'");
    // self-collision (M6b): "disabled" (default) | "final" (post-LM, independent channel; default selection annotate)
    need(self_collision == "disabled" || self_collision == "final",
         "self_collision must be 'disabled' or 'final' (in-loop / hard modes are not implemented)");
    const bool self_on = self_collision == "final";
    if (!self_on) need(self_pairs.is_none() && self_margin == 0.0 && self_selection == "annotate",
                       "self_pairs / self_margin / self_selection require self_collision='final'");
    need(env_on || self_on || robot_spheres.is_none(), "robot_spheres requires environment_collision or self_collision = 'final'");
    SelfHost<T> sh;
    py::array_t<unsigned char> xst({P_, S_}), xsst({P_, M_}); carr<T> xmin({P_, S_}), xsmin({P_, M_}); carr<int> xpi({P_, S_}), xpj({P_, S_});
    py::array_t<long long> xne({P_, S_}); carr<int> xfree({P_}), xcoll({P_}), xinv({P_}), xsf({P_});
    hjcd_tree::SelfOut<T> xo{};
    if (self_on) {
        build_self<T>(sh, robot_spheres, self_pairs, self_margin, self_selection);
        xo.status = xst.mutable_data(); xo.min_clearance = xmin.mutable_data(); xo.pair_i = xpi.mutable_data(); xo.pair_j = xpj.mutable_data();
        xo.evaluated_pairs = xne.mutable_data(); xo.num_free = xfree.mutable_data(); xo.num_colliding = xcoll.mutable_data();
        xo.num_invalid = xinv.mutable_data(); xo.num_solved_free = xsf.mutable_data(); xo.sel_status = xsst.mutable_data();
        xo.sel_min_clearance = xsmin.mutable_data();
    }
    EnvHost<T> eh;
    py::array_t<unsigned char> est({P_, S_}), sst({P_, M_}); carr<T> emin({P_, S_}), smin({P_, M_}); carr<int> earg({P_, S_});
    carr<int> nfree({P_}), ncoll({P_}), ninv({P_}), nsf({P_});
    hjcd_tree::EnvOut<T> eo{};
    if (env_on) {
        build_env<T>(eh, robot_spheres, environment, environment_margin, environment_selection);
        eo.status = est.mutable_data(); eo.min_clearance = emin.mutable_data(); eo.argmin_sphere = earg.mutable_data();
        eo.num_free = nfree.mutable_data(); eo.num_colliding = ncoll.mutable_data(); eo.num_invalid = ninv.mutable_data();
        eo.num_solved_free = nsf.mutable_data(); eo.sel_status = sst.mutable_data(); eo.sel_min_clearance = smin.mutable_data();
    }
    hjcd_tree::BatchTiming tm{};
    { py::gil_scoped_release r; hjcd_tree::solve_batch<T>(seeds.data(), int(P_), int(S_), int(M_), Pa.in, cfg, &raw, so, &tm,
                                                          run_coarse ? &ccfg : nullptr, run_coarse ? &cio : nullptr,
                                                          env_on ? &eh.ec : nullptr, env_on ? &eo : nullptr,
                                                          self_on ? &sh.sc : nullptr, self_on ? &xo : nullptr); }
    py::dict d = select_dict<T>(so, sq, spe, soe, scost, seph, sseed, sexact, svalid, ns, nv, nc);
    // gather per-slot diagnostics and identities on the host (pads: -1 / false / 0)
    carr<long long> scid({P_, M_}); carr<int> sit({P_, M_}), sfl({P_, M_}); py::array_t<bool> slm({P_, M_});
    for (py::ssize_t p = 0; p < P_; ++p) for (py::ssize_t m = 0; m < M_; ++m) {
        const int s = sseed.data()[p * M_ + m];
        const py::ssize_t b = p * S_ + s;
        scid.mutable_data()[p * M_ + m] = s < 0 ? -1 : (cid.size() ? cid.data()[b] : (long long)s);
        sit.mutable_data()[p * M_ + m] = s < 0 ? 0 : iters.data()[b];
        sfl.mutable_data()[p * M_ + m] = s < 0 ? 0 : sfail.data()[b];
        slm.mutable_data()[p * M_ + m] = s < 0 ? false : bool(succ.data()[b]);
    }
    d["selected_candidate_ids"] = scid; d["selected_iterations"] = sit; d["selected_solve_failures"] = sfl;
    d["selected_lm_converged"] = slm; d["num_problems"] = P_; d["seeds_per_problem"] = S_;
    d["precision"] = sizeof(T) == 4 ? "float32" : "float64"; d["uniform_lambda"] = uniform_lambda;
    py::dict t; t["model_init_ms"] = tm.init_ms; t["upload_ms"] = tm.upload_ms; t["lm_kernel_ms"] = tm.lm_ms;
    t["select_kernel_ms"] = tm.select_ms; t["coarse_kernel_ms"] = tm.coarse_ms; t["download_ms"] = tm.download_ms; t["native_total_ms"] = tm.total_ms;
    d["timing"] = t;
    d["floating_base"] = floating; d["base_update_enabled"] = bool(cfg.base_enabled);
    d["coarse_mode"] = coarse_mode; d["coarse_ran"] = run_coarse;
    d["environment_collision"] = environment_collision; d["environment_evaluated"] = env_on;
    if (env_on) {
        // success / num_solved / problem_success keep their M5 meaning (contact-exactness only); the environment
        // verdict is reported beside them, never folded in.
        d["environment_selection"] = environment_selection; d["environment_margin"] = environment_margin;
        d["environment_source"] = "grid_collision robot spheres, final joints + final base, post-LM";
        d["selected_environment_status"] = sst; d["selected_environment_min_clearance"] = smin;
        d["selected_environment_free"] = sst.attr("__eq__")(0);
        d["num_environment_free"] = nfree; d["num_environment_colliding"] = ncoll; d["num_environment_invalid"] = ninv;
        d["num_solved_environment_free"] = nsf; d["problem_success_environment_free"] = nsf.attr("__gt__")(0);
        t["environment_prep_ms"] = eo.prep_ms; t["environment_upload_ms"] = eo.upload_ms; t["environment_kernel_ms"] = eo.kernel_ms;
        if (return_all) { d["all_environment_status"] = est; d["all_environment_min_clearance"] = emin; d["all_environment_argmin_sphere"] = earg; }
    }
    d["self_collision"] = self_collision; d["self_evaluated"] = self_on;
    py::list chans; if (env_on) chans.append("environment"); if (self_on) chans.append("self");
    d["collision_channels"] = chans;   // which checks actually ran; *_collision_free below = free in EVERY listed channel
    if (self_on) {
        d["self_selection"] = self_selection; d["self_margin"] = self_margin;
        d["self_source"] = "grid_collision robot spheres, base frame, final joints, post-LM; pairs = caller's body-pair policy";
        d["self_eligible_pairs"] = xo.eligible_pairs; d["self_num_bodies"] = xo.n_bodies; d["self_num_body_pairs"] = xo.n_body_pairs;
        d["selected_self_status"] = xsst; d["selected_self_min_clearance"] = xsmin; d["selected_self_free"] = xsst.attr("__eq__")(0);
        d["num_self_free"] = xfree; d["num_self_colliding"] = xcoll; d["num_self_invalid"] = xinv;
        d["num_solved_collision_free"] = xsf; d["problem_success_collision_free"] = xsf.attr("__gt__")(0);
        t["self_prep_ms"] = xo.prep_ms; t["self_upload_ms"] = xo.upload_ms; t["self_kernel_ms"] = xo.kernel_ms;
        if (return_all) {
            d["all_self_status"] = xst; d["all_self_min_clearance"] = xmin; d["all_self_evaluated_pairs"] = xne;
            carr<int> sp({P_, S_, py::ssize_t(2)});
            for (py::ssize_t i = 0; i < P_ * S_; ++i) { sp.mutable_data()[2*i] = xpi.data()[i]; sp.mutable_data()[2*i + 1] = xpj.data()[i]; }
            d["all_self_pair"] = sp; d["all_self_body_pair"] = pair_bodies<T>(sh, xpi.data(), xpj.data(), P_ * S_).attr("reshape")(P_, S_, 2);
        }
    }
    if (env_on || self_on) {   // combined verdict over the channels that ran (never claims an unevaluated channel)
        py::object sel_free = env_on ? sst.attr("__eq__")(0) : xsst.attr("__eq__")(0);
        py::object all_free = env_on ? est.attr("__eq__")(0) : xst.attr("__eq__")(0);
        if (env_on && self_on) { sel_free = sel_free.attr("__and__")(xsst.attr("__eq__")(0)); all_free = all_free.attr("__and__")(xst.attr("__eq__")(0)); }
        d["selected_collision_free"] = sel_free;
        if (return_all) d["all_collision_free"] = all_free;   // (P,S) arrays are always computed; exposed with return_all
    }
    if (run_coarse && return_all) {
        d["all_use_coarse"] = py::array_t<bool>(py::array_t<unsigned char>({P_, S_}, use_c.data()).attr("astype")("bool"));
        d["all_coarse_joint_config"] = cx; d["all_coarse_cost_initial"] = c0; d["all_coarse_cost_final"] = c1;
        d["all_coarse_position_errors"] = cpn; d["all_coarse_orientation_errors"] = con; d["all_coarse_iterations"] = cit;
        d["all_coarse_kicks"] = ckick; d["all_coarse_success"] = csucc.attr("astype")("bool");
    }
    if (floating) {
        py::slice p3(0, 3, 1), q4(3, 7, 1);
        d["base_position"] = sel_base[py::make_tuple(py::ellipsis(), p3)];
        d["base_quaternion"] = sel_base[py::make_tuple(py::ellipsis(), q4)];
        carr<int> sb({P_, M_, py::ssize_t(4)});
        for (py::ssize_t p = 0; p < P_; ++p) for (py::ssize_t m = 0; m < M_; ++m) {
            const int sd = sseed.data()[p * M_ + m];
            for (int i = 0; i < 4; ++i) sb.mutable_data()[(p * M_ + m) * 4 + i] = sd < 0 ? 0 : bdiag.data()[((p * S_) + sd) * 4 + i];
        }
        d["selected_base_updates"] = sb;   // (attempted, accepted, numerical failures, rejected)
        if (return_all) {
            d["all_base_position"] = all_base[py::make_tuple(py::ellipsis(), p3)];
            d["all_base_quaternion"] = all_base[py::make_tuple(py::ellipsis(), q4)];
            d["all_base_updates"] = bdiag;
            d["all_base_in"] = carr<T>({P_, S_, py::ssize_t(7)}, bcopy.data());
        }
    }
    if (return_all) {
        d["all_joint_config"] = x; d["all_position_errors"] = pn; d["all_orientation_errors"] = on; d["all_cost_lm"] = cost;
        d["all_lm_converged"] = succ.attr("astype")("bool"); d["all_iterations"] = iters; d["all_solve_failures"] = sfail;
        d["all_kicks"] = kicks; d["all_accepted"] = acc; d["all_initial_position_errors"] = pn0;
        d["all_initial_orientation_errors"] = on0; d["all_cost_physical"] = eph;
    }
    return d;
}

template<typename T>
py::dict select_py(carr<T> q, carr<T> pe, carr<T> oe, carr<T> cost, py::array_t<unsigned char, py::array::c_style | py::array::forcecast> succ,
                   carr<T> seeds, carr<unsigned int> active, int M, double eps_pos, double eps_ori, py::object env_status,
                   py::object env_min_clearance, const std::string& env_selection, py::object self_status,
                   const std::string& self_selection) {
    need(q.ndim() == 3 && q.shape(2) == NUM_JOINTS, "q must be (P,S,N)");
    const py::ssize_t P_ = q.shape(0), S_ = q.shape(1), M_ = M;
    need(pe.size() == P_ * S_ * NUM_TARGETS && oe.size() == P_ * S_ * NUM_TARGETS && cost.size() == P_ * S_
         && succ.size() == P_ * S_ && seeds.size() == q.size() && active.size() == P_, "inconsistent shapes");
    need(M_ >= 1 && M_ <= S_, "M must be in [1, S]");
    HJCD_SEL_ARRAYS(P_, M_)
    const int esel = env_selection_code(env_selection);
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> est; carr<T> emc;
    py::array_t<unsigned char> sst({P_, M_}); carr<int> nfree({P_}), ncoll({P_}), ninv({P_}), nsf({P_});
    hjcd_tree::EnvOut<T> eo{};
    const bool env_on = !env_status.is_none();
    const int xsel = env_selection_code(self_selection);
    const bool self_on = !self_status.is_none();
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> xs;
    py::array_t<unsigned char> xsst({P_, M_}); carr<int> xfree({P_}), xcoll({P_}), xinv({P_}), xsf({P_});
    hjcd_tree::SelfOut<T> xo{};
    if (self_on) {
        xs = self_status.cast<py::array_t<unsigned char, py::array::c_style | py::array::forcecast>>();
        need(xs.size() == P_ * S_, "self_status must be (P,S)");
        for (py::ssize_t i = 0; i < xs.size(); ++i) need(xs.data()[i] <= 2, "self_status values must be 0 free, 1 colliding, 2 invalid");
        xo.num_free = xfree.mutable_data(); xo.num_colliding = xcoll.mutable_data(); xo.num_invalid = xinv.mutable_data();
        xo.num_solved_free = xsf.mutable_data(); xo.sel_status = xsst.mutable_data();
    }
    if (env_on) {
        est = env_status.cast<py::array_t<unsigned char, py::array::c_style | py::array::forcecast>>();
        need(est.size() == P_ * S_, "env_status must be (P,S)");
        for (py::ssize_t i = 0; i < est.size(); ++i) need(est.data()[i] <= 2, "env_status values must be 0 free, 1 colliding, 2 invalid");
        if (!env_min_clearance.is_none()) { emc = env_min_clearance.cast<carr<T>>(); need(emc.size() == P_ * S_, "env_min_clearance must be (P,S)"); }
        eo.num_free = nfree.mutable_data(); eo.num_colliding = ncoll.mutable_data(); eo.num_invalid = ninv.mutable_data();
        eo.num_solved_free = nsf.mutable_data(); eo.sel_status = sst.mutable_data();
    }
    { py::gil_scoped_release r; hjcd_tree::select_only<T>(q.data(), pe.data(), oe.data(), cost.data(), succ.data(), seeds.data(),
                                                          active.data(), int(P_), int(S_), int(M_), eps_pos, eps_ori, so,
                                                          env_on ? est.data() : nullptr, (env_on && emc.size()) ? emc.data() : nullptr,
                                                          esel, env_on ? &eo : nullptr,
                                                          self_on ? xs.data() : nullptr, xsel, self_on ? &xo : nullptr); }
    py::dict d = select_dict<T>(so, sq, spe, soe, scost, seph, sseed, sexact, svalid, ns, nv, nc);
    if (self_on) { d["selected_self_status"] = xsst; d["num_self_free"] = xfree; d["num_self_colliding"] = xcoll;
                   d["num_self_invalid"] = xinv; d["num_solved_collision_free"] = xsf; }
    if (env_on) { d["selected_environment_status"] = sst; d["num_environment_free"] = nfree; d["num_environment_colliding"] = ncoll;
                  d["num_environment_invalid"] = ninv; d["num_solved_environment_free"] = nsf; }
    return d;
}

}  // namespace

PYBIND11_MODULE(HJCDIK_TREE_MODULE_NAME, m) {
    m.doc() = "HJCD-IK fixed-base tree model: target metadata and batched base-frame target FK (no IK)";
    m.attr("TREE_ABI") = HJCDIK_TREE_ABI;
    m.attr("ROBOT") = HJCDIK_ROBOT_NAME;
    m.attr("MODEL_SHA256") = MODEL_SHA256;
    m.attr("TARGETS_SHA256") = HJCDIK_TARGETS_SHA256;
    m.attr("NUM_JOINTS") = NUM_JOINTS;
    m.attr("NUM_TARGETS") = NUM_TARGETS;
    std::vector<std::string> names(TARGET_NAMES, TARGET_NAMES + NUM_TARGETS), frames(TARGET_FRAMES, TARGET_FRAMES + NUM_TARGETS);
    m.attr("TARGET_NAMES") = py::tuple(py::cast(names));
    m.attr("TARGET_FRAMES") = py::tuple(py::cast(frames));
    m.attr("TARGET_ANCHOR_JID") = py::tuple(py::cast(vec(H_TARGET_ANCHOR_JID)));
    m.attr("TARGET_FIXED_FRAME_IDX") = py::tuple(py::cast(vec(H_TARGET_FIXED_FRAME_IDX)));
    m.attr("TARGET_ANCESTOR_MASK") = py::tuple(py::cast(vec(H_TARGET_ANCESTOR_MASK)));
    m.attr("JOINT_TARGET_MASK") = py::tuple(py::cast(vec(H_JOINT_TARGET_MASK)));
    m.attr("JOINT_PARENT") = py::tuple(py::cast(vec(H_JOINT_PARENT)));
    m.attr("TARGET_TOOL_COLUMN_MAJOR") = py::tuple(py::cast(vec(H_TARGET_TOOL)));
    m.def("target_poses_f32", &poses<float>, py::arg("q"), py::arg("warps_per_block") = 4,
          py::arg("source") = "metadata", py::arg("repeats") = 0);
    m.def("target_poses_f64", &poses<double>, py::arg("q"), py::arg("warps_per_block") = 4,
          py::arg("source") = "metadata", py::arg("repeats") = 0);
    m.def("kernel_info_f32", &info<float>, py::arg("warps_per_block") = 4);
    m.def("kernel_info_f64", &info<double>, py::arg("warps_per_block") = 4);
    m.def("proposals_f32", &proposals_py<float>, py::arg("q"), py::arg("target"), py::arg("active"), py::arg("w_pos"), py::arg("w_ori"),
          py::arg("mode"), py::arg("axis"), py::arg("twist") = py::none(), py::arg("tool") = py::none(), py::arg("base") = py::none());
    m.def("proposals_f64", &proposals_py<double>, py::arg("q"), py::arg("target"), py::arg("active"), py::arg("w_pos"), py::arg("w_ori"),
          py::arg("mode"), py::arg("axis"), py::arg("twist") = py::none(), py::arg("tool") = py::none(), py::arg("base") = py::none());
    m.def("coarse_kernel_info_f32", [] { auto k = hjcd_tree::coarse_kernel_info<float>(); py::dict d; d["num_regs"] = k.num_regs;
          d["local_bytes"] = k.local_bytes; d["dynamic_shared_bytes"] = k.dynamic_shared_bytes; return d; });
    m.def("coarse_kernel_info_f64", [] { auto k = hjcd_tree::coarse_kernel_info<double>(); py::dict d; d["num_regs"] = k.num_regs;
          d["local_bytes"] = k.local_bytes; d["dynamic_shared_bytes"] = k.dynamic_shared_bytes; return d; });
    m.attr("ORI_NONE") = 0; m.attr("ORI_AXIS") = 1; m.attr("ORI_FULL") = 2; m.attr("ORI_AXIS_BOUNDED_TWIST") = 3;
    m.attr("JOINT_LOWER") = py::tuple(py::cast(vec(H_JOINT_LOWER)));
    m.attr("JOINT_UPPER") = py::tuple(py::cast(vec(H_JOINT_UPPER)));
#define HJCD_EVAL_ARGS py::arg("q"), py::arg("target"), py::arg("active"), py::arg("w_pos"), py::arg("w_ori"), \
    py::arg("mode"), py::arg("axis"), py::arg("twist") = py::none(), py::arg("tool") = py::none()
    m.def("evaluate_f32", &evaluate_py<float>, HJCD_EVAL_ARGS, py::arg("lam") = 5e-3, py::arg("base") = py::none());
    m.def("evaluate_f64", &evaluate_py<double>, HJCD_EVAL_ARGS, py::arg("lam") = 5e-3, py::arg("base") = py::none());
    m.def("base_step_f32", &base_step_py<float>, HJCD_EVAL_ARGS, py::arg("base"), py::arg("base_update") = py::none(),
          py::arg("base_bounds") = py::none(), py::arg("eps_pos") = 1e-4, py::arg("eps_ori") = 1e-3);
    m.def("base_step_f64", &base_step_py<double>, HJCD_EVAL_ARGS, py::arg("base"), py::arg("base_update") = py::none(),
          py::arg("base_bounds") = py::none(), py::arg("eps_pos") = 1e-4, py::arg("eps_ori") = 1e-3);
#define HJCD_LM_ARGS HJCD_EVAL_ARGS, py::arg("eps_pos") = 1e-4, py::arg("eps_ori") = 1e-3, py::arg("lambda_init") = 5e-3, \
    py::arg("k_max") = 60, py::arg("stag_patience") = 2, py::arg("stag_rel") = 1e-3, py::arg("warps_per_block") = 1, \
    py::arg("repeats") = 0, py::arg("uniform_lambda") = false
    m.def("lm_f32", &lm_py<float>, HJCD_LM_ARGS);
    m.def("lm_f64", &lm_py<double>, HJCD_LM_ARGS);
#define HJCD_BATCH_ARGS py::arg("seed_configs"), py::arg("target_poses"), py::arg("active_masks"), py::arg("position_weights"), \
    py::arg("orientation_weights"), py::arg("orientation_modes"), py::arg("orientation_axes"), py::arg("twist_spec") = py::none(), \
    py::arg("tool_override") = py::none(), py::arg("num_solutions") = 1, py::arg("candidate_ids") = py::none(), \
    py::arg("position_tol") = 1e-4, py::arg("orientation_tol") = 1e-3, py::arg("lambda_init") = 5e-3, py::arg("lm_iters") = 60, \
    py::arg("stag_patience") = 2, py::arg("stag_rel") = 1e-3, py::arg("warps_per_block") = 1, py::arg("uniform_lambda") = false, \
    py::arg("return_all_candidates") = false, py::arg("base_poses") = py::none(), py::arg("base_update") = py::none(), \
    py::arg("base_bounds") = py::none(), py::arg("coarse_mode") = "none", py::arg("coarse_iters") = 120, py::arg("seed") = 0, \
    py::arg("problem_seeds") = py::none(), py::arg("environment_collision") = "disabled", py::arg("environment") = py::none(), \
    py::arg("robot_spheres") = py::none(), py::arg("environment_margin") = 0.0, py::arg("environment_selection") = "gate", \
    py::arg("self_collision") = "disabled", py::arg("self_pairs") = py::none(), py::arg("self_margin") = 0.0, \
    py::arg("self_selection") = "annotate"
    m.def("solve_problems_f32", &solve_problems_py<float>, HJCD_BATCH_ARGS);
    m.def("solve_problems_f64", &solve_problems_py<double>, HJCD_BATCH_ARGS);
    m.def("select_f32", &select_py<float>, py::arg("q"), py::arg("pe"), py::arg("oe"), py::arg("cost"), py::arg("succ"),
          py::arg("seeds"), py::arg("active"), py::arg("M"), py::arg("eps_pos") = 1e-4, py::arg("eps_ori") = 1e-3,
          py::arg("env_status") = py::none(), py::arg("env_min_clearance") = py::none(), py::arg("env_selection") = "gate",
          py::arg("self_status") = py::none(), py::arg("self_selection") = "annotate");
    m.def("self_collision_f32", &self_collision_py<float>, py::arg("q"), py::arg("robot_spheres"), py::arg("pairs"),
          py::arg("margin") = 0.0, py::arg("base") = py::none());
    m.def("self_kernel_info_f32", []() { auto i = hjcd_tree::self_kernel_info<float>();
        py::dict d; d["num_regs"] = i.num_regs; d["local_bytes"] = i.local_bytes; d["static_shared_bytes"] = i.static_shared_bytes;
        d["dynamic_shared_bytes"] = i.dynamic_shared_bytes; d["max_threads_per_block"] = i.max_threads_per_block; return d; });
    m.def("env_collision_f32", &env_collision_py<float>, py::arg("q"), py::arg("robot_spheres"), py::arg("environment") = py::none(),
          py::arg("base") = py::none(), py::arg("margin") = 0.0, py::arg("per_sphere") = false);
    m.def("env_kernel_info_f32", []() { auto i = hjcd_tree::env_kernel_info<float>();
        py::dict d; d["num_regs"] = i.num_regs; d["local_bytes"] = i.local_bytes; d["static_shared_bytes"] = i.static_shared_bytes;
        d["dynamic_shared_bytes"] = i.dynamic_shared_bytes; d["max_threads_per_block"] = i.max_threads_per_block; return d; });
    m.def("select_f64", &select_py<double>, py::arg("q"), py::arg("pe"), py::arg("oe"), py::arg("cost"), py::arg("succ"),
          py::arg("seeds"), py::arg("active"), py::arg("M"), py::arg("eps_pos") = 1e-4, py::arg("eps_ori") = 1e-3,
          py::arg("env_status") = py::none(), py::arg("env_min_clearance") = py::none(), py::arg("env_selection") = "gate",
          py::arg("self_status") = py::none(), py::arg("self_selection") = "annotate");
    m.def("self_collision_f64", &self_collision_py<double>, py::arg("q"), py::arg("robot_spheres"), py::arg("pairs"),
          py::arg("margin") = 0.0, py::arg("base") = py::none());
    m.def("self_kernel_info_f64", []() { auto i = hjcd_tree::self_kernel_info<double>();
        py::dict d; d["num_regs"] = i.num_regs; d["local_bytes"] = i.local_bytes; d["static_shared_bytes"] = i.static_shared_bytes;
        d["dynamic_shared_bytes"] = i.dynamic_shared_bytes; d["max_threads_per_block"] = i.max_threads_per_block; return d; });
    m.def("env_collision_f64", &env_collision_py<double>, py::arg("q"), py::arg("robot_spheres"), py::arg("environment") = py::none(),
          py::arg("base") = py::none(), py::arg("margin") = 0.0, py::arg("per_sphere") = false);
    m.def("env_kernel_info_f64", []() { auto i = hjcd_tree::env_kernel_info<double>();
        py::dict d; d["num_regs"] = i.num_regs; d["local_bytes"] = i.local_bytes; d["static_shared_bytes"] = i.static_shared_bytes;
        d["dynamic_shared_bytes"] = i.dynamic_shared_bytes; d["max_threads_per_block"] = i.max_threads_per_block; return d; });
    m.def("lm_kernel_info_f32", &lm_info<float>, py::arg("warps_per_block") = 1);
    m.def("lm_kernel_info_f64", &lm_info<double>, py::arg("warps_per_block") = 1);
}
