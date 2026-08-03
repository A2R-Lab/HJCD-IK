// Standalone sidecar pybind module (Checkpoint 2, Stage 6). Behaviorally isolated from HJCD:
// this is a SEPARATE extension (_sidecar); it does not touch _hjcdik, grid.cuh, or solve dispatch.
// One translation unit: it #includes the sidecar .cu so the generated header's __constant__ arrays
// (which must have exactly one definition) are defined once.
#include "collision_sidecar.cu"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <map>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace g1_sidecar;   // header constants + hashes

static py::array_t<uint8_t> full_check(py::array_t<float, py::array::c_style | py::array::forcecast> q,
                                       float margin) {
    int B = q.shape(0);
    if (q.shape(1) != N_JOINTS) throw std::runtime_error("q must be [B, 29]");
    auto out = py::array_t<uint8_t>({(py::ssize_t)B, (py::ssize_t)N_CHECKED_PAIRS});
    g1sc::sidecar_full_check(q.data(), out.mutable_data(), B, margin);
    return out;
}

static py::array_t<uint8_t> incr_check(py::array_t<float, py::array::c_style | py::array::forcecast> qbase,
                                       py::array_t<uint8_t, py::array::c_style | py::array::forcecast> base,
                                       py::array_t<int32_t, py::array::c_style | py::array::forcecast> jidx,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> newval,
                                       float margin) {
    int J = qbase.shape(0);
    if (base.shape(0) != J || base.shape(1) != N_CHECKED_PAIRS) throw std::runtime_error("base must be [J, NP]");
    if (jidx.shape(0) != J || newval.shape(0) != J) throw std::runtime_error("jidx/newval must be [J]");
    auto out = py::array_t<uint8_t>({(py::ssize_t)J, (py::ssize_t)N_CHECKED_PAIRS});
    g1sc::sidecar_incr_check(qbase.data(), base.data(), jidx.data(), newval.data(),
                             out.mutable_data(), J, margin);
    return out;
}

static void upload_sdf(int cid, py::array_t<int16_t, py::array::c_style | py::array::forcecast> grid) {
    g1sc::sidecar_upload_sdf(cid, grid.data(), (int)grid.size());
}
static void upload_convex(py::array_t<double, py::array::c_style | py::array::forcecast> verts) {
    if (verts.shape(1) != 3) throw std::runtime_error("verts must be [N, 3]");
    g1sc::sidecar_upload_convex(verts.data(), (int)verts.shape(0));
}

// -- CUDA environment checker (Task A) --
using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
static void upload_scene(int nobj, iarr otype, darr broad, darr box, darr plane, darr sph,
                         darr reg, iarr rallow, iarr plink, darr poff, darr ptype) {
    if (otype.shape(0)!=nobj) throw std::runtime_error("otype must be [nobj]");
    if (broad.size()!=(size_t)nobj*4) throw std::runtime_error("broad must be [nobj,4]");
    if (box.size()!=(size_t)nobj*10) throw std::runtime_error("box must be [nobj,10]");
    if (plane.size()!=(size_t)nobj*6) throw std::runtime_error("plane must be [nobj,6]");
    if (sph.size()!=(size_t)nobj*4) throw std::runtime_error("sph must be [nobj,4]");
    if (reg.size()!=(size_t)nobj*16) throw std::runtime_error("reg must be [nobj,16]");
    if (rallow.shape(0)!=nobj) throw std::runtime_error("rallow must be [nobj]");
    if (plink.size()!=4 || poff.size()!=12 || ptype.size()!=4) throw std::runtime_error("proxy dims");
    g1sc::sidecar_upload_scene(nobj, otype.data(), broad.data(), box.data(), plane.data(),
                               sph.data(), reg.data(), rallow.data(), plink.data(), poff.data(),
                               ptype.data());
}
static py::array_t<uint8_t> env_check(darr q, iarr assign) {
    int B = q.shape(0);
    if (q.shape(1)!=36) throw std::runtime_error("q must be [B,36] (base pos3+quat4+joints29)");
    if (assign.shape(0)!=B || assign.shape(1)!=4) throw std::runtime_error("assign must be [B,4]");
    if (!g1sc::sidecar_env_ready()) throw std::runtime_error("upload_scene first");
    auto out = py::array_t<uint8_t>({(py::ssize_t)B, (py::ssize_t)6});
    g1sc::sidecar_env_check(q.data(), assign.data(), out.mutable_data(), B);
    return out;
}
static float env_bench(int B, int iters) { return g1sc::sidecar_env_bench(B, iters); }
static int env_nalloc() { return g1sc::sidecar_env_nalloc(); }
// timed variant -> (flags[B,6], (h2d_ms, kernel_ms, d2h_ms))
static py::tuple env_check_timed(darr q, iarr assign) {
    int B = q.shape(0);
    if (q.shape(1)!=36) throw std::runtime_error("q must be [B,36]");
    if (assign.shape(0)!=B || assign.shape(1)!=4) throw std::runtime_error("assign must be [B,4]");
    if (!g1sc::sidecar_env_ready()) throw std::runtime_error("upload_scene first");
    auto out = py::array_t<uint8_t>({(py::ssize_t)B, (py::ssize_t)6});
    float ms3[3] = {0,0,0};
    g1sc::sidecar_env_check_timed(q.data(), assign.data(), out.mutable_data(), B, ms3);
    return py::make_tuple(out, py::make_tuple(ms3[0], ms3[1], ms3[2]));
}

// -- optional diagnostics: per-pair gaps by narrow phase --
static py::array_t<float> prim_gaps(py::array_t<float, py::array::c_style | py::array::forcecast> q) {
    int B = q.shape(0);
    auto out = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)N_CHECKED_PAIRS});
    g1sc::sidecar_prim_gaps(q.data(), out.mutable_data(), B);
    return out;
}
static py::tuple cluster_gaps(py::array_t<float, py::array::c_style | py::array::forcecast> q) {
    int B = q.shape(0);
    auto g = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)N_CHECKED_PAIRS});
    auto e = py::array_t<int32_t>({(py::ssize_t)B, (py::ssize_t)N_CHECKED_PAIRS});
    g1sc::sidecar_cluster_gaps(q.data(), g.mutable_data(), e.mutable_data(), B);
    return py::make_tuple(g, e);
}
static py::tuple gjk_gaps(py::array_t<float, py::array::c_style | py::array::forcecast> q) {
    int B = q.shape(0);
    auto g = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)N_GJK_PAIRS});
    auto it = py::array_t<int32_t>({(py::ssize_t)B, (py::ssize_t)N_GJK_PAIRS});
    g1sc::sidecar_gjk_gaps(q.data(), g.mutable_data(), it.mutable_data(), B);
    return py::make_tuple(g, it);
}

static py::dict model_info() {
    py::dict d;
    d["n_joints"] = N_JOINTS; d["n_links"] = N_LINKS; d["n_primitives"] = N_PRIMITIVES;
    d["n_checked_pairs"] = N_CHECKED_PAIRS; d["n_prim_pairs"] = N_PRIM_PAIRS;
    d["n_cluster_pairs"] = N_CLUSTER_PAIRS; d["n_gjk_pairs"] = N_GJK_PAIRS;
    d["n_clusters"] = N_CLUSTERS; d["n_convex_verts"] = N_CONVEX_VERTS;
    d["broad_margin"] = BROAD_MARGIN; d["sdf_max_evals"] = SDF_MAX_EVALS;
    py::dict h;
    h["urdf"] = HASH_URDF; h["joint_order"] = HASH_JOINT_ORDER; h["proxy_yaml"] = HASH_PROXY_YAML;
    h["torso_sdf"] = HASH_TORSO_SDF; h["pelvis_sdf"] = HASH_PELVIS_SDF; h["convex"] = HASH_CONVEX;
    h["pair_policy"] = HASH_PAIR_POLICY;
    d["hashes"] = h;
    return d;
}

PYBIND11_MODULE(_sidecar, m) {
    m.doc() = "Standalone GPU G1 self-collision sidecar (Checkpoint 2). Isolated from HJCD solve.";
    m.def("upload_sdf", &upload_sdf, "cid"_a, "grid_i16"_a);
    m.def("upload_convex", &upload_convex, "verts"_a);
    m.def("full_check", &full_check, "q"_a, "margin"_a = 0.0f);
    m.def("incr_check", &incr_check, "qbase"_a, "base"_a, "jidx"_a, "newval"_a, "margin"_a = 0.0f);
    m.def("prim_gaps", &prim_gaps, "q"_a);
    m.def("cluster_gaps", &cluster_gaps, "q"_a);
    m.def("gjk_gaps", &gjk_gaps, "q"_a);
    m.def("model_info", &model_info);
    // CUDA environment checker (Task A)
    m.def("upload_scene", &upload_scene, "nobj"_a, "otype"_a, "broad"_a, "box"_a, "plane"_a,
          "sph"_a, "reg"_a, "rallow"_a, "plink"_a, "poff"_a, "ptype"_a);
    m.def("env_check", &env_check, "q"_a, "assign"_a);
    m.def("env_check_timed", &env_check_timed, "q"_a, "assign"_a);
    m.def("env_bench", &env_bench, "B"_a, "iters"_a);
    m.def("env_nalloc", &env_nalloc);
}
