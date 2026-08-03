// Standalone CUDA collision sidecar -- device functions (Checkpoint 2).
// Behaviorally isolated from the HJCD solver: this header is NOT included by grid.cuh,
// hjcd_targets.cuh, the kernel, or the pybind solve path. FP32 throughout.
#pragma once
#include "g1_collision_sidecar.cuh"   // generated SoA metadata (namespace g1_sidecar), -I generated

namespace g1sc {

using namespace g1_sidecar;

// ----- 4x4 homogeneous transforms, COLUMN-MAJOR (element (r,c) = m[4*c + r]) -----
__device__ __forceinline__ void mat4_identity(float* m) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}
__device__ __forceinline__ void mat4_copy(const float* a, float* o) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = a[i];
}
// o = a * b  (both column-major)
__device__ __forceinline__ void mat4_mul(const float* a, const float* b, float* o) {
    #pragma unroll
    for (int c = 0; c < 4; ++c)
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) s += a[4 * k + r] * b[4 * c + k];
            o[4 * c + r] = s;
        }
}
// Revolute joint transform about a unit axis by angle (Rodrigues), column-major, no translation.
__device__ __forceinline__ void axis_angle(const float* ax, float ang, float* m) {
    float n = sqrtf(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    float x = ax[0] / n, y = ax[1] / n, z = ax[2] / n;
    float c = cosf(ang), s = sinf(ang), t = 1.0f - c;
    mat4_identity(m);
    m[0] = t * x * x + c;       m[4] = t * x * y - s * z;   m[8]  = t * x * z + s * y;
    m[1] = t * x * y + s * z;   m[5] = t * y * y + c;       m[9]  = t * y * z - s * x;
    m[2] = t * x * z - s * y;   m[6] = t * y * z + s * x;   m[10] = t * z * z + c;
}
// Transform a point (link frame) by a column-major 4x4.
__device__ __forceinline__ void xform_point(const float* T, const float* p, float* o) {
    o[0] = T[0] * p[0] + T[4] * p[1] + T[8]  * p[2] + T[12];
    o[1] = T[1] * p[0] + T[5] * p[1] + T[9]  * p[2] + T[13];
    o[2] = T[2] * p[0] + T[6] * p[1] + T[10] * p[2] + T[14];
}

// ----- generic fixed-base FK: all N_LINKS transforms (root at identity) -----
// T_out: N_LINKS * 16 floats (column-major). Parents precede children (generated BFS order).
__device__ inline void sidecar_fk(const float* q, float* T_out) {
    mat4_identity(T_out);   // link 0 == root
    for (int L = 1; L < N_LINKS; ++L) {
        const int par = LINK_PARENT[L];
        float To[16];
        mat4_mul(&T_out[par * 16], &LINK_T_ORIGIN[L * 16], To);   // T_parent * T_parent_to_joint
        const int qi = LINK_QINDEX[L];
        if (qi >= 0) {
            float Rj[16];
            axis_angle(&LINK_AXIS[L * 3], q[qi], Rj);
            mat4_mul(To, Rj, &T_out[L * 16]);                     // ... * T_joint(q_j)
        } else {
            mat4_copy(To, &T_out[L * 16]);
        }
    }
}

// ================= Stage 2: primitive narrow phases =================
// Exact device ports of collision_cpu.pt_seg_dist / seg_seg_dist (Ericson).
__device__ __forceinline__ float v3dot(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ __forceinline__ float v3dist(const float* a, const float* b) {
    float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}
__device__ __forceinline__ float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// distance from point p to segment [a,b]
__device__ __forceinline__ float pt_seg_dist(const float* p, const float* a, const float* b) {
    float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    float ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    float t = clamp01(v3dot(ap, ab) / (v3dot(ab, ab) + 1e-12f));
    float c[3] = {a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t};
    return v3dist(p, c);
}
// minimum distance between segments [p1,q1] and [p2,q2] (Ericson, matches CPU seg_seg_dist)
__device__ __forceinline__ float seg_seg_dist(const float* p1, const float* q1,
                                              const float* p2, const float* q2) {
    float d1[3] = {q1[0]-p1[0], q1[1]-p1[1], q1[2]-p1[2]};
    float d2[3] = {q2[0]-p2[0], q2[1]-p2[1], q2[2]-p2[2]};
    float r[3]  = {p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]};
    float a = v3dot(d1, d1), e = v3dot(d2, d2), f = v3dot(d2, r);
    const float EPS = 1e-12f;
    float s, t;
    if (a <= EPS && e <= EPS) return v3dist(p1, p2);
    if (a <= EPS) { s = 0.0f; t = clamp01(f / e); }
    else {
        float c = v3dot(d1, r);
        if (e <= EPS) { t = 0.0f; s = clamp01(-c / a); }
        else {
            float b = v3dot(d1, d2);
            float denom = a * e - b * b;
            s = (denom > EPS) ? clamp01((b * f - c * e) / denom) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f)      { t = 0.0f; s = clamp01(-c / a); }
            else if (t > 1.0f) { t = 1.0f; s = clamp01((b - c) / a); }
        }
    }
    float c1[3] = {p1[0]+d1[0]*s, p1[1]+d1[1]*s, p1[2]+d1[2]*s};
    float c2[3] = {p2[0]+d2[0]*t, p2[1]+d2[1]*t, p2[2]+d2[2]*t};
    return v3dist(c1, c2);
}

// World primitive: transform prim `pi` by its link transform. Returns type (0 sphere/1 capsule);
// fills a0 (center/p0), a1 (p1, capsule only), r (radius).
__device__ __forceinline__ int world_prim(int pi, const float* T_all, float* a0, float* a1, float* r) {
    const float* T = &T_all[PRIM_LINK[pi] * 16];
    const float* pp = &PRIM_PARAM[pi * 7];
    xform_point(T, &pp[0], a0);
    *r = pp[3];
    if (PRIM_TYPE[pi] == 1) { xform_point(T, &pp[4], a1); return 1; }
    return 0;
}
// Signed gap between two primitives (matches CPU _pair_gap): dist(cores) - (ri+rj).
__device__ __forceinline__ float prim_pair_gap(int pi, int pj, const float* T_all) {
    float a0[3], a1[3], ri, b0[3], b1[3], rj;
    int ti = world_prim(pi, T_all, a0, a1, &ri);
    int tj = world_prim(pj, T_all, b0, b1, &rj);
    float d;
    if (ti == 0 && tj == 0)      d = v3dist(a0, b0);
    else if (ti == 0)            d = pt_seg_dist(a0, b0, b1);
    else if (tj == 0)            d = pt_seg_dist(b0, a0, a1);
    else                         d = seg_seg_dist(a0, a1, b0, b1);
    return d - (ri + rj);
}
// Primitive link-pair: min gap over the FULL cross product of both links' primitives
// (matches _linkpair_colliding "prim" branch, incl. 4-primitive feet). Sets *any_below.
__device__ __forceinline__ float prim_linkpair_gap(int la, int lb, const float* T_all,
                                                   float margin, int* any_below) {
    float mg = 1e30f; *any_below = 0;
    for (int ia = LINK_PRIM_OFF[la]; ia < LINK_PRIM_OFF[la + 1]; ++ia)
        for (int ib = LINK_PRIM_OFF[lb]; ib < LINK_PRIM_OFF[lb + 1]; ++ib) {
            float g = prim_pair_gap(LINK_PRIM[ia], LINK_PRIM[ib], T_all);
            if (g < mg) mg = g;
            if (g < margin) *any_below = 1;
        }
    return mg;
}

// ================= Stage 3: cluster SDF narrow phase =================
// Device SDF grids (int16, 0.1mm quant), one per cluster, set at host init via cudaMemcpyToSymbol.
__device__ const short* g_sdf[8];

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
// Trilinear signed distance in cluster-local frame, matching collision_cpu.TorsoSDF.trilinear_sdf
// (incl. the out-of-grid euclidean "outside" term). p is cluster-local.
__device__ inline float trilinear_sdf(int cid, const float* p) {
    const short* S = g_sdf[cid];
    const float* o = &CLUSTER_ORIGIN[cid * 3];
    float sp = CLUSTER_SPACING[cid], scale = CLUSTER_SCALE[cid];
    int nx = CLUSTER_DIMS[cid * 3 + 0], ny = CLUSTER_DIMS[cid * 3 + 1], nz = CLUSTER_DIMS[cid * 3 + 2];
    float g[3], gc[3]; int dimv[3] = {nx, ny, nz};
    #pragma unroll
    for (int k = 0; k < 3; ++k) { g[k] = (p[k] - o[k]) / sp; gc[k] = clampf(g[k], 0.0f, dimv[k] - 1.0001f); }
    int i0 = (int)floorf(gc[0]), i1 = (int)floorf(gc[1]), i2 = (int)floorf(gc[2]);
    float f0 = gc[0] - i0, f1 = gc[1] - i1, f2 = gc[2] - i2;
    float val = 0.0f;
    #pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
        float wx = dx ? f0 : 1.0f - f0;
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            float wy = dy ? f1 : 1.0f - f1;
            #pragma unroll
            for (int dz = 0; dz < 2; ++dz) {
                float wz = dz ? f2 : 1.0f - f2;
                int ix = i0 + dx, iy = i1 + dy, iz = i2 + dz;
                val += wx * wy * wz * ((float)S[((size_t)ix * ny + iy) * nz + iz] / scale);
            }
        }
    }
    float ex = (g[0] - gc[0]) * sp, ey = (g[1] - gc[1]) * sp, ez = (g[2] - gc[2]) * sp;
    return val + sqrtf(ex * ex + ey * ey + ez * ez);
}
// sphere-vs-SDF: 1 eval. center is cluster-local.
__device__ __forceinline__ float sphere_sdf_gap(int cid, const float* center, float radius, int* evals) {
    *evals = 1;
    return trilinear_sdf(cid, center) - radius;
}
// capsule-vs-SDF: adaptive Lipschitz branch-and-bound, exact port of capsule_torso_sdf_distance
// (tol=SDF_TOL, cap=SDF_MAX_EVALS, LIFO stack, same best/bs update order). p0,p1 cluster-local.
__device__ inline float capsule_sdf_gap(int cid, const float* p0, const float* p1, float radius, int* evals) {
    float seg[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    float L = sqrtf(seg[0] * seg[0] + seg[1] * seg[1] + seg[2] * seg[2]);
    float ps[3];
    #define SDF_F(s) ( ps[0]=p0[0]+(s)*seg[0], ps[1]=p0[1]+(s)*seg[1], ps[2]=p0[2]+(s)*seg[2], trilinear_sdf(cid, ps) )
    float fa = SDF_F(0.0f), fb = SDF_F(1.0f);
    int ev = 2;
    float best = fa <= fb ? fa : fb;
    float bs = fa <= fb ? 0.0f : 1.0f;
    // LIFO stack of (sa, sb, fsa, fsb)
    float st_sa[128], st_sb[128], st_fa[128], st_fb[128]; int sp2 = 0;
    st_sa[sp2] = 0.0f; st_sb[sp2] = 1.0f; st_fa[sp2] = fa; st_fb[sp2] = fb; ++sp2;
    while (sp2 > 0 && ev < SDF_MAX_EVALS) {
        --sp2;
        float sa = st_sa[sp2], sb = st_sb[sp2], fsa = st_fa[sp2], fsb = st_fb[sp2];
        float lo = (fsa < fsb ? fsa : fsb) - 0.5f * (sb - sa) * L;
        if (lo >= best - SDF_TOL) continue;
        float sm = 0.5f * (sa + sb);
        float fm = SDF_F(sm); ++ev;
        if (fm < best) { best = fm; bs = sm; }
        if (sp2 < 126) {
            st_sa[sp2] = sa; st_sb[sp2] = sm; st_fa[sp2] = fsa; st_fb[sp2] = fm; ++sp2;
            st_sa[sp2] = sm; st_sb[sp2] = sb; st_fa[sp2] = fm; st_fb[sp2] = fsb; ++sp2;
        }
    }
    #undef SDF_F
    *evals = ev;
    return best - radius;
}

// Cluster link-pair gap: min over the limb link's primitives of (broad enclosing capsule reject ->
// cluster SDF). Matches _linkpair_colliding "cluster" branch. Accumulates SDF evals.
__device__ inline float cluster_linkpair_gap(int cid, int cl_link, int limb_link, const float* T_all,
                                             float margin, int* any_below, int* evals_acc) {
    const float* Tc = &T_all[cl_link * 16];
    // broad enclosing capsule (attached to cluster link) -> world
    int bp = CLUSTER_BROAD_PRIM[cid];
    const float* bpar = &PRIM_PARAM[bp * 7];
    float b0[3], b1[3]; xform_point(Tc, &bpar[0], b0); xform_point(Tc, &bpar[4], b1);
    float br = bpar[3];
    // cluster-local rotation^T and translation for limb->local mapping
    float R[9] = {Tc[0], Tc[1], Tc[2], Tc[4], Tc[5], Tc[6], Tc[8], Tc[9], Tc[10]};  // columns
    float t[3] = {Tc[12], Tc[13], Tc[14]};
    float mg = 1e30f; *any_below = 0;
    for (int ip = LINK_PRIM_OFF[limb_link]; ip < LINK_PRIM_OFF[limb_link + 1]; ++ip) {
        int pi = LINK_PRIM[ip];
        float a0[3], a1[3], r; int typ = world_prim(pi, T_all, a0, a1, &r);
        float bgap = (typ == 0 ? pt_seg_dist(a0, b0, b1) : seg_seg_dist(a0, a1, b0, b1)) - (r + br);
        float gap;
        if (bgap > BROAD_MARGIN) {
            gap = bgap;                                   // broad-rejected (no SDF eval)
        } else {
            // map limb prim into cluster-local: local = R^T (world - t)
            float d0[3] = {a0[0]-t[0], a0[1]-t[1], a0[2]-t[2]};
            float l0[3] = {R[0]*d0[0]+R[1]*d0[1]+R[2]*d0[2],
                           R[3]*d0[0]+R[4]*d0[1]+R[5]*d0[2],
                           R[6]*d0[0]+R[7]*d0[1]+R[8]*d0[2]};
            int ev;
            if (typ == 0) { gap = sphere_sdf_gap(cid, l0, r, &ev); }
            else {
                float d1[3] = {a1[0]-t[0], a1[1]-t[1], a1[2]-t[2]};
                float l1[3] = {R[0]*d1[0]+R[1]*d1[1]+R[2]*d1[2],
                               R[3]*d1[0]+R[4]*d1[1]+R[5]*d1[2],
                               R[6]*d1[0]+R[7]*d1[1]+R[8]*d1[2]};
                gap = capsule_sdf_gap(cid, l0, l1, r, &ev);
            }
            *evals_acc += ev;
        }
        if (gap < mg) mg = gap;
        if (gap < margin) *any_below = 1;
    }
    return mg;
}

// ================= Stage 4: convex GJK narrow phase (DOUBLE precision) =================
// GJK runs the penetration/separation boundary; it is decided in f64 to match the CPU oracle
// exactly (f32 GJK disagrees with the f64 oracle by up to ~0.5 mm at sub-mm contact, and no
// threshold reconciles the two distributions). Vertices + the GJK links' FK are f64.
// Exact convex vertices (double3-packed), uploaded once; CSR ranges in CONVEX_VERT_OFF (by slot).
__device__ const double* g_cverts;   // [N_CONVEX_VERTS * 3]

// -- f64 transforms (column-major) for the GJK links --
__device__ __forceinline__ void dmat4_identity(double* m){
    #pragma unroll
    for (int i=0;i<16;++i) m[i]=0.0; m[0]=m[5]=m[10]=m[15]=1.0;
}
__device__ __forceinline__ void dmat4_mul(const double* a, const double* b, double* o){
    #pragma unroll
    for (int c=0;c<4;++c)
        #pragma unroll
        for (int r=0;r<4;++r){ double s=0.0;
            #pragma unroll
            for (int k=0;k<4;++k) s+=a[4*k+r]*b[4*c+k]; o[4*c+r]=s; }
}
__device__ __forceinline__ void daxis_angle(const float* ax, double ang, double* m){
    double n=sqrt((double)ax[0]*ax[0]+(double)ax[1]*ax[1]+(double)ax[2]*ax[2]);
    double x=ax[0]/n, y=ax[1]/n, z=ax[2]/n, c=cos(ang), s=sin(ang), t=1.0-c;
    dmat4_identity(m);
    m[0]=t*x*x+c;    m[4]=t*x*y-s*z;  m[8]=t*x*z+s*y;
    m[1]=t*x*y+s*z;  m[5]=t*y*y+c;    m[9]=t*y*z-s*x;
    m[2]=t*x*z-s*y;  m[6]=t*y*z+s*x;  m[10]=t*z*z+c;
}
// full-tree FK in double (root=identity), matching the CPU f64 oracle chain.
__device__ inline void sidecar_fk_d(const float* q, double* T_out){
    dmat4_identity(T_out);
    for (int L=1; L<N_LINKS; ++L){
        int par=LINK_PARENT[L];
        double To[16], Orig[16];
        #pragma unroll
        for (int i=0;i<16;++i) Orig[i]=(double)LINK_T_ORIGIN[L*16+i];
        dmat4_mul(&T_out[par*16], Orig, To);
        int qi=LINK_QINDEX[L];
        if (qi>=0){ double Rj[16]; daxis_angle(&LINK_AXIS[L*3], (double)q[qi], Rj);
                    dmat4_mul(To, Rj, &T_out[L*16]); }
        else { for (int i=0;i<16;++i) T_out[L*16+i]=To[i]; }
    }
}
__device__ __forceinline__ double dv3dot(const double* a, const double* b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
__device__ __forceinline__ void dv3sub(const double* a, const double* b, double* o){ o[0]=a[0]-b[0];o[1]=a[1]-b[1];o[2]=a[2]-b[2]; }
__device__ __forceinline__ double dv3norm(const double* a){ return sqrt(dv3dot(a,a)); }
__device__ __forceinline__ void dv3cross(const double* a, const double* b, double* o){
    o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
}
__device__ __forceinline__ double dclamp01(double x){ return x<0.0?0.0:(x>1.0?1.0:x); }
__device__ __forceinline__ void dxform_point(const double* T, const double* p, double* o){
    o[0]=T[0]*p[0]+T[4]*p[1]+T[8]*p[2]+T[12];
    o[1]=T[1]*p[0]+T[5]*p[1]+T[9]*p[2]+T[13];
    o[2]=T[2]*p[0]+T[6]*p[1]+T[10]*p[2]+T[14];
}
__device__ __forceinline__ void drotT_apply(const double* T, const double* d, double* o){
    o[0]=T[0]*d[0]+T[1]*d[1]+T[2]*d[2];
    o[1]=T[4]*d[0]+T[5]*d[1]+T[6]*d[2];
    o[2]=T[8]*d[0]+T[9]*d[1]+T[10]*d[2];
}
// warp-cooperative hull support (f64), argmax v.(R^T d), lowest-index tiebreak (== np.argmax)
// support over an EXPLICIT vertex range [v0,v1) of one convex piece (world = R v + t).
__device__ inline void dhull_support_warp(int v0, int v1, const double* T, const double* d, int lane, double* out){
    double dl[3]; drotT_apply(T, d, dl);
    double bestdot=-1e300; int bestidx=-1;
    for (int i=v0+lane; i<v1; i+=32){
        const double* v=&g_cverts[i*3];
        double dp=v[0]*dl[0]+v[1]*dl[1]+v[2]*dl[2];
        if (dp>bestdot || (dp==bestdot && i<bestidx)){ bestdot=dp; bestidx=i; }
    }
    #pragma unroll
    for (int off=16; off>0; off>>=1){
        double odot=__shfl_down_sync(0xffffffffu, bestdot, off);
        int    oidx=__shfl_down_sync(0xffffffffu, bestidx, off);
        if (odot>bestdot || (odot==bestdot && oidx<bestidx)){ bestdot=odot; bestidx=oidx; }
    }
    bestidx=__shfl_sync(0xffffffffu, bestidx, 0);
    const double* v=&g_cverts[bestidx*3];
    dxform_point(T, v, out);
}
__device__ inline int dclosest_seg(const double* a, const double* b, double* out, int* keep){
    double ab[3]; dv3sub(b,a,ab);
    double t=-dv3dot(a,ab)/(dv3dot(ab,ab)+1e-12); t=dclamp01(t);
    out[0]=a[0]+t*ab[0]; out[1]=a[1]+t*ab[1]; out[2]=a[2]+t*ab[2];
    if (t<=0.0){ keep[0]=0; return 1; } if (t>=1.0){ keep[0]=1; return 1; }
    keep[0]=0; keep[1]=1; return 2;
}
__device__ inline int dclosest_tri(const double* a, const double* b, const double* c, double* out, int* keep){
    double ab[3],ac[3],ao[3]; dv3sub(b,a,ab); dv3sub(c,a,ac); ao[0]=-a[0];ao[1]=-a[1];ao[2]=-a[2];
    double d1=dv3dot(ab,ao), d2=dv3dot(ac,ao);
    if (d1<=0&&d2<=0){ out[0]=a[0];out[1]=a[1];out[2]=a[2]; keep[0]=0; return 1; }
    double bo[3]={-b[0],-b[1],-b[2]}; double d3=dv3dot(ab,bo), d4=dv3dot(ac,bo);
    if (d3>=0&&d4<=d3){ out[0]=b[0];out[1]=b[1];out[2]=b[2]; keep[0]=1; return 1; }
    double vc=d1*d4-d3*d2;
    if (vc<=0&&d1>=0&&d3<=0){ double v=d1/(d1-d3);
        out[0]=a[0]+v*ab[0];out[1]=a[1]+v*ab[1];out[2]=a[2]+v*ab[2]; keep[0]=0;keep[1]=1; return 2; }
    double co[3]={-c[0],-c[1],-c[2]}; double d5=dv3dot(ab,co), d6=dv3dot(ac,co);
    if (d6>=0&&d5<=d6){ out[0]=c[0];out[1]=c[1];out[2]=c[2]; keep[0]=2; return 1; }
    double vb=d5*d2-d1*d6;
    if (vb<=0&&d2>=0&&d6<=0){ double w=d2/(d2-d6);
        out[0]=a[0]+w*ac[0];out[1]=a[1]+w*ac[1];out[2]=a[2]+w*ac[2]; keep[0]=0;keep[1]=2; return 2; }
    double va=d3*d6-d5*d4;
    if (va<=0&&(d4-d3)>=0&&(d5-d6)>=0){ double w=(d4-d3)/((d4-d3)+(d5-d6));
        double cb[3]; dv3sub(c,b,cb);
        out[0]=b[0]+w*cb[0];out[1]=b[1]+w*cb[1];out[2]=b[2]+w*cb[2]; keep[0]=1;keep[1]=2; return 2; }
    double denom=1.0/(va+vb+vc), v=vb*denom, w=vc*denom;
    out[0]=a[0]+ab[0]*v+ac[0]*w; out[1]=a[1]+ab[1]*v+ac[1]*w; out[2]=a[2]+ab[2]*v+ac[2]*w;
    keep[0]=0;keep[1]=1;keep[2]=2; return 3;
}
__device__ inline int dclosest_simplex(const double W[4][3], int n, double* out, int* keep, int* enc){
    *enc=0;
    if (n==1){ out[0]=W[0][0];out[1]=W[0][1];out[2]=W[0][2]; keep[0]=0; *enc=dv3dot(out,out)<1e-12; return 1; }
    if (n==2){ int k=dclosest_seg(W[0],W[1],out,keep); *enc=dv3dot(out,out)<1e-12; return k; }
    if (n==3){ int k=dclosest_tri(W[0],W[1],W[2],out,keep); *enc=dv3dot(out,out)<1e-12; return k; }
    const int faces[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}}; const int fourth[4]={3,2,1,0};
    double best_p[3]; int best_keep[3]; int best_nk=0; double best_d2=1e300; int inside=1;
    for (int fi=0; fi<4; ++fi){
        const int* f=faces[fi];
        const double* pa=W[f[0]]; const double* pb=W[f[1]]; const double* pc=W[f[2]]; const double* other=W[fourth[fi]];
        double e1[3],e2[3],nrm[3]; dv3sub(pb,pa,e1); dv3sub(pc,pa,e2); dv3cross(e1,e2,nrm);
        double oa[3]; dv3sub(other,pa,oa); double na[3]={-pa[0],-pa[1],-pa[2]};
        if (dv3dot(nrm,oa)*dv3dot(nrm,na) < 0){
            inside=0; double p[3]; int lk[3]; int nk=dclosest_tri(pa,pb,pc,p,lk); double d2=dv3dot(p,p);
            if (d2<best_d2){ best_d2=d2; best_nk=nk; for(int i=0;i<nk;++i) best_keep[i]=f[lk[i]];
                best_p[0]=p[0];best_p[1]=p[1];best_p[2]=p[2]; }
        }
    }
    if (inside){ out[0]=out[1]=out[2]=0.0; keep[0]=0;keep[1]=1;keep[2]=2;keep[3]=3; *enc=1; return 4; }
    out[0]=best_p[0];out[1]=best_p[1];out[2]=best_p[2]; for(int i=0;i<best_nk;++i) keep[i]=best_keep[i]; return best_nk;
}
// Typed piece support (native-completion checkpoint, Task A): dispatch on PIECE_TYPE. A HULL uses
// the warp argmax over its vertex range; a SPHERE returns center(world) + radius * d/||d|| -- a
// sphere is rotation-invariant, so the WORLD search direction is used directly (matching the CPU
// oracle's world_support). Lane-independent for a sphere, so it stays warp-uniform.
__device__ inline void dpiece_support_warp(int pi, const double* T, const double* d, int lane, double* out){
    if (PIECE_TYPE[pi] == 0) {                        // hull
        dhull_support_warp(PIECE_VERT_OFF[pi], PIECE_VERT_OFF[pi+1], T, d, lane, out);
    } else {                                          // sphere: center + r * dir
        const float* sp = &PIECE_SPHERE[pi*4];
        double c[3] = {sp[0], sp[1], sp[2]};
        double cw[3]; dxform_point(T, c, cw);
        double n = dv3norm(d);
        double inv = (n > 1e-300) ? (1.0/n) : 0.0;
        out[0] = cw[0] + (double)sp[3]*d[0]*inv;
        out[1] = cw[1] + (double)sp[3]*d[1]*inv;
        out[2] = cw[2] + (double)sp[3]*d[2]*inv;
    }
}
// warp-cooperative GJK in f64 over one PIECE PAIR (typed pieces), matching gjk() +
// link_pieces_collide gap convention.
__device__ inline double dgjk_gap_warp(int pia, const double* Ta,
                                       int pib, const double* Tb, int lane, int* iters){
    const int MAXIT=64; const double TOL=1e-9;
    double W[4][3]; int n=0; double d0[3]={1.0,0.0,0.0}; double sa[3],sb[3],negd[3];
    dpiece_support_warp(pia,Ta,d0,lane,sa); negd[0]=-d0[0];negd[1]=-d0[1];negd[2]=-d0[2];
    dpiece_support_warp(pib,Tb,negd,lane,sb);
    W[0][0]=sa[0]-sb[0]; W[0][1]=sa[1]-sb[1]; W[0][2]=sa[2]-sb[2]; n=1;
    double closest[3]={W[0][0],W[0][1],W[0][2]};
    for (int it=1; it<=MAXIT; ++it){
        double d[3]={-closest[0],-closest[1],-closest[2]};
        if (dv3dot(d,d)<TOL){ *iters=it; return -1e-9; }
        dpiece_support_warp(pia,Ta,d,lane,sa); negd[0]=-d[0];negd[1]=-d[1];negd[2]=-d[2];
        dpiece_support_warp(pib,Tb,negd,lane,sb);
        double a[3]={sa[0]-sb[0],sa[1]-sb[1],sa[2]-sb[2]};
        double ad=dv3dot(a,d), cd=dv3dot(closest,d);
        if (ad-cd < TOL*(1.0+fabs(cd))){ *iters=it; return dv3norm(closest); }
        W[n][0]=a[0];W[n][1]=a[1];W[n][2]=a[2]; ++n;
        int keep[4],enc; double cp[3]; int nk=dclosest_simplex(W,n,cp,keep,&enc);
        double NW[4][3];
        for (int i=0;i<nk;++i){ NW[i][0]=W[keep[i]][0];NW[i][1]=W[keep[i]][1];NW[i][2]=W[keep[i]][2]; }
        for (int i=0;i<nk;++i){ W[i][0]=NW[i][0];W[i][1]=NW[i][1];W[i][2]=NW[i][2]; }
        n=nk; closest[0]=cp[0];closest[1]=cp[1];closest[2]=cp[2];
        if (enc){ *iters=it; return -1e-9; }
    }
    *iters=MAXIT; return (dv3norm(closest)<1e-4)? -1e-9 : dv3norm(closest);
}
// GJK link-pair gap (f64), MULTI-piece: link-level broad phase, then every piece-pair (each with a
// conservative per-piece bounding-sphere broad phase) through exact GJK. Colliding iff any piece
// pair intersects; gap = min over piece pairs. *iters accumulates GJK iters; matches CPU verdict.
__device__ inline double gjk_linkpair_gap_d(int la, int lb, const double* Td, double margin, int lane, int* iters){
    int sa=LINK_CONVEX[la], sb=LINK_CONVEX[lb];
    const double* Ta=&Td[la*16]; const double* Tb=&Td[lb*16];
    // link-level broad phase (enclosing spheres)
    double baC[3]={CONVEX_BOUND[sa*4+0],CONVEX_BOUND[sa*4+1],CONVEX_BOUND[sa*4+2]};
    double bbC[3]={CONVEX_BOUND[sb*4+0],CONVEX_BOUND[sb*4+1],CONVEX_BOUND[sb*4+2]};
    double cAw[3],cBw[3]; dxform_point(Ta,baC,cAw); dxform_point(Tb,bbC,cBw);
    double dd[3]; dv3sub(cAw,cBw,dd);
    double link_bgap=dv3norm(dd)-((double)CONVEX_BOUND[sa*4+3]+(double)CONVEX_BOUND[sb*4+3]);
    *iters=0;
    if (link_bgap>margin) return link_bgap;
    // piece-pair loop
    double mg=1e300;
    for (int pia=LINK_PIECE_OFF[la]; pia<LINK_PIECE_OFF[la+1]; ++pia){
        double pac[3]={PIECE_BOUND[pia*4+0],PIECE_BOUND[pia*4+1],PIECE_BOUND[pia*4+2]};
        double paw[3]; dxform_point(Ta,pac,paw); double par=PIECE_BOUND[pia*4+3];
        for (int pib=LINK_PIECE_OFF[lb]; pib<LINK_PIECE_OFF[lb+1]; ++pib){
            double pbc[3]={PIECE_BOUND[pib*4+0],PIECE_BOUND[pib*4+1],PIECE_BOUND[pib*4+2]};
            double pbw[3]; dxform_point(Tb,pbc,pbw); double pbr=PIECE_BOUND[pib*4+3];
            double pd[3]; dv3sub(paw,pbw,pd);
            double pbgap=dv3norm(pd)-(par+pbr);
            double gap;
            if (pbgap>margin){ gap=pbgap; }                          // per-piece broad reject
            else if (PIECE_TYPE[pia]==1 && PIECE_TYPE[pib]==1){
                // sphere<->sphere: exact analytic signed separation (piece bound IS the sphere).
                gap = pbgap;
            } else {
                int it;
                gap=dgjk_gap_warp(pia,Ta,pib,Tb,lane,&it);
                *iters+=it;
            }
            if (gap<mg) mg=gap;
        }
    }
    return mg;
}

// ================= Stage 5: full + incremental link-pair checker =================
// Colliding verdict for a NON-GJK checked pair g (primitive or cluster-SDF). Single-lane cheap.
__device__ inline int linkpair_colliding_nongjk(int g, const float* T, float margin) {
    int a = PAIR_LINK_A[g], b = PAIR_LINK_B[g], any = 0;
    if (PAIR_TYPE[g] == PAIR_PRIMITIVE) {
        prim_linkpair_gap(a, b, T, margin, &any);
    } else {  // PAIR_CLUSTER_SDF
        int cid, cl_link, limb;
        if (LINK_CLUSTER[a] >= 0) { cid = LINK_CLUSTER[a]; cl_link = a; limb = b; }
        else                      { cid = LINK_CLUSTER[b]; cl_link = b; limb = a; }
        int ev = 0;
        cluster_linkpair_gap(cid, cl_link, limb, T, margin, &any, &ev);
    }
    return any;
}
// Colliding verdict for a GJK pair g (full warp). Uses f64 transforms.
__device__ inline int linkpair_colliding_gjk(int g, const double* Td, float margin, int lane) {
    int it;
    double gap = gjk_linkpair_gap_d(PAIR_LINK_A[g], PAIR_LINK_B[g], Td, (double)margin, lane, &it);
    return gap < margin;
}

}  // namespace g1sc
