// CUDA native ENVIRONMENT collision checker (checkpoint: CUDA env parity, Task A).
// Exact device port of production/native_env_checker.py. Behaviourally isolated like the
// self-collision sidecar; reuses its on-device f64 FK (sidecar_fk_d) + typed-piece machinery
// (dpiece_support_warp, dclosest_simplex, g_cverts, PIECE_*/LINK_PIECE_OFF).
//
// Gating verdict = wall(box) AND floor(plane) AND intended-contact(proxy vs assigned hold region).
// Physical hold overlap + unrelated-hold overlap are DIAGNOSTIC only (never gate) -- policy
// wall_floor_contact_v1. All boundary math is f64 to match the numpy/GJK CPU oracle bitwise-closely.
#pragma once
#include "collision_sidecar.cuh"

namespace g1sc {

// ---- runtime-uploaded scene (device pointers set by sidecar_upload_scene) --------------------
__device__ int          g_env_nobj;
__device__ const int*   g_env_otype;    // [nobj] 0 box(wall) / 1 plane(floor) / 2 sphere(hold)
__device__ const double* g_env_broad;   // [nobj*4] broad center xyz, radius
__device__ const double* g_env_box;     // [nobj*10] pos xyz, quat wxyz, half xyz
__device__ const double* g_env_plane;   // [nobj*6] point xyz, normal xyz
__device__ const double* g_env_sph;     // [nobj*4] center xyz, radius
__device__ const double* g_env_reg;     // [nobj*16] fp xyz, n xyz, t1 xyz, t2 xyz, tup, tlat, outtol, pentol
__device__ const int*    g_env_rallow;  // [nobj] bit0 hand, bit1 foot
// proxies (4 limbs, order left_hand,right_hand,left_foot,right_foot)
__device__ const int*    g_env_plink;   // [4] contact link index (FK link id)
__device__ const double* g_env_poff;    // [4*3] local offset
__device__ const double* g_env_ptype;   // [4] 1 hand / 2 foot   (double for packing simplicity)

__device__ __constant__ double DESIGNATED_PEN_LIMIT = 0.03;  // matches CPU

// ---- base compose: T_world[L] = T_base * T_fixed[L] (f64, column-major) ----------------------
__device__ __forceinline__ void dquat2mat_cols(const double* qwxyz, double* Rcol /*9, col-major*/) {
    double w=qwxyz[0], x=qwxyz[1], y=qwxyz[2], z=qwxyz[3];
    Rcol[0]=1-2*(y*y+z*z); Rcol[3]=2*(x*y-w*z);   Rcol[6]=2*(x*z+w*y);
    Rcol[1]=2*(x*y+w*z);   Rcol[4]=1-2*(x*x+z*z); Rcol[7]=2*(y*z-w*x);
    Rcol[2]=2*(x*z-w*y);   Rcol[5]=2*(y*z+w*x);   Rcol[8]=1-2*(x*x+y*y);
}
// Tw = Tbase(4x4 col-major) * Tl. Build Tbase once, multiply each link transform.
__device__ inline void dworld_from_base(const double* base_pos, const double* base_quat,
                                        const double* Tfix, double* Tworld) {
    double Rb[9]; dquat2mat_cols(base_quat, Rb);
    double Tb[16];
    Tb[0]=Rb[0];Tb[1]=Rb[1];Tb[2]=Rb[2];Tb[3]=0;
    Tb[4]=Rb[3];Tb[5]=Rb[4];Tb[6]=Rb[5];Tb[7]=0;
    Tb[8]=Rb[6];Tb[9]=Rb[7];Tb[10]=Rb[8];Tb[11]=0;
    Tb[12]=base_pos[0];Tb[13]=base_pos[1];Tb[14]=base_pos[2];Tb[15]=1;
    for (int L=0; L<N_LINKS; ++L) dmat4_mul(Tb, &Tfix[L*16], &Tworld[L*16]);
}

// ---- analytic sphere narrow phases (exact ports of native_env_checker) -----------------------
__device__ __forceinline__ double dsphere_plane_gap(const double* c, double r, const double* pt, const double* n) {
    return (n[0]*(c[0]-pt[0])+n[1]*(c[1]-pt[1])+n[2]*(c[2]-pt[2])) - r;
}
__device__ __forceinline__ double dsphere_box_gap(const double* c, double r, const double* box) {
    const double* pos=&box[0]; double Rc[9]; dquat2mat_cols(&box[3], Rc); const double* half=&box[7];
    double d[3]={c[0]-pos[0],c[1]-pos[1],c[2]-pos[2]};
    // local = R^T d  (Rc is column-major world<-local; R^T d picks columns as rows)
    double loc[3]={Rc[0]*d[0]+Rc[1]*d[1]+Rc[2]*d[2],
                   Rc[3]*d[0]+Rc[4]*d[1]+Rc[5]*d[2],
                   Rc[6]*d[0]+Rc[7]*d[1]+Rc[8]*d[2]};
    double q0=fabs(loc[0])-half[0], q1=fabs(loc[1])-half[1], q2=fabs(loc[2])-half[2];
    double ox=q0>0?q0:0, oy=q1>0?q1:0, oz=q2>0?q2:0;
    double d_out=sqrt(ox*ox+oy*oy+oz*oz);
    double mx=q0>q1?(q0>q2?q0:q2):(q1>q2?q1:q2);
    double d_in=mx<0?mx:0;
    return (d_out+d_in)-r;
}
__device__ __forceinline__ double dsphere_sphere_gap(const double* c, double r, const double* sph) {
    double dx=c[0]-sph[0],dy=c[1]-sph[1],dz=c[2]-sph[2];
    return sqrt(dx*dx+dy*dy+dz*dz)-(r+sph[3]);
}

// ---- box support (analytic, lane-uniform) for GJK; matches gjk.py box support ----------------
__device__ __forceinline__ void dbox_support(const double* box, const double* d, double* out) {
    const double* pos=&box[0]; double Rc[9]; dquat2mat_cols(&box[3], Rc); const double* half=&box[7];
    double dl[3]={Rc[0]*d[0]+Rc[1]*d[1]+Rc[2]*d[2],
                  Rc[3]*d[0]+Rc[4]*d[1]+Rc[5]*d[2],
                  Rc[6]*d[0]+Rc[7]*d[1]+Rc[8]*d[2]};
    double s[3]={dl[0]>=0?half[0]:-half[0], dl[1]>=0?half[1]:-half[1], dl[2]>=0?half[2]:-half[2]};
    out[0]=pos[0]+Rc[0]*s[0]+Rc[3]*s[1]+Rc[6]*s[2];
    out[1]=pos[1]+Rc[1]*s[0]+Rc[4]*s[1]+Rc[7]*s[2];
    out[2]=pos[2]+Rc[2]*s[0]+Rc[5]*s[1]+Rc[8]*s[2];
}
// point "support" (a fixed point regardless of direction) -- for hull-vs-sphere GJK (sphere r
// subtracted after). Lane-uniform.
__device__ __forceinline__ void dpoint_support(const double* p, double* out){ out[0]=p[0];out[1]=p[1];out[2]=p[2]; }

// GJK between robot PIECE pi (typed, transform Tp) and a target whose support is (box|point).
// tgt_kind: 0 box (params=box[10]), 1 point (params=pt[3]). Returns signed gap (>=0) or -1e-9.
__device__ inline double dgjk_piece_target(int pi, const double* Tp, int tgt_kind, const double* tparam,
                                           int lane, int* iters) {
    const int MAXIT=64; const double TOL=1e-9;
    double W[4][3]; int n=0; double d0[3]={1,0,0}; double sa[3],sb[3],negd[3];
    dpiece_support_warp(pi,Tp,d0,lane,sa);
    negd[0]=-1;negd[1]=0;negd[2]=0;
    if (tgt_kind==0) dbox_support(tparam,negd,sb); else dpoint_support(tparam,sb);
    W[0][0]=sa[0]-sb[0];W[0][1]=sa[1]-sb[1];W[0][2]=sa[2]-sb[2]; n=1;
    double closest[3]={W[0][0],W[0][1],W[0][2]};
    for (int it=1; it<=MAXIT; ++it){
        double d[3]={-closest[0],-closest[1],-closest[2]};
        if (dv3dot(d,d)<TOL){*iters=it; return -1e-9;}
        dpiece_support_warp(pi,Tp,d,lane,sa);
        negd[0]=-d[0];negd[1]=-d[1];negd[2]=-d[2];
        if (tgt_kind==0) dbox_support(tparam,negd,sb); else dpoint_support(tparam,sb);
        double a[3]={sa[0]-sb[0],sa[1]-sb[1],sa[2]-sb[2]};
        double ad=dv3dot(a,d), cd=dv3dot(closest,d);
        if (ad-cd < TOL*(1.0+fabs(cd))){*iters=it; return dv3norm(closest);}
        W[n][0]=a[0];W[n][1]=a[1];W[n][2]=a[2]; ++n;
        int keep[4],enc; double cp[3]; int nk=dclosest_simplex(W,n,cp,keep,&enc);
        double NW[4][3];
        for(int i=0;i<nk;++i){NW[i][0]=W[keep[i]][0];NW[i][1]=W[keep[i]][1];NW[i][2]=W[keep[i]][2];}
        for(int i=0;i<nk;++i){W[i][0]=NW[i][0];W[i][1]=NW[i][1];W[i][2]=NW[i][2];}
        n=nk; closest[0]=cp[0];closest[1]=cp[1];closest[2]=cp[2];
        if (enc){*iters=it; return -1e-9;}
    }
    *iters=MAXIT; return (dv3norm(closest)<1e-4)?-1e-9:dv3norm(closest);
}

// hull piece vs plane: min over verts of (v - point) . normal (warp argmin). matches CPU min-vertex.
__device__ inline double dhull_plane_gap(int pi, const double* Tp, const double* pt, const double* nrm, int lane){
    int v0=PIECE_VERT_OFF[pi], v1=PIECE_VERT_OFF[pi+1];
    double best=1e300;
    for (int i=v0+lane;i<v1;i+=32){
        double vw[3]; dxform_point(Tp,&g_cverts[i*3],vw);
        double g=(vw[0]-pt[0])*nrm[0]+(vw[1]-pt[1])*nrm[1]+(vw[2]-pt[2])*nrm[2];
        if (g<best) best=g;
    }
    #pragma unroll
    for (int off=16;off>0;off>>=1){ double o=__shfl_down_sync(0xffffffffu,best,off); if(o<best)best=o; }
    return __shfl_sync(0xffffffffu,best,0);
}

// one robot piece (link L, piece pi) vs one env object -> signed gap (min over the object's pieces;
// each env object has exactly one piece here). Exact per native_env_checker._piece_vs_object.
__device__ inline double denv_piece_vs_object(int pi, const double* Tp, int oid, int lane){
    int ot=g_env_otype[oid];
    int is_hull = (PIECE_TYPE[pi]==0);
    if (ot==1){ // plane
        const double* pl=&g_env_plane[oid*6];
        if (is_hull) return dhull_plane_gap(pi,Tp,&pl[0],&pl[3],lane);
        // sphere piece
        const float* sp=&PIECE_SPHERE[pi*4]; double c[3]; double cl[3]={sp[0],sp[1],sp[2]}; dxform_point(Tp,cl,c);
        return dsphere_plane_gap(c,(double)sp[3],&pl[0],&pl[3]);
    } else if (ot==0){ // box (wall)
        const double* bx=&g_env_box[oid*10];
        if (is_hull){ int it; return dgjk_piece_target(pi,Tp,0,bx,lane,&it); }
        const float* sp=&PIECE_SPHERE[pi*4]; double c[3]; double cl[3]={sp[0],sp[1],sp[2]}; dxform_point(Tp,cl,c);
        return dsphere_box_gap(c,(double)sp[3],bx);
    } else { // sphere (hold solid)
        const double* sph=&g_env_sph[oid*4];
        if (is_hull){ int it; double d=dgjk_piece_target(pi,Tp,1,&sph[0],lane,&it); return d-sph[3]; }
        const float* sp=&PIECE_SPHERE[pi*4]; double c[3]; double cl[3]={sp[0],sp[1],sp[2]}; dxform_point(Tp,cl,c);
        return dsphere_sphere_gap(c,(double)sp[3],sph);
    }
}

// intended-contact proxy vs region test (exact port of _proxy_in_region). limb_type 1 hand/2 foot.
__device__ inline int dproxy_in_region(const double* pc, int oid, int limb_type, int require_type){
    const double* r=&g_env_reg[oid*16];
    int allow=g_env_rallow[oid];
    if (require_type && !((limb_type==1 && (allow&1)) || (limb_type==2 && (allow&2)))) return 0;
    double dlt[3]={pc[0]-r[0],pc[1]-r[1],pc[2]-r[2]};
    double d_n=r[3]*dlt[0]+r[4]*dlt[1]+r[5]*dlt[2];
    double d1=r[6]*dlt[0]+r[7]*dlt[1]+r[8]*dlt[2];
    double d2=r[9]*dlt[0]+r[10]*dlt[1]+r[11]*dlt[2];
    double outtol=r[14], pentol=r[15], tup=r[12], tlat=r[13];
    if (d_n>outtol || d_n<-pentol) return 0;
    if (fabs(d1)>tup || fabs(d2)>tlat) return 0;
    return 1;
}

// ---- the env kernel: one WARP per candidate (GJK needs a warp). Output flags[b*6]. ----
// flags: [0]wall_valid [1]floor_valid [2]intended_valid [3]env_valid [4]phys_overlap [5]unrelated_overlap
#define ENV_WPB 2
__global__ void env_check_kernel(const double* __restrict__ q, const int* __restrict__ assign,
                                 unsigned char* __restrict__ flags, int B){
    __shared__ double shTfix[ENV_WPB*N_LINKS*16];
    __shared__ double shTw[ENV_WPB*N_LINKS*16];
    int w=threadIdx.x>>5, lane=threadIdx.x&31;
    int b=blockIdx.x*ENV_WPB+w; if (b>=B) return;
    double* Tfix=&shTfix[w*N_LINKS*16]; double* Tw=&shTw[w*N_LINKS*16];
    const double* qb=&q[b*36]; const int* asg=&assign[b*4];
    if (lane==0){
        // sidecar_fk_d expects joints (float); q joints are qb[7..35]. Convert to float buffer.
        float jf[N_JOINTS];
        for (int i=0;i<N_JOINTS;++i) jf[i]=(float)qb[7+i];
        sidecar_fk_d(jf, Tfix);
        dworld_from_base(&qb[0], &qb[3], Tfix, Tw);
    }
    __syncwarp();

    int wall_ok=1, floor_ok=1, phys=0;
    // PHYSICAL wall/floor (gating) + physical-hold diagnostic, over every robot piece.
    for (int L=0; L<N_LINKS; ++L){
        const double* Tp=&Tw[L*16];
        for (int pi=LINK_PIECE_OFF[L]; pi<LINK_PIECE_OFF[L+1]; ++pi){
            // per-piece world bounding sphere for broad phase
            double pbc[3]={PIECE_BOUND[pi*4+0],PIECE_BOUND[pi*4+1],PIECE_BOUND[pi*4+2]};
            double pbw[3]; dxform_point(Tp,pbc,pbw); double pbr=PIECE_BOUND[pi*4+3];
            for (int o=0;o<g_env_nobj;++o){
                int ot=g_env_otype[o];
                // broad phase (skip for floor/plane: infinite)
                if (ot!=1){
                    double bc[3]={g_env_broad[o*4+0],g_env_broad[o*4+1],g_env_broad[o*4+2]};
                    double dd[3]={pbw[0]-bc[0],pbw[1]-bc[1],pbw[2]-bc[2]};
                    if (dv3norm(dd)-(pbr+g_env_broad[o*4+3])>0.0) continue;
                }
                double g=denv_piece_vs_object(pi,Tp,o,lane);
                double thresh=0.0;
                if (ot==2){
                    // designated-link shallow-rest allowance at its OWN assigned hold
                    for (int limb=0; limb<4; ++limb)
                        if (asg[limb]==o && g_env_plink[limb]==L){ thresh=-DESIGNATED_PEN_LIMIT; break; }
                    if (g<thresh){ phys=1; }         // diagnostic only
                    continue;
                }
                if (g>=thresh) continue;
                if (ot==0) wall_ok=0; else floor_ok=0;   // gating
            }
        }
    }
    // reduce wall/floor/phys across lanes (hull GJK computed same on all lanes; flags set by any)
    #pragma unroll
    for (int off=16;off>0;off>>=1){
        wall_ok = min(wall_ok, __shfl_down_sync(0xffffffffu,wall_ok,off));
        floor_ok= min(floor_ok,__shfl_down_sync(0xffffffffu,floor_ok,off));
        phys    = max(phys,   __shfl_down_sync(0xffffffffu,phys,off));
    }
    wall_ok=__shfl_sync(0xffffffffu,wall_ok,0);
    floor_ok=__shfl_sync(0xffffffffu,floor_ok,0);
    phys=__shfl_sync(0xffffffffu,phys,0);

    int intended_ok=1, unrelated=0;
    if (lane==0){
        for (int limb=0; limb<4; ++limb){
            int oid=asg[limb]; if (oid<0) continue;
            int L=g_env_plink[limb];
            const double* Tp=&Tw[L*16];
            // proxy world = link_xpos + R * offset
            const double* off=&g_env_poff[limb*3];
            double pc[3];
            pc[0]=Tp[12]+Tp[0]*off[0]+Tp[4]*off[1]+Tp[8]*off[2];
            pc[1]=Tp[13]+Tp[1]*off[0]+Tp[5]*off[1]+Tp[9]*off[2];
            pc[2]=Tp[14]+Tp[2]*off[0]+Tp[6]*off[1]+Tp[10]*off[2];
            int lt=(int)(g_env_ptype[limb]+0.5);
            if (!dproxy_in_region(pc,oid,lt,1)) intended_ok=0;
            for (int o=0;o<g_env_nobj;++o){
                if (g_env_otype[o]!=2 || o==oid) continue;
                if (dproxy_in_region(pc,o,lt,0)) unrelated++;
            }
        }
        int env_ok = wall_ok && floor_ok && intended_ok;
        unsigned char* f=&flags[b*6];
        f[0]=wall_ok; f[1]=floor_ok; f[2]=intended_ok; f[3]=env_ok;
        f[4]=(phys?1:0); f[5]=(unrelated>0?1:0);
    }
}

}  // namespace g1sc
