// FROZEN benchmark reference data (paper's 59-sphere Panda model), vendored from the former
// csrc/robots/panda.cuh when the kernel migrated to GRiD grid_collision (which now bakes the SAME
// spheres, sourced from external/foam/assets/panda/smaller_panda_spherized.urdf, into grid.cuh).
// This copy is the INDEPENDENT oracle for benchmark/panda_model.py -> panda_collision.py: it lets the
// Table II collision-free column validate any solver's returned q against the exact published model.
// Data-only (sphere arrays + fixed transforms + joint types + #defines); the pRRTC fk<>/self-collision
// device code was intentionally dropped. Do not wire into the build.
namespace ppln_reference {
namespace ppln::collision {
    #define PANDA_SPHERE_COUNT 59
    #define PANDA_JOINT_COUNT 8
    #define PANDA_SELF_CC_RANGE_COUNT 24
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    #define BATCH_SIZE 16

    __device__ __constant__ float4 panda_spheres_array[59] = {
        { 0.0f, 0.0f, 0.05f, 0.08f },
        { 0.0f, -0.08f, 0.0f, 0.06f },
        { 0.0f, -0.03f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.12f, 0.06f },
        { 0.0f, 0.0f, -0.17f, 0.06f },
        { 0.0f, 0.0f, 0.03f, 0.06f },
        { 0.0f, 0.0f, 0.08f, 0.06f },
        { 0.0f, -0.12f, 0.0f, 0.06f },
        { 0.0f, -0.17f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.1f, 0.06f },
        { 0.0f, 0.0f, -0.06f, 0.05f },
        { 0.08f, 0.06f, 0.0f, 0.055f },
        { 0.08f, 0.02f, 0.0f, 0.055f },
        { -0.08f, 0.095f, 0.0f, 0.06f },
        { 0.0f, 0.0f, 0.02f, 0.055f },
        { 0.0f, 0.0f, 0.06f, 0.055f },
        { -0.08f, 0.06f, 0.0f, 0.055f },
        { 0.0f, 0.055f, 0.0f, 0.06f },
        { 0.0f, 0.075f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.22f, 0.06f },
        { 0.0f, 0.05f, -0.18f, 0.05f },
        { 0.01f, 0.08f, -0.14f, 0.025f },
        { 0.01f, 0.085f, -0.11f, 0.025f },
        { 0.01f, 0.09f, -0.08f, 0.025f },
        { 0.01f, 0.095f, -0.05f, 0.025f },
        { -0.01f, 0.08f, -0.14f, 0.025f },
        { -0.01f, 0.085f, -0.11f, 0.025f },
        { -0.01f, 0.09f, -0.08f, 0.025f },
        { -0.01f, 0.095f, -0.05f, 0.025f },
        { 0.0f, 0.0f, 0.0f, 0.05f },
        { 0.08f, -0.01f, 0.0f, 0.05f },
        { 0.08f, 0.035f, 0.0f, 0.052f },
        { 0.0f, 0.0f, 0.07f, 0.05f },
        { 0.02f, 0.04f, 0.08f, 0.025f },
        { 0.04f, 0.02f, 0.08f, 0.025f },
        { 0.04f, 0.06f, 0.085f, 0.02f },
        { 0.06f, 0.04f, 0.085f, 0.02f },
        { -0.053033f, -0.053033f, 0.117f, 0.028f },
        { -0.03182f, -0.03182f, 0.117f, 0.028f },
        { -0.010607f, -0.010607f, 0.117f, 0.028f },
        { 0.010607f, 0.010607f, 0.117f, 0.028f },
        { 0.03182f, 0.03182f, 0.117f, 0.028f },
        { 0.053033f, 0.053033f, 0.117f, 0.028f },
        { -0.053033f, -0.053033f, 0.137f, 0.026f },
        { -0.03182f, -0.03182f, 0.137f, 0.026f },
        { -0.010607f, -0.010607f, 0.137f, 0.026f },
        { 0.010607f, 0.010607f, 0.137f, 0.026f },
        { 0.03182f, 0.03182f, 0.137f, 0.026f },
        { 0.053033f, 0.053033f, 0.137f, 0.026f },
        { -0.053033f, -0.053033f, 0.157f, 0.024f },
        { -0.03182f, -0.03182f, 0.157f, 0.024f },
        { -0.010607f, -0.010607f, 0.157f, 0.024f },
        { 0.010607f, 0.010607f, 0.157f, 0.024f },
        { 0.03182f, 0.03182f, 0.157f, 0.024f },
        { 0.053033f, 0.053033f, 0.157f, 0.024f },
        { 0.056569f, 0.056569f, 0.1874f, 0.012f },
        { 0.051619f, 0.051619f, 0.2094f, 0.012f },
        { -0.056569f, -0.056569f, 0.1874f, 0.012f },
        { -0.051619f, -0.051619f, 0.2094f, 0.012f }
    };

    __device__ __constant__ float panda_fixed_transforms[] = {
        // joint 0
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.333,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, -0.316,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.0825,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, -0.0825,
        0.0, 0.0, 1.0, 0.384,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, 0.0, 0.0, 0.088,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };

    __device__ __constant__ int panda_sphere_to_joint[] = {
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    };

    __device__ __constant__ int panda_joint_types[] = {
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        5
    };

    __device__ __constant__ int panda_self_cc_ranges[24][3] = {
        { 0, 17, 58 },
        { 1, 17, 58 },
        { 2, 17, 58 },
        { 3, 17, 58 },
        { 4, 17, 58 },
        { 5, 17, 28 },
        { 5, 32, 58 },
        { 6, 17, 28 },
        { 6, 32, 58 },
        { 7, 17, 28 },
        { 7, 32, 58 },
        { 8, 17, 28 },
        { 8, 32, 58 },
        { 17, 32, 58 },
        { 18, 32, 58 },
        { 19, 32, 58 },
        { 20, 32, 58 },
        { 21, 32, 58 },
        { 22, 32, 58 },
        { 23, 32, 58 },
        { 24, 32, 58 },
        { 25, 32, 58 },
        { 26, 32, 58 },
        { 27, 32, 58 }
    };

}
