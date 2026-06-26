#include "collision/prrtc_collision.cuh"
#include <cuda_runtime.h>

namespace pRRTC {

void setup_environment_on_device(
    ppln::collision::Environment<float>*& d_env,
    const ppln::collision::Environment<float>& h_env)
{
    cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));
    cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

    if (h_env.num_spheres > 0) {
        ppln::collision::Sphere<float>* d_spheres;
        cudaMalloc(&d_spheres, sizeof(*d_spheres) * h_env.num_spheres);
        cudaMemcpy(d_spheres, h_env.spheres,
                   sizeof(*d_spheres) * h_env.num_spheres,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(d_spheres), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(h_env.num_spheres), cudaMemcpyHostToDevice);
    }

    if (h_env.num_capsules > 0) {
        ppln::collision::Capsule<float>* d_capsules;
        cudaMalloc(&d_capsules, sizeof(*d_capsules) * h_env.num_capsules);
        cudaMemcpy(d_capsules, h_env.capsules,
                   sizeof(*d_capsules) * h_env.num_capsules,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(d_capsules), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(h_env.num_capsules), cudaMemcpyHostToDevice);
    }

    if (h_env.num_cuboids > 0) {
        ppln::collision::Cuboid<float>* d_cuboids;
        cudaMalloc(&d_cuboids, sizeof(*d_cuboids) * h_env.num_cuboids);
        cudaMemcpy(d_cuboids, h_env.cuboids,
                   sizeof(*d_cuboids) * h_env.num_cuboids,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(d_cuboids), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(h_env.num_cuboids), cudaMemcpyHostToDevice);
    }
}

void cleanup_environment_on_device(
    ppln::collision::Environment<float>* d_env,
    const ppln::collision::Environment<float>& h_env)
{
    ppln::collision::Sphere<float>* d_spheres = nullptr;
    if (h_env.num_spheres > 0) {
        cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(d_spheres), cudaMemcpyDeviceToHost);
        cudaFree(d_spheres);
    }

    ppln::collision::Capsule<float>* d_capsules = nullptr;
    if (h_env.num_capsules > 0) {
        cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(d_capsules), cudaMemcpyDeviceToHost);
        cudaFree(d_capsules);
    }

    ppln::collision::Cuboid<float>* d_cuboids = nullptr;
    if (h_env.num_cuboids > 0) {
        cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(d_cuboids), cudaMemcpyDeviceToHost);
        cudaFree(d_cuboids);
    }

    cudaFree(d_env);
}

}