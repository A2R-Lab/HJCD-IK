#pragma once

#include <vector>
#include <optional>
#include "shapes.hh"

/* Adapted from https://github.com/KavrakiLab/vamp/blob/main/src/impl/vamp/collision/environment.hh */

namespace ppln::collision
{
    template <typename DataT>
    struct Environment
    {
        Sphere<DataT> *spheres = nullptr;
        unsigned int num_spheres = 0;

        Capsule<DataT> *capsules = nullptr;
        unsigned int num_capsules = 0;

        Capsule<DataT> *z_aligned_capsules = nullptr;
        unsigned int num_z_aligned_capsules = 0;

        Cylinder<DataT> *cylinders = nullptr;
        unsigned int num_cylinders = 0;

        Cuboid<DataT> *cuboids = nullptr;
        unsigned int num_cuboids = 0;

        Cuboid<DataT> *z_aligned_cuboids = nullptr;
        unsigned int num_z_aligned_cuboids = 0;

        // HeightField<DataT> *heightfields;
        // unsigned int num_heightfields;

        Environment() = default;

        Environment(const Environment&) = delete;
        auto operator=(const Environment&) -> Environment& = delete;

        Environment(Environment&& other) noexcept
          : spheres(other.spheres)
          , num_spheres(other.num_spheres)
          , capsules(other.capsules)
          , num_capsules(other.num_capsules)
          , z_aligned_capsules(other.z_aligned_capsules)
          , num_z_aligned_capsules(other.num_z_aligned_capsules)
          , cylinders(other.cylinders)
          , num_cylinders(other.num_cylinders)
          , cuboids(other.cuboids)
          , num_cuboids(other.num_cuboids)
          , z_aligned_cuboids(other.z_aligned_cuboids)
          , num_z_aligned_cuboids(other.num_z_aligned_cuboids)
        {
            other.spheres = nullptr;
            other.num_spheres = 0;
            other.capsules = nullptr;
            other.num_capsules = 0;
            other.z_aligned_capsules = nullptr;
            other.num_z_aligned_capsules = 0;
            other.cylinders = nullptr;
            other.num_cylinders = 0;
            other.cuboids = nullptr;
            other.num_cuboids = 0;
            other.z_aligned_cuboids = nullptr;
            other.num_z_aligned_cuboids = 0;
        }

        auto operator=(Environment&& other) noexcept -> Environment&
        {
            if (this == &other) {
                return *this;
            }

            reset();

            spheres = other.spheres;
            num_spheres = other.num_spheres;
            capsules = other.capsules;
            num_capsules = other.num_capsules;
            z_aligned_capsules = other.z_aligned_capsules;
            num_z_aligned_capsules = other.num_z_aligned_capsules;
            cylinders = other.cylinders;
            num_cylinders = other.num_cylinders;
            cuboids = other.cuboids;
            num_cuboids = other.num_cuboids;
            z_aligned_cuboids = other.z_aligned_cuboids;
            num_z_aligned_cuboids = other.num_z_aligned_cuboids;

            other.spheres = nullptr;
            other.num_spheres = 0;
            other.capsules = nullptr;
            other.num_capsules = 0;
            other.z_aligned_capsules = nullptr;
            other.num_z_aligned_capsules = 0;
            other.cylinders = nullptr;
            other.num_cylinders = 0;
            other.cuboids = nullptr;
            other.num_cuboids = 0;
            other.z_aligned_cuboids = nullptr;
            other.num_z_aligned_cuboids = 0;

            return *this;
        }

        void reset()
        {
            delete[] spheres;
            delete[] capsules;
            delete[] cuboids;
            delete[] z_aligned_capsules;
            delete[] cylinders;
            delete[] z_aligned_cuboids;

            spheres = nullptr;
            num_spheres = 0;
            capsules = nullptr;
            num_capsules = 0;
            z_aligned_capsules = nullptr;
            num_z_aligned_capsules = 0;
            cylinders = nullptr;
            num_cylinders = 0;
            cuboids = nullptr;
            num_cuboids = 0;
            z_aligned_cuboids = nullptr;
            num_z_aligned_cuboids = 0;
        }

        ~Environment() {
            reset();
        }
    };
}  // namespace ppln::collision
