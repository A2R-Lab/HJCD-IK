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
        Sphere<DataT> *spheres;
        unsigned int num_spheres;

        Capsule<DataT> *capsules;
        unsigned int num_capsules;

        Capsule<DataT> *z_aligned_capsules;
        unsigned int num_z_aligned_capsules;

        Cylinder<DataT> *cylinders;
        unsigned int num_cylinders;

        Cuboid<DataT> *cuboids;
        unsigned int num_cuboids;

        Cuboid<DataT> *z_aligned_cuboids;
        unsigned int num_z_aligned_cuboids;

        // HeightField<DataT> *heightfields;
        // unsigned int num_heightfields;

        Environment() = default;

        ~Environment() {
            delete[] spheres;
            delete[] capsules;
            delete[] cuboids;
            delete[] z_aligned_capsules;
            delete[] cylinders;
            delete[] z_aligned_cuboids;
        }
    };
}  // namespace ppln::collision