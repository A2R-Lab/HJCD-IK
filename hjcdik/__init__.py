import os

if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = os.path.join(conda_prefix, "Library", "bin")
        if os.path.isdir(conda_bin):
            os.add_dll_directory(conda_bin)

from ._hjcdik import generate_solutions, sample_targets, num_joints
