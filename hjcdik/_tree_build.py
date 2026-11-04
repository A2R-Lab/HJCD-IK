"""Runtime builds of fixed-base TREE models with several target frames (humanoid-v1 M2b).

Opt-in and separate from `hjcdik._build` (the single-target serial-chain solver). It reuses the same
GRiD consumer-artifact store (`grid_rbd.build_artifact`): the identity covers the URDF, the codegen
options, our sources and the staged side headers by content (hjcd_joint_model.h + hjcd_targets.cuh,
which carry the validated target set, its full rigid tools and masks), defines, toolchain and arch.
An artifact currently exposes target metadata and batched base-frame target FK only (no IK).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import shutil
import sys
import sysconfig
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ._build import BuildError, CONSUMER, _check_name, _grid_rbd, _nvcc_version, robot_name, source_root
from ._robot_model import describe_tree_robot, model_sha256, stage_tree_headers, tree_fixed_frame_index

TREE_ABI = 7   # must equal HJCDIK_TREE_ABI in csrc/bindings/tree_fk_module.cpp
_LOADED: dict = {}


@dataclass(frozen=True)
class TreeArtifact:
    name: str
    robot: str
    path: Path
    header: Path
    content_key: str
    cuda_arch: int
    module_name: str
    compiled: bool
    urdf_sha256: str
    targets_sha256: str
    model_sha256: str
    cuda_compiler_version: str = ""


def _module_name(name: str) -> str:
    return "_hjcdik_tree_" + hashlib.sha256(name.encode()).hexdigest()[:12]


def load_targets_spec(path):
    """(names-and-frames list, sha256 of the spec bytes). Rejects a spec pinned to another URDF."""
    data = Path(path).expanduser().read_bytes()
    spec = json.loads(data)
    if spec.get("schema") != 1 or spec.get("kind") != "fixed_base_tree":
        raise BuildError(f"{path}: expected a schema-1 fixed_base_tree target spec")
    targets = [(t["name"], t["frame"]) for t in spec["targets"]]
    return spec, targets, hashlib.sha256(data).hexdigest()


def _parse(urdf_path):
    _grid_rbd()
    from URDFParser import URDFParser  # noqa: E402  (grid-rbd ships it)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return URDFParser().parse(str(urdf_path), floating_base=False)


def _fixed_indices(urdf_path, urdf_sha, frames, cache_dir):
    """Each target's s_XmatsHom slot, read back from kinematics-only GRiD generations (cached per URDF)."""
    cache = Path(cache_dir).expanduser().resolve() / "hjcdik-inputs" / f"tree-frames-{urdf_sha}.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    if all(f in known for f in frames):
        return [int(known[f]) for f in frames]
    from grid_codegen import GRiDCodeGenerator  # noqa: E402
    for frame in frames:
        if frame in known:
            continue
        with tempfile.TemporaryDirectory(prefix="hjcdik-tree-frame-") as tmp:
            out = Path(tmp) / "grid.cuh"
            with contextlib.redirect_stdout(io.StringIO()):
                GRiDCodeGenerator(_parse(urdf_path), DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid").gen_all_code(
                    codegen_profile="kinematics", vendor_glass=False, include_homogenous_transforms=True,
                    fixed_target_name=frame, output_path=str(out))
            known[frame] = tree_fixed_frame_index(out.read_text(), frame)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, sort_keys=True))
    return [int(known[f]) for f in frames]


def describe(urdf, targets_spec):
    """Validate the URDF + target spec and return (model description, spec sha256, urdf sha256)."""
    urdf_path = Path(urdf).expanduser().resolve()
    if not urdf_path.is_file():
        raise BuildError(f"URDF not found: {urdf_path}")
    spec, targets, spec_sha = load_targets_spec(targets_spec)
    urdf_sha = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if spec.get("urdf_sha256") not in (None, urdf_sha):
        raise BuildError(f"target spec is pinned to URDF {spec['urdf_sha256'][:12]} but {urdf_path.name} is {urdf_sha[:12]}")
    try:
        model = describe_tree_robot(_parse(urdf_path), urdf_path, targets)
    except ValueError as error:
        raise BuildError(f"unsupported tree model {urdf_path}: {error}") from error
    return urdf_path, model, spec_sha, urdf_sha


def build_tree(urdf, targets_spec, *, name: Optional[str] = None, cache_dir=None, cuda_arch: Optional[int] = None,
               force_rebuild: bool = False, plan_only: bool = False):
    """Build (or reuse) the tree-model artifact for `urdf` + `targets_spec` and return a `TreeArtifact`."""
    grid_rbd = _grid_rbd()
    urdf_path, model, spec_sha, urdf_sha = describe(urdf, targets_spec)
    name = _check_name(name or f"{robot_name(urdf_path)}-tree")
    store = cache_dir if cache_dir is not None else grid_rbd.default_cache_dir()
    frames = [t["fixed_joint"] for t in model["targets"]]
    indices = _fixed_indices(urdf_path, urdf_sha, frames, store)
    side = stage_tree_headers(model, indices, store)
    root = source_root()
    try:
        import pybind11
    except ImportError as error:
        raise BuildError("runtime robot builds need pybind11: pip install 'hjcdik[build]'") from error
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    module = _module_name(name)
    defines = {"HJCDIK_TREE_MODULE_NAME": module, "HJCDIK_TREE_ABI": str(TREE_ABI),
               "HJCDIK_ROBOT_NAME": f'"{robot_name(urdf_path)}"', "HJCDIK_TARGETS_SHA256": f'"{spec_sha}"'}
    try:
        result = grid_rbd.build_artifact(
            CONSUMER, name, urdf_path, cache_dir=cache_dir, cuda_arch=cuda_arch, force_rebuild=force_rebuild,
            plan_only=plan_only, ee_joint_names=[frames[0]], algorithm_list="kinematics", emit_plant=False,
            collision=False,
            sources=[root / "tree" / "hjcd_tree_fk.cu", root / "tree" / "hjcd_tree_lm.cu", root / "bindings" / "tree_fk_module.cpp"],
            headers=[root / "tree" / "hjcd_tree_fk.h", root / "tree" / "hjcd_tree_lm.h", root / "tree" / "hjcd_tree_math.cuh", root / "kernel" / "cuda_memory.h", root / "kernel" / "util.h",
                     side / "hjcd_joint_model.h", side / "hjcd_targets.cuh"],
            include_dirs=[side, root, sysconfig.get_paths()["include"], pybind11.get_include()],
            defines=[f"{k}={v}" for k, v in defines.items()],
            extra_flags=["-Xptxas=-O3", "--expt-relaxed-constexpr",
                         "--compiler-options=-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique"],
            link_flags=["--cudart=shared", f"-Xlinker=-rpath={Path(nvcc).resolve().parent.parent / 'lib64'}"],
            artifact_name=f"hjcdik_tree_{module[13:]}.so")
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise BuildError(f"tree build of {name!r} failed: {error}") from error
    if plan_only:
        return result
    if not Path(result.path).exists():
        raise BuildError(f"tree build of {name!r} published no artifact at {result.path}; see build.log there")
    return TreeArtifact(name=name, robot=robot_name(urdf_path), path=Path(result.path), header=Path(result.header),
                        content_key=result.content_key, cuda_arch=int(result.cuda_arch), module_name=module,
                        compiled=bool(result.compiled), urdf_sha256=urdf_sha, targets_sha256=spec_sha,
                        model_sha256=model_sha256(model), cuda_compiler_version=_nvcc_version(result.meta))


def load_tree(name: str, *, urdf=None, targets_spec=None, cache_dir=None, cuda_arch: Optional[int] = None):
    """A prepared tree artifact by name, with no toolchain. With `urdf` and `targets_spec` the artifact
    must have been built from exactly those inputs (URDF bytes, spec bytes, validated model)."""
    grid_rbd = _grid_rbd()
    _check_name(name)
    try:
        record = grid_rbd.load_artifact(CONSUMER, name, cache_dir=cache_dir, cuda_arch=cuda_arch)
    except grid_rbd.ArtifactNotFoundError as error:
        raise BuildError(f"no prepared tree build named {name!r}; run hjcdik._tree_build.build_tree(...)") from error
    except grid_rbd.ConsumerArtifactError as error:
        raise BuildError(f"the prepared tree build {name!r} cannot be used here: {error}") from error
    meta = record.meta or {}
    defines = {d.split("=", 1)[0]: d.split("=", 1)[1] if "=" in d else "" for d in meta.get("defines", [])}
    if "HJCDIK_TREE_MODULE_NAME" not in defines:
        raise BuildError(f"prepared build {name!r} is not a tree-model artifact")
    spec_sha = defines.get("HJCDIK_TARGETS_SHA256", "").strip('"')
    urdf_sha = meta.get("urdf_sha256", "")
    artifact = TreeArtifact(name=name, robot=defines.get("HJCDIK_ROBOT_NAME", "").strip('"'), path=Path(record.path),
                            header=Path(record.header), content_key=record.content_key,
                            cuda_arch=int(record.cuda_arch), module_name=_module_name(name), compiled=False,
                            urdf_sha256=urdf_sha, targets_sha256=spec_sha, model_sha256="",
                            cuda_compiler_version=_nvcc_version(meta))
    if urdf is not None or targets_spec is not None:
        if urdf is None or targets_spec is None:
            raise BuildError("checking a prepared tree build needs both urdf and targets_spec")
        _, model, want_spec, want_urdf = describe(urdf, targets_spec)
        if (want_urdf, want_spec) != (urdf_sha, spec_sha):
            raise BuildError(f"prepared tree build {name!r} does not match the requested URDF/target spec; "
                             "rebuild it with build_tree(...)")
        artifact = TreeArtifact(**{**artifact.__dict__, "model_sha256": model_sha256(model)})
    return artifact


def import_tree(artifact: TreeArtifact):
    """Import the artifact's module once per process per content key; verify ABI and model identity."""
    cached = _LOADED.get(artifact.module_name)
    if cached is not None:
        key, module = cached
        if key != artifact.content_key:
            raise BuildError(f"tree build {artifact.name!r} changed after it was imported into this process "
                             f"({key[:12]} -> {artifact.content_key[:12]}); restart the process to use it")
        return module
    loader = importlib.machinery.ExtensionFileLoader(artifact.module_name, str(artifact.path))
    spec = importlib.util.spec_from_loader(artifact.module_name, loader)
    try:
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except ImportError as error:
        raise BuildError(f"cannot import the prepared tree build {artifact.name!r}: {error}") from error
    if getattr(module, "TREE_ABI", None) != TREE_ABI:
        raise BuildError(f"tree build {artifact.name!r} has ABI {getattr(module, 'TREE_ABI', None)}, expected {TREE_ABI}")
    if artifact.targets_sha256 and module.TARGETS_SHA256 != artifact.targets_sha256:
        raise BuildError(f"tree build {artifact.name!r} module does not carry its recorded target spec")
    if artifact.model_sha256 and module.MODEL_SHA256 != artifact.model_sha256:
        raise BuildError(f"tree build {artifact.name!r} module does not carry the requested validated model")
    sys.modules[artifact.module_name] = module
    _LOADED[artifact.module_name] = (artifact.content_key, module)
    return module
