"""Runtime robot builds over GRiD's consumer-artifact store (`grid_rbd.build_artifact`).

A build compiles HJCD-IK's own translation units (the solver kernel and the Python binding) against
the `grid.cuh` GRiD generates for a URDF, under GRiD's store rules: an identity over the URDF, the
codegen options and tree, GLASS, our sources and headers by content, defines, flags, toolchain and
CUDA architecture; per-unit object reuse; atomic publish. `load` returns a prepared artifact with
no toolchain. GRiD never binds our symbols: the artifact is a Python extension module that this
file imports from its store path under a per-artifact module name.

Requires the `hjcdik[build]` extra (grid-rbd, pybind11) and nvcc to build. Loading a prepared
artifact still needs grid-rbd, but needs neither nvcc nor pybind11.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import io
import importlib.util
import os
import re
import shutil
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ._robot_model import validate_solver_robot, describe_robot, stage_joint_header, fixed_frame_index

CONSUMER = "hjcdik"
BINDING_ABI = 5   # must equal HJCDIK_BINDING_ABI in csrc/bindings/pybind_module.cpp

_PACKAGE_DIR = Path(__file__).resolve().parent
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LOADED: dict = {}   # module name -> (content_key, module)


class BuildError(RuntimeError):
    """A runtime robot build or load could not be completed; the message names the next action."""


@dataclass(frozen=True)
class Artifact:
    """A prepared per-robot build: where it is and what it was built for."""

    name: str
    robot: str
    ee_frame: str
    collision_enabled: bool
    path: Path            # the extension module
    header: Path          # the grid.cuh it was compiled against
    content_key: str
    cuda_arch: int
    module_name: str
    compiled: bool        # nvcc ran in this call (False: reused or loaded)
    cuda_compiler_version: str = ""   # from GRiD's build record when it names nvcc

    @property
    def grid_header_sha256(self) -> str:
        return hashlib.sha256(self.header.read_bytes()).hexdigest()


def source_root() -> Path:
    """The `csrc/` tree: inside the installed package, or the repository for an editable install."""
    for candidate in (_PACKAGE_DIR / "_csrc", _PACKAGE_DIR.parent / "csrc"):
        if (candidate / "kernel" / "hjcd_kernel.cu").is_file():
            return candidate
    raise BuildError("HJCD-IK sources not found beside the package; reinstall hjcdik (the wheel carries csrc/)")


def sphere_presets() -> dict:
    """Named, validated sphere models shipped with the package (keyed by preset name)."""
    root = source_root()
    return {
        "panda_foam": root / "collision" / "panda_foam_rebound.urdf",
        "iiwa14": root / "collision" / "iiwa14_spheres.urdf",
        "ur5e": root / "collision" / "ur5e_spheres.urdf",
    }


def presets() -> dict:
    """Robots shipped with the package as URDF + end-effector frame + sphere model.

    `Solver("<preset>")` builds the preset on first use (or loads it when already built); the
    packaged robot ("panda") is compiled into the wheel and needs no build. Every preset is a
    fixed-base serial chain with independent revolute, continuous or prismatic joints.
    """
    root = source_root()
    return {
        "panda": dict(urdf=root / "urdf" / "panda.urdf", ee_link="panda_grasptarget_hand",
                      spheres="panda_foam", source="franka_description (panda_arm_hand.urdf.xacro)"),
        "iiwa14": dict(urdf=root / "urdf" / "iiwa14.urdf", ee_link="iiwa_joint_ee", spheres="iiwa14",
                       source="Drake manipulation/models/iiwa_description (iiwa14_primitive_collision.urdf)"),
        "ur5e": dict(urdf=root / "urdf" / "ur5e.urdf", ee_link="flange-tool0", spheres="ur5e",
                     source="UniversalRobots/Universal_Robots_ROS2_Description at 22f055da (nominal calibration)"),
    }


def robot_name(urdf_path) -> str:
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except (ET.ParseError, OSError) as error:
        raise BuildError(f"cannot read the URDF {urdf_path}: {error}") from error
    name = root.get("name") if root.tag == "robot" else None
    if not name:
        raise BuildError(f"{urdf_path} has no <robot name=...>")
    return name


def default_name(urdf_path, ee_link: str, spheres) -> str:
    tag = f"{robot_name(urdf_path)}-{ee_link}" + ("-cc" if spheres else "")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:64]


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise BuildError(f"artifact names are short identifiers (letters, digits, '_', '.', '-'); got {name!r}")
    return name


def _grid_rbd():
    try:
        import grid_rbd  # noqa: F401
        return grid_rbd
    except ImportError as error:
        raise BuildError("runtime robot builds need the hjcdik[build] extra: pip install 'hjcdik[build]'") from error


def _resolve_spheres(spheres):
    if spheres is None:
        return None
    presets = sphere_presets()
    if isinstance(spheres, str) and spheres in presets:
        return presets[spheres]
    path = Path(spheres).expanduser()
    if not path.is_file():
        raise BuildError(f"sphere model {spheres!r} is neither a preset ({', '.join(presets)}) nor a file")
    return path.resolve()


def _prepare_robot(urdf_path, ee_link: str):
    """Where GRiD places the named fixed target inside the homogeneous-transform block.

    The index is robot-specific and read back from a kinematics-only generation, exactly as
    scripts/codegen/generate_grid.py does for the committed header; a runtime build cannot post-process
    the header GRiD generates, so the value travels as -DHJCDIK_EE_FIXED_FRAME_IDX.
    """
    _grid_rbd()
    from URDFParser import URDFParser  # noqa: E402  (grid-rbd ships it)
    from grid_codegen import GRiDCodeGenerator  # noqa: E402
    try:
        robot = URDFParser().parse(str(urdf_path), floating_base=False)
        validate_solver_robot(robot, ee_link)
    except ValueError as error:
        raise BuildError(f"unsupported robot {urdf_path}: {error}") from error
    with tempfile.TemporaryDirectory(prefix="hjcdik-frame-") as tmp:
        out = Path(tmp) / "grid.cuh"
        with contextlib.redirect_stdout(io.StringIO()):   # the generator narrates its frame table
            GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid").gen_all_code(
                codegen_profile="kinematics", vendor_glass=False, include_homogenous_transforms=True,
                fixed_target_name=ee_link, output_path=str(out))
        text = out.read_text()
    return fixed_frame_index(text, ee_link), describe_robot(robot, urdf_path, ee_link)


def _nvcc_version(meta) -> str:
    """nvcc's release number from GRiD's recorded toolchain identity, or an empty string."""
    text = str((meta or {}).get("build", {}).get("nvcc", "")) if isinstance(meta, dict) else ""
    m = re.search(r"release (\d+(?:\.\d+)+)", text) or re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return m.group(1) if m else ""


def _module_name(name: str) -> str:
    return "_hjcdik_" + hashlib.sha256(name.encode()).hexdigest()[:12]


def _build_inputs(name: str, urdf_path: Path, ee_link: str, spheres: Optional[Path], defines_extra: dict,
                  cache_dir=None) -> dict:
    root = source_root()
    include = sysconfig.get_paths()["include"]
    try:
        import pybind11
    except ImportError as error:
        raise BuildError("runtime robot builds need pybind11: pip install 'hjcdik[build]'") from error
    index, model = _prepare_robot(urdf_path, ee_link)
    side = stage_joint_header(model, cache_dir if cache_dir is not None else _grid_rbd().default_cache_dir())
    defines = {
        "HJCDIK_MODULE_NAME": _module_name(name),
        "HJCDIK_ROBOT_NAME": f'"{robot_name(urdf_path)}"',
        "HJCDIK_EE_TARGET": f'"{ee_link}"',
        "HJCDIK_EE_FIXED_FRAME_IDX": str(index),
    }
    if spheres is not None:
        defines["HJCD_HAS_COLLISION"] = "1"
    defines.update(defines_extra)
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuda_lib = Path(nvcc).resolve().parent.parent / "lib64"
    return dict(
        sources=[root / "kernel" / "hjcd_kernel.cu", root / "bindings" / "pybind_module.cpp"],
        headers=sorted(p for p in (root / "kernel").iterdir() if p.suffix in (".h", ".cuh")) + [side],
        include_dirs=[side.parent, root, include, pybind11.get_include()],
        defines=[f"{k}={v}" for k, v in defines.items()],
        # Hidden visibility and no STB_GNU_UNIQUE: every artifact is its own module in one process, and
        # default-visibility template statics (the cached robot model, the solver mutex, GRiD's pools)
        # would otherwise bind process-wide across modules and make a second robot solve with the first
        # robot's kinematics.
        extra_flags=["-use_fast_math", "-Xptxas=-O3", "--expt-relaxed-constexpr",
                     "--compiler-options=-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique"],
        link_flags=["--cudart=shared", f"-Xlinker=-rpath={cuda_lib}"],
        artifact_name=f"hjcdik_{_module_name(name)[8:]}.so",   # store file name; the init symbol is the module name
    )


def build(urdf, *, ee_link: str, spheres=None, name: Optional[str] = None, collision_res=0.05,
          cache_dir=None, cuda_arch: Optional[int] = None, force_rebuild: bool = False,
          plan_only: bool = False, _diagnostics: bool = False):
    """Build (or reuse) the solver for one robot and return its `Artifact`.

    `urdf` is a fixed-base serial-chain URDF, `ee_link` the fixed joint naming the end-effector target
    frame. `spheres` enables collision checking: a preset name (`"panda_foam"`) or a spherized URDF
    whose kinematic frames match `urdf` (GRiD validates it; `grid-spherize --rebind-to` adapts a model
    made for another URDF of the same robot). `name` defaults to `<robot>-<ee_link>[-cc]`.
    `plan_only=True` reports the identity and whether the artifact is already there, without building.
    """
    grid_rbd = _grid_rbd()
    urdf_path = Path(urdf).expanduser().resolve()
    if not urdf_path.is_file():
        raise BuildError(f"URDF not found: {urdf_path}")
    sphere_path = _resolve_spheres(spheres)
    name = _check_name(name or default_name(urdf_path, ee_link, sphere_path))
    try:
        inputs = _build_inputs(name, urdf_path, ee_link, sphere_path,
                               {"HJCDIK_DIAGNOSTICS": "1"} if _diagnostics else {}, cache_dir)
        result = grid_rbd.build_artifact(
            CONSUMER, name, urdf_path, cache_dir=cache_dir, cuda_arch=cuda_arch,
            force_rebuild=force_rebuild, plan_only=plan_only,
            ee_joint_names=[ee_link], algorithm_list="kinematics", emit_plant=False,
            # GRiD's collision switches are exclusive: a spherized URDF is the collision request by itself.
            collision=False, collision_res=collision_res,
            spherized_urdf=None if sphere_path is None else str(sphere_path),
            **inputs)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise BuildError(f"build of {name!r} failed: {error}") from error
    if plan_only:
        return result
    if not Path(result.path).exists():
        # A damaged store entry (its .so removed) keeps its directory, and grid_rbd 0.6.0's repair
        # cannot publish over it without force_rebuild; rebuild the entry from scratch once.
        if force_rebuild:
            raise BuildError(f"build of {name!r} published no artifact at {result.path}; see build.log there")
        return build(urdf, ee_link=ee_link, spheres=spheres, name=name, collision_res=collision_res,
                     cache_dir=cache_dir, cuda_arch=cuda_arch, force_rebuild=True, _diagnostics=_diagnostics)
    return Artifact(name=name, robot=robot_name(urdf_path), ee_frame=ee_link,
                    collision_enabled=sphere_path is not None, path=Path(result.path),
                    header=Path(result.header), content_key=result.content_key,
                    cuda_arch=int(result.cuda_arch), module_name=_module_name(name),
                    compiled=bool(result.compiled), cuda_compiler_version=_nvcc_version(result.meta))


def load(name: str, cache_dir=None, cuda_arch: Optional[int] = None) -> Artifact:
    """A prepared artifact by name, with no toolchain. Raises `BuildError` naming the build command."""
    return _load(name, cache_dir, cuda_arch)


def load_for_urdf(urdf, ee_link, spheres, cache_dir=None) -> Artifact:
    """Load without compilation, requiring the requested model rather than just its display name."""
    sphere_path = _resolve_spheres(spheres)
    try:
        expected = (hashlib.sha256(Path(urdf).expanduser().read_bytes()).hexdigest(),
                    None if sphere_path is None else hashlib.sha256(sphere_path.read_bytes()).hexdigest(),
                    ee_link)
    except OSError as error:
        raise BuildError(f"cannot read requested model: {error}; restore the input files or load by name") from error
    return _load(default_name(Path(urdf).expanduser(), ee_link, sphere_path), cache_dir,
                 expected=expected)


def _load(name, cache_dir=None, cuda_arch=None, *, expected=None):
    grid_rbd = _grid_rbd()
    _check_name(name)
    try:
        record = grid_rbd.load_artifact(CONSUMER, name, cache_dir=cache_dir, cuda_arch=cuda_arch)
    except grid_rbd.ArtifactNotFoundError as error:
        raise BuildError(f"no prepared build named {name!r}; run `hjcdik build <urdf> --ee-link <joint> "
                         f"--name {name}` or hjcdik.build(...) on a machine with nvcc") from error
    except grid_rbd.ConsumerArtifactError as error:
        raise BuildError(f"the prepared build {name!r} cannot be used here: {error}; "
                         "rebuild from the original URDF, tool frame and sphere model with "
                         "hjcdik.build(..., force_rebuild=True) on a compatible machine with nvcc") from error
    meta = record.meta or {}
    defines = {d.split("=", 1)[0]: d.split("=", 1)[1] if "=" in d else "" for d in meta.get("defines", [])}
    if expected is not None:
        options = meta.get("code_options", {})
        actual = (meta.get("urdf_sha256"), options.get("spherized_urdf_sha256"),
                  defines.get("HJCDIK_EE_TARGET", "").strip('"'))
        collision = expected[1] is not None
        if (actual != expected or options.get("ee_joint_names") != [expected[2]]
                or ("HJCD_HAS_COLLISION" in defines) != collision
                or bool(options.get("spherized_urdf")) != collision):
            raise BuildError(f"prepared build {name!r} does not match the requested model; rebuild "
                             "from the requested URDF, tool frame and sphere model with hjcdik.build(...), "
                             "or explicitly load the prepared artifact by name")
    return Artifact(name=name, robot=defines.get("HJCDIK_ROBOT_NAME", "").strip('"'),
                    ee_frame=defines.get("HJCDIK_EE_TARGET", "").strip('"'),
                    collision_enabled="HJCD_HAS_COLLISION" in defines, path=Path(record.path),
                    header=Path(record.header), content_key=record.content_key,
                    cuda_arch=int(record.cuda_arch), module_name=_module_name(name), compiled=False,
                    cuda_compiler_version=_nvcc_version(meta))


def list_builds(cache_dir=None) -> list:
    """Names and metadata of every prepared build in the store."""
    grid_rbd = _grid_rbd()
    return grid_rbd.list_artifacts(CONSUMER, cache_dir=cache_dir)


def import_artifact(artifact: Artifact):
    """Import the artifact's extension module (once per process per content key)."""
    cached = _LOADED.get(artifact.module_name)
    if cached is not None:
        key, module = cached
        if key != artifact.content_key:
            raise BuildError(f"build {artifact.name!r} changed after it was imported into this process "
                             f"({key[:12]} -> {artifact.content_key[:12]}); restart the process to use it")
        return module
    loader = importlib.machinery.ExtensionFileLoader(artifact.module_name, str(artifact.path))
    spec = importlib.util.spec_from_loader(artifact.module_name, loader)
    try:
        # Extension initialization can fail in create_module, before exec_module.
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except ImportError as error:
        raise BuildError(f"cannot import the prepared build {artifact.name!r} from {artifact.path}: {error}; "
                         "check CUDA runtime/library compatibility, or rebuild with "
                         "hjcdik.build(..., force_rebuild=True)") from error
    abi = getattr(module, "BINDING_ABI", None)
    if abi != BINDING_ABI:
        raise BuildError(f"build {artifact.name!r} has binding ABI {abi}, this hjcdik expects {BINDING_ABI}; "
                         f"rebuild it with hjcdik.build(..., force_rebuild=True)")
    sys.modules[artifact.module_name] = module
    _LOADED[artifact.module_name] = (artifact.content_key, module)
    return module
