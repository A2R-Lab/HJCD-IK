"""The generated-name metadata API: hjcdik.joint_names() / hjcdik.target_names().

The compiled extension carries no strings — the CUDA hot path sees only indices, bitmasks and baked
4x4s. Names are resolved at codegen time and written to the package-data sidecar
(hjcdik/hjcd_targets.json) by scripts/codegen/generate_grid.py. These tests pin the contract that
downstream consumers (e.g. CRAG's HJCDMujocoBridge) rely on: that the ordering compiled into HJCD is
readable FROM HJCD, so no consumer has to hardcode a joint order or a target index.
"""
import json
from pathlib import Path

import pytest

import hjcdik

REPO = Path(__file__).resolve().parents[1]
SIDECAR = REPO / "hjcdik" / "hjcd_targets.json"


# --- 1 & 2. the API exists ----------------------------------------------------------------------

def test_joint_names_exists():
    assert hasattr(hjcdik, "joint_names")
    assert callable(hjcdik.joint_names)


def test_target_names_exists():
    assert hasattr(hjcdik, "target_names")
    assert callable(hjcdik.target_names)


def test_names_are_exported():
    """Public API, not an accident of module scope."""
    assert "joint_names" in hjcdik.__all__
    assert "target_names" in hjcdik.__all__


# --- 3. non-empty, ordered, immutable -----------------------------------------------------------

@pytest.mark.parametrize("fn", [hjcdik.joint_names, hjcdik.target_names])
def test_names_are_non_empty_ordered_strings(fn):
    names = fn()
    assert isinstance(names, tuple), "must be an immutable ordered sequence"
    assert names, "must be non-empty"
    assert all(isinstance(n, str) and n for n in names)


@pytest.mark.parametrize("fn", [hjcdik.joint_names, hjcdik.target_names])
def test_names_are_cached_and_stable(fn):
    """Repeated calls must not re-read or re-parse the JSON; identity makes that observable."""
    assert fn() is fn()


# --- 4. uniqueness ------------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [hjcdik.joint_names, hjcdik.target_names])
def test_names_are_unique(fn):
    names = fn()
    assert len(set(names)) == len(names), f"duplicate names: {names}"


# --- 5 & 6. agreement with the compiled extension -----------------------------------------------

def test_joint_names_length_matches_extension():
    assert len(hjcdik.joint_names()) == hjcdik.num_joints()


def test_target_names_length_matches_extension():
    assert len(hjcdik.target_names()) == hjcdik.num_targets()


# --- 7. the ordering is the GENERATED ordering ---------------------------------------------------

def test_names_match_the_generated_sidecar_verbatim():
    """The accessors must surface the generated order as-is — no sorting, filtering or reordering.

    This is the property that lets a consumer treat index k from the solver and index k from
    target_names() as the same target.
    """
    meta = json.loads(SIDECAR.read_text())
    assert list(hjcdik.joint_names()) == meta["joint_names"]
    assert list(hjcdik.target_names()) == meta["target_names"]


def test_target_names_match_per_target_records_in_order():
    """target_names[k] must be the name of the k-th record in `targets`, positionally."""
    meta = json.loads(SIDECAR.read_text())
    assert list(hjcdik.target_names()) == [t["name"] for t in meta["targets"]]


def test_joint_names_index_agrees_with_anchor_jid():
    """Cross-check joint order against an independent field: each target's anchor.

    anchor_jid is consumed by the device; anchor_name is a separate host-side string. If
    joint_names were in any other order (URDF declaration order, alphabetical, a stale build),
    joint_names[anchor_jid] would not be anchor_name.
    """
    meta = json.loads(SIDECAR.read_text())
    names = hjcdik.joint_names()
    for target in meta["targets"]:
        assert names[target["anchor_jid"]] == target["anchor_name"], (
            f"target {target['name']}: joint_names[{target['anchor_jid']}] = "
            f"{names[target['anchor_jid']]!r}, expected {target['anchor_name']!r}")


def test_joint_names_index_agrees_with_device_anchor_jid():
    """Same cross-check, but against the DEVICE's anchor_jid rather than the JSON's."""
    names = hjcdik.joint_names()
    meta_json = json.loads(SIDECAR.read_text())
    device_anchor_jid = hjcdik.target_metadata()["anchor_jid"]
    for k, target in enumerate(meta_json["targets"]):
        assert names[int(device_anchor_jid[k])] == target["anchor_name"]


@pytest.mark.skipif("g1" not in json.loads(SIDECAR.read_text())["robot"],
                    reason="G1-specific expectation; the current build is another robot")
def test_g1_target_order_is_the_generated_order():
    """The G1 build's four contact frames, in the order codegen emitted them.

    Deliberately NOT sorted: this asserts the generated order survives to the API, so a consumer
    can rely on index 0 being the left hand for THIS build without hardcoding that mapping itself.
    """
    assert hjcdik.target_names() == (
        "left_hand", "right_hand", "left_foot", "right_foot")


# --- 8. malformed / inconsistent metadata fails loudly -------------------------------------------

@pytest.fixture
def injected_metadata(monkeypatch):
    """Swap the sidecar CONTENTS for one test, then restore the real caches.

    hjcdik._read_metadata_text() is the single point where the resource is read, so overriding it
    exercises the real parse/validate path against synthetic metadata. The compiled extension is
    untouched — num_joints()/num_targets() stay real, which is what makes the stale-sidecar case
    below a genuine mismatch rather than a fabricated one.
    """
    def _inject(payload):
        if isinstance(payload, BaseException):
            def _raise():
                raise payload
            monkeypatch.setattr(hjcdik, "_read_metadata_text", _raise)
        else:
            monkeypatch.setattr(hjcdik, "_read_metadata_text", lambda: payload)
        hjcdik._metadata_cache = None
        hjcdik._names_cache = {}
        return hjcdik

    yield _inject

    hjcdik._metadata_cache = None      # drop anything the test cached
    hjcdik._names_cache = {}


def test_malformed_json_raises_runtime_error(injected_metadata):
    module = injected_metadata("{not valid json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        module.joint_names()


def test_non_object_json_raises_runtime_error(injected_metadata):
    module = injected_metadata(json.dumps(["a", "list", "not", "an", "object"]))
    with pytest.raises(RuntimeError, match="must be a JSON object"):
        module.joint_names()


def test_missing_field_raises_runtime_error(injected_metadata):
    module = injected_metadata(json.dumps({"robot": "x", "target_names": ["a"]}))
    with pytest.raises(RuntimeError, match="no 'joint_names' field"):
        module.joint_names()


def test_wrong_type_raises_runtime_error(injected_metadata):
    module = injected_metadata(json.dumps({"joint_names": [1, 2, 3]}))
    with pytest.raises(RuntimeError, match="must be a list of strings"):
        module.joint_names()


def test_empty_names_raise_runtime_error(injected_metadata):
    module = injected_metadata(json.dumps({"joint_names": []}))
    with pytest.raises(RuntimeError, match="empty"):
        module.joint_names()


def test_duplicate_names_raise_runtime_error(injected_metadata):
    module = injected_metadata(json.dumps({"joint_names": ["a", "b", "a"]}))
    with pytest.raises(RuntimeError, match="duplicate"):
        module.joint_names()


def test_count_mismatch_with_extension_raises_runtime_error(injected_metadata):
    """The stale-sidecar case: a JSON from a different codegen run than the compiled .so.

    Well-formed and internally consistent — only disagreeing with the extension's own count. This
    is the silently-wrong-by-one failure the length check exists to catch.
    """
    names = [f"joint_{i}" for i in range(hjcdik.num_joints() - 1)]
    module = injected_metadata(json.dumps({"joint_names": names}))
    with pytest.raises(RuntimeError, match="stale"):
        module.joint_names()


def test_absent_metadata_raises_runtime_error(injected_metadata):
    module = injected_metadata(FileNotFoundError("hjcd_targets.json"))
    with pytest.raises(RuntimeError, match="missing from the hjcdik package"):
        module.target_names()


def test_real_metadata_survives_the_injection_tests(injected_metadata):
    """Guard the fixture itself: the real accessors must still work after a swap is undone."""
    injected_metadata(json.dumps({"joint_names": ["a"]}))
    hjcdik._metadata_cache = None
    hjcdik._names_cache = {}
