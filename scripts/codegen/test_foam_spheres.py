"""CPU unit tests for scripts/codegen/foam_spheres.py against foam's committed example spherized URDF
(external/foam/spherized.urdf, a UR10e). Run: pytest scripts/codegen/test_foam_spheres.py

Outside tests/ on purpose (tests/ is auto-marked gpu_proof; this is CPU-only).
"""
import os

import pytest

from foam_spheres import (
    parse_spherized_urdf,
    link_to_joint_index,
    build_sphere_model,
    emit_cuh_fragment,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE = os.path.join(REPO, "external", "foam", "spherized.urdf")

pytestmark = pytest.mark.skipif(not os.path.exists(EXAMPLE),
                                reason="foam submodule not initialized (external/foam/spherized.urdf)")


def test_parse_finds_ur10e_links():
    by_link = parse_spherized_urdf(EXAMPLE)
    assert by_link, "no spheres parsed"
    # UR arm links should be present
    assert any("shoulder" in k for k in by_link)
    assert any("upper_arm" in k for k in by_link)
    assert any("forearm" in k for k in by_link)
    # every sphere is a 4-tuple with a positive radius
    for link, spheres in by_link.items():
        for (x, y, z, r) in spheres:
            assert r > 0.0
            assert all(isinstance(v, float) for v in (x, y, z, r))


def test_first_sphere_matches_known_value():
    by_link = parse_spherized_urdf(EXAMPLE)
    # base_link_inertia carries a single big sphere r=0.107272 at ~(0,0,0.0496)
    base = next(v for k, v in by_link.items() if k == "base_link_inertia")
    x, y, z, r = base[0]
    assert r == pytest.approx(0.107272, abs=1e-6)
    assert z == pytest.approx(0.0496443, abs=1e-5)


def test_joint_index_is_base0_and_monotone_down_chain():
    l2j = link_to_joint_index(EXAMPLE)
    # a root link exists at index 0
    assert min(l2j.values()) == 0
    # UR10e has 6 actuated joints -> deepest link index is 6
    assert max(l2j.values()) == 6
    # the arm chain is strictly increasing base(0) < shoulder < upper_arm < forearm
    def idx(sub):
        return next(v for k, v in l2j.items() if sub in k)
    assert idx("base_link_inertia") < idx("shoulder") < idx("upper_arm") < idx("forearm")


def test_build_sphere_model_aligned():
    spheres, s2j, links = build_sphere_model(EXAMPLE)
    assert len(spheres) == len(s2j) == len(links) > 0
    # sphere_to_joint values are all valid joint indices
    assert all(0 <= j <= 6 for j in s2j)
    # spheres on the same link share a joint index
    from collections import defaultdict
    per_link = defaultdict(set)
    for link, j in zip(links, s2j):
        per_link[link].add(j)
    assert all(len(js) == 1 for js in per_link.values())


def test_emit_cuh_fragment_format():
    spheres, s2j, _ = build_sphere_model(EXAMPLE)
    frag = emit_cuh_fragment("ur10e", spheres, s2j)
    assert f"#define UR10E_SPHERE_COUNT {len(spheres)}" in frag
    assert f"float4 ur10e_spheres_array[{len(spheres)}]" in frag
    assert "ur10e_sphere_to_joint[]" in frag
    # one brace-line per sphere
    assert frag.count("f },") + frag.count("f }, ") >= len(spheres) - 1
