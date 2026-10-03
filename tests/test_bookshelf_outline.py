"""Gates of the bookshelf protocol pieces: hyperedges kept by the parser, hard blocks as fixed shapes,
the fixed-outline transform (outline area, pins inside the outline, reference layout dropped) and the
net-HPWL / dead-space / fits metric; the MCNC gates skip without the data folder."""

from pathlib import Path

import numpy as np
import pytest

import trinity.floorplan.legalize  # noqa: F401  (register: scale_pack)
from trinity.floorplan.data import fixed_outline, load_mcnc_case, with_fixed_outline
from trinity.floorplan.data.bookshelf import parse_bookshelf_case
from trinity.floorplan.data.paths import project_data_root
from trinity.floorplan.legalize.scale_pack import last_info, scale_pack_legalize
from trinity.floorplan.scoring.bookshelf import (
    clamp_aspect,
    net_hpwl,
    orient_targets,
    score_bookshelf,
)
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.types import FloorplanInstance, Placement

MCNC = project_data_root() / "mcnc"


def _toy_bookshelf(tmp_path: Path) -> tuple[Path, Path, Path]:
    (tmp_path / "t.blocks").write_text(
        "UCSC blocks 1.0\nNumSoftRectangularBlocks : 0\nNumHardRectilinearBlocks : 3\nNumTerminals : 2\n"
        "a hardrectilinear 4 (0, 0) (0, 2) (4, 2) (4, 0)\nb hardrectilinear 4 (0, 0) (0, 3) (2, 3) (2, 0)\n"
        "c hardrectilinear 4 (0, 0) (0, 1) (3, 1) (3, 0)\np1 terminal\np2 terminal\n"
    )
    (tmp_path / "t.pl").write_text("UCSC pl 1.0\na 0 0\nb 4 0\nc 0 2\np1 0 5\np2 6 0\n")
    (tmp_path / "t.nets").write_text(
        "UCSC nets 1.0\nNumNets : 2\nNumPins : 5\nNetDegree : 3\na B\nb B\np1 B\nNetDegree : 2\nc B\np2 B\n"
    )
    return tmp_path / "t.blocks", tmp_path / "t.pl", tmp_path / "t.nets"


def test_parser_keeps_hyperedges_and_hard_shapes(tmp_path):
    inst = parse_bookshelf_case(*_toy_bookshelf(tmp_path), hard_shapes=True)
    assert inst.block_count == 3 and inst.nets == [[0, 1, 3], [2, 4]]
    assert inst.is_fixed.all() and np.allclose(
        inst.target_positions[:, 2:], [[4, 2], [2, 3], [3, 1]]
    )
    assert inst.gt_positions is not None and inst.outline is None


def test_fixed_outline_transform_and_metric(tmp_path):
    inst = parse_bookshelf_case(*_toy_bookshelf(tmp_path), hard_shapes=True)
    W, H = fixed_outline(inst, gamma=0.10, aspect=2.0)
    assert np.isclose(W * H, 1.1 * inst.area_targets.sum()) and np.isclose(H / W, 2.0)
    out = with_fixed_outline(inst, gamma=0.10, aspect=2.0)
    assert out.outline == (W, H) and out.gt_positions is None and out.nets == inst.nets
    assert np.allclose(out.pins_pos, inst.pins_pos)
    mapped = with_fixed_outline(inst, gamma=0.10, aspect=2.0, map_pins=True)
    assert (
        (mapped.pins_pos >= -1e-6).all()
        and (mapped.pins_pos[:, 0] <= W + 1e-6).all()
        and (mapped.pins_pos[:, 1] <= H + 1e-6).all()
    )
    p = Placement(xywh=inst.gt_positions.astype(np.float64), instance=inst)
    hp = net_hpwl(p)
    assert np.isclose(hp, (5 - 0) + (5 - 1) + (6 - 1.5) + (2.5 - 0))
    s = score_bookshelf(p, outline=(6.0, 5.0))
    assert s.overlap_free and s.shapes_kept and s.fits_outline and s.n_nets == 2
    assert np.isclose(s.area, 6 * 3) and np.isclose(s.dead_space, 1 - 17 / 18)
    assert not score_bookshelf(p, outline=(5.0, 5.0)).fits_outline


def _toy_soft(tmp_path: Path) -> tuple[Path, Path, Path]:
    (tmp_path / "s.blocks").write_text(
        "UCSC blocks 1.0\nNumSoftRectangularBlocks : 3\nNumHardRectilinearBlocks : 0\nNumTerminals : 1\n"
        "a softrectangular 8 0.5 2.0\nb softrectangular 6 1.5 1.5\nc softrectangular 3\np1 terminal\n"
    )
    (tmp_path / "s.pl").write_text("UCSC pl 1.0\np1 0 5\n")
    (tmp_path / "s.nets").write_text(
        "UCSC nets 1.0\nNumNets : 1\nNumPins : 3\nNetDegree : 3\na B\nb B\np1 B\n"
    )
    return tmp_path / "s.blocks", tmp_path / "s.pl", tmp_path / "s.nets"


def test_soft_blocks_carry_aspect_bounds_and_the_checker_enforces_them(tmp_path):
    inst = parse_bookshelf_case(*_toy_soft(tmp_path))
    assert inst.is_soft.all() and np.allclose(inst.area_targets, [8, 6, 3])
    assert np.allclose(inst.aspect_bounds, [[0.5, 2.0], [1 / 3, 3.0], [1 / 3, 3.0]])
    out = with_fixed_outline(inst, gamma=0.5, aspect=1.0)
    assert np.allclose(out.aspect_bounds, inst.aspect_bounds)
    wide = np.array([[0.0, 0.0, 8.0, 1.0], [0.0, 2.0, 3.0, 2.0], [4.0, 2.0, 1.0, 3.0]])
    rep = check_feasibility(Placement(xywh=wide, instance=out))
    assert rep.aspect_violations == [0] and not rep.is_feasible()
    assert not score_bookshelf(Placement(xywh=wide, instance=out)).shapes_kept
    clipped = clamp_aspect(out, wide)
    assert np.allclose(clipped[:, 2] * clipped[:, 3], [8, 6, 3]) and np.isclose(
        clipped[0, 2] / clipped[0, 3], 2.0
    )
    assert (
        check_feasibility(Placement(xywh=clipped, instance=out)).aspect_violations == []
    )


def test_scale_pack_keeps_soft_blocks_within_their_bounds_and_fits_by_reshaping(
    tmp_path,
):
    inst = with_fixed_outline(
        parse_bookshelf_case(*_toy_soft(tmp_path)), gamma=0.30, aspect=1.0
    )
    tall = np.array([[0.0, 0.0, 1.0, 8.0], [1.0, 0.0, 1.0, 6.0], [2.0, 0.0, 1.0, 3.0]])
    out = scale_pack_legalize(Placement(xywh=tall, instance=inst))
    s = score_bookshelf(out)
    ratio = out.xywh[:, 2] / out.xywh[:, 3]
    assert s.overlap_free and s.shapes_kept and s.fits_outline
    assert np.all(ratio >= inst.aspect_bounds[:, 0] - 1e-6) and np.all(
        ratio <= inst.aspect_bounds[:, 1] + 1e-6
    )
    assert np.allclose(out.xywh[:, 2] * out.xywh[:, 3], inst.area_targets, rtol=1e-6)
    assert last_info()["reshaped"] >= 0


def test_orient_targets_accepts_a_rotated_hard_block(tmp_path):
    inst = with_fixed_outline(
        parse_bookshelf_case(*_toy_bookshelf(tmp_path), hard_shapes=True),
        gamma=1.0,
        aspect=1.0,
    )
    rotated = np.array(
        [[0.0, 0.0, 2.0, 4.0], [2.0, 0.0, 2.0, 3.0], [4.0, 0.0, 1.0, 3.0]]
    )
    assert not score_bookshelf(Placement(xywh=rotated, instance=inst)).shapes_kept
    turned = orient_targets(inst, rotated)
    assert (
        np.allclose(turned.target_positions[:, 2:], [[2, 4], [2, 3], [1, 3]])
        and turned.area_targets is inst.area_targets
    )
    s = score_bookshelf(Placement(xywh=rotated, instance=turned))
    assert s.shapes_kept and s.overlap_free and s.fits_outline
    out = scale_pack_legalize(
        Placement(
            xywh=rotated
            + [[0.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            instance=turned,
        )
    )
    assert (
        np.allclose(out.xywh[:, 2:], rotated[:, 2:])
        and score_bookshelf(out).shapes_kept
    )
    assert orient_targets(inst, inst.target_positions * [[0, 0, 1, 1]]) is inst


def test_scale_pack_caps_the_outline_of_a_bookshelf_case(tmp_path):
    base = parse_bookshelf_case(*_toy_bookshelf(tmp_path), hard_shapes=True)
    spread = np.array(
        [[0.0, 0.0, 4.0, 2.0], [0.5, 6.0, 2.0, 3.0], [2.5, 5.6, 3.0, 1.0]]
    )
    inst = with_fixed_outline(base, gamma=1.0, aspect=1.0)
    out = scale_pack_legalize(Placement(xywh=spread, instance=inst))
    s = score_bookshelf(out)
    assert (
        s.overlap_free and s.shapes_kept and s.fits_outline and last_info()["rung"] == 1
    )
    tight = with_fixed_outline(base, gamma=0.30, aspect=1.0)
    out = scale_pack_legalize(Placement(xywh=spread, instance=tight))
    s = score_bookshelf(out)
    assert (
        s.overlap_free
        and s.shapes_kept
        and not s.fits_outline
        and last_info()["rung"] == 5
    )
    loose = scale_pack_legalize(Placement(xywh=spread, instance=tight), outline=False)
    assert score_bookshelf(loose).overlap_free and last_info()["rung"] == 1


@pytest.mark.skipif(
    not (MCNC / "HARD" / "ami33.blocks").is_file(), reason="MCNC files not downloaded"
)
def test_mcnc_loads_with_terminals_and_outline():
    inst = load_mcnc_case("ami33", "HARD", allow_download=False)
    assert inst.block_count == 33 and len(inst.pins_pos) == 42 and inst.is_fixed.all()
    assert inst.nets is not None and all(len(net) >= 2 for net in inst.nets)
    out = with_fixed_outline(inst, 0.15, 1.0)
    assert out.outline is not None and np.isclose(out.outline[0], out.outline[1])
    soft = load_mcnc_case("ami33", "SOFT", allow_download=False)
    assert soft.block_count == 33 and not soft.is_fixed.any()
    inst2 = FloorplanInstance(**{**inst.__dict__})
    assert inst2.outline is None
