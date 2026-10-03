"""Render a :class:`Placement` to a PNG.

Blocks are colored by kind (soft / fixed / preplaced / cluster member / boundary-coded),
overlapping blocks are hatched red, b2b nets are faint steelblue lines between block
centers, pins are green dots with faint green lines to their blocks, and the bounding
box is dashed. The title carries the score summary (feasibility, cost, gaps, ``V_rel``).
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

from trinity.floorplan.geometry import (  # noqa: E402
    bounding_box,
    component_labels,
    overlapping_pairs,
)
from trinity.floorplan.registry import RENDERER  # noqa: E402
from trinity.floorplan.scoring.cost import CaseScore  # noqa: E402
from trinity.floorplan.scoring.soft import check_boundary, check_grouping  # noqa: E402
from trinity.floorplan.types import Placement  # noqa: E402

_KIND_COLOR = {
    "soft": "silver",
    "fixed": "violet",
    "preplaced": "gray",
    "cluster": "lightcoral",
    "boundary": "khaki",
}

_COMPONENT_COLORS = ["magenta", "cyan", "yellow", "lime", "orange", "deeppink"]


def _line_style(n_edges: int) -> tuple[float, float]:
    """``(alpha, linewidth)`` of a net layer with
    ``n_edges`` lines (fading as ``1/sqrt(n)``)."""
    if n_edges <= 1:
        return 0.7, 0.8
    alpha = float(np.clip(4.5 / np.sqrt(n_edges), 0.12, 0.7))
    width = float(np.clip(1.6 / np.sqrt(n_edges) * 3, 0.25, 0.9))
    return alpha, width


def _block_kind(inst, i: int) -> str:
    if inst.is_preplaced[i]:
        return "preplaced"
    if inst.is_fixed[i]:
        return "fixed"
    if inst.cluster_id[i] > 0:
        return "cluster"
    if inst.boundary_code[i] != 0:
        return "boundary"
    return "soft"


def _b2b_segments(inst, centers):
    """The in-range b2b center-to-center segments ``(M, 2, 2)``."""
    edges = inst.b2b
    if not edges.shape[0]:
        return np.empty((0, 2, 2))
    keep = (
        (edges[:, 0] != -1)
        & (edges[:, 0] < inst.block_count)
        & (edges[:, 1] < inst.block_count)
    )
    edges = edges[keep].astype(np.int64)
    return np.stack([centers[edges[:, 0]], centers[edges[:, 1]]], axis=1)


def _p2b_segments(inst, centers):
    """The in-range pin-to-block segments ``(M, 2,
    2)`` and the valid pin positions ``(P, 2)``."""
    edges = inst.p2b
    keep = (
        (edges[:, 0] >= 0)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < inst.block_count)
        & (edges[:, 0] < inst.pins_pos.shape[0])
    )
    edges = edges[keep].astype(np.int64)
    pin_xy = inst.pins_pos[edges[:, 0]]
    placed = pin_xy[:, 0] != -1
    edges, pin_xy = edges[placed], pin_xy[placed]
    segments = np.stack([pin_xy, centers[edges[:, 1]]], axis=1)
    valid_pins = inst.pins_pos[inst.pins_pos[:, 0] != -1]
    return segments, valid_pins


def _violation_marks(placement: Placement):
    """The soft-violation overlay of ``placement``.

    Returns ``(V_boundary, V_grouping, boundary-violating blocks, {block: (color,
    fragment)} for the members of split clusters, [(centroid_a, centroid_b, color)]
    links between consecutive fragments)``.
    """
    v_boundary, bad = check_boundary(placement)
    v_grouping, split_groups = check_grouping(placement)
    block_group: dict[int, tuple[str, int]] = {}
    gap_links: list[tuple[tuple, tuple, str]] = []
    cluster = placement.instance.cluster_id
    for gi, g in enumerate(split_groups):
        color = _COMPONENT_COLORS[gi % len(_COMPONENT_COLORS)]
        members = np.nonzero(cluster == g)[0]
        component = component_labels(placement.xywh, members)
        fragment_centroid = {}
        for fragment, c in enumerate(sorted(set(component.values()))):
            blocks = [int(m) for m in members if component[int(m)] == c]
            for m in blocks:
                block_group[m] = (color, fragment + 1)
            centers = placement.centers[blocks]
            fragment_centroid[fragment] = (
                float(centers[:, 0].mean()),
                float(centers[:, 1].mean()),
            )
        ordered = [fragment_centroid[k] for k in sorted(fragment_centroid)]
        for a, b in zip(ordered, ordered[1:], strict=False):
            gap_links.append((a, b, color))
    return v_boundary, v_grouping, {int(i) for i in bad}, block_group, gap_links


def _draw_block(
    ax, inst, i, box, overlapped, boundary_bad, block_group, labels
) -> None:
    """One block: its kind-colored rectangle and the overlap / violation overlays."""
    x, y, w, h = box
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=_KIND_COLOR[_block_kind(inst, i)],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            zorder=2,
        )
    )
    if i in overlapped:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="none",
                edgecolor="red",
                hatch="///",
                linewidth=0.8,
                zorder=3,
            )
        )
    if i in boundary_bad:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="orange", alpha=0.45, zorder=5))
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="none",
                edgecolor="darkorange",
                linewidth=3.5,
                zorder=6,
            )
        )
        ax.text(
            x + w - 0.04 * w,
            y + h - 0.04 * h,
            "B",
            ha="right",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="square,pad=0.1", fc="orange", ec="black", lw=0.8),
            zorder=7,
        )
    if i in block_group:
        color, fragment = block_group[i]
        ax.add_patch(
            Rectangle(
                (x, y), w, h, facecolor="none", edgecolor=color, linewidth=3.0, zorder=6
            )
        )
        ax.text(
            x + 0.05 * w,
            y + h - 0.05 * h,
            str(fragment),
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.1", fc=color, ec="black", lw=0.6),
            zorder=7,
        )
    if labels:
        ax.text(
            x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=5, zorder=4
        )


def _draw_nets(ax, inst, centers, draw_nets: bool, draw_pins: bool) -> None:
    """The b2b nets and the pins with their p2b
    lines, one ``LineCollection`` per layer."""
    if draw_nets:
        segments = _b2b_segments(inst, centers)
        if segments.shape[0]:
            alpha, width = _line_style(segments.shape[0])
            ax.add_collection(
                LineCollection(
                    segments,
                    colors="steelblue",
                    linewidths=width,
                    alpha=alpha,
                    zorder=0,
                )
            )
    if draw_pins and inst.pins_pos.shape[0]:
        segments, valid_pins = _p2b_segments(inst, centers)
        if segments.shape[0]:
            alpha, width = _line_style(segments.shape[0])
            ax.add_collection(
                LineCollection(
                    segments, colors="green", linewidths=width, alpha=alpha, zorder=0
                )
            )
        for px, py in valid_pins:
            ax.add_patch(Circle((px, py), radius=1.0, color="green", zorder=1))


def draw_placement(
    ax,
    placement: Placement,
    score: CaseScore | None = None,
    title: str | None = None,
    draw_nets: bool = True,
    draw_pins: bool = True,
    labels: bool = True,
    mark_violations: bool = False,
) -> None:
    """Draw ``placement`` onto the matplotlib axes ``ax``.

    Blocks by kind, nets, pins, bounding box, legend and a score title. ``labels``
    writes the block indices; ``mark_violations`` overlays an orange "B" on
    boundary-violating blocks and outlines every member of a split cluster in its group
    color with its fragment number, a dashed arrow linking consecutive fragments.
    """
    inst = placement.instance
    centers = placement.centers
    overlapped = set()
    for i, j in overlapping_pairs(placement.xywh):
        overlapped.update((i, j))

    v_boundary = v_grouping = 0
    boundary_bad: set[int] = set()
    block_group: dict[int, tuple[str, int]] = {}
    gap_links: list = []
    if mark_violations:
        v_boundary, v_grouping, boundary_bad, block_group, gap_links = _violation_marks(
            placement
        )

    for i in range(inst.block_count):
        _draw_block(
            ax,
            inst,
            i,
            placement.xywh[i],
            overlapped,
            boundary_bad,
            block_group,
            labels,
        )
    for (x0, y0), (x1, y1), color in gap_links:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="<|-|>", color=color, lw=2.0, linestyle="dashed"
            ),
            zorder=8,
        )
    _draw_nets(ax, inst, centers, draw_nets, draw_pins)

    x_min, y_min, x_max, y_max = bounding_box(placement.xywh)
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="none",
            edgecolor="navy",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
    )

    legend = [
        Patch(facecolor=c, edgecolor="black", label=k) for k, c in _KIND_COLOR.items()
    ]
    if draw_nets:
        legend.append(Line2D([], [], color="steelblue", label="b2b net"))
    if draw_pins:
        legend.append(Line2D([], [], color="green", label="p2b pin net"))
    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title="block kind",
        borderaxespad=0.0,
    )

    heading = title or f"FloorSet case n={inst.block_count}"
    if score is not None:
        heading += (
            f"\nfeasible={score.feasible}  cost={score.cost:.4f}  "
            f"hpwl_gap={score.hpwl_gap:+.3f}  area_gap={score.area_gap:+.3f}  "
            f"V_rel={score.v_rel:.3f}"
        )
    if mark_violations:
        heading += f"\n[B] V_boundary: {v_boundary}    [G] V_grouping: {v_grouping}"
    ax.set_title(heading, fontsize=9)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.05)


@RENDERER.register("placement")
def render_placement(
    placement: Placement,
    path: str,
    score: CaseScore | None = None,
    title: str | None = None,
    draw_nets: bool = True,
    draw_pins: bool = True,
    mark_violations: bool = True,
) -> str:
    """Render ``placement`` to the PNG ``path``; return ``path``."""
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_placement(
        ax,
        placement,
        score,
        title,
        draw_nets,
        draw_pins,
        mark_violations=mark_violations,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
