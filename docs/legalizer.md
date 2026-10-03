# The legalizer: expansion and constraint-graph packing (`scale_pack`)

`trinity/floorplan/legalize/scale_pack.py`, registered as `LEGALIZER` key `scale_pack`, and
the default route of `trinity.floorplan.legalize.legalize`.

The legalizer takes a refined layout that may still overlap and returns a legal one. Every
step is a one-line rule or a standard constraint-graph method.

```python
from trinity.floorplan.legalize import legalize

result = legalize(placement)  # portfolio=None -> [["scale_pack"]]
result.placement, result.score.feasible, result.score.cost
```

## Problem

A case has `n` blocks with lower-left corner `(x_i, y_i)` and size `(w_i, h_i)`,
`s = √Σ area`. Hard constraints: no pair overlaps; a soft block's area is within 1 % of its
target; a fixed block keeps its `(w, h)`; a preplaced block keeps its `(x, y, w, h)`. Soft
constraints (counted): cluster members form one edge-connected component, MIB members share
one `(w, h)`, a boundary-coded block touches its outline edge(s).

The input is the refined layout (areas exact by the latent parameterization). The output
changes positions, and in step H the aspect of soft blocks at exact area; fixed and MIB shapes
are kept, so area and shape constraints hold by construction.

## Steps

| step | rule |
|---|---|
| A anchors | preplaced blocks to their target box; fixed blocks to their target shape about their centre; soft blocks to their target area about their centre (aspect clipped into the instance's bounds when it carries them); an all-soft MIB group with a common area to its median aspect |
| B expansion | every free block's centre moves outward from the layout centre by the smallest factor `λ ≥ 1` (capped at 10) at which no pair overlaps; preplaced blocks stay |
| C relations | for every pair, the axis on which the two blocks are separated by the larger gap (for an overlapping pair, the axis of least penetration) and their order along it, read from the restored layout (sequence pair / constraint graph; Murata et al. 1996, Lin & Chang 2001) |
| C′ rewrites | a block ordered between an abutting pair on its contact axis, or outside a boundary-coded block on its coded side, is re-ordered on the perpendicular axis |
| D abutment tree | per cluster, a Kruskal spanning tree over member pairs ranked by gap, missing perpendicular overlap, the number of blocks between them, and collisions with preplaced blocks |
| E packing LP | one GLOP LP over block positions with shapes fixed (below; Murata & Kuh 1998) |
| F ladder | when the LP is infeasible, in order: abutment and boundary as penalties; the relations of pinned pairs the LP cannot hold flipped to the other axis (up to six rounds); free blocks above the preplaced blocks of those pairs; every free block above every preplaced block (feasible by construction) |
| G contact snap | abutting tree pairs within 1e-6 of contact set to exact contact |
| H aspect pass | one more LP with the width and height of every soft non-MIB block free within a trust factor of 1.3, area kept by the tangent `h₀·W + w₀·H = 2A`, the same relations; shapes projected back to exact area, re-packed if needed, contacts snapped; kept only when the checker scores it feasible and cheaper. With a fixed outline the pass repeats (up to 4 rounds) while it gains |

## The packing LP (step E)

Variables: `X_i, Y_i` for every block (fixed for preplaced blocks), the outline `L, R, B, T`,
and non-negative auxiliaries for absolute values and slacks.

Constraints:

* outline: `L ≤ X_i`, `X_i + w_i ≤ R`, `B ≤ Y_i`, `Y_i + h_i ≤ T`; with a fixed outline
  `(W, H)` also `R − L ≤ W`, `T − B ≤ H` on every rung but the last;
* relations: `P_lo + size_lo ≤ P_hi` on the pair's axis;
* abutment (tree edge): `P_lo + size_lo = P_hi` with perpendicular overlap `≥ ε`, or the
  deviation penalized on the soft rung;
* boundary: `X_i = L` / `X_i + w_i = R` / `Y_i = B` / `Y_i + h_i = T` per code bit, or the
  deviation penalized on the soft rung.

Objective: `(R − L)·H₀/A₀ + (T − B)·W₀/A₀` (the outline, linearized at the input bounding box
`W₀ × H₀ = A₀`) `+ μ/(n·s)·Σ_i (|X_i − x̂_i| + |Y_i − ŷ_i|)` (displacement from the restored
positions) `+ penalty/s · (soft slacks)`.

## Knobs

| knob | default | meaning |
|---|---|---|
| `relation` | `restored` | read relations from the restored (`λ → 1`) or the expanded layout |
| `mu` | 0.5 | displacement weight |
| `penalty` | 10 | soft-rung penalty |
| `contact` | 1e-3 | perpendicular overlap `ε` of an abutment |
| `cap` | 10 | upper bound of `λ` |
| `reshape` / `trust` | on / 1.3 | the aspect pass and its trust factor |
| `outline` | on | honour a fixed outline when the instance carries one |

`λ` is reported per draw as the price of residual overlap.

## References

* H. Murata, K. Fujiyoshi, S. Nakatake, Y. Kajitani. VLSI module placement based on
  rectangle-packing by the sequence-pair. IEEE TCAD, 1996.
* H. Murata, E. S. Kuh. Sequence-pair based placement method for hard/soft/pre-placed
  modules. ISPD, 1998.
* J.-M. Lin, Y.-W. Chang. TCG: a transitive closure graph-based representation for
  non-slicing floorplans. DAC, 2001.

Tests: `tests/test_scale_pack.py`, `tests/test_bookshelf_outline.py`.
