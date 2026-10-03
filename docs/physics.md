# The physics: six constraint terms

Trinity carries one set of differentiable quantities, one per FloorSet constraint or
objective. The same six functions are

* the **auxiliary loss** in training (`trinity/losses/constraint.py`),
* the **energy** the refiner descends on a finished sample (`trinity/sampling/refine_closed.py`),
* the **soft metrics** in evaluation (`trinity/floorplan/scoring/`).

Each term is the violated quantity of its constraint, made dimensionless by an instance
property, and zero exactly when the constraint holds. A term has one constant, its `weight`.
Fixed and preplaced channels are detached in every term, so a frozen block is an obstacle
that exerts force but receives none.

Notation: block `i` is the rectangle `[x_i, x_i+w_i] × [y_i, y_i+h_i]`; `s² = Σ_i w_i h_i` is
the total block area of the instance and `s` its length scale.

## 1. Overlap (`overlap`)

No two blocks intersect. With the intersection area of a pair

```
A_ij = max(0, min(x_i+w_i, x_j+w_j) − max(x_i, x_j)) · max(0, min(y_i+h_i, y_j+h_j) − max(y_i, y_j))
```

the term is `L_overlap = Σ_{i<j} A_ij / s²`, the metric `overlap_ratio`. For an overlapping
pair the force on each block along x equals the penetration along y (and vice versa): zero at
contact, growing with penetration.

## 2. Grouping (`group`)

The blocks of a cluster form one edge-connected component. The distance-to-contact of a pair
is `gap_ij = max(gap_x, gap_y)` with `gap_x = max(0, max(x_i, x_j) − min(x_i+w_i, x_j+w_j))`;
it is zero exactly when the two rectangles touch or overlap on both axes. The violated
quantity of a cluster is the total gap of the minimum spanning tree of its members,

```
G_p = Σ_{(i,j) ∈ MST_p} gap_ij,        L_group = Σ_p G_p / s
```

which is zero exactly when the cluster is connected. Each tree edge pulls its two blocks
together along the axis that defines its gap. The tree is computed by a batched Prim over all
clusters at once.

## 3. MIB (`mib`)

All blocks of an MIB group share one `(w, h)`. Members share their area and the decode keeps
area, so a shape has one degree of freedom, `ρ_i = log(w_i / h_i)`. The violated quantity is
the total aspect change that makes the group identical:

```
M_q = Σ_{i ∈ q} |ρ_i − median_q(ρ)|,        L_mib = Σ_q M_q
```

Each member receives a unit force in `ρ` toward the group median.

## 4. Boundary (`boundary`)

A boundary-coded block touches every outline edge its code names (1 left, 2 right, 4 top,
8 bottom; corners are sums) of the layout's bounding box:

```
B_i = [left](x_i − x_min) + [right](x_max − x_i − w_i) + [top](y_max − y_i − h_i) + [bottom](y_i − y_min)
L_boundary = Σ_{i coded} B_i / s
```

The coded block moves toward its edge, and the block that currently owns that edge moves
toward the coded block.

## 5. Wirelength (`wl`)

An objective. With block centres `c_i`, block-to-block weights `a_ij` and pins `p` at `q_p`
on block `b_p` with weight `w_p`:

```
H = Σ_{i<j} a_ij |c_i − c_j|_1 + Σ_p w_p |c_{b_p} − q_p|_1
L_wl = H / (s · (Σ a_ij + Σ w_p))
```

the mean net length in units of `s` (the metric `wl_norm`).

## 6. Area (`area`)

An objective: the bounding-box area, `L_area = (x_max − x_min)(y_max − y_min) / s²` (the
metric `compactness`, floor 1). Only the four frontier blocks receive force.

## The outline term (bookshelf protocol only)

The fixed-outline protocol of the GSRC / MCNC benchmarks adds a hinge on the bounding box
beyond the given outline `W × H`:

```
L_outline = (max(0, x_max − x_min − W) + max(0, y_max − y_min − H)) / s
```

FloorSet instances carry no outline, so the term is absent there.

## Summary

| term | registry key | quantity | normalizer | zero iff |
|---|---|---|---|---|
| overlap | `overlap` | Σ pairwise intersection area | `s²` | no overlap |
| grouping | `group` | Σ clusters, MST total gap-to-contact | `s` | every cluster connected |
| MIB | `mib` | Σ groups Σ members \|ρ − median ρ\| | — | identical shapes |
| boundary | `boundary` | Σ coded blocks, distance to required edges | `s` | coded blocks on their edges |
| wirelength | `wl` | weighted Manhattan net length | `s · Σ weights` | (objective) |
| area | `area` | bounding-box area | `s²` | (objective, floor 1) |

In training each term's per-sample value is weighted by `weight`, or by `weight · t` with
`t_weight: true` (`t = 1` at pure noise, `0` at data), and averaged over the batch. The
decode clamps `ρ` to `[−4, 4]` (`trinity.decode.RHO_CLAMP`).

Tests: `tests/test_losses_constraint.py`, `tests/test_refine_closed.py`.
