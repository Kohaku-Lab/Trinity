# The closed-form refiner

`trinity/sampling/refine_closed.py`, registered as `REFINER` key `closed`.

The refiner descends the six constraint terms of [physics.md](physics.md) on the latent of a
finished sample. Nothing is learned: the refiner is the hand-written gradient of those terms
with respect to the latent, and an Adam step without momentum. `constraint_latent`
(`trinity/sampling/refine_constraint.py`) is the same energy through autograd; the tests check
that the two gradients agree.

```python
from trinity.registry import REFINER, build
from trinity.sampling import build_refine_case

refiner = build({"name": "closed", "steps": 400}, REFINER)
case = build_refine_case(instances, samples_per_case, device)
z_refined = refiner.refine(z, case)  # z: (B, N, 3) latents
```

## Coordinates

The latent of a block is `z = (u, v, ρ)` with `cx = u·s`, `cy = v·s`, `w = √a·e^{ρ/2}`,
`h = √a·e^{−ρ/2}` (`a` the target area; the refiner works in normalized units, `s = 1`). Every
term is written on the box edges `L, R, B, T`, so each derivative is taken with respect to the
edges first and then folded back to the latent:

```
dE/dcx = dE/dL + dE/dR          dE/dw = (dE/dR − dE/dL) / 2
dE/dcy = dE/dB + dE/dT          dE/dh = (dE/dT − dE/dB) / 2
dE/du  = s · dE/dcx             dE/dv = s · dE/dcy
dE/dρ  = (w/2) · dE/dw − (h/2) · dE/dh       (0 where ρ sits on the ±4 clamp)
```

Preplaced blocks get no position gradient; fixed and preplaced blocks get no shape gradient;
padding tokens get zero. Anchored coordinates are re-clamped after every step.

## The derivatives

* **overlap** — a pair with `ox·oy > 0` pushes apart: `dE/dR_i = oy·[R_i < R_j] / s²`,
  `dE/dL_i = −oy·[L_i > L_j] / s²`, and the same in `y` with `ox`.
* **group** — Prim's tree is recomputed every step; every tree edge with a positive gap pulls
  its two blocks together along the active axis with force `1/s`.
* **mib** — `dE/dρ_i = sign(ρ_i − median)` for groups of more than one block.
* **boundary** — a left-coded block gets `dE/dL = +1/s`; the block that owns `x_min` gets
  `−1/s` per left-coded block. Right, top and bottom are symmetric.
* **wl** — `dE/dcx_i = [Σ_j ā_ij sign(cx_i − cx_j) + Σ_{p→i} w_p sign(cx_i − q_p^x)] / (s·W)`
  with `ā = (a + aᵀ)/2`; no shape force.
* **area** — only the four owners of the bounding box move: `dE/dL_{argmin L} = −H_bb/s²`,
  `dE/dR_{argmax R} = +H_bb/s²`, and in `y` with `W_bb`.
* **outline** (bookshelf protocol only) — the two owners of the axis that exceeds the outline
  move inward with force `1/s`.

## The step

Adam with `β₁ = 0`:

```
v ← β₂·v + (1 − β₂)·g²
z ← z − lr_t · g / (√(v / (1 − β₂^t)) + ε)
```

`lr_t` is a precomputed vector from a named schedule (`constant`, `cosine`, `linear`) or an
AnySchedule dict. The loop runs `chunk` steps inside one `torch.compile`d function and passes
`(z, v, t)` between chunks, so snapshots fall on chunk boundaries.

## Defaults

`ClosedFormRefiner()` is the paper's setting: `PAPER_WEIGHTS` (overlap 10, the other five
terms 1), 400 steps, linear decay from `lr = 1e-3`, `betas = (0, 0.99)`. Every argument is a
config knob; the configs under `configs/eval/` show the settings of each experiment.

## Tests

`tests/test_refine_closed.py`: closed form equals autograd per term and summed (float64,
1e-6); padding invariance and zero gradient on padding and frozen channels; the loop equals
`torch.optim.Adam` at the same betas; compiled equals eager; the lr vectors of the named
schedules.
