"""A validator over a :class:`Placement`: one
score and a readable list of violations."""

from dataclasses import dataclass

from trinity.floorplan.registry import SCORER, resolve
from trinity.floorplan.scoring.cost import CaseScore
from trinity.floorplan.types import Placement


@dataclass
class ValidationResult:
    """Feasibility and the full score of one layout."""

    feasible: bool
    score: CaseScore

    def summary(self) -> str:
        """A multi-line report naming every violated constraint."""
        s = self.score
        lines = [
            f"feasible={self.feasible}  cost={s.cost:.4f}  "
            f"hpwl_gap={s.hpwl_gap:+.3f}  area_gap={s.area_gap:+.3f}  "
            f"V_rel={s.v_rel:.3f}",
        ]
        f = s.feasibility
        if f.overlap_count:
            lines.append(
                f"  HARD overlap: {f.overlap_count} pair(s) e.g. {f.overlap_pairs[:5]}"
            )
        if f.area_violations:
            lines.append(f"  HARD area-tolerance: blocks {f.area_violations}")
        if f.fixed_violations:
            lines.append(f"  HARD fixed-shape moved: blocks {f.fixed_violations}")
        if f.preplaced_violations:
            lines.append(f"  HARD preplaced moved: blocks {f.preplaced_violations}")
        if s.soft is not None:
            so = s.soft
            if so.grouping:
                lines.append(
                    f"  SOFT grouping: {so.grouping} "
                    f"(split groups {so.grouping_groups})"
                )
            if so.mib:
                lines.append(f"  SOFT mib: {so.mib} (groups {so.mib_groups})")
            if so.boundary:
                lines.append(
                    f"  SOFT boundary: {so.boundary} (blocks {so.boundary_blocks})"
                )
        if len(lines) == 1:
            lines.append("  no violations")
        return "\n".join(lines)


def validate(placement: Placement, scorer: str = "full") -> ValidationResult:
    """Score ``placement`` with the ``scorer`` backend."""
    score_fn = resolve(scorer, SCORER)
    score = score_fn(placement)
    return ValidationResult(feasible=score.feasible, score=score)
