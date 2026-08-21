from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from core.associator import PersonAssociation

@dataclass
class ComplianceResult:
    person_id: str
    status: str  # "COMPLIANT", "VIOLATION", "INSUFFICIENT EVIDENCE"
    has_glasses: bool
    has_shoes: bool
    feet_visible: bool
    face_visible: bool
    glasses_conf: Optional[float]
    shoes_conf: Optional[float]
    missing_ppe: List[str] = field(default_factory=list)
    reason: str = ""

class ComplianceEngine:
    """Evaluates compliance status per person based on associated PPE and body region visibility."""

    def evaluate(self, assoc: PersonAssociation) -> ComplianceResult:
        has_glasses = len(assoc.glasses) > 0
        has_shoes = len(assoc.shoes) > 0

        glasses_conf = max([g.confidence for g in assoc.glasses]) if has_glasses else None
        shoes_conf = max([s.confidence for s in assoc.shoes]) if has_shoes else None

        feet_vis = assoc.feet_visible
        face_vis = assoc.face_visible

        # If safety shoes are detected on the person, feet are effectively visible
        if has_shoes:
            feet_vis = True

        missing_ppe = []

        # Glasses evaluation
        glasses_missing_definite = False
        if not has_glasses:
            if face_vis:
                glasses_missing_definite = True
                missing_ppe.append("safety_glasses")

        # Shoes evaluation
        shoes_missing_definite = False
        if not has_shoes:
            if feet_vis:
                shoes_missing_definite = True
                missing_ppe.append("safety_shoes")

        # Determine overall status
        if has_glasses and has_shoes:
            status = "COMPLIANT"
            reason = "All required PPE present and verified"
        elif glasses_missing_definite or shoes_missing_definite:
            status = "VIOLATION"
            reason = f"Missing: {', '.join(missing_ppe)}"
        elif not has_shoes and not feet_vis:
            status = "INSUFFICIENT EVIDENCE"
            reason = "Feet not visible"
        elif not has_glasses and not face_vis:
            status = "INSUFFICIENT EVIDENCE"
            reason = "Face not visible"
        else:
            status = "INSUFFICIENT EVIDENCE"
            reason = "Insufficient body visibility"

        return ComplianceResult(
            person_id=assoc.person_id,
            status=status,
            has_glasses=has_glasses,
            has_shoes=has_shoes,
            feet_visible=feet_vis,
            face_visible=face_vis,
            glasses_conf=glasses_conf,
            shoes_conf=shoes_conf,
            missing_ppe=missing_ppe,
            reason=reason
        )
