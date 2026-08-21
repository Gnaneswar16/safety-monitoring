from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from core.detector import Detection
import config

@dataclass
class PersonAssociation:
    person_detection: Detection
    person_id: str
    glasses: List[Detection] = field(default_factory=list)
    shoes: List[Detection] = field(default_factory=list)
    eye_roi: Tuple[float, float, float, float] = (0, 0, 0, 0)
    feet_roi: Tuple[float, float, float, float] = (0, 0, 0, 0)
    feet_visible: bool = True
    face_visible: bool = True
    debug_info: Dict[str, Any] = field(default_factory=dict)

class PPEAssociator:
    """Associates glasses and shoes detections to person detections based on spatial ROI rules."""

    @staticmethod
    def calculate_bbox_intersection(boxA: Tuple[float, float, float, float],
                                     boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interWidth = max(0.0, xB - xA)
        interHeight = max(0.0, yB - yA)
        return interWidth * interHeight

    @staticmethod
    def get_eye_roi(person_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        eye_y1 = py1 + 0.04 * ph
        eye_y2 = py1 + 0.42 * ph
        eye_x1 = px1 - 0.10 * pw
        eye_x2 = px2 + 0.10 * pw
        return (eye_x1, eye_y1, eye_x2, eye_y2)

    @staticmethod
    def get_feet_roi(person_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        feet_y1 = py1 + 0.60 * ph
        feet_y2 = py2 + 0.08 * ph
        feet_x1 = px1 - 0.15 * pw
        feet_x2 = px2 + 0.15 * pw
        return (feet_x1, feet_y1, feet_x2, feet_y2)

    def validate_glasses_candidate(self, person_bbox: Tuple[float, float, float, float],
                                    glasses_bbox: Tuple[float, float, float, float]) -> Tuple[bool, str, float]:
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        gx1, gy1, gx2, gy2 = glasses_bbox
        gw = gx2 - gx1
        gh = gy2 - gy1
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0

        if gw <= 0 or gh <= 0 or pw <= 0 or ph <= 0:
            return False, "Invalid bbox dimensions", 0.0

        # 1. Reject forehead/hair (too high) or chest/torso (too low)
        if gcy < py1 + 0.03 * ph:
            return False, f"Glasses center y={gcy:.1f} on forehead/hair (<{py1+0.03*ph:.1f})", 0.0
        if gcy > py1 + 0.45 * ph:
            return False, f"Glasses center y={gcy:.1f} on chest/body (>{py1+0.45*ph:.1f})", 0.0

        eye_roi = self.get_eye_roi(person_bbox)
        eye_x1, eye_y1, eye_x2, eye_y2 = eye_roi

        # 2. Check center position within eye ROI
        if not (eye_x1 <= gcx <= eye_x2 and eye_y1 <= gcy <= eye_y2):
            return False, f"Glasses center ({gcx:.1f},{gcy:.1f}) outside eye ROI [{eye_x1:.1f},{eye_y1:.1f},{eye_x2:.1f},{eye_y2:.1f}]", 0.0

        # 3. Check plausible relative size
        if gw > 0.75 * pw or gh > 0.40 * ph:
            return False, f"Glasses size ({gw:.1f}x{gh:.1f}) implausible for person size ({pw:.1f}x{ph:.1f})", 0.0

        # 4. Check overlap with eye ROI
        inter_area = self.calculate_bbox_intersection(glasses_bbox, eye_roi)
        glasses_area = gw * gh
        overlap_ratio = inter_area / glasses_area if glasses_area > 0 else 0.0

        if overlap_ratio < 0.15:
            return False, f"Eye ROI overlap ratio {overlap_ratio:.2f} < 0.15", overlap_ratio

        return True, "ACCEPTED", overlap_ratio

    def validate_shoes_candidate(self, person_bbox: Tuple[float, float, float, float],
                                  shoes_bbox: Tuple[float, float, float, float]) -> Tuple[bool, str, float]:
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        sx1, sy1, sx2, sy2 = shoes_bbox
        sw = sx2 - sx1
        sh = sy2 - sy1
        scx = (sx1 + sx2) / 2.0
        scy = (sy1 + sy2) / 2.0

        if sw <= 0 or sh <= 0 or pw <= 0 or ph <= 0:
            return False, "Invalid bbox dimensions", 0.0

        # 1. Must be in lower portion of body
        if scy < py1 + 0.50 * ph:
            return False, f"Shoes center y={scy:.1f} in upper/mid body (<{py1+0.50*ph:.1f})", 0.0

        feet_roi = self.get_feet_roi(person_bbox)
        feet_x1, feet_y1, feet_x2, feet_y2 = feet_roi

        # 2. Check center position within feet ROI
        if not (feet_x1 <= scx <= feet_x2 and feet_y1 <= scy <= feet_y2):
            return False, f"Shoes center ({scx:.1f},{scy:.1f}) outside feet ROI [{feet_x1:.1f},{feet_y1:.1f},{feet_x2:.1f},{feet_y2:.1f}]", 0.0

        # 3. Check plausible relative size
        if sw > 0.75 * pw or sh > 0.50 * ph:
            return False, f"Shoes size ({sw:.1f}x{sh:.1f}) implausible for person size ({pw:.1f}x{ph:.1f})", 0.0

        # 4. Overlap ratio
        inter_area = self.calculate_bbox_intersection(shoes_bbox, feet_roi)
        shoes_area = sw * sh
        overlap_ratio = inter_area / shoes_area if shoes_area > 0 else 0.0

        if overlap_ratio < 0.15:
            return False, f"Feet ROI overlap ratio {overlap_ratio:.2f} < 0.15", overlap_ratio

        return True, "ACCEPTED", overlap_ratio

    def associate(self, detections: List[Detection], frame_shape: Tuple[int, int, int] = (480, 640, 3)) -> List[PersonAssociation]:
        frame_h, frame_w = frame_shape[:2]

        raw_person_dets = [d for d in detections if d.class_id == config.CLASS_PERSON]
        glasses_dets = [d for d in detections if d.class_id == config.CLASS_GLASSES]
        shoes_dets = [d for d in detections if d.class_id == config.CLASS_SHOES]

        # Deduplicate overlapping person detections (Non-Maximum Suppression)
        raw_person_dets.sort(key=lambda d: d.confidence, reverse=True)
        person_dets: List[Detection] = []
        for p in raw_person_dets:
            keep = True
            for existing in person_dets:
                inter = self.calculate_bbox_intersection(p.bbox, existing.bbox)
                p_area = (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])
                e_area = (existing.bbox[2] - existing.bbox[0]) * (existing.bbox[3] - existing.bbox[1])
                union = p_area + e_area - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.45 or inter / min(p_area, e_area) > 0.70:
                    keep = False
                    break
            if keep:
                person_dets.append(p)

        associations: List[PersonAssociation] = []

        # Sort persons left-to-right to maintain stable spatial order
        person_dets.sort(key=lambda d: d.bbox[0])

        for idx, person in enumerate(person_dets):
            p_id = f"P{idx+1:03d}"
            px1, py1, px2, py2 = person.bbox
            ph = py2 - py1

            eye_roi = self.get_eye_roi(person.bbox)
            feet_roi = self.get_feet_roi(person.bbox)

            # Feet visibility: person bottom near frame bottom edge
            feet_visible = (py2 < frame_h - 15)
            # Face visibility: person top near frame top edge
            face_visible = (py1 > 10) and (py1 + 0.38 * ph < frame_h)

            assoc = PersonAssociation(
                person_detection=person,
                person_id=p_id,
                eye_roi=eye_roi,
                feet_roi=feet_roi,
                feet_visible=feet_visible,
                face_visible=face_visible,
                debug_info={"glasses_logs": [], "shoes_logs": []}
            )
            associations.append(assoc)

        # Match glasses detections to candidate persons
        for g_idx, g_det in enumerate(glasses_dets):
            best_person = None
            best_score = -1.0
            best_log = ""

            for assoc in associations:
                valid, reason, score = self.validate_glasses_candidate(assoc.person_detection.bbox, g_det.bbox)
                assoc.debug_info["glasses_logs"].append({
                    "glasses_idx": g_idx,
                    "bbox": g_det.bbox,
                    "confidence": g_det.confidence,
                    "accepted": valid,
                    "reason": reason,
                    "score": score
                })
                if valid and score > best_score:
                    best_score = score
                    best_person = assoc
                    best_log = reason

            if best_person is not None:
                best_person.glasses.append(g_det)

        # Match shoes detections to candidate persons
        for s_idx, s_det in enumerate(shoes_dets):
            best_person = None
            best_score = -1.0

            for assoc in associations:
                valid, reason, score = self.validate_shoes_candidate(assoc.person_detection.bbox, s_det.bbox)
                assoc.debug_info["shoes_logs"].append({
                    "shoes_idx": s_idx,
                    "bbox": s_det.bbox,
                    "confidence": s_det.confidence,
                    "accepted": valid,
                    "reason": reason,
                    "score": score
                })
                if valid and score > best_score:
                    best_score = score
                    best_person = assoc

            if best_person is not None:
                best_person.shoes.append(s_det)

        return associations
