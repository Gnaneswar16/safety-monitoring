from collections import deque
import time
from typing import Dict, Tuple, List, Optional
from core.compliance import ComplianceResult
import config

class PersonTrackState:
    def __init__(self, window_size: int = config.SMOOTHING_WINDOW_SIZE):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.consecutive_violations: int = 0
        self.last_violation_event_time: float = 0.0

class TemporalSmoother:
    """Provides 5-frame temporal smoothing and violation debouncing/cooldown management."""

    def __init__(self, window_size: int = config.SMOOTHING_WINDOW_SIZE,
                 violation_frames: int = config.VIOLATION_CONSECUTIVE_FRAMES,
                 cooldown_seconds: float = config.VIOLATION_COOLDOWN_SECONDS):
        self.window_size = window_size
        self.violation_frames = violation_frames
        self.cooldown_seconds = cooldown_seconds
        self.tracks: Dict[str, PersonTrackState] = {}

    def reset(self):
        self.tracks.clear()

    def process(self, person_id: str, raw_result: ComplianceResult, current_time: Optional[float] = None) -> Tuple[ComplianceResult, bool]:
        if current_time is None:
            current_time = time.time()

        if person_id not in self.tracks:
            self.tracks[person_id] = PersonTrackState(self.window_size)

        track = self.tracks[person_id]
        track.history.append(raw_result)

        # Track consecutive raw violation frames
        if raw_result.status == "VIOLATION":
            track.consecutive_violations += 1
        else:
            track.consecutive_violations = 0

        # Majority vote for smoothed status across history
        statuses = [res.status for res in track.history]
        violation_count = statuses.count("VIOLATION")
        compliant_count = statuses.count("COMPLIANT")
        insufficient_count = statuses.count("INSUFFICIENT EVIDENCE")

        # Smooth status decision
        if compliant_count > 0 and raw_result.status == "VIOLATION" and len(track.history) < self.window_size:
            # Prevent instant status flip on initial single missed frame
            smoothed_status = "COMPLIANT" if compliant_count >= violation_count else raw_result.status
        elif track.consecutive_violations >= self.violation_frames:
            smoothed_status = "VIOLATION"
        elif compliant_count > violation_count:
            smoothed_status = "COMPLIANT"
        elif insufficient_count >= compliant_count and insufficient_count >= violation_count:
            smoothed_status = "INSUFFICIENT EVIDENCE"
        else:
            smoothed_status = raw_result.status

        # Create smoothed result
        smoothed_result = ComplianceResult(
            person_id=person_id,
            status=smoothed_status,
            has_glasses=raw_result.has_glasses,
            has_shoes=raw_result.has_shoes,
            feet_visible=raw_result.feet_visible,
            face_visible=raw_result.face_visible,
            glasses_conf=raw_result.glasses_conf,
            shoes_conf=raw_result.shoes_conf,
            missing_ppe=raw_result.missing_ppe,
            reason=raw_result.reason
        )

        # Check violation event trigger (requires ~5 consecutive violation frames & 30s cooldown)
        trigger_event = False
        if track.consecutive_violations >= self.violation_frames:
            if (current_time - track.last_violation_event_time) >= self.cooldown_seconds:
                trigger_event = True
                track.last_violation_event_time = current_time

        return smoothed_result, trigger_event
