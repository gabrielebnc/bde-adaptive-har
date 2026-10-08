"""Small stateful power-mode controller; no battery/energy model is implied."""
import math
from dataclasses import dataclass


@dataclass
class BatteryController:
    low_at: float = 35.0
    recover_at: float = 40.0
    critical_below: float = 20.0
    mode: str = "normal"

    def __post_init__(self):
        if not 0 <= self.critical_below < self.low_at < self.recover_at <= 100:
            raise ValueError("Battery thresholds must satisfy 0 <= critical_below < low_at < recover_at <= 100")
        if self.mode not in ("normal", "low-power", "critical"):
            raise ValueError("Unknown initial power mode")

    def update(self, battery):
        battery = float(battery)
        if not math.isfinite(battery) or not 0 <= battery <= 100:
            raise ValueError("Battery percentage must be finite and in [0, 100]")
        if battery < self.critical_below:
            self.mode = "critical"
        elif battery >= self.recover_at:
            self.mode = "normal"
        elif self.mode != "normal" or battery <= self.low_at:
            self.mode = "low-power"
        return self.mode

    def reset(self):
        self.mode = "normal"
