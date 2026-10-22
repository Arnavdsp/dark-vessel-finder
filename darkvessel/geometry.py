"""Imaging geometry for a small LEO SAR satellite.

Image axes: x = ground range (columns), y = azimuth / along-track (rows).
All positions are in metres from the scene origin.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SarGeometry:
    altitude_m: float = 520e3
    incidence_deg: float = 30.0
    platform_speed_ms: float = 7600.0
    wavelength_m: float = 0.031       # X-band
    prf_hz: float = 3500.0
    pixel_m: float = 5.0
    aasr_db: float = -18.0            # azimuth ambiguity-to-signal ratio

    @property
    def slant_range_m(self) -> float:
        # flat-earth approximation is good to a few percent at 30 deg
        return self.altitude_m / math.cos(math.radians(self.incidence_deg))

    def azimuth_shift_m(self, v_ground_range_ms):
        """Apparent azimuth displacement of a target moving in ground range.

        A radial velocity v_r adds a Doppler offset 2 v_r / lambda, which the
        azimuth compression maps to a position offset of -R v_r / V.
        Works on scalars and numpy arrays.
        """
        v_r = v_ground_range_ms * math.sin(math.radians(self.incidence_deg))
        return -self.slant_range_m * v_r / self.platform_speed_ms

    @property
    def ambiguity_offset_m(self) -> float:
        """Azimuth distance between a bright target and its first ambiguity."""
        return (self.wavelength_m * self.slant_range_m * self.prf_hz
                / (2.0 * self.platform_speed_ms))
