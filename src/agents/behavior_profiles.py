# src/agents/behavior_profiles.py


PROFILES = {

    # ==========================================================
    # AGGRESSIVE
    # ==========================================================

    "aggressive": {

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------
        # Values immediate physical progress strongly.
        "progress_scale": 0.30,
        "progress_distance_weight": 0.75,
        "progress_safety_floor": 0.80,

        # ------------------------------------------------------
        # Risk
        # ------------------------------------------------------
        # Lowest risk sensitivity of the three behaviors.
        "risk_weight": 0.45,
        "risk_power": 1.25,
        "near_collision_penalty": 0.55,

        # ------------------------------------------------------
        # Terminal events
        # ------------------------------------------------------
        "collision_penalty": 10.0,
        "completion_bonus": 4.0,

        # ------------------------------------------------------
        # Action preference
        # ------------------------------------------------------
        "action_scale": 0.10,

        # Accelerate remains acceptable to a higher risk level.
        "accelerate_limit": 0.65,

        # Maintain is useful around low-to-medium risk.
        "maintain_center": 0.22,
        "maintain_width": 0.58,

        # Aggressive waits/yields later.
        "yield_start": 0.70,

        # Aggressive brakes only for genuinely high risk.
        "brake_start": 0.82,

        # ------------------------------------------------------
        # Unnecessary defensive action penalties
        # ------------------------------------------------------
        "unnecessary_yield_penalty": 0.025,
        "unnecessary_brake_penalty": 0.035,

        # ------------------------------------------------------
        # Recovery
        # ------------------------------------------------------
        "recovery_delta": -0.08,
        "recovery_risk_limit": 0.35,
        "recovery_accelerate": 0.08,
        "recovery_maintain": 0.05,
        "recovery_yield": 0.04,
        "recovery_brake": 0.06,

        # ------------------------------------------------------
        # Low-speed / deadlock protection
        # ------------------------------------------------------
        "low_speed_threshold": 1.5,
        "low_speed_risk_limit": 0.35,
        "low_speed_penalty": 0.025,

        # ------------------------------------------------------
        # Time efficiency
        # ------------------------------------------------------
        "step_penalty": 0.006,
    },


    # ==========================================================
    # NEUTRAL
    # ==========================================================

    "neutral": {

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------
        "progress_scale": 0.30,
        "progress_distance_weight": 0.65,
        "progress_safety_floor": 0.60,

        # ------------------------------------------------------
        # Risk
        # ------------------------------------------------------
        "risk_weight": 0.60,
        "risk_power": 1.15,
        "near_collision_penalty": 0.70,

        # ------------------------------------------------------
        # Terminal events
        # ------------------------------------------------------
        "collision_penalty": 13.0,
        "completion_bonus": 5.0,

        # ------------------------------------------------------
        # Action preference
        # ------------------------------------------------------
        "action_scale": 0.10,

        "accelerate_limit": 0.45,

        "maintain_center": 0.25,
        "maintain_width": 0.55,

        "yield_start": 0.58,

        "brake_start": 0.72,

        # ------------------------------------------------------
        # Unnecessary defensive action penalties
        # ------------------------------------------------------
        "unnecessary_yield_penalty": 0.012,
        "unnecessary_brake_penalty": 0.020,

        # ------------------------------------------------------
        # Recovery
        # ------------------------------------------------------
        "recovery_delta": -0.08,
        "recovery_risk_limit": 0.35,
        "recovery_accelerate": 0.06,
        "recovery_maintain": 0.05,
        "recovery_yield": 0.025,
        "recovery_brake": 0.04,

        # ------------------------------------------------------
        # Low-speed / deadlock protection
        # ------------------------------------------------------
        "low_speed_threshold": 1.5,
        "low_speed_risk_limit": 0.35,
        "low_speed_penalty": 0.020,

        # ------------------------------------------------------
        # Time efficiency
        # ------------------------------------------------------
        "step_penalty": 0.007,
    },


    # ==========================================================
    # CONSERVATIVE
    # ==========================================================

    "conservative": {

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------
        # Progress still matters, but becomes more safety-gated.
        "progress_scale": 0.30,
        "progress_distance_weight": 0.55,
        "progress_safety_floor": 0.40,

        # ------------------------------------------------------
        # Risk
        # ------------------------------------------------------
        "risk_weight": 0.80,
        "risk_power": 1.05,
        "near_collision_penalty": 0.90,

        # ------------------------------------------------------
        # Terminal events
        # ------------------------------------------------------
        "collision_penalty": 16.0,
        "completion_bonus": 6.0,

        # ------------------------------------------------------
        # Action preference
        # ------------------------------------------------------
        "action_scale": 0.10,

        # Conservative becomes uncomfortable with acceleration early.
        "accelerate_limit": 0.25,

        "maintain_center": 0.20,
        "maintain_width": 0.50,

        "yield_start": 0.45,

        "brake_start": 0.62,

        # ------------------------------------------------------
        # Unnecessary defensive action penalties
        # ------------------------------------------------------
        # Very small because caution is part of the style,
        # but not zero so the policy cannot simply stop.
        "unnecessary_yield_penalty": 0.004,
        "unnecessary_brake_penalty": 0.008,

        # ------------------------------------------------------
        # Recovery
        # ------------------------------------------------------
        # Recovery is intentionally important here.
        "recovery_delta": -0.08,
        "recovery_risk_limit": 0.35,
        "recovery_accelerate": 0.07,
        "recovery_maintain": 0.06,
        "recovery_yield": 0.015,
        "recovery_brake": 0.025,

        # ------------------------------------------------------
        # Low-speed / deadlock protection
        # ------------------------------------------------------
        "low_speed_threshold": 1.5,
        "low_speed_risk_limit": 0.35,
        "low_speed_penalty": 0.018,

        # ------------------------------------------------------
        # Time efficiency
        # ------------------------------------------------------
        "step_penalty": 0.006,
    },
}


def get_behavior_profile(behavior):
    """
    Return a copy so the environment cannot accidentally
    modify the global profile.
    """

    if behavior not in PROFILES:
        raise ValueError(
            f"Unknown behavior: {behavior}. "
            f"Expected one of {list(PROFILES.keys())}"
        )

    return PROFILES[behavior].copy()