"""MPC controller for autonomous vehicles in the mixed-traffic simulation."""

import numpy as np
from scipy.optimize import minimize

from .config import (
    DT,
    MPC_HORIZON,
    AV_ACCEL_MIN,
    AV_ACCEL_MAX,
    AV_SPEED_MAX,
    AV_DESIRED_SPEED,
    AV_SPEED_WEIGHT,
    AV_SMOOTHNESS_WEIGHT,
    VEHICLE_LENGTH,
    MIN_GAP,
    AV_HEADWAY_TARGET,
    AV_HEADWAY_WEIGHT,
    AV_EPSILON
)
from .dynamics import AVDynamics, HDVDynamics, VehicleDynamics


def _predict_leader(current_position, current_speed, current_acceleration, horizon):
    """Predict the leader using its current acceleration over the MPC horizon."""
    positions = []
    speeds = []

    for _ in range(horizon):
        speed = HDVDynamics.update_velocity(current_speed, current_acceleration)
        position = VehicleDynamics.update_position(current_position, current_speed)
        positions.append(position)
        speeds.append(speed)
        current_speed = speed
        current_position = position

    positions = np.asarray(positions)
    speeds = np.asarray(speeds)

    return positions, speeds 


def _predict_av(current_position, current_speed, acceleration_sequence):
    """Predict the AV trajectory for a candidate acceleration sequence."""
    positions, speeds = AVDynamics.predict_trajectory(current_position, current_speed, acceleration_sequence)
    positions = np.asarray(positions)
    speeds = np.asarray(speeds)

    return positions, speeds



def _objective(acceleration_sequence, current_position, current_speed, previous_acceleration, leader_position, leader_speed, leader_acceleration):
    """MPC objective: desired speed tracking + smooth acceleration + soft time-headway safety."""
    av_positions, av_speeds = _predict_av(current_position, current_speed, acceleration_sequence)
    leader_positions, _ = _predict_leader(leader_position, leader_speed, leader_acceleration, MPC_HORIZON)

    # Speed tracking.
    speed_error = av_speeds - AV_DESIRED_SPEED
    speed_cost = AV_SPEED_WEIGHT * np.sum(speed_error ** 2)

    # Penalize changes in acceleration
    acceleration_changes = np.diff(np.concatenate(([previous_acceleration], acceleration_sequence))) # calculate changes in acceleration, including the previous acceleration; total length = MPC_HORIZON
    jerk = np.diff(acceleration_changes) / DT
    jerk_cost = (AV_SMOOTHNESS_WEIGHT * np.sum(jerk ** 2) * DT) #

    # penalize predicted time headway below the target.
    gaps = leader_positions - av_positions - VEHICLE_LENGTH
    positive_speeds = np.maximum(av_speeds, AV_EPSILON)
    time_headways = gaps / positive_speeds
    headway_deficit = np.maximum(0.0, AV_HEADWAY_TARGET - time_headways) # penalize only when the predicted time headway is below the target
    safety_cost = AV_HEADWAY_WEIGHT * np.sum(headway_deficit ** 2)

    return speed_cost + jerk_cost + safety_cost


def _constraint_values(acceleration_sequence, current_position, current_speed, leader_position, leader_speed, leader_acceleration):
    """Return predicted speed and gap margins for the MPC constraints."""
    av_positions, av_speeds = _predict_av(current_position, current_speed, acceleration_sequence)
    leader_positions, _ = _predict_leader(leader_position, leader_speed, leader_acceleration, MPC_HORIZON)

    # Speed must remain in the physical AV range: 0 <= v <= AV_SPEED_MAX.
    speed_margins = np.concatenate((av_speeds, AV_SPEED_MAX - av_speeds))

    # Front-bumper positions are used, so subtract vehicle length to obtain the bumper-to-bumper gap.
    gaps = leader_positions - av_positions - VEHICLE_LENGTH  
    gap_margins = gaps - MIN_GAP

    return np.concatenate((speed_margins, gap_margins))


def compute_av_acceleration(current_position, current_speed, leader_position, leader_speed, leader_acceleration, previous_acceleration=0.0, previous_optimal_sequence=None):
    """
    Compute the AV acceleration using receding-horizon MPC.
    A warm start is used after the first MPC call. If the previous optimal sequence was [a0, a1, a2], the next optimization starts from [a1, a2, a2].
    The optimizer chooses a sequence of accelerations over the MPC horizon. 
    Only the first acceleration is applied by the simulation. The full optimal sequence is returned so it can be stored and shifted at the next step.
    """
    # -------------------------------------------------
    # Warm-start initial guess
    # -------------------------------------------------
    if previous_optimal_sequence is not None:
        previous_optimal_sequence = np.asarray(previous_optimal_sequence, dtype=float)

        if previous_optimal_sequence.shape == (MPC_HORIZON,):
            # Shift the previous optimal sequence by one step.
            initial_guess = np.concatenate(
                (
                    previous_optimal_sequence[1:],
                    [previous_optimal_sequence[-1]],
                )
            )
        else:
            # Safe fallback if the stored sequence has the wrong size.
            initial_guess = np.full(MPC_HORIZON, previous_acceleration, dtype=float) # creates an array of length MPC_HORIZON filled with previous_acceleration
    else:
        # First MPC call: there is no previous optimal sequence yet.
        initial_guess = np.full(MPC_HORIZON, previous_acceleration, dtype=float) # creates an array of length MPC_HORIZON filled with previous_acceleration
    
    initial_guess = np.clip(initial_guess, AV_ACCEL_MIN, AV_ACCEL_MAX) # ensure the initial guess is within the AV acceleration limits

    # -------------------------------------------------
    # MPC optimization
    # -------------------------------------------------
    result = minimize(
        _objective,
        initial_guess,
        args=(current_position, current_speed, previous_acceleration, leader_position, leader_speed, leader_acceleration),
        method="SLSQP",
        bounds=[(AV_ACCEL_MIN, AV_ACCEL_MAX)] * MPC_HORIZON,
        constraints={
            "type": "ineq",
            "fun": lambda a: _constraint_values(
                a,
                current_position,
                current_speed,
                leader_position,
                leader_speed,
                leader_acceleration,
            ),
        },
        options={
            "maxiter": 100,
            "ftol": 1e-6,
        },
    )

    # Evaluate the candidate returned by SLSQP.
    candidate = (np.asarray(result.x, dtype=float)
        if hasattr(result, "x")
        else initial_guess
    )
    constraint_values = _constraint_values(candidate, current_position, current_speed, leader_position, leader_speed, leader_acceleration)
    speed_margins = constraint_values[:2 * MPC_HORIZON] # The first 2 * MPC_HORIZON values correspond to speed constraints.
    gap_margins = constraint_values[2 * MPC_HORIZON:]   # The last MPC_HORIZON values correspond to gap constraints.
    diagnostics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun)
        if np.isfinite(result.fun)
        else np.inf,
        "min_constraint_margin": float(np.min(constraint_values)),
        "min_speed_margin": float(np.min(speed_margins)),
        "min_gap_margin": float(np.min(gap_margins)),
        "iterations": int(getattr(result, "nit", -1)),
        "initial_guess": initial_guess.copy(),
        "optimal_sequence": None,
    }

    # If optimization fails, use a safe bounded fallback.
    if not result.success:
        fallback_acceleration = float(np.clip(previous_acceleration, AV_ACCEL_MIN, AV_ACCEL_MAX))
        diagnostics["optimal_sequence"] = None
        diagnostics["fallback_acceleration"] = fallback_acceleration
        return fallback_acceleration, initial_guess, diagnostics
    
    optimal_sequence = np.clip(result.x, AV_ACCEL_MIN, AV_ACCEL_MAX)
    diagnostics["optimal_sequence"] = optimal_sequence.copy()

    return float(optimal_sequence[0]), optimal_sequence, diagnostics