"""MPC controller for autonomous vehicles in the mixed-traffic simulation."""

import numpy as np
from scipy.optimize import minimize
from functools import lru_cache # remembers function results for faster repeated calls

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
    AV_EPSILON,
    HDV_SPEED_MAX
)


@lru_cache(maxsize=1)
def _prediction_matrices(horizon, dt):
    """Return constant sensitivity matrices for the discrete AV model.

    The model is
        v[k+1] = v[k] + a[k] * dt
        x[k+1] = x[k] + v[k] * dt

    For a decision vector a = [a0, ..., a(N-1)],
        v_pred = v_base + Jv @ a
        x_pred = x_base + Jx @ a

    Jv and Jx are independent of the current state, so they can be cached.
    Jjerk is the Jacobian of the jerk sequence with respect to a.
    """
    n = int(horizon)
    idx = np.arange(n, dtype=float)

    # Speed sensitivity:
    #   dv_j / da_i = dt, for i <= j.
    jv = dt * np.tril(np.ones((n, n), dtype=float))

    # Position sensitivity:
    #   dx_j / da_i = dt^2 * (j-i), for i < j.
    # The diagonal is zero because x[k+1] uses the old speed v[k].
    position_coeff = np.maximum(idx[:, None] - idx[None, :], 0.0)
    jx = (dt ** 2) * position_coeff

    # Acceleration-change matrix for
    #   da = diff([previous_acceleration, a]).
    d_acc = np.eye(n, dtype=float)
    if n > 1:
        d_acc[np.arange(1, n), np.arange(n - 1)] = -1.0 

    # Jerk = diff(acceleration_changes) / dt.
    if n > 1:
        d_diff = np.zeros((n - 1, n), dtype=float)
        rows_j = np.arange(n - 1)
        d_diff[rows_j, rows_j] = -1.0
        d_diff[rows_j, rows_j + 1] = 1.0
        jjerk = (d_diff @ d_acc) / dt
    else:
        jjerk = np.empty((0, n), dtype=float)

    return jv, jx, jjerk

# Cache matrices once at import time for the configured MPC horizon.
J_V, J_X, J_JERK = _prediction_matrices(MPC_HORIZON, float(DT))


# ============================================================
# Vectorized trajectory prediction
# ============================================================
def _predict_av(current_position, current_speed, acceleration_sequence):
    """Vectorized prediction of the AV trajectory."""
    acceleration_sequence = np.asarray(acceleration_sequence, dtype=float)

    horizon = acceleration_sequence.size
    if horizon != MPC_HORIZON:
        jv, jx, _ = _prediction_matrices(horizon, float(DT))
    else:
        jv, jx = J_V, J_X

    # No-acceleration baseline.
    steps = np.arange(1, horizon + 1, dtype=float)
    base_speeds = np.full(horizon, current_speed, dtype=float)
    base_positions = current_position + current_speed * DT * steps

    speeds = base_speeds + jv @ acceleration_sequence
    positions = base_positions + jx @ acceleration_sequence

    return positions, speeds

def _predict_leader(current_position, current_speed, current_acceleration, horizon):
    """Vectorized prediction of the leader under constant acceleration."""
    steps = np.arange(1, horizon + 1, dtype=float)
    speeds = np.clip(current_speed + DT * current_acceleration * steps, 0.0, HDV_SPEED_MAX,)

    # Same integration convention as the simulation:
    # x[k+1] = x[k] + v[k] * DT.
    positions = (
        current_position
        + current_speed * DT * steps
        + 0.5 * current_acceleration * (DT ** 2) * steps * (steps - 1.0)
    )

    return positions, speeds

# ============================================================
# Objective and analytic Jacobian
# ============================================================
def _objective_with_jacobian(acceleration_sequence, current_position, current_speed, previous_acceleration, leader_positions):
    """Return MPC objective and its analytic gradient."""
    a = np.asarray(acceleration_sequence, dtype=float)
    av_positions, av_speeds = _predict_av(current_position, current_speed, a)

    # --------------------------------------------------------
    # 1. Desired-speed tracking
    # --------------------------------------------------------
    speed_error = av_speeds - AV_DESIRED_SPEED
    speed_cost = AV_SPEED_WEIGHT * np.sum(speed_error ** 2)
    grad_speed = 2.0 * AV_SPEED_WEIGHT * (J_V.T @ speed_error)

    # --------------------------------------------------------
    # 2. Jerk/smoothness penalty
    # --------------------------------------------------------
    acceleration_changes = np.diff(np.concatenate(([previous_acceleration], a)))
    jerk = acceleration_changes / DT

    jerk_cost = AV_SMOOTHNESS_WEIGHT * np.sum(jerk ** 2) * DT

    if J_JERK.size:
        grad_jerk = (
            2.0
            * AV_SMOOTHNESS_WEIGHT
            * DT
            * (J_JERK.T @ jerk)
        )
    else:
        grad_jerk = np.zeros_like(a)

    # --------------------------------------------------------
    # 3. Soft time-headway penalty
    # --------------------------------------------------------
    gaps = leader_positions - av_positions - VEHICLE_LENGTH
    positive_speeds = np.maximum(av_speeds, AV_EPSILON)
    time_headways = gaps / positive_speeds

    headway_deficit = np.maximum(0.0, AV_HEADWAY_TARGET - time_headways)

    safety_cost = AV_HEADWAY_WEIGHT * np.sum(headway_deficit ** 2)

    # h = gap / positive_speed
    # gap = leader_position - av_position - L
    # dgap/da = -J_X
    # d(positive_speed)/da = J_V where speed > epsilon,
    #                                  0 otherwise.
    d_headway = np.empty((a.size, a.size), dtype=float)

    positive_mask = av_speeds > AV_EPSILON

    if np.any(positive_mask):
        v = av_speeds[positive_mask]
        g = gaps[positive_mask]
        d_headway[positive_mask] = (
            -J_X[positive_mask] / v[:, None]
            - (g / (v ** 2))[:, None] * J_V[positive_mask]
        )

    if np.any(~positive_mask):
        v = positive_speeds[~positive_mask]
        d_headway[~positive_mask] = -J_X[~positive_mask] / v[:, None]

    active_headway = headway_deficit > 0.0

    if np.any(active_headway):
        # d(deficit)/da = -d(headway)/da.
        grad_headway = 2.0 * AV_HEADWAY_WEIGHT * (
            (-headway_deficit[active_headway])[:, None]
            * d_headway[active_headway]
        ).sum(axis=0)
    else:
        grad_headway = np.zeros_like(a)

    objective = speed_cost + jerk_cost + safety_cost
    gradient = grad_speed + grad_jerk + grad_headway

    return float(objective), gradient

# ============================================================
# Constraint values and analytic Jacobian
# ============================================================
def _constraint_values(acceleration_sequence, current_position, current_speed, leader_positions):
    """Return speed and minimum-gap constraint margins."""
    av_positions, av_speeds = _predict_av(current_position, current_speed, acceleration_sequence)

    speed_margins = np.concatenate(av_speeds, AV_SPEED_MAX - av_speeds)

    gaps = leader_positions - av_positions - VEHICLE_LENGTH
    gap_margins = gaps - MIN_GAP

    return np.concatenate((speed_margins, gap_margins))


def _constraint_jacobian():
    """Return the analytic Jacobian of all SLSQP inequality margins.

    Constraint order is exactly the same as _constraint_values:
        [v,
         AV_SPEED_MAX-v,
         gap-MIN_GAP]

    Because leader predictions are independent of the AV decision variables,
    d(gap)/da = -J_X.
    """
    return np.vstack(
        (
            J_V,
            -J_V,
            -J_X,
        )
    )


CONSTRAINT_JAC = _constraint_jacobian()


# ============================================================
# Public MPC controller
# ============================================================
def compute_av_acceleration(
    current_position,
    current_speed,
    leader_position,
    leader_speed,
    leader_acceleration,
    previous_acceleration=0.0,
    previous_optimal_sequence=None,
):
    """Compute the AV acceleration using receding-horizon MPC.
    A warm start is used after the first call. Only the first acceleration of
    the optimized sequence is applied by the simulator.
    """
    horizon = MPC_HORIZON

    if previous_optimal_sequence is not None:
        previous_optimal_sequence = np.asarray(previous_optimal_sequence, dtype=float)

        if previous_optimal_sequence.shape == (horizon,):
            initial_guess = np.concatenate(previous_optimal_sequence[1:], [previous_optimal_sequence[-1]])
        else:
            initial_guess = np.full(horizon, previous_acceleration, dtype=float)
    else:
        initial_guess = np.full(horizon, previous_acceleration, dtype=float)

    initial_guess = np.clip(initial_guess, AV_ACCEL_MIN, AV_ACCEL_MAX)

    # Leader prediction is fixed during this MPC solve, so compute it once.
    leader_positions, leader_speeds = _predict_leader(leader_position, leader_speed, leader_acceleration, horizon)

    del leader_speeds  # retained only for clarity; objective/constraints use positions.

    # --------------------------------------------------------
    # Evaluation cache
    # --------------------------------------------------------
    # SLSQP may ask for objective, Jacobian, constraints and constraint
    # Jacobian at the same point. Keep one cached evaluation to avoid running
    # the trajectory rollout more than necessary.
    cache = {
        "x": None,
        "objective": None,
        "objective_gradient": None,
        "constraints": None,
    }

    def _same_point(x):
        return (
            cache["x"] is not None
            and cache["x"].shape == x.shape
            and np.array_equal(cache["x"], x)
        )

    def _evaluate_objective(x):
        x = np.asarray(x, dtype=float)
        if not _same_point(x):
            value, gradient = _objective_with_jacobian(
                x,
                current_position,
                current_speed,
                previous_acceleration,
                leader_positions,
            )
            cache["x"] = x.copy()
            cache["objective"] = value
            cache["objective_gradient"] = gradient
            cache["constraints"] = None
        elif cache["objective"] is None:
            value, gradient = _objective_with_jacobian(
                x,
                current_position,
                current_speed,
                previous_acceleration,
                leader_positions,
            )
            cache["objective"] = value
            cache["objective_gradient"] = gradient

        return cache["objective"]

    def _evaluate_objective_jacobian(x):
        x = np.asarray(x, dtype=float)
        _ = _evaluate_objective(x)
        return cache["objective_gradient"]

    def _evaluate_constraints(x):
        x = np.asarray(x, dtype=float)
        if not _same_point(x):
            cache["x"] = x.copy()
            cache["objective"] = None
            cache["objective_gradient"] = None
            cache["constraints"] = _constraint_values(
                x,
                current_position,
                current_speed,
                leader_positions,
            )
        elif cache["constraints"] is None:
            cache["constraints"] = _constraint_values(
                x,
                current_position,
                current_speed,
                leader_positions,
            )

        return cache["constraints"]

    def _evaluate_constraint_jacobian(x):
        # The Jacobian is constant for this model, because both speed and
        # position depend affinely on the acceleration sequence.
        return CONSTRAINT_JAC

    # --------------------------------------------------------
    # SLSQP optimization
    # --------------------------------------------------------
    result = minimize(
        _evaluate_objective,
        initial_guess,
        jac=_evaluate_objective_jacobian,
        method="SLSQP",
        bounds=[(AV_ACCEL_MIN, AV_ACCEL_MAX)] * horizon,
        constraints={
            "type": "ineq",
            "fun": _evaluate_constraints,
            "jac": _evaluate_constraint_jacobian,
        },
        options={
            "maxiter": 100,
            "ftol": 1e-6,
            "disp": False,
        },
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    candidate = (
        np.asarray(result.x, dtype=float)
        if hasattr(result, "x")
        else initial_guess
    )

    constraint_values = _constraint_values(
        candidate,
        current_position,
        current_speed,
        leader_positions,
    )

    speed_margins = constraint_values[: 2 * horizon]
    gap_margins = constraint_values[2 * horizon:]

    diagnostics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "objective": (
            float(result.fun)
            if np.isfinite(result.fun)
            else np.inf
        ),
        "min_constraint_margin": float(np.min(constraint_values)),
        "min_speed_margin": float(np.min(speed_margins)),
        "min_gap_margin": float(np.min(gap_margins)),
        "iterations": int(getattr(result, "nit", -1)),
        "initial_guess": initial_guess.copy(),
        "optimal_sequence": None,
    }

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------
    if not result.success:
        fallback_acceleration = float(np.clip(previous_acceleration, AV_ACCEL_MIN, AV_ACCEL_MAX))
        diagnostics["fallback_acceleration"] = fallback_acceleration
        return fallback_acceleration, initial_guess, diagnostics

    optimal_sequence = np.clip(candidate, AV_ACCEL_MIN, AV_ACCEL_MAX)
    diagnostics["optimal_sequence"] = optimal_sequence.copy()

    return float(optimal_sequence[0]), optimal_sequence, diagnostics
