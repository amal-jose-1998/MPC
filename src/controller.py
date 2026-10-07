"""
MPC controller for autonomous vehicles in the mixed-traffic simulation.
- during the warning period, check whether the feasible MPC trajectory can
  clear the pedestrian crossing before the pedestrian arrives;
- if it cannot, switch to a stopping trajectory;
- while the pedestrian is active, keep the AV at or before the safe stop
  position until the pedestrian leaves.
"""

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
    HDV_SPEED_MAX,
    PEDESTRIAN_TIME,
    PEDESTRIAN_POSITION,
    PEDESTRIAN_REQUIRE_FULL_CLEARANCE,
    AV_STOP_POSITION_WEIGHT,
    PEDESTRIAN_QUEUE_STANDSTILL_GAP,
)
from .pedestrian import get_pedestrian_state, X_STOP, T_PEDESTRIAN_END


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

    # Jerk Jacobian
    # jerk[0] = (a[0] - previous_acceleration) / dt
    # jerk[k] = (a[k] - a[k-1]) / dt
    # Therefore J_JERK has shape (n, n).

    jjerk = np.zeros((n, n), dtype=float)

    jjerk[0, 0] = 1.0 / dt

    if n > 1:
        rows = np.arange(1, n)
        jjerk[rows, rows] = 1.0 / dt
        jjerk[rows, rows - 1] = -1.0 / dt

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

    # x[k+1] = x[k] + v[k] * DT.
    positions = (
        current_position
        + DT 
        * np.cumsum(
            np.concatenate(
                ([current_speed], speeds[:-1]))
        )
    )

    return positions, speeds


# ============================================================
# Pedestrian geometry
# ============================================================
def _pedestrian_clearance_position():
    """Position required for the AV to have cleared the crossing."""
    if PEDESTRIAN_REQUIRE_FULL_CLEARANCE:
        return PEDESTRIAN_POSITION + VEHICLE_LENGTH
    return PEDESTRIAN_POSITION

def _crossing_index(current_time, horizon):
    """Index corresponding to pedestrian arrival, if it lies in the horizon."""

    time_to_pedestrian = PEDESTRIAN_TIME - current_time
    if time_to_pedestrian <= 0.0:
        return None

    steps = int(np.ceil(time_to_pedestrian / DT - 1e-10))
    if steps < 1 or steps > horizon:
        return None

    return steps - 1

def _pedestrian_constraint_indices(current_time, horizon):
    """Return prediction indices that must stay before X_STOP."""
    state = get_pedestrian_state(current_time)

    if state == "warning":
        crossing_idx = _crossing_index(current_time, horizon)
        if crossing_idx is None:
            return np.empty(0, dtype=int)
        # In warning mode, the stop constraint is used only by the STOP MPC and is imposed at pedestrian arrival.
        return np.array([crossing_idx], dtype=int)

    if state != "active":
        return np.empty(0, dtype=int)

    future_times = current_time + DT * np.arange(1, horizon + 1)
    indices = np.flatnonzero(future_times <= T_PEDESTRIAN_END + 1e-9)
    return indices.astype(int)

# ============================================================
# Objective and analytic Jacobian
# ============================================================
def _objective_with_jacobian(acceleration_sequence, current_position, current_speed, previous_acceleration, 
                             leader_positions, speed_reference, stop_target_positions=None):
    """Return MPC objective and its analytic gradient."""
    a = np.asarray(acceleration_sequence, dtype=float)
    av_positions, av_speeds = _predict_av(current_position, current_speed, a)

    # --------------------------------------------------------
    # 1. Desired-speed tracking
    # --------------------------------------------------------
    speed_error = av_speeds - speed_reference
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

    # ========================================================
    # 4. stopping-position objective
    # ========================================================
    stop_position_cost = 0.0
    grad_stop_position = (np.zeros_like(a))

    if stop_target_positions is not None:
        position_error = av_positions - stop_target_positions
        stop_position_cost = AV_STOP_POSITION_WEIGHT  * np.sum(position_error ** 2)
        grad_stop_position = 2.0 * AV_STOP_POSITION_WEIGHT * (J_X.T @ position_error)

    # ========================================================
    # Total objective
    # ========================================================
    objective = speed_cost + jerk_cost + safety_cost + stop_position_cost
    gradient = grad_speed + grad_jerk + grad_headway + grad_stop_position

    return float(objective), gradient

# ============================================================
# Constraint values and analytic Jacobian
# ============================================================
def _constraint_values(acceleration_sequence, current_position, current_speed, leader_positions, current_time, pedestrian_stop=False,):
    """Return speed and minimum-gap constraint margins."""
    av_positions, av_speeds = _predict_av(current_position, current_speed, acceleration_sequence)
    speed_margins = np.concatenate((av_speeds, AV_SPEED_MAX - av_speeds))

    gaps = leader_positions - av_positions - VEHICLE_LENGTH
    gap_margins = gaps - MIN_GAP

    margins = [speed_margins, gap_margins]

    if pedestrian_stop:
        indices = _pedestrian_constraint_indices(current_time, acceleration_sequence.size)

        if indices.size:
            # Hard pedestrian safety condition: x_pred <= X_STOP
            pedestrian_margin = X_STOP - av_positions[indices]
            margins.append(pedestrian_margin)

    return np.concatenate(margins)


def _constraint_jacobian(current_time, pedestrian_stop=False):
    """Return the analytic Jacobian of all SLSQP inequality margins."""
    rows = [
        J_V,
        -J_V,
        -J_X,
    ]
    if pedestrian_stop:
        indices = _pedestrian_constraint_indices(current_time, MPC_HORIZON)
        if indices.size:
            rows.append(-J_X[indices])

    return np.vstack(rows)


# Default Jacobian for the normal MPC.
CONSTRAINT_JAC = _constraint_jacobian(current_time=0.0, pedestrian_stop=False)


# ============================================================
# One MPC solve
# ============================================================
def _solve_mpc(current_position, current_speed, leader_position, leader_speed, leader_acceleration,
    previous_acceleration, initial_guess, current_time, speed_reference, pedestrian_stop=False, leader_is_stopping=False):
    """Solve one MPC problem."""
    horizon = MPC_HORIZON

    leader_positions, _ = _predict_leader(leader_position, leader_speed, leader_acceleration, horizon)

    stop_target_positions = None
    if pedestrian_stop:
        if leader_is_stopping:
            stop_target = leader_positions - VEHICLE_LENGTH - PEDESTRIAN_QUEUE_STANDSTILL_GAP
        else:
            stop_target = X_STOP

        # Keep one fixed stopping target over the whole MPC horizon.
        stop_target_positions = np.full(horizon, stop_target, dtype=float)

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
                speed_reference,
                stop_target_positions,
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
                speed_reference,
                stop_target_positions,
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
                current_time,
                pedestrian_stop,
            )
        elif cache["constraints"] is None:
            cache["constraints"] = _constraint_values(
                x,
                current_position,
                current_speed,
                leader_positions,
                current_time,
                pedestrian_stop,
            )

        return cache["constraints"]

    def _evaluate_constraint_jacobian(x):
        # Horizon and pedestrian phase are fixed during this solve.
        return _constraint_jacobian(
            current_time=current_time,
            pedestrian_stop=pedestrian_stop,
        )

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

    candidate = (
        np.asarray(result.x, dtype=float)
        if hasattr(result, "x")
        else np.asarray(initial_guess, dtype=float)
    )

    constraint_values = _constraint_values(
        candidate,
        current_position,
        current_speed,
        leader_positions,
        current_time,
        pedestrian_stop,
    )

    return result, candidate, constraint_values, leader_positions

# ============================================================
# Public MPC controller
# ============================================================
def compute_av_acceleration(current_position, current_speed, leader_position, leader_speed, leader_acceleration, previous_acceleration=0.0,
    previous_optimal_sequence=None, current_time=0.0, leader_is_stopping=False,):
    """Compute the AV acceleration using receding-horizon MPC.
    
    Pedestrian decision logic:
        1. Before warning: normal MPC.
        2. During warning: solve normal MPC first.
           - If its feasible trajectory reaches the clearance position by
             pedestrian arrival, continue with that trajectory.
           - Otherwise, solve a stopping MPC with x(t_p) <= X_STOP.
        3. During pedestrian activity: use stopping MPC with a hard
           x <= X_STOP constraint until the pedestrian leaves.
        4. After pedestrian clearance: return to normal MPC.

    Only the first acceleration of the chosen optimal sequence is applied.
    """
    horizon = MPC_HORIZON

    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------
    if previous_optimal_sequence is not None:
        previous_optimal_sequence = np.asarray(previous_optimal_sequence, dtype=float)

        if previous_optimal_sequence.shape == (horizon,):
            initial_guess = np.concatenate((previous_optimal_sequence[1:], [previous_optimal_sequence[-1]]))
        else:
            initial_guess = np.full(horizon, previous_acceleration, dtype=float)
    else:
        initial_guess = np.full(horizon, previous_acceleration, dtype=float)

    initial_guess = np.clip(initial_guess, AV_ACCEL_MIN, AV_ACCEL_MAX)

    state = get_pedestrian_state(current_time)
    diagnostics = {
        "success": False,
        "status": -1,
        "message": "",
        "objective": np.inf,
        "min_constraint_margin": np.inf,
        "min_speed_margin": np.inf,
        "min_gap_margin": np.inf,
        "iterations": -1,
        "initial_guess": initial_guess.copy(),
        "optimal_sequence": None,
        "pedestrian_state": state,
        "pedestrian_mode": "normal",
        "pedestrian_clearance_possible": None,
        "pedestrian_clearance_position": _pedestrian_clearance_position(),
        "pedestrian_stop_position": X_STOP,
        "predicted_position_at_pedestrian_time": None,
        "stop_mpc_success": None,
    }

    # --------------------------------------------------------
    # 1. Normal MPC solve.
    # --------------------------------------------------------
    normal_result, normal_candidate, normal_constraints, leader_positions = _solve_mpc(
        current_position=current_position,
        current_speed=current_speed,
        leader_position=leader_position,
        leader_speed=leader_speed,
        leader_acceleration=leader_acceleration,
        previous_acceleration=previous_acceleration,
        initial_guess=initial_guess,
        current_time=current_time,
        speed_reference=AV_DESIRED_SPEED,
        pedestrian_stop=False,
        leader_is_stopping=leader_is_stopping,
    )

    normal_feasible = bool(normal_result.success and np.all(normal_constraints >= -1e-6))

    chosen_result = normal_result
    chosen_candidate = normal_candidate
    chosen_constraints = normal_constraints
    chosen_mode = "normal"

    # --------------------------------------------------------
    # 2. Warning: use predicted position at pedestrian arrival to decide CLEAR vs STOP.
    # --------------------------------------------------------
    if state == "warning":
        crossing_idx = _crossing_index(current_time, horizon)

        if crossing_idx is not None and normal_feasible: # Only check clearance if the normal MPC is feasible and the pedestrian arrival is within the horizon.
            predicted_at_pedestrian = _predict_av(current_position, current_speed, normal_candidate)[0][crossing_idx] # Predicted position at pedestrian arrival.
            diagnostics["predicted_position_at_pedestrian_time"] = float(predicted_at_pedestrian)

            clearance_target = _pedestrian_clearance_position()
            clearance_possible = (predicted_at_pedestrian >= clearance_target - 1e-6)
            diagnostics["pedestrian_clearance_possible"] = bool(clearance_possible)
        else: # If the normal MPC is infeasible or the pedestrian arrival is outside the horizon, we cannot guarantee clearance.
            clearance_possible = False
            diagnostics["pedestrian_clearance_possible"] = False

        if clearance_possible:
            chosen_mode = "committed"
        else:
            # If the normal MPC cannot guarantee clearance, we attempt a stopping MPC with a hard x <= X_STOP constraint.
            stop_guess = normal_candidate if normal_feasible else initial_guess

            stop_result, stop_candidate, stop_constraints, _ = _solve_mpc(
                current_position=current_position,
                current_speed=current_speed,
                leader_position=leader_position,
                leader_speed=leader_speed,
                leader_acceleration=leader_acceleration,
                previous_acceleration=previous_acceleration,
                initial_guess=stop_guess,
                current_time=current_time,
                speed_reference=0.0,
                pedestrian_stop=True,
                leader_is_stopping=leader_is_stopping
            )

            stop_feasible = bool(stop_result.success and np.all(stop_constraints >= -1e-6))
            diagnostics["stop_mpc_success"] = stop_feasible

            if stop_feasible:
                chosen_result = stop_result
                chosen_candidate = stop_candidate
                chosen_constraints = stop_constraints
                chosen_mode = "stopping"
            else:
                # A true dilemma/infeasible situation: neither a normal crossing trajectory nor a safe stopping trajectory is feasible.
                chosen_mode = "dilemma"

    # --------------------------------------------------------
    # 3. Active pedestrian: enforce x <= X_STOP during the active interval.
    # --------------------------------------------------------
    elif state == "active" and current_position < PEDESTRIAN_POSITION:
        active_guess = normal_candidate if normal_feasible else initial_guess

        stop_result, stop_candidate, stop_constraints, _ = _solve_mpc(
            current_position=current_position,
            current_speed=current_speed,
            leader_position=leader_position,
            leader_speed=leader_speed,
            leader_acceleration=leader_acceleration,
            previous_acceleration=previous_acceleration,
            initial_guess=active_guess,
            current_time=current_time,
            speed_reference=0.0,
            pedestrian_stop=True,
            leader_is_stopping=leader_is_stopping
        )

        stop_feasible = bool(stop_result.success and np.all(stop_constraints >= -1e-6))
        diagnostics["stop_mpc_success"] = stop_feasible

        if stop_feasible:
            chosen_result = stop_result
            chosen_candidate = stop_candidate
            chosen_constraints = stop_constraints
            chosen_mode = "stopping"
        else:
            chosen_mode = "dilemma"

    # --------------------------------------------------------
    # Diagnostics for chosen solution.
    # --------------------------------------------------------
    diagnostics["success"] = bool(chosen_result.success and np.all(chosen_constraints >= -1e-6))
    diagnostics["status"] = int(chosen_result.status)
    diagnostics["message"] = str(chosen_result.message)
    diagnostics["objective"] = (float(chosen_result.fun) if np.isfinite(chosen_result.fun) else np.inf)

    speed_margins = chosen_constraints[: 2 * horizon]
    gap_margins = chosen_constraints[2 * horizon:]

    diagnostics["min_constraint_margin"] = float(np.min(chosen_constraints))
    diagnostics["min_speed_margin"] = float(np.min(speed_margins))

    # The first horizon entries after speed margins are the gap constraints.
    diagnostics["min_gap_margin"] = float(np.min(gap_margins[:horizon]))
    diagnostics["iterations"] = int(getattr(chosen_result, "nit", -1))
    diagnostics["pedestrian_mode"] = chosen_mode

    # --------------------------------------------------------
    # Safe fallback.
    # --------------------------------------------------------
    if not diagnostics["success"]:
        if chosen_mode == "dilemma":
            # Use the strongest available AV braking as the emergency action.
            fallback_acceleration = AV_ACCEL_MIN
        else:
            fallback_acceleration = float(np.clip(previous_acceleration, AV_ACCEL_MIN, AV_ACCEL_MAX))

        diagnostics["fallback_acceleration"] = fallback_acceleration
        diagnostics["optimal_sequence"] = initial_guess.copy()
        return (fallback_acceleration, initial_guess, diagnostics)

    optimal_sequence = np.clip(chosen_candidate, AV_ACCEL_MIN, AV_ACCEL_MAX,)
    diagnostics["optimal_sequence"] = optimal_sequence.copy()

    return (float(optimal_sequence[0]), optimal_sequence, diagnostics)
