"""
Pedestrian logic for the all-HDV scenario.
 
The crossing is treated as a virtual stopped obstacle at
    X_STOP = PEDESTRIAN_POSITION - PEDESTRIAN_HDV_SAFE_DISTANCE.
Every HDV behind the crossing (leader and followers alike) uses the same rule.
 
Driver mode (stored per vehicle by the simulation):
    MODE_FREE      : not braking for the pedestrian
    MODE_STOPPING  : latched, braking to stop at X_STOP
    MODE_COMMITTED : cannot stop in time but clears the crossing before the pedestrian appears
"""

import numpy as np

from .config import (
    PEDESTRIAN_TIME,
    PEDESTRIAN_POSITION,
    PEDESTRIAN_DURATION,
    PEDESTRIAN_WARNING_DURATION,
    PEDESTRIAN_HDV_ACTIVE_BRAKE,
    PEDESTRIAN_HDV_COMFORT_BRAKE,
    PEDESTRIAN_HDV_REACTION_TIME,
    PEDESTRIAN_HDV_SAFE_DISTANCE,
    PEDESTRIAN_QUEUE_STANDSTILL_GAP,
    PEDESTRIAN_REQUIRE_FULL_CLEARANCE,
    VEHICLE_LENGTH,
    DT,
)


MODE_FREE = 0
MODE_STOPPING = 1
MODE_COMMITTED = 2
 
B_MAX = abs(PEDESTRIAN_HDV_ACTIVE_BRAKE)                      # maximum braking [m/s^2]
X_STOP = PEDESTRIAN_POSITION - PEDESTRIAN_HDV_SAFE_DISTANCE   # target stopping position [m]
T_WARNING_START = PEDESTRIAN_TIME - PEDESTRIAN_WARNING_DURATION
T_PEDESTRIAN_END = PEDESTRIAN_TIME + PEDESTRIAN_DURATION
T_REACTION_END = T_WARNING_START + PEDESTRIAN_HDV_REACTION_TIME   # driver starts to act here

def get_pedestrian_state(time):
    """
    Determine the pedestrian state at the given simulation time.
    States:
        "none"     : no pedestrian warning or pedestrian present
        "warning"  : possible pedestrian warning is active
        "active"   : pedestrian is present at the crossing
        "cleared"  : pedestrian has left the crossing
    """
    eps = 1e-9
    if time < T_WARNING_START - eps:
        return "none"
    elif time < PEDESTRIAN_TIME - eps:
        return "warning"
    elif time < T_PEDESTRIAN_END - eps:
        return "active"
    return "cleared"


def calculate_hdv_stopping_distance(vehicle_speed):
    """Reaction distance plus braking distance at the maximum braking rate (diagnostic helper)."""
    return vehicle_speed * PEDESTRIAN_HDV_REACTION_TIME + vehicle_speed ** 2 / (2.0 * B_MAX)


def _clears_crossing_in_time(position, speed, time):
    """True if the vehicle, keeping its speed, is past the crossing when the pedestrian appears."""
    time_left = max(PEDESTRIAN_TIME - time, 0.0)
    target = PEDESTRIAN_POSITION + (VEHICLE_LENGTH if PEDESTRIAN_REQUIRE_FULL_CLEARANCE else 0.0)
    return position + speed * time_left >= target


def calculate_queue_stop_target(leader_mode, leader_stop_target):
    """
    Stop position of a vehicle. If its leader is stopping for the pedestrian, the vehicle stops
    behind the leader's stop position, so the queue forms without relying on the gap constraint.
    Otherwise it stops at X_STOP.
    """
    if leader_mode == MODE_STOPPING and leader_stop_target is not None:
        return min(X_STOP, leader_stop_target - VEHICLE_LENGTH - PEDESTRIAN_QUEUE_STANDSTILL_GAP)
    return X_STOP


def calculate_hdv_pedestrian_acceleration(vehicle_position, vehicle_speed, time, pedestrian_state, mode, stop_target=X_STOP):
    """
    Pedestrian braking command of one HDV.
    Returns:
        (acceleration or None, new_mode, infeasible_flag)
        stop_target is the position where the vehicle aims to stop (X_STOP, or behind a stopping leader).
        acceleration is None when the pedestrian does not affect the vehicle.
        infeasible_flag is True only on the step where the vehicle is found unable to stop
        AND unable to clear the crossing in time.
    """
    # -------------------------------------------------
    # Pedestrian is not relevant (no warning/active phase) or vehicle already passed the crossing
    # -------------------------------------------------
    if pedestrian_state not in ("warning", "active") or vehicle_position >= PEDESTRIAN_POSITION:
        return None, MODE_FREE, False

    # A committed vehicle that is still before the crossing when the pedestrian appears
    # (for example held back by its leader) must brake now.
    if mode == MODE_COMMITTED:
        if time < PEDESTRIAN_TIME:
            return None, MODE_COMMITTED, False
        mode = MODE_FREE

    # Driver reaction delay: no pedestrian response before the driver has reacted
    if time < T_REACTION_END - 1e-9:
        return None, mode, False

    def required_deceleration(target):
        distance = target - vehicle_position
        if distance > 1e-6:
            return vehicle_speed ** 2 / (2.0 * distance)
        return np.inf if vehicle_speed > 1e-3 else 0.0
 
    required_decel_line = required_deceleration(X_STOP)       # can the vehicle stop before the crossing at all?
    required_decel = required_deceleration(stop_target)       # braking needed for its actual stop position
 
    infeasible = False
 
    # Dilemma zone: stopping needs more than the physical braking limit
    if mode == MODE_FREE and required_decel_line > B_MAX:
        if time < PEDESTRIAN_TIME and _clears_crossing_in_time(vehicle_position, vehicle_speed, time):
            return None, MODE_COMMITTED, False
        infeasible = True
 
    # Engage at the last comfortable moment, then stay engaged (latched)
    if mode == MODE_STOPPING or infeasible or required_decel >= PEDESTRIAN_HDV_COMFORT_BRAKE:
        return -min(required_decel, B_MAX), MODE_STOPPING, infeasible
 
    return None, MODE_FREE, False


def apply_pedestrian_wall(new_position, previous_position, time):
    """
    Hard safety invariant: while the pedestrian is active, no vehicle whose front bumper is
    still before the crossing may reach it.

    Returns:
        (position, speed or None, wall_hit)
    """
    if get_pedestrian_state(time) != "active" or previous_position >= PEDESTRIAN_POSITION:
        return new_position, None, False

    wall = max(previous_position, PEDESTRIAN_POSITION - 0.01)   # never move backwards
    if new_position > wall:
        return wall, max(0.0, (wall - previous_position) / DT), True
    return new_position, None, False