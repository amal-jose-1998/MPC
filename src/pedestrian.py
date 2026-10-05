from .config import (
    HDV_ACCEL_MIN,
    PEDESTRIAN_TIME,
    PEDESTRIAN_POSITION,
    PEDESTRIAN_DURATION,
    PEDESTRIAN_WARNING_DURATION,
    PEDESTRIAN_HDV_WARNING_BRAKE,
    PEDESTRIAN_HDV_ACTIVE_BRAKE,
    PEDESTRIAN_HDV_REACTION_TIME,
    PEDESTRIAN_HDV_SAFE_DISTANCE,
)


def get_pedestrian_state(time):
    """
    Determine the pedestrian state at the given simulation time.
    States:
        "none"     : no pedestrian warning or pedestrian present
        "warning"  : possible pedestrian warning is active
        "active"   : pedestrian is present at the crossing
        "cleared"  : pedestrian has left the crossing
    """
    warning_start = PEDESTRIAN_TIME - PEDESTRIAN_WARNING_DURATION
    pedestrian_end = PEDESTRIAN_TIME + PEDESTRIAN_DURATION

    if time < warning_start:
        return "none"
    elif time < PEDESTRIAN_TIME:
        return "warning"
    elif time < pedestrian_end:
        return "active"
    else:
        return "cleared"

def is_pedestrian_ahead_of_vehicle(vehicle_position, pedestrian_position):
    """
    Check whether the pedestrian crossing is ahead of the vehicle.
    """
    return pedestrian_position > vehicle_position


def is_pedestrian_between_vehicle_and_leader(vehicle_position, leader_position, pedestrian_position):
    """
    Check whether the pedestrian crossing is between the vehicle and its leader.
    """
    return (vehicle_position < pedestrian_position  and pedestrian_position < leader_position)


def calculate_hdv_stopping_distance(vehicle_speed):
    """
    Calculate the approximate stopping distance of an HDV.
    """
    reaction_distance = vehicle_speed * PEDESTRIAN_HDV_REACTION_TIME   # Distance travelled during driver reaction time.
    braking_distance = vehicle_speed ** 2 / (2.0 * abs(PEDESTRIAN_HDV_ACTIVE_BRAKE))   # Distance required during braking.
    
    stopping_distance = reaction_distance + braking_distance
    return stopping_distance



def calculate_hdv_pedestrian_acceleration(vehicle_position, vehicle_speed, pedestrian_state, pedestrian_position, leader_position):
    """
    Determine whether an HDV should react to the pedestrian.
    For the front-most vehicle:
        There is no leader, so the vehicle reacts if the pedestrian is ahead of it.
    For a following HDV:
        The pedestrian must be between the HDV and its leader.

    Warning phase:
        Apply mild braking when the pedestrian is within the HDV's reaction/stopping region.
    Active phase:
        Apply stronger braking when the pedestrian is present and the HDV needs to stop before the crossing.

    Returns:
        acceleration value if pedestrian braking is required None otherwise.
    """
    # -------------------------------------------------
    # No pedestrian information or pedestrian cleared.
    # -------------------------------------------------
    if pedestrian_state not in ("warning", "active"):
        return None

    # -------------------------------------------------
    # Determine whether the pedestrian is relevant.
    # Front-most vehicle:
    #     no leader -> only check whether pedestrian is ahead of the vehicle.
    # Following HDV:
    #     pedestrian must be between the HDV and leader.
    # -------------------------------------------------
    if leader_position is None:
        pedestrian_is_relevant = is_pedestrian_ahead_of_vehicle(vehicle_position, pedestrian_position)

    else:
        pedestrian_is_relevant = is_pedestrian_between_vehicle_and_leader(vehicle_position, leader_position, pedestrian_position)

    if not pedestrian_is_relevant:
        return None
    
    # -------------------------------------------------
    # Distance from HDV to pedestrian crossing.
    # -------------------------------------------------
    distance_to_pedestrian = pedestrian_position - vehicle_position

    # -------------------------------------------------
    # Required stopping distance.
    # -------------------------------------------------
    stopping_distance = calculate_hdv_stopping_distance(vehicle_speed)
    required_distance = stopping_distance + PEDESTRIAN_HDV_SAFE_DISTANCE    # Add an extra safety margin.

    # -------------------------------------------------
    # WARNING PHASE
    # -------------------------------------------------
    if pedestrian_state == "warning":
        # The driver becomes cautious when the crossing  is close enough that braking may be needed.
        if distance_to_pedestrian <= required_distance:
            return PEDESTRIAN_HDV_WARNING_BRAKE

        return None

    # -------------------------------------------------
    # ACTIVE PHASE
    # -------------------------------------------------
    if pedestrian_state == "active":
        distance_to_stop = pedestrian_position - vehicle_position

        # Already beyond the pedestrian.
        if distance_to_stop <= 0.0:
            return None

        # Braking acceleration required to stop at the safe stopping position.
        return max((-(vehicle_speed ** 2) / (2.0 * distance_to_stop)), HDV_ACCEL_MIN)

    return None