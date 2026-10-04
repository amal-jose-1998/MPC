"""
Vehicle dynamics for the mixed-traffic simulation.
"""

from .config import (
    DT,
    ALPHA,
    BETA,
    DESIRED_GAP,
    VEHICLE_LENGTH,
    LEADER_ALPHA,
    HDV_ACCEL_MIN,
    HDV_ACCEL_MAX,
    HDV_SPEED_MAX
)


# ============================================================
# General traffic functions
# ============================================================
def calculate_gap(leader_position, follower_position):
    """
    Calculate the bumper-to-bumper gap between two vehicles.
    Positions represent the front bumper of each vehicle.
    """
    gap = leader_position - follower_position - VEHICLE_LENGTH
    return gap


def enforce_gap_constraint(next_position, leader_next_position, min_gap):
    """
    Ensure that the next position of a vehicle does not violate
    the minimum gap constraint with respect to the leader's next position.
    """
    max_position = (leader_next_position - VEHICLE_LENGTH - min_gap)
    return min(next_position, max_position)


# ============================================================
# Common vehicle dynamics
# ============================================================
class VehicleDynamics:
    """
    Common kinematic dynamics shared by HDVs and AVs.
    """
    @staticmethod
    def update_velocity(current_speed, acceleration):
        """
        Calculate the vehicle speed at the next time step.
            v(t+1) = v(t) + a(t) * DT
        """
        return current_speed + acceleration * DT

    @staticmethod
    def update_position(current_position, current_speed):
        """
        Calculate the vehicle position at the next time step.
            x(k+1) = x(k) + v(k) * DT
        """
        return current_position + current_speed * DT

   
# ============================================================
# HDV dynamics
# ============================================================
class HDVDynamics(VehicleDynamics):
    """
    Human-driven vehicle dynamics.
    HDVs determine their acceleration using the car-following model.
    """
    @staticmethod
    def calculate_acceleration(follower_speed, leader_speed, gap):
        """
        Calculate HDV acceleration using the car-following model.
            a(k) = alpha * (v_lead - v) + beta * (gap - desired_gap)
        """
        raw_acceleration = ALPHA * (leader_speed - follower_speed) + BETA * (gap - DESIRED_GAP)
        acceleration = max(HDV_ACCEL_MIN, min(raw_acceleration, HDV_ACCEL_MAX))
        return acceleration

    @staticmethod
    def calculate_leader_acceleration(current_speed, desired_speed):
        """
        Calculate leader acceleration when the leader moves toward a desired speed.
        """
        raw_acceleration = LEADER_ALPHA * (desired_speed - current_speed)
        acceleration = max(HDV_ACCEL_MIN, min(raw_acceleration, HDV_ACCEL_MAX))
        return acceleration

    @staticmethod
    def update_velocity(current_speed, current_acceleration):
        """
        Update HDV speed while respecting the HDV speed limit.
        """
        next_speed = VehicleDynamics.update_velocity(current_speed, current_acceleration)
        next_speed = max(0.0, min(next_speed, HDV_SPEED_MAX))
        return next_speed

    @staticmethod
    def update_position(current_position, current_speed, leader_next_position, min_gap):
        """
        Update an HDV position while enforcing the minimum gap to its leader.
        """
        candidate_position = VehicleDynamics.update_position(current_position, current_speed)
        safe_position = enforce_gap_constraint(candidate_position, leader_next_position, min_gap)
        
        if safe_position < candidate_position:
            next_speed = max(0.0, (safe_position - current_position) / DT) # Make speed consistent with the constrained position
            next_position = safe_position
        else:
            next_position = candidate_position
            next_speed = None

        return next_position, next_speed

# ============================================================
# AV dynamics
# ============================================================
class AVDynamics(VehicleDynamics):
    """
    Autonomous vehicle kinematic prediction model. The acceleration is supplied by the MPC controller.
    """
    @staticmethod
    def predict_trajectory(current_position, current_speed, acceleration_sequence):
        """
        Predict the AV trajectory for a given acceleration sequence.
        """
        positions = []
        speeds = []

        for acceleration in acceleration_sequence:
            speed = VehicleDynamics.update_velocity(current_speed, acceleration)
            position = VehicleDynamics.update_position(current_position, current_speed)
            speeds.append(speed)
            positions.append(position)
            current_position = position
            current_speed = speed

        return positions, speeds

    