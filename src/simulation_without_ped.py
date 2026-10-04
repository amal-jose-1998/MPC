import numpy as np
import pandas as pd

from .config import DT, SIM_TIME, LEADER_MODE, LEADER_DESIRED_SPEED, MIN_GAP
from .dynamics import calculate_gap, VehicleDynamics, HDVDynamics
from .controller import compute_av_acceleration


def simulate(df):
    """
    Simulate mixed traffic with HDV car-following and AV MPC control.
    """
    vehicles = df.copy().reset_index(drop=True)
    n_vehicles = len(vehicles)
    n_steps = int(SIM_TIME / DT) + 1

    # =================================================
    # Initial states
    # =================================================
    positions = vehicles["position_m"].to_numpy(dtype=float)
    speeds = vehicles["speed_mps"].to_numpy(dtype=float)
    vehicle_types = vehicles["type"].to_numpy()

    # One warm-start sequence per AV.
    previous_optimal_sequences = {i: None for i in range(n_vehicles) if vehicle_types[i] == "AV"}

    # Store simulation history
    time_history = np.zeros(n_steps)
    position_history = np.zeros((n_steps, n_vehicles))
    speed_history = np.zeros((n_steps, n_vehicles))
    acceleration_history = np.zeros((n_steps, n_vehicles))
    gap_history = np.full((n_steps, n_vehicles), np.nan)

    # =================================================
    # k = 0
    # =================================================
    position_history[0] = positions
    speed_history[0] = speeds
    
    # Initial gaps
    for i in range(1, n_vehicles):
        gap_history[0, i] = calculate_gap(positions[i - 1], positions[i])
    
    # Initial accelerations
    # Leader
    if LEADER_MODE == "constant":
        acceleration_history[0, 0] = 0.0

    elif LEADER_MODE == "desired_speed":
        acceleration_history[0, 0] = HDVDynamics.calculate_leader_acceleration(speeds[0], LEADER_DESIRED_SPEED)

    else:
        raise ValueError(f"Unknown LEADER_MODE: {LEADER_MODE}")

    # Following vehicles
    for i in range(1, n_vehicles):
        if vehicle_types[i] == "AV":
            acceleration, optimal_sequence = compute_av_acceleration(
                current_position=positions[i],
                current_speed=speeds[i],
                leader_position=positions[i - 1],
                leader_speed=speeds[i - 1],
                leader_acceleration=acceleration_history[0, i - 1],
                previous_acceleration=0.0,
                previous_optimal_sequence=None,
            )
            acceleration_history[0, i] = acceleration
            previous_optimal_sequences[i] = optimal_sequence
        else:
            acceleration_history[0, i] = HDVDynamics.calculate_acceleration(speeds[i], speeds[i - 1], gap_history[0, i])

    # =================================================
    # Simulation loop: k = 1, 2, ..., n_steps-1
    # =================================================
    for k in range(1, n_steps):

        # Previous state: k - 1
        previous_positions = position_history[k - 1]
        previous_speeds = speed_history[k - 1]
        previous_accelerations = acceleration_history[k - 1]

        # New state: k
        new_speeds = np.zeros(n_vehicles)
        new_positions = np.zeros(n_vehicles)

        # =================================================
        # 1. State propagation: speed.  (v(k) = v(k-1) + a(k-1) * DT)
        # =================================================
        for i in range(n_vehicles):
            if vehicle_types[i] == "AV":
                # Pure AV dynamics: execute the acceleration selected by MPC.
                new_speeds[i] = VehicleDynamics.update_velocity(previous_speeds[i], previous_accelerations[i])
            else:
                new_speeds[i] = HDVDynamics.update_velocity(previous_speeds[i], previous_accelerations[i])

        # =================================================
        # 2. State propagation: position.  (x(k) = x(k-1) + v(k-1) * DT)
        # =================================================
        # Leader
        new_positions[0] = VehicleDynamics.update_position(previous_positions[0], previous_speeds[0])

        # Following vehicles
        for i in range(1, n_vehicles):
            if vehicle_types[i] == "AV":
                # No rule-based AV gap correction.
                new_positions[i] = VehicleDynamics.update_position(previous_positions[i], previous_speeds[i])
            else:
                new_positions[i], constrained_speed = HDVDynamics.update_position(previous_positions[i], previous_speeds[i], new_positions[i - 1], MIN_GAP)

                # If the gap constraint changed the position, update the speed so that the position and speed remain consistent.
                if constrained_speed is not None:
                    new_speeds[i] = constrained_speed

        # =================================================
        # 3. Calculate gaps at k for evaluation and HDV control: g(k) = x_leader(k) - x_follower(k) - L
        # =================================================
        for i in range(1, n_vehicles):
            gap_history[k, i] = calculate_gap(new_positions[i - 1], new_positions[i])

        # =================================================
        # 4. Calculate acceleration at k
        # =================================================
        # Leader
        if LEADER_MODE == "constant":
            acceleration_history[k, 0] = 0.0

        elif LEADER_MODE == "desired_speed":
            acceleration_history[k, 0] = HDVDynamics.calculate_leader_acceleration(new_speeds[0], LEADER_DESIRED_SPEED)

        else:
            raise ValueError(f"Unknown LEADER_MODE: {LEADER_MODE}")

        # Following vehicles
        for i in range(1, n_vehicles):
            if vehicle_types[i] == "AV":
                acceleration, optimal_sequence = compute_av_acceleration(
                    current_position=new_positions[i],
                    current_speed=new_speeds[i],
                    leader_position=new_positions[i - 1],
                    leader_speed=new_speeds[i - 1],
                    leader_acceleration=acceleration_history[k, i - 1],
                    previous_acceleration=previous_accelerations[i],
                    previous_optimal_sequence=previous_optimal_sequences[i],
                )
                acceleration_history[k, i] = acceleration
                previous_optimal_sequences[i] = optimal_sequence
            else:
                acceleration_history[k, i] = HDVDynamics.calculate_acceleration(new_speeds[i], new_speeds[i - 1], gap_history[k, i])

        # =================================================
        # 5. Store state at k
        # =================================================
        time_history[k] = k * DT
        position_history[k] = new_positions
        speed_history[k] = new_speeds

    return {
        "time": time_history,
        "positions": position_history,
        "speeds": speed_history,
        "accelerations": acceleration_history,
        "gaps": gap_history,
        "vehicle_types": vehicle_types,
    }