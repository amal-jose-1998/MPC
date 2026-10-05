import numpy as np
import pandas as pd

from .config import (
    DT,
    SIM_TIME,
    LEADER_MODE,
    LEADER_DESIRED_SPEED,
    MIN_GAP,
    PEDESTRIAN_POSITION,
)
from .dynamics import calculate_gap, VehicleDynamics, HDVDynamics
from .controller import compute_av_acceleration
from .pedestrian import (
    get_pedestrian_state,
    calculate_hdv_pedestrian_acceleration,
)



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

    pedestrian_state_history = np.empty(n_steps, dtype=object)             # Pedestrian state at every simulation timestep.

    # MPC diagnostics for every timestep and vehicle.
    mpc_diagnostics = [[None for _ in range(n_vehicles)] for _ in range(n_steps)]

    # =================================================
    # k = 0
    # =================================================
    time_history[0] = 0.0
    position_history[0] = positions
    speed_history[0] = speeds
    pedestrian_state_history[0] = get_pedestrian_state(time_history[0])
    
    # Initial gaps
    for i in range(1, n_vehicles):
        gap_history[0, i] = calculate_gap(positions[i - 1], positions[i])
    
    # Initial accelerations
    # Leader
    if LEADER_MODE == "constant":
        acceleration_history[0, 0] = 0.0

    elif LEADER_MODE == "desired_speed":
        leader_acceleration = HDVDynamics.calculate_leader_acceleration(speeds[0], LEADER_DESIRED_SPEED)
        # The front-most vehicle has no leader. Therefore, if the pedestrian is ahead of it, it must react directly. 
        pedestrian_acceleration = (
            calculate_hdv_pedestrian_acceleration(
                vehicle_position=positions[0],
                vehicle_speed=speeds[0],
                pedestrian_state=pedestrian_state_history[0],
                pedestrian_position=PEDESTRIAN_POSITION,
                leader_position=None,
            )
        )
        if pedestrian_acceleration is not None:
            leader_acceleration = min(leader_acceleration, pedestrian_acceleration)

        acceleration_history[0, 0] = leader_acceleration

    else:
        raise ValueError(f"Unknown LEADER_MODE: {LEADER_MODE}")

    # Following vehicles
    for i in range(1, n_vehicles):
        if vehicle_types[i] == "AV":
            acceleration, optimal_sequence, diagnostics = compute_av_acceleration(
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
            mpc_diagnostics[0][i] = diagnostics
        else:
            # Normal HDV car-following acceleration
            hdv_acceleration = HDVDynamics.calculate_acceleration(speeds[i], speeds[i - 1], gap_history[0, i])
            # Pedestrian response for a following HDV. The pedestrian must be between this HDV and its leader. 
            pedestrian_acceleration = calculate_hdv_pedestrian_acceleration(
                vehicle_position=positions[i],
                vehicle_speed=speeds[i],
                pedestrian_state=pedestrian_state_history[0],
                pedestrian_position=PEDESTRIAN_POSITION,
                leader_position=positions[i - 1],
            )

            if pedestrian_acceleration is not None:
                hdv_acceleration = min(hdv_acceleration, pedestrian_acceleration)

            acceleration_history[0, i] = hdv_acceleration

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
                # HDV retains its existing gap-enforcing dynamics.
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
        time_history[k] = k * DT # Current simulation time.
        pedestrian_state_history[k] = get_pedestrian_state(time_history[k]) # Current pedestrian state.

        # Leader
        if LEADER_MODE == "constant":
            acceleration_history[k, 0] = 0.0

        elif LEADER_MODE == "desired_speed":
            leader_acceleration = HDVDynamics.calculate_leader_acceleration(new_speeds[0], LEADER_DESIRED_SPEED)
            # The leader has no vehicle ahead of it. Therefore, the pedestrian can directly affect it when the pedestrian is ahead of the leader.
            pedestrian_acceleration = (
                calculate_hdv_pedestrian_acceleration(
                    vehicle_position=new_positions[0],
                    vehicle_speed=new_speeds[0],
                    pedestrian_state=pedestrian_state_history[k],
                    pedestrian_position=PEDESTRIAN_POSITION,
                    leader_position=None,
                )
            )

            if pedestrian_acceleration is not None:
                leader_acceleration = min(leader_acceleration, pedestrian_acceleration)

            acceleration_history[k, 0] = leader_acceleration


        else:
            raise ValueError(f"Unknown LEADER_MODE: {LEADER_MODE}")

        # Following vehicles
        for i in range(1, n_vehicles):
            if vehicle_types[i] == "AV":
                acceleration, optimal_sequence, diagnostics = compute_av_acceleration(
                    current_position=new_positions[i],
                    current_speed=new_speeds[i],
                    leader_position=new_positions[i - 1],
                    leader_speed=new_speeds[i - 1],
                    leader_acceleration=acceleration_history[k, i - 1],
                    previous_acceleration=previous_accelerations[i],
                    previous_optimal_sequence=previous_optimal_sequences[i],
                )
                acceleration_history[k, i] = acceleration
                mpc_diagnostics[k][i] = diagnostics
                previous_optimal_sequences[i] = optimal_sequence
            else:
                # Normal HDV car-following acceleration
                hdv_acceleration = HDVDynamics.calculate_acceleration(new_speeds[i], new_speeds[i - 1], gap_history[k, i])
                # Additional pedestrian response
                pedestrian_acceleration = calculate_hdv_pedestrian_acceleration(
                    vehicle_position=new_positions[i],
                    vehicle_speed=new_speeds[i],
                    pedestrian_state=pedestrian_state_history[k],
                    pedestrian_position=PEDESTRIAN_POSITION,
                    leader_position=new_positions[i - 1],
                )

                if pedestrian_acceleration is not None:
                    hdv_acceleration = min(hdv_acceleration, pedestrian_acceleration)

                acceleration_history[k, i] = hdv_acceleration


        # =================================================
        # 5. Store state at k
        # =================================================
        position_history[k] = new_positions
        speed_history[k] = new_speeds

    return {
        "time": time_history,
        "positions": position_history,
        "speeds": speed_history,
        "accelerations": acceleration_history,
        "gaps": gap_history,
        "vehicle_types": vehicle_types,
        "mpc_diagnostics": mpc_diagnostics,
        "pedestrian_state": pedestrian_state_history,
        "pedestrian_position": PEDESTRIAN_POSITION,
    }


def trace_av_failures(results, vehicle_index=10):
    """Print the first few MPC failures for a selected AV."""
    diagnostics_history = results["mpc_diagnostics"] 
    positions = results["positions"]
    speeds = results["speeds"]
    accelerations = results["accelerations"]
    gaps = results["gaps"]
    vehicle_types = results["vehicle_types"]
    time = results["time"]

    print(f"\n=== MPC failure trace: vehicle {vehicle_index} ({vehicle_types[vehicle_index]}) ===")

    failure_count = 0

    for k, row in enumerate(diagnostics_history):
        d = row[vehicle_index]

        if d is None or d["success"]:
            continue

        print(f"\nFailure #{failure_count + 1} at t = {time[k]:.1f} s")
        print(f"  current position       : {positions[k, vehicle_index]:.3f} m")
        print(f"  current speed          : {speeds[k, vehicle_index]:.3f} m/s")
        print(f"  current acceleration   : {accelerations[k, vehicle_index]:.3f} m/s²")
        print(f"  current gap            : {gaps[k, vehicle_index]:.3f} m")
        print(f"  objective              : {d['objective']:.6f}")
        print(f"  status                 : {d['status']}")
        print(f"  message                : {d['message']}")
        print(f"  iterations             : {d['iterations']}")
        print(f"  min speed margin       : {d['min_speed_margin']:.6f}")
        print(f"  min gap margin         : {d['min_gap_margin']:.6f}")
        print(f"  min constraint margin  : {d['min_constraint_margin']:.6f}")
        print(f"  warm-start guess       : {np.asarray(d['initial_guess'])}")

        if "fallback_acceleration" in d:
            print(f"  fallback acceleration  : {d['fallback_acceleration']:.3f} m/s²")

        failure_count += 1

        if failure_count >= 5:
            print("\n... failures more than 5, stopping ...")
            break

    if failure_count == 0:
        print("No MPC failures found.")

    return failure_count