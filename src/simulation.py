"""
Mixed-traffic simulation with a pedestrian crossing.
Scenario:
    - Vehicle 0 is the leader and is assumed to be an HDV.
    - Other vehicles can be HDVs or AVs according to the input dataframe.
    - HDVs use the existing car-following model plus the existing pedestrian braking/mode logic.
    - AVs use the pedestrian-aware MPC controller.
    - HDVs are subject to the external pedestrian safety wall.
"""


import numpy as np

from .config import (
    DT,
    PEDESTRIAN_POSITION,
    SIM_TIME,
    LEADER_MODE,
    LEADER_DESIRED_SPEED,
    MIN_GAP,
)

from .dynamics import (
    calculate_gap,
    VehicleDynamics,
    HDVDynamics
)
from .pedestrian import (
    get_pedestrian_state,
    calculate_hdv_pedestrian_acceleration,
    apply_pedestrian_wall,
    calculate_queue_stop_target,
    X_STOP,
    MODE_FREE,
    MODE_STOPPING,
    MODE_COMMITTED,
    VEHICLE_LENGTH,
    PEDESTRIAN_QUEUE_STANDSTILL_GAP,
)
from .controller import compute_av_acceleration


def _compute_leader_acceleration(speed):
    """Compute the normal acceleration of the HDV leader."""
    if LEADER_MODE == "constant":
        return 0.0

    if LEADER_MODE == "desired_speed":
        return HDVDynamics.calculate_leader_acceleration(
            speed,
            LEADER_DESIRED_SPEED,
        )

    raise ValueError(f"Unknown LEADER_MODE: {LEADER_MODE}")


def _av_mode_to_numeric(diagnostics):
    """
    Convert the AV controller's pedestrian mode to the same numeric labels used by the HDV pedestrian module.
    This is only used so that downstream HDVs can recognize an AV that is stopping for the pedestrian and form a queue behind it.
    """
    if diagnostics is None:
        return MODE_FREE

    mode = diagnostics.get("pedestrian_mode", "normal")

    if mode in ("stopping", "dilemma"):
        return MODE_STOPPING

    if mode == "committed":
        return MODE_COMMITTED

    return MODE_FREE


def _leader_stop_target(leader_index, vehicle_types, positions, modes, mpc_diagnostics_row, stop_targets):
    """
    Determine where an HDV should aim to stop relative to its leader.
    If the leader is an HDV that is stopping, use the existing queue rule.
    If the leader is an AV and its MPC is stopping/dilemma, also create a target behind the AV. 
    Otherwise the pedestrian stop target is X_STOP.
    """
    if leader_index < 0:
        return X_STOP

    if vehicle_types[leader_index] == "AV":
        leader_mode = _av_mode_to_numeric(mpc_diagnostics_row[leader_index])

        if leader_mode == MODE_STOPPING:
            leader_position = positions[leader_index]
            return min(X_STOP, leader_position - VEHICLE_LENGTH - PEDESTRIAN_QUEUE_STANDSTILL_GAP)

        return X_STOP

    if modes[leader_index] == MODE_STOPPING:
        return calculate_queue_stop_target(modes[leader_index], stop_targets[leader_index])

    return X_STOP


def _compute_accelerations(positions, speeds, gaps, current_time, pedestrian_state, vehicle_types, modes, av_pedestrian_modes,
                           previous_accelerations, previous_optimal_sequences, pedestrian_acceleration_row, mpc_diagnostics_row,):
    """
    HDV:
        normal car-following/leader dynamics, combined with pedestrian braking.
    AV:
        pedestrian-aware MPC controller.
    Returns:
        accelerations, infeasible_vehicle_indices
    """
    n_vehicles = len(positions)
    accelerations = np.zeros(n_vehicles)
    infeasible = []
    stop_targets = np.full(n_vehicles, X_STOP) # Default pedestrian stopping target for every vehicle.
 
    for i in range(n_vehicles):
        # -------------------------------------------------
        # 1. Normal vehicle control
        # -------------------------------------------------
        if i == 0:
            normal_acceleration = _compute_leader_acceleration(speeds[i])
 
        elif vehicle_types[i] == "AV":
            # AV control is handled by the MPC controller. It receives the current leader state, including the leader's current acceleration.
            acceleration, optimal_sequence, diagnostics = compute_av_acceleration(
                current_position=positions[i],
                current_speed=speeds[i],
                leader_position=positions[i - 1],
                leader_speed=speeds[i - 1],
                leader_acceleration=accelerations[i - 1],
                previous_acceleration=previous_accelerations[i],
                previous_optimal_sequence=previous_optimal_sequences.get(i),
                current_time=current_time,
                previous_pedestrian_mode=av_pedestrian_modes[i],
            )
            accelerations[i] = acceleration
            previous_optimal_sequences[i] = optimal_sequence
            mpc_diagnostics_row[i] = diagnostics
            av_pedestrian_modes[i] = diagnostics["pedestrian_mode"]
            continue

        else:
            # HDV follower.
            normal_acceleration = HDVDynamics.calculate_acceleration(speeds[i], speeds[i - 1], gaps[i])

        # -------------------------------------------------
        # 2. Pedestrian response for HDVs
        # -------------------------------------------------
        if i == 0:
            stop_targets[i] = X_STOP
        else:
            stop_targets[i] = _leader_stop_target(
                leader_index=i - 1,
                vehicle_types=vehicle_types,
                positions=positions,
                modes=modes,
                mpc_diagnostics_row=mpc_diagnostics_row,
                stop_targets=stop_targets,
            )

        pedestrian_acceleration, new_mode, is_infeasible = (
            calculate_hdv_pedestrian_acceleration(
                vehicle_position=positions[i],
                vehicle_speed=speeds[i],
                time=current_time,
                pedestrian_state=pedestrian_state,
                mode=modes[i],
                stop_target=stop_targets[i],
            )
        )

        modes[i] = new_mode

        if is_infeasible:
            infeasible.append(i)

        if pedestrian_acceleration is not None:
            # Pedestrian braking overrides normal behavior only when it asks
            # for stronger braking.
            accelerations[i] = min(
                normal_acceleration,
                pedestrian_acceleration,
            )
            pedestrian_acceleration_row[i] = pedestrian_acceleration
        else:
            accelerations[i] = normal_acceleration

    return accelerations, infeasible


def simulate(df):
    """Simulate mixed traffic with the pedestrian crossing."""
    vehicles = df.copy().reset_index(drop=True)
    required_columns = {"position_m", "speed_mps", "type"}
    missing = required_columns.difference(vehicles.columns)

    if missing:
        raise ValueError(
            f"Missing required dataframe columns: {sorted(missing)}"
        )

    if len(vehicles) == 0:
        raise ValueError("The vehicle dataframe is empty.")

    vehicle_types = vehicles["type"].astype(str).to_numpy()

    if vehicle_types[0] != "HDV":
        raise ValueError(
            "Vehicle 0 is the traffic leader and must be an HDV in the "
            "current simulation structure."
        )

    n_vehicles = len(vehicles)
    n_steps = int(SIM_TIME / DT) + 1

    # =================================================
    # Initial states
    # =================================================
    positions = vehicles["position_m"].to_numpy(dtype=float)
    speeds = vehicles["speed_mps"].to_numpy(dtype=float)
    if np.any(speeds < 0.0):
        raise ValueError("Initial vehicle speeds must be non-negative.")

    # One warm-start sequence per AV.
    previous_optimal_sequences = {i: None for i in range(n_vehicles) if vehicle_types[i] == "AV"}

    # Store simulation history
    time_history = np.zeros(n_steps)
    position_history = np.zeros((n_steps, n_vehicles))
    speed_history = np.zeros((n_steps, n_vehicles))
    acceleration_history = np.zeros((n_steps, n_vehicles))
    gap_history = np.full((n_steps, n_vehicles), np.nan)

    # Pedestrian diagnostics
    pedestrian_state_history = np.empty(n_steps, dtype=object)
    pedestrian_acceleration_history = np.full((n_steps, n_vehicles), np.nan, dtype=float)
    mode_history = np.zeros((n_steps, n_vehicles), dtype=int)
    wall_hit_history = np.zeros((n_steps, n_vehicles), dtype=bool)
    infeasible_events = []     # (time, vehicle index, position, speed)

    # MPC diagnostics for every AV and time step.
    mpc_diagnostics = [
        [None for _ in range(n_vehicles)]
        for _ in range(n_steps)
    ]
 
    modes = np.full(n_vehicles, MODE_FREE, dtype=int) # Current pedestrian mode state for HDVs.
    # Persistent pedestrian mode state for AVs.
    # Values: "normal", "committed", "stopping", "dilemma"
    av_pedestrian_modes = {i: "normal" for i in range(n_vehicles) if vehicle_types[i] == "AV"}

    # =================================================
    # k = 0
    # =================================================
    position_history[0] = positions
    speed_history[0] = speeds
    pedestrian_state_history[0] = get_pedestrian_state(0.0)
    
    # Initial gaps
    for i in range(1, n_vehicles):
        gap_history[0, i] = calculate_gap(positions[i - 1], positions[i])

    
    # Initial accelerations
    acceleration_history[0], infeasible = _compute_accelerations(
        positions=positions,
        speeds=speeds,
        gaps=gap_history[0],
        current_time=0.0,
        pedestrian_state=pedestrian_state_history[0],
        vehicle_types=vehicle_types,
        modes=modes,
        av_pedestrian_modes=av_pedestrian_modes,
        previous_accelerations=np.zeros(n_vehicles, dtype=float),
        previous_optimal_sequences=previous_optimal_sequences,
        pedestrian_acceleration_row=pedestrian_acceleration_history[0],
        mpc_diagnostics_row=mpc_diagnostics[0],
    )

    mode_history[0] = modes
    for i in infeasible:
        infeasible_events.append((0.0, i, positions[i], speeds[i]))

    # =================================================
    # Simulation loop: k = 1, 2, ..., n_steps-1
    # =================================================
    for k in range(1, n_steps):
        # Previous state: k - 1
        current_time = k * DT
        previous_positions = position_history[k - 1]
        previous_speeds = speed_history[k - 1]
        previous_accelerations = acceleration_history[k - 1]

        # New state: k
        new_speeds = np.zeros(n_vehicles, dtype=float)
        new_positions = np.zeros(n_vehicles, dtype=float)

        # =================================================
        # 1. Calculate speed at k: v(k) = v(k-1) + a(k-1) * DT
        # =================================================
        for i in range(n_vehicles):
            if vehicle_types[i] == "AV":
                # Pure AV dynamics: execute the acceleration selected by MPC.
                new_speeds[i] = VehicleDynamics.update_velocity(previous_speeds[i], previous_accelerations[i])
            else:
                new_speeds[i] = HDVDynamics.update_velocity(previous_speeds[i], previous_accelerations[i])

        # =================================================
        # 2. Calculate position at k: x(k) = x(k-1) + v(k-1) * DT
        # =================================================
        for i in range(n_vehicles):
            if i == 0: # Leader
                new_positions[0] = VehicleDynamics.update_position(previous_positions[0], previous_speeds[0])

            elif vehicle_types[i] == "AV":
                # AV position is produced by the MPC-selected acceleration. No separate rule-based gap correction is added here.
                new_positions[i] = VehicleDynamics.update_position(previous_positions[i], previous_speeds[i])
            
            else:
                # HDV position with minimum-gap enforcement.
                new_positions[i], constrained_speed = HDVDynamics.update_position(previous_positions[i], previous_speeds[i], new_positions[i - 1], MIN_GAP)

                # If the gap constraint changed the position, update the speed so that the position and speed remain consistent.
                if constrained_speed is not None:
                    new_speeds[i] = constrained_speed

            if vehicle_types[i] != "AV":
                # Apply the pedestrian wall to HDVs only. The wall is only active when the pedestrian is present.
                new_positions[i], wall_speed, wall_hit = apply_pedestrian_wall(new_positions[i], previous_positions[i], current_time)
                if wall_hit:
                    new_speeds[i] = wall_speed
                    wall_hit_history[k, i] = True

        # =================================================
        # 3. Calculate gaps at k: g(k) = x_leader(k) - x_follower(k) - L
        # =================================================
        for i in range(1, n_vehicles):
            gap_history[k, i] = calculate_gap(new_positions[i - 1], new_positions[i])

        # =================================================
        # 4. Calculate acceleration at k
        # =================================================
        pedestrian_state = get_pedestrian_state(current_time)
        pedestrian_state_history[k] = pedestrian_state
        acceleration_history[k], infeasible = _compute_accelerations(
            positions=new_positions,
            speeds=new_speeds,
            gaps=gap_history[k],
            current_time=current_time,
            pedestrian_state=pedestrian_state,
            vehicle_types=vehicle_types,
            modes=modes,
            av_pedestrian_modes=av_pedestrian_modes,
            previous_accelerations=previous_accelerations,
            previous_optimal_sequences=previous_optimal_sequences,
            pedestrian_acceleration_row=pedestrian_acceleration_history[k],
            mpc_diagnostics_row=mpc_diagnostics[k],
        )
        mode_history[k] = modes
        for i in infeasible:
            infeasible_events.append((current_time, i, new_positions[i], new_speeds[i]))

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
        "pedestrian_state": pedestrian_state_history,
        "pedestrian_accelerations": pedestrian_acceleration_history,
        "pedestrian_position": PEDESTRIAN_POSITION,
        "pedestrian_stop_position": X_STOP,
        "modes": mode_history,
        "wall_hits": wall_hit_history,
        "infeasible_events": infeasible_events,
        "mpc_diagnostics": mpc_diagnostics,
    }


def trace_av_failures(results, vehicle_index):
    """
    Print the first few failed MPC solves for one AV.
    """
    diagnostics_history = results["mpc_diagnostics"]
    positions = results["positions"]
    speeds = results["speeds"]
    accelerations = results["accelerations"]
    gaps = results["gaps"]
    vehicle_types = results["vehicle_types"]
    time = results["time"]

    if vehicle_index < 0 or vehicle_index >= len(vehicle_types):
        raise IndexError("vehicle_index is outside the simulation range.")

    if vehicle_types[vehicle_index] != "AV":
        raise ValueError(
            f"Vehicle {vehicle_index} is {vehicle_types[vehicle_index]}, not an AV."
        )

    print(
        f"\n=== MPC failure trace: vehicle "
        f"{vehicle_index} ({vehicle_types[vehicle_index]}) ==="
    )

    failure_count = 0

    for k, row in enumerate(diagnostics_history):
        diagnostics = row[vehicle_index]

        if diagnostics is None or diagnostics["success"]:
            continue

        print(
            f"\nFailure #{failure_count + 1} at t = "
            f"{time[k]:.1f} s"
        )
        print(
            f"  position        : {positions[k, vehicle_index]:.3f} m"
        )
        print(
            f"  speed           : {speeds[k, vehicle_index]:.3f} m/s"
        )
        print(
            f"  acceleration    : "
            f"{accelerations[k, vehicle_index]:.3f} m/s²"
        )
        print(
            f"  gap             : "
            f"{gaps[k, vehicle_index]:.3f} m"
        )
        print(
            f"  pedestrian mode : "
            f"{diagnostics.get('pedestrian_mode')}"
        )
        print(
            f"  objective       : "
            f"{diagnostics['objective']:.6f}"
        )
        print(
            f"  status          : "
            f"{diagnostics['status']}"
        )
        print(
            f"  message         : "
            f"{diagnostics['message']}"
        )

        if "fallback_acceleration" in diagnostics:
            print(
                f"  fallback accel  : "
                f"{diagnostics['fallback_acceleration']:.3f} m/s²"
            )

        failure_count += 1

        if failure_count >= 5:
            print("\n... more than 5 failures; stopping trace ...")
            break

    if failure_count == 0:
        print("No MPC failures found.")

    return failure_count
