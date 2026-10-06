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
)


def _compute_accelerations(positions, speeds, gaps, time, state, modes, pedestrian_acceleration_row):
    """
    Acceleration of every vehicle at one time step.
    Leader: leader model (constant / desired speed). Followers: car following.
    All vehicles: min(normal acceleration, pedestrian braking). Updates modes in place.
    Returns (accelerations, list of infeasible vehicle indices).
    """
    n_vehicles = len(positions)
    accelerations = np.zeros(n_vehicles)
    infeasible = []
    stop_targets = np.full(n_vehicles, X_STOP)
 
    for i in range(n_vehicles):
        if i == 0:
            if LEADER_MODE == "constant":
                normal_acceleration = 0.0
            elif LEADER_MODE == "desired_speed":
                normal_acceleration = HDVDynamics.calculate_leader_acceleration(speeds[0], LEADER_DESIRED_SPEED)
            else:
                raise ValueError(f"Unknown LEADER_MODE: {LEADER_MODE}")
        else:
            normal_acceleration = HDVDynamics.calculate_acceleration(speeds[i], speeds[i - 1], gaps[i])
 
        if i > 0:
            stop_targets[i] = calculate_queue_stop_target(modes[i - 1], stop_targets[i - 1])
 
        pedestrian_acceleration, modes[i], is_infeasible = calculate_hdv_pedestrian_acceleration(
            vehicle_position=positions[i],
            vehicle_speed=speeds[i],
            time=time,
            pedestrian_state=state,
            mode=modes[i],
            stop_target=stop_targets[i],
        )
        if is_infeasible:
            infeasible.append(i)
 
        if pedestrian_acceleration is not None:
            # Pedestrian braking overrides normal behaviour whenever it requires stronger braking
            accelerations[i] = min(normal_acceleration, pedestrian_acceleration)
            pedestrian_acceleration_row[i] = pedestrian_acceleration
        else:
            accelerations[i] = normal_acceleration
 
    return accelerations, infeasible


def simulate(df):
    """
    Simulate the all-HDV traffic scenario.
    Vehicle 0 is the leader (leader model). The others use HDV car following.
    Every vehicle behind the crossing treats the pedestrian as a virtual stopped obstacle
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

    # Store simulation history
    time_history = np.zeros(n_steps)
    position_history = np.zeros((n_steps, n_vehicles))
    speed_history = np.zeros((n_steps, n_vehicles))
    acceleration_history = np.zeros((n_steps, n_vehicles))
    gap_history = np.full((n_steps, n_vehicles), np.nan)

    # Pedestrian diagnostics
    pedestrian_state_history = np.empty(n_steps, dtype=object)
    pedestrian_acceleration_history = np.full((n_steps, n_vehicles), np.nan)
    mode_history = np.zeros((n_steps, n_vehicles), dtype=int)
    wall_hit_history = np.zeros((n_steps, n_vehicles), dtype=bool)
    infeasible_events = []     # (time, vehicle index, position, speed)
 
    modes = np.full(n_vehicles, MODE_FREE, dtype=int)

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
        positions, speeds, gap_history[0], 0.0, pedestrian_state_history[0], modes, pedestrian_acceleration_history[0]
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
        new_speeds = np.zeros(n_vehicles)
        new_positions = np.zeros(n_vehicles)

        # =================================================
        # 1. Calculate speed at k: v(k) = v(k-1) + a(k-1) * DT
        # =================================================
        for i in range(n_vehicles):
            new_speeds[i] = HDVDynamics.update_velocity(previous_speeds[i], previous_accelerations[i])

        # =================================================
        # 2. Calculate position at k: x(k) = x(k-1) + v(k-1) * DT
        # =================================================
        for i in range(n_vehicles):
            if i == 0:
                new_positions[0] = VehicleDynamics.update_position(previous_positions[0], previous_speeds[0])
            
            else:
                new_positions[i], constrained_speed = HDVDynamics.update_position(previous_positions[i], previous_speeds[i], new_positions[i - 1], MIN_GAP)

                # If the gap constraint changed the position, update the speed so that the position and speed remain consistent.
                if constrained_speed is not None:
                    new_speeds[i] = constrained_speed

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
        pedestrian_state_history[k] = get_pedestrian_state(current_time)
        acceleration_history[k], infeasible = _compute_accelerations(
            new_positions, new_speeds, gap_history[k], current_time,
            pedestrian_state_history[k], modes, pedestrian_acceleration_history[k]
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
        "modes": mode_history,
        "wall_hits": wall_hit_history,
        "infeasible_events": infeasible_events,
    }