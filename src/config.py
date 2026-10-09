# Simulation
DT = 0.04                    # Simulation time step [s] (25 Hz simulation frequency)
SIM_TIME = 60.0              # Total simulation time [s]

# HDV model
ALPHA = 0.5                  # Speed adaptation parameter
BETA = 0.2                   # Gap adaptation parameter
DESIRED_GAP = 10.0           # [m] Desired gap between vehicles
# Assumptions
HDV_ACCEL_MIN = -8.0           # [m/s^2]
HDV_ACCEL_MAX = 3.0            # [m/s^2]
HDV_SPEED_MAX = 25.0           # [m/s]
MIN_GAP = 3.0                  # Minimum bumper-to-bumper gap [m]

# Vehicle geometry
VEHICLE_LENGTH = 4.5         # Assumed common vehicle length [m]

# position_m = front-bumper longitudinal position

# ============================================================
# Pedestrian disturbance
# ============================================================
PEDESTRIAN_TIME = 20.0       # Time at which the pedestrian appears [s]
PEDESTRIAN_POSITION = 400.0  # [m]
PEDESTRIAN_DURATION = 10.0   # [s]
PEDESTRIAN_WARNING_DURATION = 3.0   # Warning before pedestrian becomes active [s]

# ============================================================
# HDV pedestrian response
# ============================================================
PEDESTRIAN_HDV_WARNING_BRAKE = -2.0    # [m/s²] it is assumed that the HDV brakes gently when it receives a warning about the pedestrian, but not as hard as when the pedestrian is actually present.
PEDESTRIAN_HDV_ACTIVE_BRAKE = -8.0     # [m/s²]  it is assumed that the HDV can brake harder when the pedestrian is actually present than when it is only a warning.
PEDESTRIAN_HDV_REACTION_TIME = 0     # [s] Driver reaction time is the time it takes for the driver to react to the pedestrian warning and start braking.
PEDESTRIAN_HDV_SAFE_DISTANCE = 5.0     # [m] Extra distance maintained before crossing
PEDESTRIAN_HDV_COMFORT_BRAKE = 2.0     # [m/s^2] A driver starts braking once the required deceleration reaches this value
PEDESTRIAN_QUEUE_STANDSTILL_GAP = 3.0  # [m] Bumper-to-bumper gap between stopped vehicles in the queue at the crossing
PEDESTRIAN_REQUIRE_FULL_CLEARANCE = True  

# ============================================================
# Visualization
# ============================================================
WINDOW_MARGIN = 50.0     # Margin around traffic for plots [m]

# ============================================================
# Leader HDV behavior
# ============================================================
LEADER_MODE = "desired_speed"   # "constant" or "desired_speed"
LEADER_DESIRED_SPEED = 25.0  # m/s
LEADER_ALPHA = 0.5                # Speed adaptation

# ============================================================
# AV physical limits
# ============================================================
AV_ACCEL_MIN = -8.0       # m/s²
AV_ACCEL_MAX = 3.0        # m/s²
AV_SPEED_MAX = 25.0       # m/s
AV_DESIRED_SPEED = 20.0   # m/s
AV_SAFE_BRAKE = 2.0       # m/s² 

# ============================================================
# AV MPC
# ============================================================
MPC_HORIZON = 75      # Prediction horizon (3 seconds at 25 Hz)
AV_HEADWAY_TARGET = 1.5       # Desired time headway [s]
# MPC objective weights
AV_SPEED_WEIGHT = 1.0 
AV_SMOOTHNESS_WEIGHT = 0.01
AV_HEADWAY_WEIGHT = 20.0   
AV_EPSILON = 1e-3    
AV_STOP_POSITION_WEIGHT = 2