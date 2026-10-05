# Simulation
DT = 0.1                     # Simulation time step [s] (10 Hz simulation frequency)
SIM_TIME = 60.0              # Total simulation time [s]

# HDV model
ALPHA = 0.5                  # Speed adaptation parameter
BETA = 0.2                   # Gap adaptation parameter
DESIRED_GAP = 10.0           # [m]
# Assumptions
HDV_ACCEL_MIN = -3.0           # [m/s^2]
HDV_ACCEL_MAX = 2.0            # [m/s^2]
HDV_SPEED_MAX = 30.0           # [m/s]
MIN_GAP = 2.0                  # Minimum bumper-to-bumper gap [m]

# Vehicle geometry
VEHICLE_LENGTH = 4.5         # Assumed common vehicle length [m]

# position_m = front-bumper longitudinal position

# Pedestrian disturbance
PEDESTRIAN_TIME = 20.0       # Time at which the pedestrian appears [s]
PEDESTRIAN_POSITION = 400.0  # [m]
PEDESTRIAN_DURATION = 10.0   # [s]
PEDESTRIAN_WARNING_DURATION = 2.0   # Warning before pedestrian becomes active [s]

# Visualization
WINDOW_MARGIN = 50.0     # Margin around traffic for plots [m]

# Leader HDV behavior
LEADER_MODE = "desired_speed"   # "constant" or "desired_speed"
LEADER_DESIRED_SPEED = 20.0  # m/s
LEADER_ALPHA = 0.5                # Speed adaptation

# AV physical limits
AV_ACCEL_MIN = -3.0       # m/s²
AV_ACCEL_MAX = 2.0        # m/s²
AV_SPEED_MAX = 30.0       # m/s
AV_DESIRED_SPEED = 20.0   # m/s

# AV MPC
MPC_HORIZON = 30      # Prediction horizon (3 seconds)
AV_HEADWAY_TARGET = 1.5       # Desired time headway [s]
# MPC objective weights
AV_SPEED_WEIGHT = 1.0 
AV_SMOOTHNESS_WEIGHT = 0.01
AV_HEADWAY_WEIGHT = 20.0   
AV_EPSILON = 1e-3    