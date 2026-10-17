# Pedestrian-Aware MPC for Mixed Traffic

A Python simulation of mixed traffic in a single lane, with human-driven vehicles (HDVs) and autonomous vehicles (AVs) approaching a pedestrian crossing.

HDVs follow a car-following model with pedestrian-response rules. AVs use model predictive control (MPC) to choose their acceleration while considering traffic ahead, speed limits, vehicle gaps, and the pedestrian event.

The Jupyter notebook runs the simulation and visualizes vehicle positions, speeds, accelerations, gaps, pedestrian states, controller modes, and solver diagnostics.

## Scenario

The reference scenario uses:

| Setting | Value |
|---|---|
| Simulation duration | 60 s |
| Vehicles | 15 total: 11 HDVs and 4 AVs |
| Simulation time step | 0.1 s (10 Hz) |
| MPC prediction horizon | 30 steps (3 s) |
| Pedestrian crossing position | 400 m |
| Pedestrian warning period | 17–20 s |
| Pedestrian present | 20–30 s |
| Safe stopping position | 395 m |
| Vehicle length | 4.5 m |
| Minimum bumper-to-bumper gap | 3 m |
| Maximum vehicle speed | 25 m/s |
| AV desired speed | 20 m/s |
| AV desired time headway | 1.5 s |
| Acceleration limits | −8 to +3 m/s² |

Vehicle positions represent the front bumper. The vehicle in the first row after sorting by position must be an HDV because it acts as the traffic leader.

## How the Controller Works

### HDV Control

HDVs use a car-following model. Their acceleration responds to the speed difference between the leader and follower and to the difference between the current gap and a desired gap of 10 m.

Acceleration and speed are limited to the configured physical ranges.

During the pedestrian warning, an HDV checks whether it can stop before the safe stopping position. If its leader is already stopping for the pedestrian, the HDV aims to stop behind that leader.

The stopping decision is latched so the vehicle continues to brake toward its target. A gap-enforcement step also prevents HDVs from moving too close to their leaders.

### AV Model Predictive Control

At each simulation step, each AV solves an optimization problem over a 3-second prediction horizon using SLSQP from `scipy.optimize`.

The controller applies only the first acceleration from the selected sequence and then solves again at the next simulation step. The previous optimal sequence is used to warm-start the next optimization when available.

The objective balances:

- **Speed tracking:** Follow the desired speed during normal driving, or target zero speed while stopping.
- **Ride comfort:** Penalize changes in acceleration (jerk).
- **Following safety:** Penalize time headways below the 1.5 s target.
- **Stopping position:** During a stop, encourage the AV to stop near the selected target behind its leader or before the crossing.

The optimization includes:

- acceleration bounds,
- speed bounds,
- a minimum-gap constraint, and
- pedestrian-related position constraints.

The time-headway target is a soft penalty, not a hard constraint. Therefore, the simulated headway can fall below 1.5 s even when the minimum-gap constraint is satisfied.

### Pedestrian Handling

The pedestrian logic has three main stages:

1. **Before the warning:** Run normal MPC.
2. **During the 3-second warning:** Predict whether the AV can fully clear the crossing before the pedestrian arrives. If it cannot, solve a stopping problem instead.
3. **While the pedestrian is present:** Require the predicted AV position to remain at or before the safe stopping position until the crossing clears.

If an MPC solve fails, the controller records diagnostics and uses a fallback acceleration.

During stopping or dilemma states, the fallback is based on the stopping target and configured braking limits.

## Project Structure

```text
.
├── data/
│   └── assignment_initial_conditions.csv
│
├── notebooks/
│   └── simulation 10Hz.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── controller.py
│   ├── dynamics.py
│   ├── pedestrian.py
│   └── simulation.py
│
└── README.md
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy pandas scipy matplotlib jupyter
```

The project uses:

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Jupyter Notebook / JupyterLab

## Running the Simulation

1. Place the source modules under `src/`, the input CSV under `data/`, and the notebook under `notebooks/`.

2. Open a terminal at the project root and start Jupyter from the notebook directory:

```bash
cd notebooks
jupyter lab "simulation 10Hz.ipynb"
```

Starting Jupyter from the `notebooks/` directory allows the notebook's `Path.cwd().parent` to resolve to the project root.

3. Run the notebook cells in order.

The notebook:

- loads the initial vehicle configuration,
- runs the mixed-traffic simulation,
- prints summary diagnostics,
- plots vehicle positions,
- plots vehicle speeds,
- plots accelerations,
- plots vehicle gaps,
- shows the pedestrian state,
- shows controller modes,
- reports MPC solver diagnostics, and
- creates an animation of the traffic around the pedestrian crossing.

The main simulation entry point is:

```python
simulate(df)
```

It returns histories for:

- time,
- vehicle positions,
- vehicle speeds,
- accelerations,
- vehicle gaps,
- pedestrian state,
- pedestrian-related HDV events,
- controller modes,
- wall hits, and
- AV MPC diagnostics.


## Limitations

- The desired time headway is encouraged through the objective but is not enforced as a hard constraint.
- A failed MPC solve can trigger fallback braking, which may cause abrupt acceleration changes and reduce ride comfort.
- The fallback braking can produce high jerk during difficult pedestrian scenarios.
- Results depend on the initial vehicle arrangement and the selected model parameters.
- The current scenario is a simulation study and is not a safety certification.

## Future Work

Potential improvements include:

- reducing MPC solver failures,
- designing a smoother fallback controller,
- improving the trade-off between safety and ride comfort,
- testing different simulation time steps,
- evaluating the controller with multiple initial traffic configurations, and
- testing the controller under a wider range of pedestrian and traffic scenarios.

## Notes

The vehicle positions represent the front bumper. Vehicle gaps are calculated as the distance between the leader's rear bumper and the follower's front bumper.

The first vehicle is the traffic leader and must be an HDV in the current simulation structure.

The AV controller is designed to make a new control decision at every simulation step while using the prediction horizon to look ahead at the expected vehicle and pedestrian situation.
