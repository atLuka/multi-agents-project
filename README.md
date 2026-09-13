
## Project structure

- `E80/` — Unity project folder
- `sim.py` — core simulation model and logistics logic
- `Push.py` — script that streams simulation state to Unity
- `RoboArena_M4.ipynb` — notebook version of the original simulation logic and visual analysis
- `environment_preview.png` — static environment overview
- `agv_simulation.gif` and `agv_simulation_m4.gif` — animation examples

## Workflow

1. Open the `E80` folder in Unity Hub.
2. Start the Unity project.
3. Run the Python simulation bridge:

```bash
python Push.py
```

This starts the simulation from `sim.py`, which contains the same core logic originally developed in `RoboArena_M4.ipynb`.

## Requirements

Install the required Python dependencies before running the simulation:

```bash
pip install agentpy numpy matplotlib
```

If you are using a virtual environment, activate it first and then install the packages above.
