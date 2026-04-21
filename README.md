# openTorsion-Based Torsional Analysis System Optimizer with SysML Integration

This repository provides a multi-objective, mixed-variable genetic-algorithm optimizer built on the [openTorsion](https://github.com/Aalto-Arotor/openTorsion) library. It can ingest SysML-derived system descriptions (JSON), extract all available design variables, discretize the system into an openTorsion-compatible model, and evaluate it using a custom objective function.

The system JSON input must follow specific structural rules so the parser can correctly interpret and discretize the model. See `models/base_model.json` for a fully worked example.

## JSON Structure Requirements

1. The root object must be a dictionary containing a `"components"` list.
2. Each entry in `"components"` must be a component dictionary.
3. Each component must define a `"type"` field. Valid types include: Actuator, Rotor, Coupling, Disk, Shaft, Gear, Gear_Set.
4. Each component must contain a `"parameters"` dictionary. Each parameter is a dictionary with `"value"` and `"units"` and/or `"options"` with discrete values (list) or a range (dict with `"min"` and/or `"max`").
5. Components may define design-choice variants in a `"choices"` list. Each entry contains a `"parameters"` dictionary and optionally a `"name"`.
6. Speed and excitation may also be defined. Speed may be a fixed number, a list, or a range dictionary. Only one component may define speed. Excitation must be a list of excitations, each `[order, amplitude, phase]`.

## Discretization Rules

- Stiffnesses and damping map to shaft elements while Inertias map to disk elements.

1. **Actuator / Rotor**: Expand into one of several disk/shaft patterns depending on neighbors.
2. **Coupling**: Disk + Shaft + Disk. Inertia is split if only one is provided.
3. **Gear Set**: Disk + Gear + Gear + Disk.
4. **Gear**: Disk + Gear.
5. **Shaft / Disk**: Added directly per openTorsion rules.

## Code Overview

1. `adapter.py`: Reads JSON, discretizes, extracts design variables, and writes optimized values back.
2. `optimizer.py`: GA engine. Stores history and final solutions and can merge results back into the original JSON.
3. `objective.py`: The default objective function. Computes maximum vibratory torque and total inertia; the optimizer minimizes this.

## Installation

1. Create and activate a Python virtual environment:
```python
uv venv # OR        
python -m venv .venv 
source .venv/Scripts/activate 
```

2. Install dependencies:

```python
uv sync # OR
pip install -r requirements.txt 
```

## Usage

1. Create a `SystemAdapter` instance with JSON path or dictionary.
2. Create an `Optimizer` instance and assign parser + objective.
3. Run with the `.run()` method.

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{malik2026sysmlv2opentorsion,
  title={Model-Based Design of Marine Powertrains: Integrating Coupling Selection with Torsional Analysis},
  author={Malik, Rohail and Al-Shami, Haitham and Ala-Laurinaho, Riku and Viitala, Raine and Veps{\"a}l{\"a}inen, Jari},
  booktitle={Proceedings of the 12th IFToMM International Conference on Rotordynamics},
  year={2026},
  address={Lahti, Finland},
  month={June},
  date={23}
}
```
