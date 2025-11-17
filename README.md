# openTorsion-Based Torsional Analysis System Optimizer with SysML Integration

This repository provides a multi-objective, mixed-variable genetic-algorithm optimizer built on the openTorsion Python library. It can ingest SysML-derived system descriptions (JSON), extract all available design variables, discretize the system into an openTorsion-compatible model, and evaluate it using a custom objective function.

The system JSON input must follow specific structural rules so the parser can correctly interpret and discretize the model. See `models/complete_model.json` for a fully worked example.

## JSON Structure Requirements

1. The root object must be a dictionary containing a \"components\" list.
2. Each entry in \"components\" must be a component dictionary.
3. Each component must define a \"type\" field. Valid types include: Actuator, Rotor, Coupling, Disk, Shaft, Gear, Gear Set.
4. Each component must contain a \"parameters\" dictionary. Each parameter is a dictionary with:
   - \"value\"
   - \"units\"
   Optionally:
   - \"options\" with discrete values (list) or a range (dict with \"min\" and/or \"max\").
5. Components may define design-choice variants in a \"choices\" list. Each entry contains a \"parameters\" dictionary and optionally a \"name\".
6. Speed and excitation may also be defined. Speed may be a fixed number, list, or range dict. Only one component may define speed. Excitation must be a list of excitations, each `[order, amplitude, phase]`.

## Discretization Rules

- Stiffnesses and damping map to shaft elements.
- Inertias map to disk elements.

**Actuator / Rotor**  
Expand into one of several disk/shaft patterns depending on neighbors.

**Coupling**  
Disk + Shaft + Disk. Inertia is split if only one is provided.

**Gear Set**  
Disk + Gear + Gear + Disk.

**Gear**  
Disk + Gear.

**Shaft / Disk**  
Added directly per openTorsion rules.

## Code Overview

**SystemParser**  
Reads JSON, discretizes, extracts design variables, and writes optimized values back.

**Optimizer**  
GA engine. Stores history and final solutions and can merge results back into the original JSON.

**Default Objective Function**  
Computes maximum vibratory torque and total inertia; the optimizer minimizes this.

## Installation

Pinned dependencies:

uv pip freeze > requirements.txt


Install:

pip install -r requirements.txt
uv sync


## Usage

1. Create a `SystemParser` with JSON path or dictionary.
2. Create an optimizer and assign parser + objective.
3. Run:

```python
optimizer.run()
