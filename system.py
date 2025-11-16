import json
from math import inf
from pathlib import Path
from pymoo.core.variable import Real, Choice
import numpy as np

from opentorsion.plots import Plots
from opentorsion.assembly import Assembly
from opentorsion.elements.disk_element import Disk
from opentorsion.elements.shaft_element import Shaft
from opentorsion.elements.gear_element import Gear

class ValidationError(Exception):
    """Custom exception class for catching cases where the assembled system is not solveable etc."""
    pass

class System:
    """Class for storing and managing systems between openTorsion and JSON"""
    
    def __init__(self, model: dict = None, model_path: str | Path = None):
        if model:
            self.system = model
        elif model_path:
            model_path = Path(model_path).resolve()
            with open(model_path, 'r') as f:
                self.system = json.load(f)
        else:
            raise ValueError("Either a model dictionary or path to it saved as a JSON file must be provided.")
        
        self._validate_system_structure()
        
        self.name = self.system["name"]
        self.components = self.system["components"]
        
        self.assembly = None # assembly object itself
        self.assembly_json = {} # dict representation of assembly
        self.design_vars = {} # stores design variables

    def _validate_system_structure(self):
        """Validate basic system structure during initialization"""
        if not isinstance(self.system, dict):
            raise ValueError(f"System must be defined as a dictionary, got {type(self.system)} instead.")

        self.name = self.system.get("name", "Unknown System")
        if not isinstance(self.name, str):
            raise ValueError(f"System's name must be a string, got {type(self.name)} instead.")

        self.components = self.system.get("components")
        if not isinstance(self.components, list) or not all(isinstance(c, dict) for c in self.components):
            raise ValueError(f"System must have 'components' as a list of dictionaries, got {type(self.components)} instead.")

    def save_as_json(self, path: str | Path | None = None):
        """Save the current system dictionary as a JSON file."""
        path = Path(path or f"./{self.name}.json").resolve()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.system, f, indent=4)

    def get_system_json(self) -> dict:
        return self.system
    
    def get_assembly(self) -> Assembly:
        self.assemble()
        self.validate()
        return self.assembly
    
    def get_assembly_json(self) -> dict:
        """Get assembly JSON for debugging"""
        try:
            self.assemble()
        except Exception:
            self.assemble(plotting_mode=True)
        return self.assembly_json

    def plot_assembly(self):
        """Plot assembly for debugging"""
        try:
            self.assemble()
        except Exception:
            self.assemble(plotting_mode=True)
        Plots(self.assembly).plot_assembly()

    def get_speeds(self, steps: int = None):
        if not self.assembly:
            self.assembly_json = self.get_assembly_json()

        for _, data in self.assembly_json.items():
            speed = data.get("speed")
            if speed:
                break

        if isinstance(speed, dict):
            min_speed = speed["min"]
            max_speed = speed["max"]
            if not steps:
                steps = int(max_speed - min_speed)
            return np.linspace(min_speed, max_speed, steps)
        else: # it's already a list
            return np.asarray(speed)
            

    def get_excitation_data(self):
        """Returns excitation data for the assembly as matrices of excitation amplitues, phases and modes"""
        
        if not self.assembly:
            _ = self.get_assembly_json()
        
        n_dofs = self.assembly.dofs

        modes_set = set()
        rows = []
        for node in range(n_dofs):
            data = self.assembly_json[str(node)]
            exc_list = data.get("excitation")
            if not exc_list:
                continue

            ratio = data.get("speed_ratio", 1.0)
            for mode, amp, phase in exc_list:
                h = mode * ratio
                modes_set.add(h)
                rows.append((node, h, amp, phase))

        modes = sorted(modes_set)
        mode_idx = {h: i for i, h in enumerate(modes)}

        amplitudes = np.zeros((n_dofs, len(modes)), float)
        phases = np.zeros((n_dofs, len(modes)), float)

        for node, h, amp, phase in rows:
            j = mode_idx[h]
            amplitudes[node, j] = amp
            phases[node, j] = phase

        return amplitudes, phases, modes

    def get_design_vars(self) -> dict:
        """Extract tunable design variables from the JSON model"""
        self.design_vars = {}
        
        for i, comp in enumerate(self.components, start=1):
            comp_name = comp.get("name", f"Component {i}")
            params = comp.get("parameters")
            choices = comp.get("choices")

            if not params and not choices:
                raise ValueError(f"{comp_name}: Must define either 'parameters' or 'choices' with parameters.")

            # Process choices (component variants)
            if choices:
                self._process_choices(comp_name, choices)

            # Process parameters with ranges/options
            if params:
                self._process_parameters(comp_name, params)

        return self.design_vars

    def _process_choices(self, comp_name: str, choices: list):
        """Process component choices and add to design variables"""
        if not isinstance(choices, list) or not all(isinstance(ch, dict) for ch in choices):
            raise TypeError(f"{comp_name}: 'choices' must be a list of dictionaries.")

        for j, ch in enumerate(choices, start=1):
            ch_name = ch.get("name", f"Option {j}")
            ch_params = ch.get("parameters")

            if not isinstance(ch_params, dict):
                raise ValueError(f"{comp_name}: {ch_name} must define a 'parameters' dictionary.")

            for p_name, p_details in ch_params.items():
                if not isinstance(p_details, dict) or not isinstance(p_details.get("value"), (int, float)):
                    raise TypeError(f"{comp_name}: {ch_name}.{p_name} must have a numeric 'value' field.")

        choices_names = [
            ch["name"] if isinstance(ch.get("name"), str) else f"Choice__{i}" 
            for i, ch in enumerate(choices, start=1)
        ]
        self.design_vars[f"{comp_name}<<>>Choice"] = Choice(options=choices_names)

    def _process_parameters(self, comp_name: str, params: dict):
        """Process component parameters and add tunable ones to design variables"""
        if not isinstance(params, dict):
            raise TypeError(f"{comp_name}: 'parameters' must be a dictionary.")

        for p_name, p_details in params.items():
            if not isinstance(p_details, dict):
                raise TypeError(f"{comp_name}.{p_name}: Parameter definition must be a dictionary.")
            
            design_var = self._create_design_variable(comp_name, p_name, p_details)
            if design_var:
                self.design_vars[f"{comp_name}<<>>{p_name}"] = design_var

    def _create_design_variable(self, comp_name: str, param_name: str, details: dict):
        """Create a design variable from parameter details"""
        value = details.get("value")
        options = details.get("options")

        if not options:
            if value is None:
                raise ValueError(f"{comp_name}.{param_name}: Must define either 'value' or 'options'.")
            return None  # Fixed parameter

        # Continuous range
        if isinstance(options, dict):
            return self._create_continuous_variable(comp_name, param_name, options)

        # Discrete options
        if isinstance(options, list):
            if not all(isinstance(opt, (int, float)) for opt in options):
                raise ValueError(f"{comp_name}.{param_name}: All 'options' values must be numeric.")
            return Choice(options=options)

        raise TypeError(f"{comp_name}.{param_name}: Invalid 'options' format.")

    def _create_continuous_variable(self, comp_name: str, param_name: str, range_opts: dict):
        """Create a continuous Real variable from range options"""
        if not isinstance(range_opts, dict):
            raise TypeError(f"{comp_name}.{param_name}: Range 'options' must be a dictionary.")
        
        min_val = self._convert_to_float(range_opts.get("min", -inf), comp_name, param_name, "min")
        max_val = self._convert_to_float(range_opts.get("max", inf), comp_name, param_name, "max")

        if min_val >= max_val:
            raise ValueError(f"{comp_name}.{param_name}: Invalid range: min ({min_val}) >= max ({max_val}).")
        
        return Real(bounds=(min_val, max_val))

    @staticmethod
    def _convert_to_float(val, comp_name: str, param_name: str, key: str):
        """Convert value to float with error handling"""
        if isinstance(val, (int, float)):
            return float(val)
        raise ValueError(f"{comp_name}.{param_name}: Invalid '{key}' value: {val}")

    def update(self, candidate: dict):
        """Update the system based on a candidate solution"""
        for path, value in candidate.items():
            comp_name, param = path.split("<<>>")
            comp = self._find_component(comp_name)

            if param == "Choice":
                self._update_choice(comp, value)
            else:
                self._update_parameter(comp, param, value)

    def _find_component(self, comp_name: str) -> dict:
        """Find component by name"""
        comp = next((c for c in self.components if c.get("name") == comp_name), None)
        if comp:
            return comp
        
        # Fallback: try to extract index from "Component N" format
        if "Component" in comp_name:
            index = int(comp_name.split()[-1]) - 1
            return self.components[index]
        
        raise ValueError(f"Component '{comp_name}' not found")

    def _update_choice(self, comp: dict, value: str):
        """Update component with selected choice"""
        choices = comp.get("choices")
        
        if "Choice__" in value:
            index = int(value.split("__")[1]) - 1
        else:
            index = next((i for i, d in enumerate(choices) if d.get("name") == value), None)
        
        selected = choices[index].get("parameters", {})
        comp.setdefault("parameters", {}).update(selected)

    def _update_parameter(self, comp: dict, param: str, value):
        """Update a specific parameter value"""
        params = comp.setdefault("parameters", {})
        params[param]["value"] = value

    def assemble(self, plotting_mode: bool = False):
        """Discretize system model and create an openTorsion assembly object"""
        elements = []
        self.assembly_json = {}
        self.excitation_by_node = []
        node = 0
        self._speed_node_data = {}

        for i, comp in enumerate(self.components):
            cname = comp.get("name", "Unknown component")
            

        for i, comp in enumerate(self.components):

            cname = comp.get("name", "Unknown component")
            ctype = comp.get("type", "unknown").lower()
            params = comp.get("parameters", {})

            # Build component using appropriate discretizer
            builder = self._get_component_builder(
                ctype, cname, params, elements, node, i, plotting_mode
            )
            
            if not builder:
                raise ValueError(f"Unknown component type '{ctype}'")

            builder()
            node += 1

        self.assembly = Assembly(
            shaft_elements=[e for e in elements if isinstance(e, Shaft)],
            disk_elements=[e for e in elements if isinstance(e, Disk)],
            gear_elements=[e for e in elements if isinstance(e, Gear)]
        )
        self._normalize_speed_ratios()

    def _normalize_speed_ratios(self):
        """Normalize speed ratios relative to the component that defines a speed or the actuator."""

        assembly = self.assembly_json

        speed_node = None
        actuator_node = None

        for node, data in assembly.items():
            if data.get("speed") is not None:
                if speed_node is not None:
                    raise ValueError(
                        f"Failed to compile. Found multiple components with speeds defined "
                        f"({speed_node}, {node}). Only one is allowed."
                    )
                speed_node = node

            if data.get("type") == "actuator" and data.get("element") == "disk":
                if actuator_node is not None:
                    raise ValueError(
                        "Failed to compile. Found multiple components of type 'actuator'. "
                        "Only one is allowed."
                    )
                actuator_node = node

        ref_node = speed_node or actuator_node
        if ref_node is None:
            return

        ref_ratio = assembly[ref_node]["speed_ratio"]

        for node in range(self.assembly.dofs):
            entry = assembly[str(node)]
            entry["speed_ratio"] = entry["speed_ratio"] / ref_ratio


    def _get_component_builder(self, ctype: str, cname: str, params: dict, 
                               elements: list, node: int, comp_index: int, plotting_mode: bool):
        """Get the appropriate component builder function"""
        builder_map = {
            "disk": self._discretize_disk,
            "shaft": self._discretize_shaft,
            "gear": self._discretize_gear,
            "gear_set": self._discretize_gear_set,
            "coupling": self._discretize_coupling,
            "actuator": self._discretize_actuator,
            "rotor": self._discretize_actuator,
        }
        
        builder_func = builder_map.get(ctype)
        if not builder_func:
            return None
        
        # Return a closure that captures the context
        return lambda: builder_func(cname, ctype, params, elements, node, comp_index, plotting_mode)

    def _get_param(self, parameters: dict, names: str | list[str], default=None, plotting_mode: bool = False):
        """Get a parameter with one of the given names"""
        
        if isinstance(names, str):
            names = [names]
        
        key = next((k for name in names for k in parameters if k.lower() == name.lower()), None)
        
        if key:
            return parameters[key].get("value") or (0.1 if plotting_mode else default)
        return 0.1 if plotting_mode else default # Dummy value for plotting


    def _process_speed_and_excitation(self, cname: str, ctype: str, params: dict, 
                                      node: int):
        """Process operation speed and excitation for a component"""
   
        if not params:
            return

        speed = params.get("speed", {}).get("value")

        if speed:
            if isinstance(speed, (float, int)):
                speed = [speed]
            elif isinstance(speed, list):
                pass
            elif isinstance(speed, dict):
                s_min = speed.get("min")
                s_max = speed.get("max")
                if s_min is None or s_max is None:
                    raise ValueError(f"{cname}: speed dict must contain 'min' and 'max'.")
                if not isinstance(s_min, (int, float)) or not isinstance(s_max, (int, float)):
                    raise TypeError(f"{cname}: 'min' and 'max' must be numeric.")
            else:
                raise TypeError(f"Error at {cname}: speed must be a number, list of numbers, or a dict with 'min' and 'max'.")
            
            self.assembly_json[str(node)]["speed"] = speed

        excitation = params.get("excitation", {}).get("value")
        if excitation:
            if not isinstance(excitation, list):
                raise ValueError(f"Error at {cname}: Excitation 'value' must be a list of [frequency, amplitude, phase] lists.")
            if ctype in ("shaft"):
                raise ValueError(f"Failed to compile {cname}: Cannot add excitation to a shaft. Excitation can only be added at nodes.")
            
            self.assembly_json[str(node)]["excitation"] = excitation

    def _set_speed_ratio(self, node: int, ratio: float = 1.0):
        if node == 0:
            self.assembly_json[str(node)]["speed_ratio"] = 1.0
        else:
            self.assembly_json[str(node)]["speed_ratio"] = self.assembly_json.get(str(node - 1), {}).get("speed_ratio") * ratio

    # Component discretizers
    def _discretize_disk(self, cname: str, ctype: str, params: dict, elements: list, 
                        node: int, comp_index: int, plotting_mode: bool):
        """Add openTorsion disk element"""
        if elements and isinstance(elements[-1], Disk):
            raise ValueError(f"Failed to compile {cname}: Cannot connect Disk directly to Disk")
        
        I = self._get_param(params, "inertia", 0.0, plotting_mode)
        c = self._get_param(params, ["damping", "torsionaldamping"], 0.0, plotting_mode)
        k = self._get_param(params, ["stiffness", "torsionalstiffness"], 0.0, plotting_mode)

        elements.append(Disk(node=node, I=I, c=c, k=k))
        self.assembly_json[f"{node}"] = {
            "component": cname, "type": "disk", "element": "disk", "I": I, "c": c, "k": k
        }
        self._process_speed_and_excitation(cname, ctype, params, node)

    def _discretize_shaft(self, cname: str, ctype: str, params: dict, elements: list, 
                        node: int, comp_index: int, plotting_mode: bool):
        """Add openTorsion shaft element"""
        if not elements or isinstance(elements[-1], Shaft):
            raise ValueError(f"Failed to compile {cname}: Cannot add shaft at start of model")
        
        I = self._get_param(params, "inertia", None, plotting_mode)
        
        if not I:
            self._add_shaft_from_geometry(cname, params, elements, node, plotting_mode)
        else:
            self._add_shaft_from_properties(cname, params, elements, node, plotting_mode)
        
        self._process_speed_and_excitation(cname, ctype, params, node)

    def _add_shaft_from_geometry(self, cname: str, params: dict, elements: list, 
                                 node: int, plotting_mode: bool):
        """Add shaft from geometric parameters"""
        L = self._get_param(params, ["length", "l"], None, plotting_mode) 
        if not L:
            raise ValueError(f"Failed to compile {cname}: Either shaft geometry or I/c/k must be provided")
        else:
            L = 1000 * L
        
        odl = self._get_param(params, ["outer_diameter", "od", "diameter", "d"], 0.0, plotting_mode) * 1000
        idl = self._get_param(params, ["inner_diameter", "id"], 0.0, plotting_mode) * 1000
        rho = self._get_param(params, ["density", "rho"], None, plotting_mode)
        G = self._get_param(params, ["shear_modulus", "G"], None, plotting_mode)
        c = self._get_param(params, ["torsionaldamping", "damping"], 0.0, plotting_mode)
        
        elements.append(Shaft(nl=node, nr=node + 1, L=L, odl=odl, idl=idl, rho=rho, G=G, c=c))
        self.assembly_json[f"{node}-{node+1}"] = {
            "component": cname, "type": "shaft", "element": "shaft", 
            "L": L, "odl": odl, "idl": idl, "rho": rho, "G": G, "c": c
        }

    def _add_shaft_from_properties(self, cname: str, params: dict, elements: list, 
                                   node: int, plotting_mode: bool):
        """Add shaft from I/c/k properties"""
        I = self._get_param(params, "inertia", None, plotting_mode)
        c = self._get_param(params, ["torsionaldamping", "damping"], 0.0, plotting_mode)
        k = self._get_param(params, ["torsionalstiffness", "stiffness"], 0.0, plotting_mode)
        
        elements.append(Shaft(nl=node, nr=node + 1, I=I, c=c, k=k))
        self.assembly_json[f"{node}-{node+1}"] = {
            "component": cname, "type": "shaft", "element": "shaft", "I": I, "c": c, "k": k
        }

    def _discretize_gear(self, cname: str, ctype: str, params: dict, elements: list, 
                        node: int, comp_index: int, plotting_mode: bool):
        """Discretize a gear into Disk+Gear"""
        if elements and isinstance(elements[-1], Disk):
            raise ValueError(f"Failed to compile {cname}: Cannot connect Gear directly after Disk")
        
        parent = elements[-1] if elements and isinstance(elements[-1], Gear) else None
        
        I = self._get_param(params, "inertia", 0.0, plotting_mode)
        c = self._get_param(params, ["torsionaldamping", "damping"], 0.0, plotting_mode)
        k = self._get_param(params, ["torsionalstiffness", "stiffness"], 0.0, plotting_mode)
        R = self._get_param(params, "radius", None, plotting_mode)

        if not (isinstance(R, (int, float))):
            raise TypeError("Radii must be numeric")
        
        if R <= 0:
            raise ValueError(f"Failed to compile {cname}: Radii must be positive numbers, got {R}")
        
        elements.append(Disk(node=node, I=I, c=c, k=k))
        elements.append(Gear(nl=node, I=0.0, R=R, parent=parent))
        
        self.assembly_json[f"{node}"] = {
            "component": cname, "type": "gear", "element": "disk", "I": I, "c": c, "k": k
        }
        self.assembly_json[f"{node}G"] = {
            "component": cname, "type": "gear", "element": "gear", "I": 0.0, "R": R
        }
        
        ratio = elements[-2].R / elements[-1].R if parent else 1.0
        self._set_speed_ratio(node, ratio)
        self._process_speed_and_excitation(cname, ctype, params, node)

    def _discretize_gear_set(self, cname: str, ctype: str, params: dict, elements: list, 
                        node: int, comp_index: int, plotting_mode: bool):
        """Discretize a gear_set into Disk+Gear+Gear+Disk"""
        I1 = self._get_param(params, "inertia1", 0.0, plotting_mode)
        c1 = self._get_param(params, ["torsionaldamping1", "damping1"], 0.0, plotting_mode)
        k1 = self._get_param(params, ["torsionalstiffness1", "stiffness1"], 0.0, plotting_mode)
        R1 = self._get_param(params, "radius1", None, plotting_mode)

        I2 = self._get_param(params, "inertia2", 0.0, plotting_mode)
        c2 = self._get_param(params, ["torsionaldamping2", "damping2"], 0.0, plotting_mode)
        k2 = self._get_param(params, ["torsionalstiffness2", "stiffness2"], 0.0, plotting_mode)
        R2 = self._get_param(params, "radius2", None, plotting_mode)

        if not (isinstance(R1, (int, float)) and isinstance(R2, (int, float))):
            raise TypeError("Radii must be numeric")
        
        if R1 <= 0 or R2 <= 0:
            raise ValueError(f"Failed to compile {cname}: Radii must be positive numbers, got {R1} and {R2}")

        # First gear (pinion)
        elements.extend([
            Disk(node, I=I1, c=c1, k=k1),
            Gear(node, I=0.0, R=R1)
        ])
        self.assembly_json[f"{node}"] = {
            "component": cname, "type": "gear_set", "element": "disk", "I": I1, "c": c1, "k": k1
        }
        self.assembly_json[f"{node}G"] = {
            "component": cname, "type": "gear_set", "element": "gear", "I": 0.0, "R": R1
        }
        self._set_speed_ratio(node)

        # Second gear (driven)
        parent = elements[-1]
        elements.extend([
            Gear(node + 1, I=0.0, R=R2, parent=parent),
            Disk(node + 1, I=I2, c=c2, k=k2)
        ])
        self.assembly_json[f"{node+1}G"] = {
            "component": cname, "type": "gear_set", "element": "gear", "I": 0.0, "R": R2
        }
        self.assembly_json[f"{node+1}"] = {
            "component": cname, "type": "gear_set", "element": "disk", "I": I2, "c": c2, "k": k2
        }
        
        self._set_speed_ratio(node+1, ratio=R2/R1)
        self._process_speed_and_excitation(cname, ctype, params, node)

    def _discretize_coupling(self, cname: str, ctype: str, params: dict, elements: list, 
                        node: int, comp_index: int, plotting_mode: bool):
        """Discretize a coupling as Disk+Shaft+Disk"""

        if isinstance(elements[-1], Disk):
            raise ValueError(f"Failed to compile {cname}: Cannot put a coupling directly after Disk")

        I = self._get_param(params, "inertia", None, plotting_mode)
        I1 = self._get_param(params, "inertia1", None, plotting_mode)
        I2 = self._get_param(params, "inertia2", None, plotting_mode)

        if I is None:
            if I1 is None or I2 is None:
                raise ValueError(f"Failed to compile {cname}: Either total inertia or both hub inertias must be provided.")
        else:
            # I is provided
            if I1 is None or I2 is None:
                I1 = I / 2
                I2 = I / 2
            # else: both I1 and I2 are provided; do nothing

        c = self._get_param(params, ["torsionaldamping", "damping"], 0.0, plotting_mode)
        k = self._get_param(params, ["torsionalstiffness", "stiffness"], 0.0, plotting_mode)

        elements.extend([
            Disk(node, I=I1, c=0.0, k=0.0),
            Shaft(nl=node, nr=node + 1, I=0.0, c=c, k=k),
            Disk(node + 1, I=I2, c=0.0, k=0.0)
        ])
        self.assembly_json[f"{node}"] = {
            "component": cname, "type": "coupling", "element": "disk", "I": I1, "c": 0.0, "k": 0.0
        }
        self.assembly_json[f"{node}-{node+1}"] = {
            "component": cname, "type": "coupling", "element": "shaft", "I": 0.0, "c": c, "k": k
        }
        self.assembly_json[f"{node+1}"] = {
            "component": cname, "type": "coupling", "element": "disk", "I": I2, "c": 0.0, "k": 0.0
        }
        self._set_speed_ratio(node)
        self._set_speed_ratio(node+1)
        self._process_speed_and_excitation(cname, ctype, params, node)

    def _discretize_actuator(self, cname: str, ctype: str, params: dict, elements: list, 
                        node: int, comp_index: int, plotting_mode: bool):
        """Discretize an actuator/rotor as Disk+Shaft, Shaft+Disk, Shaft+Disk+Shaft"""
        end_of_chain = (comp_index == len(self.components) - 1)

        I = self._get_param(params, "inertia", 0.0, plotting_mode)
        c = self._get_param(params, ["torsionaldamping", "damping"], 0.0, plotting_mode)
        k = self._get_param(params, ["torsionalstiffness", "stiffness"], 0.0, plotting_mode)

        if not elements or isinstance(elements[-1], Shaft):
            if end_of_chain:
                elements.append(Disk(node=node, I=I, c=c, k=k))
                self.assembly_json[f"{node}"] = {
                    "component": cname, "type": ctype, "element": "disk", "I": I, "c": c, "k": k
                }
            else:
                elements.extend([
                    Disk(node=node, I=I, c=0.0, k=0.0),
                    Shaft(nl=node, nr=node + 1, I=0.0, c=c, k=k)
                ])
                self.assembly_json[f"{node}"] = {
                    "component": cname, "type": ctype, "element": "disk", "I": I, "c": 0.0, "k": 0.0
                }
                self.assembly_json[f"{node}-{node+1}"] = {
                    "component": cname, "type": ctype, "element": "shaft", "I": 0.0, "c": c, "k": k
                }

        elif isinstance(elements[-1], Disk):
            if end_of_chain:
                elements.extend([
                    Shaft(nl=node - 1, nr=node, I=0.0, c=c, k=k),
                    Disk(node=node, I=I, c=0.0, k=0.0)
                ])
                self.assembly_json[f"{node-1}-{node}"] = {
                    "component": cname, "type": ctype, "element": "shaft", "I": 0.0, "c": c, "k": k
                }
                self.assembly_json[f"{node}"] = {
                    "component": cname, "type": ctype, "element": "disk", "I": I, "c": 0.0, "k": 0.0
                }
            else:
                half_c, half_k = c / 2, k / 2
                elements.extend([
                    Shaft(nl=node - 1, nr=node, I=0.0, c=half_c, k=half_k),
                    Disk(node=node, I=I, c=0.0, k=0.0),
                    Shaft(nl=node, nr=node + 1, I=0.0, c=half_c, k=half_k)
                ])
                self.assembly_json[f"{node-1}-{node}"] = {
                    "component": cname, "type": ctype, "element": "shaft", "I": 0.0, "c": half_c, "k": half_k
                }
                self.assembly_json[f"{node}"] = {
                    "component": cname, "type": ctype, "element": "disk", "I": I, "c": 0.0, "k": 0.0
                }
                self.assembly_json[f"{node}-{node+1}"] = {
                    "component": cname, "type": ctype, "element": "shaft", "I": 0.0, "c": half_c, "k": half_k
                }

        self._set_speed_ratio(node)
        self._process_speed_and_excitation(cname, ctype, params, node)

    def validate(self, atol=1e-12):
        """Validate assembled M, K, C matrices for symmetry and physical admissibility"""
        if not self.assembly:
            self.assemble()

        M, K, C = self.assembly.M, self.assembly.K, self.assembly.C
        dofs = M.shape[0]

        # Check matrix shapes
        for A, name in [(M, "M"), (K, "K"), (C, "C")]:
            if A is not None:
                if A.shape[0] != A.shape[1]:
                    raise ValueError(f"{name} must be square, got {A.shape}.")
                if A.shape[0] != dofs:
                    raise ValueError(f"{name} shape {A.shape} does not match M ({M.shape}).")

        # Check symmetry
        self._validate_symmetry(M, "M", atol)
        self._validate_symmetry(K, "K", atol)
        if C is not None:
            self._validate_symmetry(C, "C", atol)

        # Check diagonal positivity
        if np.any(np.diag(M) <= 0):
            raise ValidationError("Inertia matrix [M] has non-positive diagonal entries.")
        if np.any(np.diag(K) < 0):
            raise ValidationError("Stiffness matrix [K] has negative diagonal entries.")
        if C is not None and np.any(np.diag(C) < 0):
            raise ValidationError("Damping matrix [C] has negative diagonal entries.")

        # Check rank
        if np.linalg.matrix_rank(M) < dofs:
            raise ValidationError("Inertia matrix [M] is singular or rank deficient.")

        # Check eigenvalues
        self._validate_eigenvalues(M, K, C, atol)

    @staticmethod
    def _validate_symmetry(A: np.ndarray, name: str, tol: float = 1e-12):
        """Validate matrix symmetry"""
        if not np.allclose(A, A.T, atol=tol):
            raise ValidationError(f"{name} matrix is not symmetric.")

    @staticmethod
    def _validate_eigenvalues(M: np.ndarray, K: np.ndarray, C: np.ndarray | None, atol: float):
        """Validate matrix eigenvalues for physical admissibility"""
        try:
            eig_M = np.linalg.eigvalsh(M)
            eig_K = np.linalg.eigvalsh(K)
            
            if np.any(eig_M <= atol):
                raise ValidationError("Inertia matrix [M] is not positive definite.")
            if np.any(eig_K < -atol):
                raise ValidationError("Stiffness matrix [K] has negative eigenvalues.")
            
            #if C is not None:
            #    eig_C = np.linalg.eigvalsh(C)
            #    if np.any(eig_C < -atol):
            #        raise ValidationError("Damping matrix [C] has negative eigenvalues.")
        except np.linalg.LinAlgError:
            raise ValidationError("Eigenvalue check failed (matrices may be ill-conditioned).")