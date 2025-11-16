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


class System():
    """Class for storing and managing systems between openTorsion and JSON"""
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path).resolve()
        with open(model_path, 'r') as f:
            self.system = json.load(f)
        
        self.name = self.system.get("name", "Unknown System")
        if not isinstance(self.name, str):
            raise ValueError("System's name must be a string.")

        self.components = self.system.get("components")
        if not isinstance(self.components, list) or not all(isinstance(c, dict) for c in self.components):
            raise ValueError("System must have 'components' as a list of dictionaries.")
        
        self.assembly = None
        self.assembly_json = {}
        self.design_vars = {}
        self.harmonics_by_node = None
        self.plotting = False
        self.excitation_by_node = []

    def save_as_json(self, path: str | Path | None = None):
        """Save the current system dictionary as a JSON file."""
        if path is None:
            path = f"./{self.name}.json"

        save_path = Path(path).resolve()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.system, f, indent=4)

    def get_system_json(self):
        return self.system
    
    def get_assembly(self):
        self.assemble()
        self.validate()
        return self.assembly
    
    def get_assembly_json(self): # for debugging
        self.plotting = True
        try:
            self.assemble()  
        except:
            raise
        finally:
            self.plotting = False
        return self.assembly_json

    def plot_assembly(self): # for debugging
        self.plotting = True
        try:
            self.assemble()  
        except:
            raise
        finally:
            self.plotting = False
        
        Plots(self.assembly).plot_assembly()

    def get_excitation_matrix(self):

        if not self.excitation_by_node:
            return None, None
        
        harmonics = sorted(list(set(e[1] for e in self.excitation_by_node)))
        harmonic_index = {h: i for i, h in enumerate(harmonics)}
        
        n_dofs = self.assembly.dofs
        excitation_matrix = np.zeros((n_dofs, len(harmonics)), dtype=float)
        
        for node, h, amp in self.excitation_by_node:
            excitation_matrix[node, harmonic_index[h]] = amp

        return excitation_matrix, harmonics


    def get_design_vars(self) -> dict:
        """Extract tunable design variables from the JSON model"""

        if self.design_vars:
            self.design_vars = {} 
        
        for i, comp in enumerate(self.components, start=1):
            comp_name = comp.get("name", f"Component {i}")
            params = comp.get("parameters")
            choices = comp.get("choices")

            if not params and not choices:
                raise ValueError(f"{comp_name}: Must define either 'parameters' or 'choices' with parameters.")

            if choices: # Case where a component has choices in form of variants
                if not isinstance(choices, list) or not all(isinstance(ch, dict) for ch in choices):
                    raise TypeError(f"{comp_name}: 'options' must be a list of dictionaries.")

                for j, ch in enumerate(choices, start=1):
                    ch_name = ch.get("name", f"Option {j}")
                    ch_params = ch.get("parameters")

                    if not isinstance(ch_params, dict):
                        raise ValueError(f"{comp_name}: {ch_name} must define a 'parameters' dictionary.")

                    for p_name, p_details in ch_params.items():
                        if not isinstance(p_details, dict) or not isinstance(p_details.get("value"), (int, float)):
                            raise TypeError(f"{comp_name}: {ch_name}.{p_name} must have a numeric 'value' field.")

                choices_names = [ch["name"] if isinstance(ch.get("name"), str) else f"Choice__{i}" for i, ch in enumerate(choices, start=1)]
                self.design_vars[f"{comp_name}<<>>Choice"] = Choice(options=choices_names) 

            if params: # Case where parameters have ranges/options
                if not isinstance(params, dict):
                    raise TypeError(f"{comp_name}: 'parameters' must be a dictionary.")

                for p_name, p_details in params.items():
                    if not isinstance(p_details, dict):
                        raise TypeError(f"{comp_name}.{p_name}: Parameter definition must be a dictionary.")
                    design_var = self._process_parameter(comp_name, p_name, p_details)
                    if design_var:
                        self.design_vars[f"{comp_name}<<>>{p_name}"] = design_var

        return self.design_vars
    

    def _process_parameter(self, comp_name: str, param_name: str, details: dict):
        """Process a single parameter definition and return a design variable if tunable."""
        value = details.get("value")
        options = details.get("options")

        if not options:
            if value is None:
                raise ValueError(f"{comp_name}.{param_name}: Must define either 'value' or 'options'.")
            return None  # fixed parameter

        def convert(val, key):
            if isinstance(val, (int, float)):
                return float(val)
            raise ValueError(f"{comp_name}.{param_name}: Invalid '{key}' value: {val}")

        def parse_range(range_opts):
            if not isinstance(range_opts, dict):
                raise TypeError(f"{comp_name}.{param_name}: Range 'options' must be a dictionary.")
            min_val = convert(range_opts.get("min", -inf), "min")
            max_val = convert(range_opts.get("max", inf), "max")

            if min_val >= max_val:
                raise ValueError(f"{comp_name}.{param_name}: Invalid range: min ({min_val}) >= max ({max_val}).")
            return min_val, max_val

        # Continuous 
        if isinstance(options, dict):
            lower, upper = parse_range(options)
            return Real(bounds=(lower, upper)) 

        # Discrete 
        if isinstance(options, list):
            if not all(isinstance(opt, (int, float)) for opt in options):
                raise ValueError(f"{comp_name}.{param_name}: All 'options' values must be numeric.")
            return Choice(options=options) 

        raise TypeError(f"{comp_name}.{param_name}: Invalid 'options' format.")
    

    def update(self, candidate: dict):
        """Update the system based on a candidate solution"""

        for path, value in candidate.items():
            comp_name, param = path.split("<<>>")

            # Find component index by name
            index = next((i for i, c in enumerate(self.components) if c.get("name") == comp_name), None)
            if not index and "Component" in comp_name:
                index = int(comp_name.split()[-1]) - 1
            comp = self.components[index]

            if param == "Choice":
                # Option case
                choices = comp.get("choices")

                if "Choice__" in value:
                    _, index = value.split("__")
                    index -= 1
                else:
                    index = next((i for i, d in enumerate(choices) if d.get("name") == value), None)
                selected = choices[index].get("parameters", {})
                comp.setdefault("parameters", {}).update(selected) # Merge selected choice’s parameters into the component

            else:
                # Parameter case
                params = comp.setdefault("parameters", {})
                param_entry = params[param]
                param_entry["value"] = value

    
    def assemble(self):
        """Dicretizes system model and creates an openTorsion assembly object from it"""
        elements, self.assembly_json, self.excitation_by_node, node = [], {}, [], 0

        for i, comp in enumerate(self.components):
            cname = comp.get("name", "Unknown component")
            ctype = comp.get("type", "unknown").lower()
            params = comp.get("parameters", {})


            def _get_speed_and_excitation(ratio=1):
                if not params:
                    return
                
                speed = params.get("operation_speed", {}).get("value")
                if not speed and node != 0:
                    speed = self.assembly_json.get(str(node - 1), {}).get("speed")

                excitation = params.get("excitation", {}).get("value")
                if excitation is not None:
                    if not isinstance(excitation, list):
                        raise ValueError("Excitation 'value' must be a list of [frequency, amplitude] pairs.")
                    if ctype == "coupling" or ctype == "shaft":
                        raise ValueError(f"Failed to compile {cname}: Cannot add excitation to a shaft or coupling as it sits between the nodes")

                if speed:
                    self.assembly_json[str(node)]["speed"] = speed * ratio
                    if excitation:
                        excitation = [[freq * speed / 2*np.pi, amp] for freq, amp in excitation]

                if excitation:
                    self.assembly_json[str(node)]["excitation"] = excitation
                    for h, amp in excitation:
                        self.excitation_by_node.append((node, h, amp))

        
            def _get_param(parameters: dict, names: str | list[str], default=None):
                """Get a parameter with one of the given names in the parameters dictionary"""
                if not parameters and self.plotting: 
                    # if params are not defined and options are given, 
                    # # for plotting, return a dummy value to ensure plot
                    return 0.1 
                
                if isinstance(names, str):
                    names = [names]
                key = next((k for name in names for k in parameters if k.lower() == name.lower()), None)
                value = parameters[key].get("value") if key else default

                return value


            def _add_disk():
                """Add openTorsion disk element"""
                I=_get_param(params, "inertia", 0.0)
                c=_get_param(params, "damping", 0.0)
                k=_get_param(params, "stiffness", 0.0)

                elements.append(Disk(node=node, I=I, c=c, k=k))
                self.assembly_json[f"{node}"] = {"component": cname, "element": "disk", "I": I, "c": c, "k": k}

            def _add_shaft():
                """Add openTorsion shaft element"""
                I=_get_param(params, "inertia", None)
                if not I:
                    L=_get_param(params, ["length", "l"], None)
                    if not L:
                        raise ValueError(f"Failed to compile {cname}: Either shaft geometry or inertia, stiffness and damping must be provided.")
                    odl=_get_param(params, ["outer_diameter", "od", "diameter", "d"], 0.0)
                    idl=_get_param(params, ["inner_diameter", "id"], 0.0)
                    rho=_get_param(params, ["density", "rho"], 0.0)
                    G=_get_param(params, ["shear_modulus", "G"], 0.0)
                    elements.append(Shaft(nl=node, nr=node + 1, L=L, odl=odl, idl=idl, rho=rho, G=G))
                    self.assembly_json[f"{node}-{node+1}"] = {"component": cname, "element": "shaft", 
                                            "L": L, "odl": odl, "idl": idl, "rho": rho, "G": G}
                else: # doing this because even 0 inertia makes openTorsion choose that over geometry
                    c=_get_param(params, "damping", 0.0)
                    k=_get_param(params, "stiffness", 0.0)
                    elements.append(Shaft(nl=node, nr=node + 1, I=I, c=c, k=k))
                    self.assembly_json[f"{node}-{node+1}"] = {"component": cname, "element": "shaft", "I": I, "c": c, "k": k}
            

            def _add_gear(parent=None):
                """Add openTorsion gear element"""
                R=_get_param(params, "radius")
                elements.append(Gear(nl=node, I=0.0, R=R, parent=parent))
                self.assembly_json[f"{node}"] = {"component": cname, "element": "gear", "I": 0.0, "R": R}
            
            def _discretize_disk():
                if elements and isinstance(elements[-1], Disk):
                    raise ValueError(f"Failed to compile {cname}: Cannot connect Disk element directly to Disk element")
                _add_disk()
                _get_speed_and_excitation()
                

            def _discretize_shaft():
                if not elements or isinstance(elements[-1], Shaft):
                    raise ValueError(f"Failed to compile {cname}: Cannot add a shaft element at the start of the model")
                _add_shaft()
                _get_speed_and_excitation()

            def _discretize_gear():
                """Discretizes a gear into Disk+Gear"""
                if elements and isinstance(elements[-1], Disk):
                    raise ValueError(f"Failed to compile {cname}: Cannot connect Gear element directly after Disk element")
                
                parent = elements[-1] if isinstance(elements[-1], Gear) else None # make last gear parent

                _add_disk()
                _add_gear(parent)
                if parent:
                    ratio = elements[-2].R / elements[-1].R
                _get_speed_and_excitation(ratio)
            

            def _discretize_gear_set():
                """Discretizes a gear_set into Disk+Gear+Gear+Disk"""
                I1=_get_param(params, "inertia_1", 0.0)
                c1=_get_param(params, "damping_1", 0.0)
                k1=_get_param(params, "stiffness_1", 0.0)
                R1=_get_param(params, "radius_1")

                I2=_get_param(params, "inertia_2", 0.0)
                c2=_get_param(params, "damping_2", 0.0)
                k2=_get_param(params, "stiffness_2", 0.0)
                R2=_get_param(params, "radius_2")

                # first gear (pinion) and second gear (driven)
                elements.extend([
                    Disk(node, I=I1, c=c1, k=k1),
                    Gear(node, I=0.0, R=R1)
                ])
                self.assembly_json[f"{node}"] = {"component": cname, "element": "disk", "I": I1, "c": c1, "k": k1}
                self.assembly_json[f"{node}G"] = {"component": cname, "element": "gear", "I": 0.0, "R": R1}

                parent = elements[-1]
                elements.extend([
                    Gear(node+1, I=0.0, R=R2, parent=parent),
                    Disk(node+1, I=I2, c=c2, k=k2)
                    ])
                self.assembly_json[f"{node+1}G"] = {"component": cname, "element": "gear", "I": 0.0, "R": R2}
                self.assembly_json[f"{node+1}"] = {"component": cname, "element": "disk", "I": I2, "c": c2, "k": k2}

                _get_speed_and_excitation(ratio=R2/R1)

                
            def _discretize_coupling():
                """
                Discretizes a coupling. Inertia always goes to disk if possible.
                - Disk+Shaft+Disk: If a shaft element is before it
                - Shaft: If a disk element is before it
                """

                if not elements or isinstance(elements[-1], Shaft):
                    half_I = _get_param(params, "inertia", 0.0) / 2
                    c = _get_param(params, "damping", 0.0)
                    k = _get_param(params, "stiffness", 0.0)

                    elements.extend([
                        Disk(node, I=half_I, c=0.0, k=0.0),
                        Shaft(nl=node, nr=node+1, I=0.0, c=c, k=k),
                        Disk(node+1, I=half_I, c=0.0, k=0.0)
                    ])

                    self.assembly_json[f"{node}"] = {"component": cname, "element": "disk", "I": half_I, "c": 0.0, "k": 0.0}
                    self.assembly_json[f"{node}-{node+1}"] = {"component": cname, "element": "shaft", "I": 0.0, "c": c, "k": k}
                    self.assembly_json[f"{node+1}"] = {"component": cname, "element": "disk", "I": half_I, "c": 0.0, "k": 0.0}

                elif isinstance(elements[-1], Disk):
                    _add_shaft()
                
                _get_speed_and_excitation()

            
            def _discretize_actuator():
                """
                Discretizes an actuator/rotor. Inertia always goes to disk if possible.
                - Disk+Shaft: if a shaft element is before it
                - Disk: if a shaft element is before it and it's the end of sequence
                - Shaft+Disk: if a Disk element is before it and it's the end of sequence
                - Shaft+Disk+Shaft: if a disk element is before it. Half stiffness/damping to each shaft 
                (as per series addition of these values)
                """
                end_of_chain = (i == len(self.components) - 1)

                I = _get_param(params, "inertia", 0.0)
                c = _get_param(params, "damping", 0.0)
                k = _get_param(params, "stiffness", 0.0)

                if not elements or isinstance(elements[-1], Shaft):
                    if end_of_chain:
                        _add_disk()
                        self.assembly_json[cname] = "disk"
                    else:
                        elements.extend([
                            Disk(node=node, I=I, c=0.0, k=0.0),
                            Shaft(nl=node, nr=node + 1, I=0.0, c=c, k=k)
                        ])
                        self.assembly_json[f"{node}"] = {"component": cname, "element": "disk", "I": I, "c": 0.0, "k": 0.0}
                        self.assembly_json[f"{node}-{node + 1}"] = {"component": cname, "element": "shaft", "I": 0.0, "c": c, "k": k}

                elif isinstance(elements[-1], Disk):
                    if end_of_chain:
                        elements.extend([
                            Shaft(nl=node-1, nr=node, I=0.0, c=c, k=k),
                            Disk(node=node, I=I, c=0.0, k=0.0)
                        ])
                        self.assembly_json[f"{node-1}-{node}"] = {"component": cname, "element": "shaft", "I": 0.0, "c": c, "k": k}
                        self.assembly_json[f"{node}"] = {"component": cname, "element": "disk", "I": I, "c": 0.0, "k": 0.0}
                        self.assembly_json[cname] = "shaft+disk"
                    else:
                        half_c = c / 2
                        half_k = k / 2
                        elements.extend([
                            Shaft(nl=node-1, nr=node, I=0.0, c=half_c, k=half_k),
                            Disk(node=node, I=I, c=0.0, k=0.0),
                            Shaft(nl=node, nr=node + 1, I=0.0, c=half_c, k=half_k)
                        ])
                        self.assembly_json[f"{node-1}-{node}"] = {"component": cname, "element": "shaft", "I": 0.0, "c": half_c, "k": half_k}
                        self.assembly_json[f"{node}"] = {"component": cname, "element": "disk", "I": I, "c": 0.0, "k": 0.0}
                        self.assembly_json[f"{node}-{node+1}"] = {"component": cname, "element": "shaft", "I": 0.0, "c": half_c, "k": half_k}

                _get_speed_and_excitation()


            builder = {
                "disk": lambda: _discretize_disk(),
                "shaft": lambda: _discretize_shaft(),
                "gear": lambda: _discretize_gear(),
                "gear_set": lambda: _discretize_gear_set(),
                "coupling": lambda: _discretize_coupling(),
                "actuator": lambda: _discretize_actuator(),
                "rotor": lambda: _discretize_actuator()
            }.get(ctype)

            if not builder:
                raise ValueError(f"Unknown component type '{ctype}'")

            builder()
            node += 1

        self.assembly = Assembly(
            shaft_elements=[e for e in elements if isinstance(e, Shaft)], 
            disk_elements=[e for e in elements if isinstance(e, Disk)], 
            gear_elements=[e for e in elements if isinstance(e, Gear)]
            )
        


    def validate(self, atol=1e-12):
        """
        Validate assembled inertia (M), stiffness (K), and damping (C) matrices
        for symmetry and physical admissibility.
        """

        if not self.assembly:
            self.assemble()

        M, K, C = self.assembly.M, self.assembly.K, self.assembly.C
        
        def is_symmetric(A, tol=1e-12):
            return np.allclose(A, A.T, atol=tol)

        dofs = M.shape[0]

        for A, name in [(M, "M"), (K, "K"), (C, "C")]:
            if A is not None:
                if A.shape[0] != A.shape[1]:
                    raise ValueError(f"{name} must be square, got {A.shape}.")
                if A.shape[0] != dofs:
                    raise ValueError(f"{name} shape {A.shape} does not match M ({M.shape}).")
                
        if not is_symmetric(M):
            raise ValueError("Inertia matrix [M] is not symmetric.")
        if not is_symmetric(K):
            raise ValueError("Stiffness matrix [K] is not symmetric.")
        if C is not None and not is_symmetric(C):
            raise ValueError("Damping matrix [C] is not symmetric.")

        if np.any(np.diag(M) <= 0):
            raise ValueError("Inertia matrix [M] has non-positive diagonal entries.")
        if np.any(np.diag(K) < 0):
            raise ValueError("Stiffness matrix [K] has negative diagonal entries.")
        if C is not None and np.any(np.diag(C) < 0):
            raise ValueError("Damping matrix [C] has negative diagonal entries.")

        if np.linalg.matrix_rank(M) < dofs:
            raise ValueError("Inertia matrix [M] is singular or rank deficient.")

        try:
            eig_M = np.linalg.eigvalsh(M)
            eig_K = np.linalg.eigvalsh(K)
            if np.any(eig_M <= atol):
                raise ValueError("Inertia matrix [M] is not positive definite.")
            if np.any(eig_K < -atol):
                raise ValueError("Stiffness matrix [K] has negative eigenvalues.")
            if C is not None:
                eig_C = np.linalg.eigvalsh(C)
                if np.any(eig_C < -atol):
                    raise ValueError("Damping matrix [C] has negative eigenvalues.")
        except np.linalg.LinAlgError:
            raise ValueError("Eigenvalue check failed (matrices may be ill-conditioned).")
    
    