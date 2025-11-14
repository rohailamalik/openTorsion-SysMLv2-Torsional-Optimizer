# first get dictionary
# loop through all components in template and discretize
# for each param, check, if it not fixed, add its reference to tunable params. Also, if component has options, add to params too
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openTorsion'))
from opentorsion.assembly import Assembly
from opentorsion.plots import Plots
from opentorsion.elements.disk_element import Disk
from opentorsion.elements.shaft_element import Shaft
from opentorsion.elements.gear_element import Gear
from opentorsion.excitation import PeriodicExcitation

from pymoo.core.variable import Real, Integer, Choice
from pymoo.core.problem import ElementwiseProblem
from math import inf
from pathlib import Path
import json

# maybe define a whole class, it hold the system dict as an attribute.
# then parser method uses it to discretize and build assembly. 
# through a get method maybe we get the parameters which are modifiable
# through a set method we set those parameters in the system dict and update it
# maybe throhg a build method we first discritize fully in form of a dict
# then build method simply uses it to form an assembly



class System():
    def __init__(self, model_path: Path):
        with open(model_path, 'r') as f:
            self.system = json.load(f)
        
        self.name = self.system.get("name", "Unknown System")
        if not isinstance(self.name, str):
            raise ValueError("System's name must be a string.")

        self.components = self.system.get("components")
        if not isinstance(self.components, list) or not all(isinstance(c, dict) for c in self.components):
            raise ValueError("System must have 'components' as a list of dictionaries.")
        
        self.assembly = None
        self.design_vars = {}
        
    def get_name(self):
        return self.name

    def get_json(self):
        return self.system
    
    def get_plot(self):
        if not self.assembly:
            self.assembly = self.assemble()
        Plots(self.assembly).plot_assembly()


    def get_design_vars(self) -> dict:
        """Extract tunable design variables from the JSON model"""

        if self.design_vars:
            self.design_vars = {} 
        
        for i, comp in enumerate(self.components, start=1):
            comp_name = comp.get("name", f"Component {i}")
            params = comp.get("parameters")
            options = comp.get("options")

            if not params and not options:
                raise ValueError(f"{comp_name}: Must define either 'parameters' or 'options' with parameters.")

            if options:
                if not isinstance(options, list) or not all(isinstance(op, dict) for op in options):
                    raise TypeError(f"{comp_name}: 'options' must be a list of dictionaries.")

                for j, opt in enumerate(options, start=1):
                    opt_name = opt.get("name", f"Option {j}")
                    opt_params = opt.get("parameters")

                    if not isinstance(opt_params, dict):
                        raise ValueError(f"{comp_name}: {opt_name} must define a 'parameters' dictionary.")

                    for p_name, p_details in opt_params.items():
                        if not isinstance(p_details, dict) or not isinstance(p_details.get("value"), (int, float)):
                            raise TypeError(f"{comp_name}: {opt_name}.{p_name} must have a numeric 'value' field.")

                self.design_vars[f"{comp_name}.-.option"] = Choice(range(1, len(options) + 1))

            if params:
                if not isinstance(params, dict):
                    raise TypeError(f"{comp_name}: 'parameters' must be a dictionary.")

                for p_name, p_details in params.items():
                    if not isinstance(p_details, dict):
                        raise TypeError(f"{comp_name}.{p_name}: Parameter definition must be a dictionary.")
                    design_var = self._process_parameter(comp_name, p_name, p_details)
                    if design_var:
                        self.design_vars[f"{comp_name}.-.{p_name}"] = design_var

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
            raise ValueError(f"{comp_name}.{param_name}: Invalid '{key}' value: {val!r}")

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
            return Real(lower, upper)

        # Discrete 
        if isinstance(options, list):
            if not all(isinstance(opt, (int, float)) for opt in options):
                raise ValueError(f"{comp_name}.{param_name}: All 'options' values must be numeric.")
            return Choice(options)

        raise TypeError(f"{comp_name}.{param_name}: Invalid 'options' format.")
    

    def update(self, candidate: dict):
        """Update the system dictionary based on a candidate solution"""

        for path, value in candidate.items():
            comp_name, param = path.split(".-.")

            # Find component index by name
            index = next((i for i, c in enumerate(self.components) if c.get("name") == comp_name), None)
            comp = self.components[index]

            if param == "option":
                # Option case
                options = comp.get("options")
                selected = options[value - 1].get("parameters", {})
                comp.setdefault("parameters", {}).update(selected) # Merge selected option’s parameters into the component

            else:
                # Parameter case
                params = comp.setdefault("parameters", {})
                param_entry = params[param]
                param_entry["value"] = value

    
    def assemble(self):
        """Dicretizes system model and creates an openTorsion assembly object from it"""
        seq, node = [], 0

        for i, comp in enumerate(self.components):
            cname = comp.get("name", "Unknown component")
            ctype = comp.get("type", "unknown").lower()
            params = comp.get("parameters")

            def _get_param(parameters: dict, names: str | list[str], default=None):
                """Get a parameter with one of the given names in the parameters dictionary"""
                if isinstance(names, str):
                    names = [names]
                key = next((k for name in names for k in parameters if k.lower() == name.lower()), None)
                value = parameters[key]["value"] if key else None
                return default if value is None and default is not None else value

            def _add_disk():
                """Add openTorsion disk element"""
                seq.append(Disk(
                    node=node,
                    I=_get_param(params, "inertia", 0.0),
                    c=_get_param(params, "damping", 0.0),
                    k=_get_param(params, "stiffness", 0.0)
                ))

            def _add_shaft():
                """Add openTorsion shaft element"""
                seq.append(Shaft(
                    nl=node,
                    nr=node + 1,
                    I=_get_param(params, "inertia", 0.0),
                    c=_get_param(params, "damping", 0.0),
                    k=_get_param(params, "stiffness", 0.0),
                    L=_get_param(params, ["length", "l"], None),
                    odl=_get_param(params, ["outer_diameter", "od", "diameter", "d"], None),
                    idl=_get_param(params, ["inner_diameter", "id"], None),
                    rho=_get_param(params, ["density", "rho"], None),
                    G=_get_param(params, ["shear_modulus", "G"], None)
                ))

            def _add_gear(parent=None):
                """Add openTorsion gear element"""
                seq.append(Gear(
                    nl=node,
                    I=0.0,
                    R=_get_param(params, "radius"),
                    parent=parent
                ))
            
            def _discretize_disk():
                if seq and isinstance(seq[-1], Disk):
                    raise ValueError(f"Failed to compile {cname}: Cannot connect Disk element directly to Disk element")
                _add_disk()

            def _discretize_shaft():
                if not seq or isinstance(seq[-1], Shaft):
                    seq.append(Disk(node=len(seq), I=0.0)) # add a dummy disk to connect shafts
                _add_shaft()

            def _discretize_gear():
                """Discretizes a gear into Disk+Gear"""
                if seq and isinstance(seq[-1], Disk):
                    raise ValueError(f"Failed to compile {cname}: Cannot connect Gear element directly after Disk element")
                parent = seq[-1] if isinstance(seq[-1], Gear) else None # make last gear parent
                _add_disk()
                _add_gear(parent)

            def _discretize_gear_set():
                """Discretizes a gear_set into Disk+Gear+Gear+Disk"""
                # first gear (pinion)
                seq.extend([
                    Disk(node, 
                        I=_get_param(params, "inertia_1", 0.0),
                        c=_get_param(params, "damping_1", 0.0), 
                        k=_get_param(params, "stiffness_1", 0.0)
                        ),
                    Gear(node, I=0.0, R=_get_param(params, "radius_1"))
                    ])
                
                parent = seq[-1]
                # second gear (driven)
                seq.extend([
                    Gear(node + 1, I=0.0, R=_get_param(params, "radius_2"), parent=parent),
                    Disk(node + 1, 
                        I=_get_param(params, "inertia_2", 0.0),
                        c=_get_param(params, "damping_2", 0.0), 
                        k=_get_param(params, "stiffness_2", 0.0)
                        )
                    ])

            def _discretize_actuator():
                """
                Discretizes an actuator/rotor. Inertia always goes to disk if possible.
                - Disk+Shaft, if a shaft element is before it
                - Disk, if a shaft element is before it and it's the end of sequence
                - Shaft+Disk, if a Disk element is before it and it's the end of sequence
                - Shaft+Disk+Shaft, if a disk element is before it. Half stiffness/damping to each shaft 
                (as per series addition of these values)
                """
                end_of_chain = (i == len(self.components) - 1)

                if not seq or isinstance(seq[-1], Shaft):
                    if end_of_chain:
                        _add_disk(seq, node, params)
                    else:
                        seq.extend([
                            Disk(node=node, I=_get_param(params, "inertia", 0.0), c=0.0, k=0.0),
                            Shaft(nl=node, nr=node + 1, I=0.0,
                                c=_get_param(params, "damping", 0.0),
                                k=_get_param(params, "stiffness", 0.0))
                        ])

                elif isinstance(seq[-1], Disk):
                    if end_of_chain:
                        seq.extend([
                            Shaft(nl=node - 1, nr=node, I=0.0,
                                c=_get_param(params, "damping", 0.0),
                                k=_get_param(params, "stiffness", 0.0)),
                            Disk(node=node, I=_get_param(params, "inertia", 0.0), c=0.0, k=0.0)
                        ])
                    else:
                        half_c = _get_param(params, "damping", 0.0) / 2
                        half_k = _get_param(params, "stiffness", 0.0) / 2
                        seq.extend([
                            Shaft(nl=node - 1, nr=node, I=0.0, c=half_c, k=half_k),
                            Disk(node=len(seq), I=_get_param(params, "inertia", 0.0), c=0.0, k=0.0),
                            Shaft(nl=len(seq), nr=len(seq) + 1, I=0.0, c=half_c, k=half_k)
                        ])

            builder = {
                "disk": lambda: _discretize_disk(seq, node, params),
                "shaft": lambda: _discretize_shaft(seq, node, params),
                "gear": lambda: _discretize_gear(seq, node, params),
                "gear_set": lambda: _discretize_gear_set(seq, node, params),
                "actuator": lambda: _discretize_actuator(seq, node, params, i, self.components),
                "rotor": lambda: _discretize_actuator(seq, node, params, i, self.components)
            }.get(ctype)

            if not builder:
                raise ValueError(f"Unknown component type '{ctype}'")

            builder()
            node = len(seq)  # update node index after additions

        return Assembly(
            shaft_elements=[c for c in seq if isinstance(c, Shaft)], 
            disk_elements=[c for c in seq if isinstance(c, Disk)], 
            gear_elements=[c for c in seq if isinstance(c, Gear)]
            )

























"""
if disk then disk
if shaft then shaft
if gear then disk+gear
if gear_set then disk+gear+gear+disk
if rotor then disk+shaft or shaft+disk
if coupling then disk+shaft+disk or shaft with inertia
"""

def get_param(parameters: dict, names: str | list[str], default=None):
    if isinstance(names, str):
        names = [names]
    key = next(
        (k for name in names for k in parameters if k.lower() == name.lower()),
        None
    )
    value = parameters[key]["value"] if key else None
    if value is None and default is not None:
        return default
    return value


# Assembly is built from the updated dict.
def discretize_system(system: dict):
    components = system.get("components", [])
    
    node = 0
    seq = []

    # TODO: sort according to index if its defined for all
    
    for i, component in enumerate(components):
        
        type = component.get("type", "Unknown type").lower()
        params = component.get("parameters", {})


        if type == "disk":
            
            if isinstance(seq[-1], Disk):
                # Disks cannot directly connect to disks, last element has to be a shaft
                # unlike the case below, we cannot add a zero stiffness shaft as that essentially means no connection
                raise ValueError("Cannot compile. A disk element cannot connect to a disk element")
            
            seq.append(
                Disk(
                    node=node,
                    I=get_param(params, "inertia", 0.0),
                    c=get_param(params, "damping", 0.0),
                    k=get_param(params, "stiffness", 0.0)
                    )
                )
        
        elif type == "shaft":
            
            if len(seq) == 0 or isinstance(seq[-1], Shaft): 
                # Shafts cannot be at the beginning where they are unconnected
                # Nor can they connect directly to another shaft since they do not sit on nodes
                # rather between them
                # so in these cases we will put a mass less disk at previous node 
                seq.append(Disk(node=len(seq), I=0.0))

                I=get_param(params, "inertia", 0.0)
                c=get_param(params, "damping", 0.0)
                k=get_param(params, "stiffness", 0.0)
                odl=get_param(params, ["outer_diameter", "od", "diameter", "d"], None)
                idl=get_param(params, ["inner_diameter", "id"], None)
                L=get_param(params, ["length", "l"], None)
                rho=get_param(params, ["density", "rho"], None)
                G=get_param(params, ["shear_modulus", "G"], None)

            # Then we can easily add a shaft
            seq.append(
                Shaft(
                    nl=node,
                    nr=node+1,
                    I=I, c=c, k=k, L=L, odl=odl, idl=idl, rho=rho, G=G
                    )
                )

        elif type == "gear":

            # Gears are basically 2 Disk elements with 2 gears acting as a connection.
            # So the element before should be a shaft or gear
            if isinstance(seq[-1], Disk):
                raise ValueError("Cannot compile. A disk element cannot connect to a disk element")
            
            parent = seq[-1] if isinstance(seq[-1], Gear) else None
            
            seq.extend([
                Disk(
                    nl=node,
                    I=get_param(params, "inertia", 0.0),
                    c=get_param(params, "damping", 0.0),
                    k=get_param(params, "stiffness", 0.0)
                ),
                Gear(
                    nl=node,
                    I=0.0,
                    R=get_param(params, "radius"), 
                    parent=parent
                )
            ])


        elif type == "gear_set":

            # Gears are basically 2 Disk elements with 2 gears acting as a connection.
            # So the element before should be a shaft or gear
            # Technically gears can be defined directly as inertias
            # But to incorporate cases where damping is provided, we will model them as disk+gear
            if isinstance(seq[-1], Disk):
                raise ValueError("Cannot compile. A disk element cannot connect to a disk element")
            
            seq.extend([
                Disk(
                    nl=node,
                    I=get_param(params, "inertia_1", 0.0),
                    c=get_param(params, "damping_1", 0.0),
                    k=get_param(params, "stiffness_1", 0.0)
                ),
                Gear(
                    nl=node,
                    I=0.0,
                    R=get_param(params, "radius_1")
                ),
                Gear(
                    nl=node+1,
                    I=0.0,
                    R=get_param(params, "radius_2"),
                    parent=parent
                ),
                Disk(
                    nl=node+1,
                    I=get_param(params, "inertia_2", 0.0),
                    c=get_param(params, "damping_2", 0.0),
                    k=get_param(params, "stiffness_2", 0.0)
                )
            ])

        elif type == "actuator" or type == "rotor":  

            # rules are, shafts can sit between disks only, disks need shafts to connect

            if len(seq) == 0 or isinstance(seq[-1], Shaft):
                # If it's the first element or if there is a shaft before it,
                # then the shaft end of actuator is pointed away from it
                # if a component is also present after it so actuator = disk + shaft
                # if there is no component after it then it's only a disk
                if i == len(components): # if this is end of chain, just add a disk
                    seq.append(
                        Disk(
                            node=node,
                            I=get_param(params, "inertia", 0.0),
                            c=get_param(params, "damping", 0.0),
                            k=get_param(params, "stiffness", 0.0)
                        )
                    )
                else: # shaft before it, it is discritized: disk - shaft, shaft takes stiffness values, inertia is carried by disk
                    seq.extend([
                        Disk(
                            node=node,
                            I=get_param(params, "inertia", 0.0),
                            c=0.0,
                            k=0.0
                        ),
                        Shaft(
                            nl=node,
                            nr=node+1,
                            I=0.0,
                            c=get_param(params, "damping", 0.0),
                            k=get_param(params, "stiffness", 0.0)
                        )
                    ])
            elif isinstance(seq[-1], Disk): # if there is a disk before it, point the shaft towards that.
                if i == len(components): # if this is end of chain, add a shaft to connect to previous disk and then a disk to end the chain
                    seq.extend([
                        Shaft(
                            nl=node-1,
                            nr=node,
                            I=0.0,
                            c=get_param(params, "damping", 0.0),
                            k=get_param(params, "stiffness", 0.0)
                        ),
                        Disk(
                            node=node,
                            I=get_param(params, "inertia", 0.0),
                            c=0.0,
                            k=0.0
                        )
                    ])
                else: # if there is another component after this, then we will have two shafts, one shaft connects to previous disk, then the disk, then the shaft to next component
                    # stiffness is assumed to be in series and split via 1/keq = 1/k1 + 1/k2 where k1=k2, leading to each being 2k and 2c for damping

                    seq.extend([
                        Shaft(
                            nl=node-1,
                            nr=node,
                            I=0.0,
                            c=get_param(params, "damping", 0.0)/2,
                            k=get_param(params, "stiffness", 0.0)/2
                        ),
                        Disk(
                            node=len(seq),
                            I=get_param(params, "inertia", 0.0),
                            c=0.0,
                            k=0.0
                        ),
                        Shaft(
                            nl=len(seq),
                            nr=len(seq)+1,
                            I=0.0,
                            c=get_param(params, "damping", 0.0)/2,
                            k=get_param(params, "stiffness", 0.0)/2
                        )
                    ])
        
# now that all components have been added
# we get a sequence, and we extract elements to form assembly.

    gears, shafts, disks = [], [], []

    for comp in seq:
        if isinstance(comp, Gear):
            gears.append(comp)
        elif isinstance(comp, Shaft):
            shafts.append(comp)
        elif isinstance(comp, Disk):
            disks.append(comp)

    assembly = Assembly(shaft_elements=shafts, disk_elements=disks, gear_elements=gears)

    # Assembly can be visualized using openTorsion plotting tools.
    plot_tools = Plots(assembly)  # initialize plot tools
    plot_tools.plot_assembly()

    return assembly



                

