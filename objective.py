import numpy as np
from opentorsion.excitation import PeriodicExcitation

from parser import System, ValidationError
from utils import to_python


def calculate_natural_freqs(system: System) -> np.array:
    assembly = system.get_assembly()
    wn, wd, zeta = assembly.modal_analysis()
    if np.any(zeta < 0):
        raise ValidationError("System has a negative damping ratio and is dynamically unstable.")
    return wn[wd > 0] # ignore freqs which are non oscillating


def calculate_vibratory_torque(system: System, steps: int = None) -> np.array:
    """Calculates vibratory torque for all shafts, in a given range of speeds"""
    assembly = system.get_assembly()
    amplitudes, phases, modes = system.get_excitation_data()
    speed = system.get_speeds(steps=steps)

    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)
    modes = np.asarray(modes)

    dofs = assembly.M.shape[1]
    num_shafts = len(system.assembly.shaft_elements)

    T_vib_for_all_speeds = np.zeros((num_shafts, len(speed)))
    

    for i, w in enumerate(speed):
        omegas = w * modes
        excitation = PeriodicExcitation(dofs, omegas)

        for node in range(dofs):
            excitation.add_sines(node, omegas, amplitudes[node], phases[node])

        _, T_vib = system.assembly.vibratory_torque(excitation)
        T_vib_for_all_speeds[:, i] = T_vib

    return T_vib_for_all_speeds # rows are shafts, columns speed


def calculate_total_inertia(system: System) -> float:
    assembly = system.get_assembly()
    return (
        sum(d.I for d in assembly.disk_elements) +
        sum(s.mass for s in assembly.shaft_elements) +
        sum(g.I for g in assembly.gear_elements)
    )


def default_obj_function(system: System) -> dict:

    speed = system.get_speeds()

    T_vib_for_all_speeds = calculate_vibratory_torque(system)
    T_vib_max = np.max(T_vib_for_all_speeds)
    I_total = calculate_total_inertia(system)
    w_natural = calculate_natural_freqs(system)

    #system_json = system.get_system_json()
    #limit = system_json["components"][1]["parameters"]["continuousVibratoryTorque"]["value"]
    #limit = to_python(limit)
    #if T_vib_max >= limit:
    #    # Incurr high cost since this coupling cant handle this much torque
    #    T_vib_max = 1e12
    #    I_total = 1e12

    T_vibs = {}

    for i in range(T_vib_for_all_speeds.shape[0]):
        T_vibs[f"shaft_{i}"] = T_vib_for_all_speeds[i,:]
    
    return {
        "objectives": [T_vib_max, I_total],
        "natural freqs (rad/s)": w_natural,
        "speeds (rad/s)": speed,
        "vibratory torque (Nm)": T_vibs
    }