import numpy as np
from opentorsion.excitation import PeriodicExcitation

from adapter import SystemAdapter, ValidationError
from utils import to_python


def calculate_natural_freqs(system: SystemAdapter) -> np.array:
    assembly = system.get_assembly()
    wn, wd, zeta = assembly.modal_analysis()
    if np.any(zeta < 0):
        raise ValidationError("System has a negative damping ratio and is dynamically unstable.")
    return wn[wd > 0] # ignore freqs which are non oscillating


def calc_objectives(system: SystemAdapter, steps: int = None) -> np.array:
    """Calculates vibratory torque for all shafts, in a given range of speeds"""
    assembly = system.get_assembly()
    amplitudes, phases, orders = system.get_excitation_data()
    speed = system.get_speeds(steps=steps)

    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)
    orders = np.asarray(orders)

    dofs = assembly.M.shape[1]
    num_shafts = len(system.assembly.shaft_elements)

    T_vib = np.zeros((num_shafts, len(speed)), dtype=float)
    P_loss = np.zeros(len(speed), dtype=float)

    for i, w in enumerate(speed):

        omegas = w * orders
        excitation = PeriodicExcitation(dofs, omegas)

        for node in range(dofs):
            excitation.add_sines(node, omegas, amplitudes[node], phases[node])

        _, w_res = assembly.ss_response(excitation.U, excitation.omegas)
        _, T_vib_sys = system.assembly.vibratory_torque(excitation)
        P_loss_sys = 0.5 * np.real(np.sum( np.conj(w_res) * (assembly.C @ w_res), axis=0 )).sum() 
        
        T_vib[:, i] = T_vib_sys
        P_loss[i] = P_loss_sys

    return T_vib, P_loss


def default_obj_function(system: SystemAdapter) -> dict:
    speed = np.asarray(system.get_speeds())
    T_vib, P_loss = calc_objectives(system)

    steady_mask = speed >= 125 # operating speed

    # Global maxima
    T_vib_max = T_vib.max()
    P_loss_max = P_loss.max()

    # Steady-state maxima 
    T_vib_ss_max = T_vib[:, steady_mask].max()
    P_loss_ss_max = P_loss[steady_mask].max()

    # Natural frequencies
    w_natural = calculate_natural_freqs(system)

    # Torque per shaft
    T_vib_per_shaft = {f"shaft_{i}": T_vib[i] for i in range(T_vib.shape[0])}

    # Coupling 
    T_cpl = T_vib_per_shaft["shaft_1"]
    T_cpl_max = T_cpl.max()
    T_cpl_ss_max = T_cpl[steady_mask].max()

    # Limits
    coupling = system.get_system_json()["components"][1]["parameters"]
    ss_limit = to_python(coupling["continuousVibratoryTorque"]["value"])
    res_limit = to_python(coupling["maxTorque"]["value"])

    # Penalty if limit breached
    if T_cpl_max >= res_limit or T_cpl_ss_max > ss_limit:
        T_vib_ss_max = 1e12
        P_loss_ss_max = 1e12

    return {
        "objectives": [T_vib_ss_max, P_loss_ss_max],
        "max_Tvib": T_vib_max,
        "max_Tvib_ss": T_vib_ss_max,
        "max_P_loss": P_loss_max,
        "max_P_loss_ss": P_loss_ss_max,
        "natural_freqs_rad_s": w_natural,
        "speeds_rad_s": speed,
        "vibratory_torque_Nm": T_vib_per_shaft,
    }
