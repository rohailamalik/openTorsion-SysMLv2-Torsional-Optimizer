from utils import to_python
import numpy as np
from archive.old_system import System

w_operation = 314

def find_natural_freqs(assembly, harmonics) -> dict:
    wn, _, _ = assembly.modal_analysis()
    f_natural = wn / (2*np.pi) # convert from rad/s to Hz

    # Calculate distance b/w each natural and forced frequency
    diffs = np.abs(f_natural[:, None] - harmonics[None, :])
    diffs = to_python(np.sort(diffs, axis=None))

    return f_natural, min(diffs)


def find_ss_response(
        assembly, 
        w_operation: float, # in rad/s
        excitation_matrix: np.array, 
        harmonics: np.array, # in Hz
        steps: int = 200):
    
    ss_torque_resp = {}

    t = np.linspace(0, 2 * np.pi / w_operation, steps) 

    # get steady-state response
    q_res, _ = assembly.ss_response(excitation_matrix, harmonics)
    q_difference = (q_res.T[:, 1:] - q_res.T[:, :-1]).T
    stiffnesses = [shaft.k for shaft in assembly.shaft_elements]
    max_torque = []

    for n, _ in enumerate(stiffnesses):
        ss_torque_resp[f"shaft_{n}"] = {}

        shaft_response = q_difference[n]
        sum_wave = np.zeros_like(t)
        
        for i, (response_component, harmonic) in enumerate(zip(shaft_response, harmonics)):
            this_wave = np.real(response_component * np.exp(1j * harmonic * w_operation * t))
            sum_wave += this_wave
            ss_torque_resp[f"shaft_{n}"][f"harmonic_{i}"] = this_wave

        max_torque.append(max(sum_wave))
        
        ss_torque_resp[f"shaft_{n}"]["total"] = sum_wave

    ss_torque_resp = to_python(ss_torque_resp)
    max_torque = to_python(max_torque)

    return ss_torque_resp, min(max_torque)


def default_obj_function(system: System) -> dict:
    assembly = system.get_assembly()
    excitation_matrix, harmonics = system.get_excitation_matrix()

    excitation_matrix = np.asarray(excitation_matrix)
    harmonics = np.asarray(harmonics)

    f_natural, min_diff = find_natural_freqs(assembly, harmonics)
    ss_torque_resp, max_torque = find_ss_response(assembly, w_operation, excitation_matrix, harmonics)

    return { # only first two (i.e. float/int) will be used in optimization, rest only for documenting results
        "min_diff_bw_fn_ff": min_diff,
        "max_torque_vibration": max_torque,
        "natural_frequencies": f_natural,
        "torque_response": ss_torque_resp
    }



