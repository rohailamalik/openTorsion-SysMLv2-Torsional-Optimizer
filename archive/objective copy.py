import numpy as np
from typing import Tuple, Dict, Any
from utils import to_python
from parser import System, ValidationError


# Constants
DEFAULT_W_OPERATION = 314  # rad/s
DEFAULT_SS_STEPS = 200


def find_natural_freqs(assembly, harmonics: np.ndarray):
    """
    Calculate natural frequencies and minimum distance to forced frequencies.
    
    Args:
        assembly: OpenTorsion assembly object
        harmonics: Array of harmonic frequencies in Hz
        
    Returns:
        Tuple of (natural frequencies in Hz, minimum distance to any harmonic)
    """

    wn, wd, zeta = assembly.modal_analysis()

    if np.any(zeta < 0):
        raise ValidationError("System has a negative damping ratio and is dynamically unstable.")
    
    fn  = wn[wd > 0] / (2 * np.pi) # Convert rad/s to Hz
    # wn from .modal_analysis() is magnitudes of the solutions of which wd are imaginary parts
    # a non zero wd will always appear twice due to conjugate pairs so we only need one from it (here positive one)
    # zero wd means those modes are not vibrating and will decay, so they are not relevant to vibrations and are to be ignored
    # also very small wn can be ignored because they essentially mean rigid body movement

    # Calculate relative distance between each natural and forced frequency
    diffs = np.abs(fn[:, None] - harmonics[None, :]) / fn[:, None]
    min_diff = float(np.min(diffs))

    return fn, min_diff


def find_ss_response(
    assembly,
    w_operation: float,
    excitation_matrix: np.ndarray,
    harmonics: np.ndarray,
    steps: int = DEFAULT_SS_STEPS
    ):
    """
    Calculate steady-state torque response for each shaft element.
    
    Args:
        assembly: OpenTorsion assembly object
        w_operation: Operating frequency in rad/s
        excitation_matrix: Excitation force matrix
        harmonics: Array of harmonic frequencies in Hz
        steps: Number of time steps for waveform reconstruction
        
    Returns:
        Tuple of (torque response dict, minimum maximum torque across all shafts)
    """
    # Time vector for one complete cycle
    t = np.linspace(0, 2 * np.pi / w_operation, steps)

    q_res, _ = assembly.ss_response(excitation_matrix, harmonics)
    

    q_difference = (q_res.T[:, 1:] - q_res.T[:, :-1]).T
    stiffnesses = [shaft.k for shaft in assembly.shaft_elements]
    
    ss_torque_resp = {}
    max_torques = []

    for shaft_idx, _ in enumerate(stiffnesses):
        shaft_key = f"shaft_{shaft_idx}"
        ss_torque_resp[shaft_key] = {}

        # Get displacement response for this shaft
        shaft_response = q_difference[shaft_idx]
        
        # Reconstruct time-domain waveform from harmonic components
        total_wave = np.zeros_like(t)
        
        for harmonic_idx, (response_component, harmonic_freq) in enumerate(
            zip(shaft_response, harmonics)
        ):
            # Convert complex frequency response to time-domain
            harmonic_wave = np.real(
                response_component * np.exp(1j * harmonic_freq * w_operation * t)
            )
            total_wave += harmonic_wave
            
            # Store individual harmonic contribution
            ss_torque_resp[shaft_key][f"harmonic_{harmonic_idx}"] = harmonic_wave

        # Store total waveform and track maximum
        ss_torque_resp[shaft_key]["total"] = total_wave
        max_torques.append(np.max(np.abs(total_wave)))

    # Convert to Python native types for JSON serialization
    ss_torque_resp = to_python(ss_torque_resp)
    max_max_torque = float(np.max(max_torques))

    return ss_torque_resp, max_max_torque


def default_obj_function(system: System, w_operation: float = DEFAULT_W_OPERATION) -> Dict[str, Any]:
    """
    Default objective function for torsional vibration optimization.
    
    Evaluates two primary objectives:
    1. Distance between natural and forced frequencies (maximize)
    2. Maximum torque vibration amplitude (minimize)
    
    Args:
        system: System object containing assembly and excitation data
        w_operation: Operating frequency in rad/s (default: 314 rad/s ≈ 50 Hz)
        
    Returns:
        Dictionary containing:
            - min_diff_bw_fn_ff: Minimum distance between natural and forced frequencies
            - max_torque_vibration: Maximum torque amplitude across all shafts
            - natural_frequencies: Array of natural frequencies (Hz)
            - torque_response: Detailed torque response for each shaft
            
    Only the first two numeric values are used for optimization, the rest stored for documentation.
    """
    # Extract assembly and excitation data from system
    assembly = system.get_assembly()
    excitation_matrix, harmonics = system.get_excitation_matrix()

    # Ensure numpy arrays
    excitation_matrix = np.asarray(excitation_matrix)
    harmonics = np.asarray(harmonics)

    # Validate inputs
    if harmonics.size == 0:
        raise ValueError("Harmonics array is empty")
    if excitation_matrix.size == 0:
        raise ValueError("Excitation matrix is empty")

    # Calculate natural frequencies and resonance proximity
    f_natural, min_diff = find_natural_freqs(assembly, harmonics)
    
    # Calculate steady-state torque response
    ss_torque_resp, max_torque = find_ss_response(
        assembly, w_operation, excitation_matrix, harmonics
    )

    return {
        "min_diff_bw_fn_ff": min_diff,
        "max_torque_vibration": max_torque,
        
        # these wont be passed to GA, they are not int/float, only for documentation
        "natural_frequencies": to_python(f_natural),
        "torque_response": ss_torque_resp
    }
