import json
import opentorsion as ot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openTorsion'))
# --- CORRECTED IMPORTS ---
# We must import each element class from its specific module
from opentorsion.assembly import Assembly
from opentorsion.plots import Plots
from opentorsion.elements.disk_element import Disk
from opentorsion.elements.shaft_element import Shaft
from opentorsion.elements.gear_element import Gear
from opentorsion.excitation import PeriodicExcitation
# --- END CORRECTIONS ---

def parse_system_json_v2(input_data: dict):
    """
    Parses the new 5-node system JSON and builds the opentorsion Assembly.
    V2: Reads explicit nodes, shafts, and gear connections.
    
    Returns:
        (ot.Assembly, dict): A tuple of the assembled system and the excitations.
    """
    disk_elements = []
    shaft_elements = []
    gear_elements = []
    excitations = {}
    
    # A map to find elements by their full name (e.g., "Motor_Component.Motor")
    element_map = {}
    # A map to store gear data and objects by node index
    gear_data_by_node = {}
    gear_objects_by_node = {}

    print("--- Parsing System JSON (v2) ---")

    # --- Pass 1: Create Disks, find Gears, and get Excitations ---
    print("  Pass 1: Creating Disks and finding Shaft/Gear data...")
    for component in input_data.get('components', []):
        comp_name = component.get('name', 'Unknown_Component')
        for element in component.get('elements', []):
            el_name = element.get('name', 'Unknown_Element')
            full_name = f"{comp_name}.{el_name}"
            element_map[full_name] = element

            el_type = element.get('type')

            # Create Disk elements
            if el_type == 'Disk':
                node_index = element.get('index')
                I = element.get('inertia', 0.0)
                c = element.get('damping', 0.0)
                k = element.get('stiffness', 0.0)

                print(f"    Creating Disk: {el_name} (Node {node_index})")
                disk_elements.append(Disk(node=node_index, I=I, c=c, k=k))

                # If the disk has excitation, store it
                if element.get('excitation'):
                    print(f"      Found excitation for Node {node_index}")
                    excitations[node_index] = element['excitation']['values']

                # If the disk is part of a gear pair, store its data
                if element.get('gear_radius'):
                    gear_data_by_node[node_index] = {
                        'radius': element['gear_radius'],
                        'parent_node': element.get('parent_gear_node'),
                        'inertia': I
                    }

    # --- Pass 2: Create Gear elements using parent-child relationships ---
    print("  Pass 2: Creating Gear elements...")
    # First pass: create gears without parents (pinions)
    for node, gear_data in gear_data_by_node.items():
        if gear_data.get('parent_node') is None:
            # This is a parent gear (pinion)
            gear_obj = Gear(node=node, I=gear_data['inertia'], R=gear_data['radius'], parent=None)
            gear_objects_by_node[node] = gear_obj
            gear_elements.append(gear_obj)
            print(f"    Created Parent Gear at Node {node} (R={gear_data['radius']})")

    # Second pass: create gears with parents (bullgears)
    for node, gear_data in gear_data_by_node.items():
        parent_node = gear_data.get('parent_node')
        if parent_node is not None:
            # This is a child gear (bullgear), link to parent
            if parent_node in gear_objects_by_node:
                parent_gear = gear_objects_by_node[parent_node]
                gear_obj = Gear(node=node, I=gear_data['inertia'], R=gear_data['radius'], parent=parent_gear)
                gear_objects_by_node[node] = gear_obj
                gear_elements.append(gear_obj)
                ratio = gear_data['radius'] / gear_data_by_node[parent_node]['radius']
                print(f"    Created Child Gear at Node {node} (R={gear_data['radius']}, Parent={parent_node}, Ratio={ratio:.2f})")

    # --- Pass 3: Create Shaft elements from the 'structure' array ---
    print("  Pass 3: Creating Shafts...")
    for shaft_def in input_data.get('structure', []):
        node_a, node_b = shaft_def['nodes']
        
        if shaft_def['type'] == 'ShaftRigid':
            print(f"    Rigid Shaft: Node {node_a} -> {node_b}")
            # --- FIX: Use correct class 'Shaft' ---
            shaft_elements.append(Shaft(node_a, node_b, k=1e12, c=0.0))
            
        elif shaft_def['type'] == 'ShaftFromElement':
            el_name = shaft_def['element_name']
            if el_name in element_map:
                element = element_map[el_name]
                k = element['stiffness']
                c = element['damping']
                print(f"    Flex Shaft '{el_name}': Node {node_a} -> {node_b}")
                # --- FIX: Use correct class 'Shaft' ---
                shaft_elements.append(Shaft(node_a, node_b, k=k, c=c))

    print("--- Parsing Complete ---")

    # Create the final assembly
    # --- FIX: Use correct class 'Assembly' ---
    assembly = Assembly(shaft_elements, disk_elements, gear_elements if gear_elements else None)
    return assembly, excitations, disk_elements, shaft_elements

# --- MAIN ANALYSIS SCRIPT ---
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Update this to the path of your new, correct 5-node JSON
    file_path = './analysis_cases/system_shore50.json' 
    # --- END CONFIGURATION ---

    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Please save the corrected 5-node JSON to this path.")
        sys.exit(1)
        
    with open(file_path) as input_json:
        input_data = json.load(input_json)

        # Use the new v2 parser
        assembly, excitations, disk_elements, shaft_elements = parse_system_json_v2(input_data)

    # Display the assembly
    # --- FIX: Use correct class 'Plots' ---
    plot_tools = Plots(assembly)
    plot_tools.plot_assembly()
    plt.title("System Assembly (5 Nodes)")
    plt.savefig('assembly_plot.png', dpi=150, bbox_inches='tight')
    print("\nSaved assembly plot to 'assembly_plot.png'")

    # Extract frequency sweep configuration
    if 'frequency_sweep' in input_data:
        freq_config = input_data['frequency_sweep']
        
        # Assume 'start' and 'stop' are RPM
        rpm_start = freq_config['start']
        rpm_stop = freq_config['stop']
        n_steps = freq_config['points']
        
        # Create RPM sweep
        rpm_sweep = np.linspace(max(rpm_start, 1.0), rpm_stop, n_steps)

        # Use assembly's built-in damping matrix
        M, K = assembly.M, assembly.K
        C = assembly.C
        dofs = M.shape[0]

        print(f"\nSystem properties:")
        print(f"  DOFs: {dofs}")
        print(f"  Mass matrix shape: {M.shape}")
        
        # Check if C matrix is valid (non-zero)
        if np.any(C):
            print(f"  Damping matrix (non-zero entries):\n{C[C != 0]}")
        else:
            print("  Damping matrix is all zeros.")

        # Prepare result storage
        num_shafts = len(assembly.shaft_elements)
        VT_sum_result = np.zeros((num_shafts, n_steps))

        # --- Read harmonic excitations from parser ---
        print(f"\n  Found excitations: {excitations}")
        plotted_harmonics = sorted(list(set(int(h[0]) for v in excitations.values() for h in v)))

        # --- Run Frequency Sweep in RPM ---
        print("  Running forced response analysis...")
        for i, rpm in enumerate(rpm_sweep):
            
            # Fundamental rotational frequency in Hz
            f_hz = rpm / 60.0
            
            all_omegas, all_amplitudes, all_phases, all_nodes = [], [], [], []

            for node, harmonics in excitations.items():
                for order, amplitude_nm in harmonics:
                    
                    # Excitation frequency (rad/s) = order * shaft_speed (rad/s)
                    omega = order * (2 * np.pi * f_hz)
                    
                    all_omegas.append(omega)
                    all_amplitudes.append(amplitude_nm) # Use direct amplitude
                    all_phases.append(0.0)
                    all_nodes.append(node)

            if all_omegas:  # Only compute if there are excitations
                # Create PeriodicExcitation with unique omega values
                unique_omegas = sorted(list(set(all_omegas)))
                excitation = PeriodicExcitation(dofs, unique_omegas)

                # Add each excitation component
                for node, omega, amplitude, phase in zip(all_nodes, all_omegas, all_amplitudes, all_phases):
                    omega_idx = unique_omegas.index(omega)
                    excitation.add_sines(node, [omega], [amplitude], [phase])

                # Run analysis
                _, T_vib_sum = assembly.vibratory_torque(excitation, C=C)
                VT_sum_result[:, i] = np.ravel(T_vib_sum)

        # --- Plot Forced Response ---
        print(f"  Max vibratory torque: {np.max(np.abs(VT_sum_result)/1000):.2f} kNm")
        plt.figure(figsize=(10, 6))
        for i in range(VT_sum_result.shape[0]):
            shaft = assembly.shaft_elements[i]
            plt.plot(rpm_sweep, np.abs(VT_sum_result[i, :]/1000), label=f"Shaft {shaft.nl}-{shaft.nr}")
        plt.xlabel("RPM (rpm)")
        plt.ylabel("Vibratory Torque (kNm)")
        plt.title("Forced Response Analysis")
        output_file = 'forced_response_plot.png'
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved forced response plot to '{output_file}'")


        # --- Generate Campbell Diagram ---
        print("\n" + "="*60)
        print("Generating Campbell Diagram...")
        print("="*60)

        rpm_range = [rpm_start, rpm_stop]
        
        plot_tools.plot_campbell(
            frequency_range_rpm=rpm_range,
            num_modes=5,
            harmonics=plotted_harmonics,
            operating_speeds_rpm=[]
        )
        plt.title("Campbell Diagram")
        plt.savefig('campbell_diagram.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  Campbell diagram generated and saved to 'campbell_diagram.png'")
        print(f"  Harmonics plotted: {plotted_harmonics}")
        print(f"  RPM range: {rpm_range[0]:.1f} - {rpm_range[1]:.1f}")

        # --- Calculate Natural Frequencies ---
        wn, wd, damping_ratios = assembly.modal_analysis()
        
        # Filter out rigid body modes (frequencies near zero)
        natural_freqs_rad_s = wn[wn > 1e-3]
        natural_freqs_hz = sorted(list(natural_freqs_rad_s / (2 * np.pi)))[:5]
        
        print(f"  Calculated Natural Frequencies (Hz): {[float(f'{f:.2f}') for f in natural_freqs_hz]}")


        # --- Save results to JSON ---
        results = {
            "analysis_type": "Torsional Vibration Analysis",
            "input_file": file_path,
            "timestamp": input_data.get('date', ''),
            "system_properties": {
                "degrees_of_freedom": int(dofs),
                "total_inertia": float(np.sum([d.I for d in disk_elements])),
                "total_stiffness": float(np.sum([s.k for s in shaft_elements if s.k is not None and s.k < 1e11]))
            },
            "natural_frequencies_hz": [float(f) for f in natural_freqs_hz],
            "forced_response": {
                "frequency_range_rpm": [float(rpm_start), float(rpm_stop)],
                "num_points": int(n_steps),
                "excitation_nodes": list(excitations.keys()),
                "excitation_harmonics_orders": plotted_harmonics,
                "max_vibratory_torque_nm": float(np.max(np.abs(VT_sum_result))),
                "vibratory_torque_data": {
                    "frequencies_rpm": [float(f) for f in rpm_sweep],
                    "torque_per_shaft_nm": [[float(val) for val in np.abs(VT_sum_result[i, :])]
                                             for i in range(VT_sum_result.shape[0])]
                }
            },
            "campbell_diagram": {
                "rpm_range": [float(rpm_range[0]), float(rpm_range[1])],
                "plotted_harmonics": plotted_harmonics,
                "natural_frequencies_hz": [float(f) for f in natural_freqs_hz]
            }
        }

        results_file = 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")