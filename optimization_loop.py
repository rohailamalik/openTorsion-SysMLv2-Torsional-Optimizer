import json
import numpy as np
import os
import sys
from opentorsion.assembly import Assembly
from opentorsion.elements.disk_element import Disk
from opentorsion.elements.shaft_element import Shaft
from opentorsion.elements.gear_element import Gear
from opentorsion.excitation import PeriodicExcitation

def assemble_system_inline(template, coupling_data, coupling_option):
    """
    Creates a complete system by combining template with a specific coupling option.
    Returns the assembled system dict.
    """
    # Make a copy of the template
    system = json.loads(json.dumps(template))

    # Extract coupling properties
    coupling_name = coupling_option.get("name")
    stiffness = coupling_option.get("torsionalStiffness") * 1000  # Convert kN·m/rad to N·m/rad
    damping = coupling_option.get("relativeDamping")

    # Extract family-level properties
    coupling_family = coupling_data.get("family")
    coupling_type = coupling_data.get("type")
    coupling_size = coupling_data.get("size")
    inertia_1 = coupling_data.get("inertia_1")
    mass_1 = coupling_data.get("mass_1")
    inertia_2 = coupling_data.get("inertia_2")
    mass_2 = coupling_data.get("mass_2")
    max_speed = coupling_data.get("maxSpeed")

    # Find coupling component
    coupling_component = next((c for c in system['components'] if c['name'] == 'Coupling_Component'), None)
    if coupling_component is None:
        raise ValueError("Could not find 'Coupling_Component' in template.")

    # Create coupling elements
    hub1_disk = {
        "name": "Coupling_Hub1",
        "type": "Disk",
        "index": 1,
        "inertia": inertia_1 if inertia_1 is not None else 0.0,
        "damping": 0.0,
        "stiffness": 0.0,
        "mass": mass_1,
        "coupling_option_name": coupling_name,
        "coupling_family": coupling_family,
        "coupling_type": coupling_type,
        "coupling_size": coupling_size
    }

    flexible_element = {
        "name": "Coupling_FlexElement",
        "type": "ShaftDiscrete",
        "damping": damping,
        "stiffness": stiffness,
        "shoreHardness": coupling_option.get("shoreHardness"),
        "nominalTorque": coupling_option.get("nominalTorque"),
        "maxTorque": coupling_option.get("maxTorque"),
        "continuousVibratoryTorque": coupling_option.get("continuousVibratoryTorque"),
    }

    hub2_disk = {
        "name": "Coupling_Hub2",
        "type": "Disk",
        "index": 2,
        "inertia": inertia_2 if inertia_2 is not None else 0.0,
        "damping": 0.0,
        "stiffness": 0.0,
        "mass": mass_2,
        "gear_radius": 25.0,
        "maxSpeed": max_speed
    }

    coupling_component['elements'] = [hub1_disk, flexible_element, hub2_disk]

    # Add shaft connections
    motor_shaft = {
        "name": "Motor_Shaft",
        "nodes": [0, 1],
        "type": "ShaftFromElement",
        "element_name": "Motor_Component.Motor_Shaft_Element"
    }

    flex_shaft = {
        "name": "Coupling_FlexElement_Shaft",
        "nodes": [1, 2],
        "type": "ShaftFromElement",
        "element_name": "Coupling_Component.Coupling_FlexElement"
    }

    system['structure'] = [motor_shaft, flex_shaft] + system['structure']

    return system

def parse_and_analyze(system_data):
    """
    Parse system JSON and calculate max vibratory torque.
    Returns max vibratory torque in Nm.
    """
    # Parse system (simplified version of parse2.py logic)
    disk_elements = []
    shaft_elements = []
    gear_elements = []
    excitations = {}

    element_map = {}
    gear_data_by_node = {}
    gear_objects_by_node = {}

    # Pass 1: Create Disks
    for component in system_data.get('components', []):
        comp_name = component.get('name', 'Unknown_Component')
        for element in component.get('elements', []):
            el_name = element.get('name', 'Unknown_Element')
            full_name = f"{comp_name}.{el_name}"
            element_map[full_name] = element

            el_type = element.get('type')

            if el_type == 'Disk':
                node_index = element.get('index')
                I = element.get('inertia', 0.0)
                c = element.get('damping', 0.0)
                k = element.get('stiffness', 0.0)

                disk_elements.append(Disk(node=node_index, I=I, c=c, k=k))

                if element.get('excitation'):
                    excitations[node_index] = element['excitation']['values']

                if element.get('gear_radius'):
                    gear_data_by_node[node_index] = {
                        'radius': element['gear_radius'],
                        'parent_node': element.get('parent_gear_node'),
                        'inertia': I
                    }

    # Pass 2: Create Gear elements
    for node, gear_data in gear_data_by_node.items():
        if gear_data.get('parent_node') is None:
            gear_obj = Gear(node=node, I=gear_data['inertia'], R=gear_data['radius'], parent=None)
            gear_objects_by_node[node] = gear_obj
            gear_elements.append(gear_obj)

    for node, gear_data in gear_data_by_node.items():
        parent_node = gear_data.get('parent_node')
        if parent_node is not None:
            if parent_node in gear_objects_by_node:
                parent_gear = gear_objects_by_node[parent_node]
                gear_obj = Gear(node=node, I=gear_data['inertia'], R=gear_data['radius'], parent=parent_gear)
                gear_objects_by_node[node] = gear_obj
                gear_elements.append(gear_obj)

    # Pass 3: Create Shaft elements
    for shaft_def in system_data.get('structure', []):
        node_a, node_b = shaft_def['nodes']

        if shaft_def['type'] == 'ShaftRigid':
            shaft_elements.append(Shaft(node_a, node_b, k=1e12, c=0.0))
        elif shaft_def['type'] == 'ShaftFromElement':
            el_name = shaft_def['element_name']
            if el_name in element_map:
                element = element_map[el_name]
                k = element['stiffness']
                c = element['damping']
                shaft_elements.append(Shaft(node_a, node_b, k=k, c=c))

    # Create assembly
    assembly = Assembly(shaft_elements, disk_elements, gear_elements if gear_elements else None)

    # Extract frequency sweep configuration
    freq_config = system_data.get('frequency_sweep', {})
    rpm_start = freq_config.get('start', 1.0)
    rpm_stop = freq_config.get('stop', 2000.0)
    n_steps = freq_config.get('points', 200)

    rpm_sweep = np.linspace(max(rpm_start, 1.0), rpm_stop, n_steps)

    # Get system matrices
    M, K = assembly.M, assembly.K
    C = assembly.C
    dofs = M.shape[0]

    # Prepare result storage
    num_shafts = len(assembly.shaft_elements)
    VT_sum_result = np.zeros((num_shafts, n_steps))

    # Run frequency sweep
    for i, rpm in enumerate(rpm_sweep):
        f_hz = rpm / 60.0

        all_omegas, all_amplitudes, all_phases, all_nodes = [], [], [], []

        for node, harmonics in excitations.items():
            for order, amplitude_nm in harmonics:
                omega = order * (2 * np.pi * f_hz)
                all_omegas.append(omega)
                all_amplitudes.append(amplitude_nm)
                all_phases.append(0.0)
                all_nodes.append(node)

        if all_omegas:
            unique_omegas = sorted(list(set(all_omegas)))
            excitation = PeriodicExcitation(dofs, unique_omegas)

            for node, omega, amplitude, phase in zip(all_nodes, all_omegas, all_amplitudes, all_phases):
                omega_idx = unique_omegas.index(omega)
                excitation.add_sines(node, [omega], [amplitude], [phase])

            _, T_vib_sum = assembly.vibratory_torque(excitation, C=C)
            VT_sum_result[:, i] = np.ravel(T_vib_sum)

    # Calculate max vibratory torque across all shafts and all RPM
    max_vibratory_torque = float(np.max(np.abs(VT_sum_result)))

    return max_vibratory_torque

def main():
    print("="*70)
    print("Coupling Optimization Loop")
    print("="*70)

    # Load files
    TEMPLATE_FILE = './placeHolder.json'
    COUPLING_DB_FILE = './all_coupllings.json'
    RESULTS_DIR = './results'
    RESULTS_FILE = os.path.join(RESULTS_DIR, 'optimization_results.json')

    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")

    print(f"\nLoading template: {TEMPLATE_FILE}")
    with open(TEMPLATE_FILE, 'r') as f:
        template = json.load(f)

    print(f"Loading coupling database: {COUPLING_DB_FILE}")
    with open(COUPLING_DB_FILE, 'r') as f:
        all_couplings = json.load(f)

    total_families = len(all_couplings)
    total_options = sum(len(coupling['options']) for coupling in all_couplings)
    print(f"\nFound {total_families} coupling families with {total_options} total options")

    # Run optimization loop
    results = []
    count = 0

    print("\nStarting analysis loop...")
    print("-"*70)

    for coupling_family in all_couplings:
        family_name = coupling_family.get('name')

        for option in coupling_family['options']:
            count += 1
            option_name = option.get('name')

            print(f"[{count}/{total_options}] Analyzing: {family_name} ({option_name})...", end=' ')

            try:
                # Assemble system
                system = assemble_system_inline(template, coupling_family, option)

                # Run analysis
                max_vib_torque = parse_and_analyze(system)

                # Store result (convert catalog kN·m values to N·m)
                result = {
                    'coupling_family': family_name,
                    'coupling_type': coupling_family.get('type'),
                    'coupling_size': coupling_family.get('size'),
                    'option': option_name,
                    'shore_hardness': option.get('shoreHardness'),
                    'stiffness': option.get('torsionalStiffness') * 1000,  # kN·m/rad → N·m/rad
                    'damping': option.get('relativeDamping'),
                    'max_vibratory_torque_nm': max_vib_torque,
                    'continuous_vibratory_torque_limit_nm': option.get('continuousVibratoryTorque') * 1000,  # kN·m → N·m
                    'max_torque_limit_nm': option.get('maxTorque') * 1000  # kN·m → N·m
                }

                results.append(result)
                print(f"Max VT: {max_vib_torque:.2f} Nm ✓")

            except Exception as e:
                print(f"FAILED - {str(e)}")
                continue

    print("-"*70)
    print(f"\nCompleted analysis of {len(results)}/{total_options} coupling options")

    # Sort by max vibratory torque (ascending - lowest is best)
    results_sorted = sorted(results, key=lambda x: x['max_vibratory_torque_nm'])

    # Save all results
    output_data = {
        'total_analyzed': len(results),
        'analysis_parameters': {
            'rpm_range': [template['frequency_sweep']['start'], template['frequency_sweep']['stop']],
            'points': template['frequency_sweep']['points']
        },
        'all_results': results_sorted
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAll results saved to: {RESULTS_FILE}")

    # Display top 5
    print("\n" + "="*70)
    print("TOP 5 BEST COUPLINGS (Lowest Vibratory Torque)")
    print("="*70)

    for i, result in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. {result['coupling_family']} ({result['option']})")
        print(f"   Size: {result['coupling_size']}, Type: {result['coupling_type']}")
        print(f"   Stiffness: {result['stiffness']:.0f} N·m/rad, Damping: {result['damping']}")
        print(f"   Max Vibratory Torque: {result['max_vibratory_torque_nm']:.2f} Nm")
        print(f"   Limit: {result['continuous_vibratory_torque_limit_nm']:.0f} Nm")

        # Check if within limits
        if result['max_vibratory_torque_nm'] < result['continuous_vibratory_torque_limit_nm']:
            margin = (1 - result['max_vibratory_torque_nm'] / result['continuous_vibratory_torque_limit_nm']) * 100
            print(f"   Status: ✓ PASS (Safety margin: {margin:.1f}%)")
        else:
            print(f"   Status: ✗ EXCEEDS LIMIT")

    print("\n" + "="*70)
    print("Optimization complete!")
    print("="*70)

if __name__ == "__main__":
    main()
