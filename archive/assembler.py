import json
import os

def assemble_system(template_path, coupling_data, coupling_option, output_path):
    """
    Combines a drivetrain template with a specific coupling option to create
    a complete system JSON file for analysis.

    Args:
        template_path (str): Path to the new, corrected drivetrain_template.json file.
        coupling_data (dict): The full coupling data dictionary containing 
                              family-level properties (inertia, mass, etc.)
        coupling_option (dict): A dictionary for a single coupling option
                                (e.g., the 'shore50' object).
        output_path (str): Path to save the final assembled system JSON file.
    """
    print(f"Assembling system for coupling option: {coupling_option.get('name', 'N/A')}...")

    # 1. Load the new, corrected drivetrain template
    with open(template_path, 'r') as f:
        template = json.load(f)

    # 2. Extract properties from the coupling data and selected option
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
    
    if coupling_name is None or stiffness is None or damping is None:
        raise ValueError("Coupling option dictionary is missing required keys (name, torsionalStiffness, or relativeDamping).")

    # 3. Find the coupling component placeholder
    coupling_component = next((c for c in template['components'] if c['name'] == 'Coupling_Component'), None)
    if coupling_component is None:
        raise ValueError("Could not find 'Coupling_Component' in template.")
        
    # 4. Create the three unambiguous coupling elements
    
    # Hub1 is a Disk at Node 1
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
    
    # The Flexible Element is a Shaft, it has NO index and NO inertia
    flexible_element = {
        "name": "Coupling_FlexElement",
        "type": "ShaftDiscrete",
        "damping": damping,
        "stiffness": stiffness,
        # Pass through all other properties for reference/constraints
        "shoreHardness": coupling_option.get("shoreHardness"),
        "nominalTorque": coupling_option.get("nominalTorque"),
        "maxTorque": coupling_option.get("maxTorque"),
        "continuousVibratoryTorque": coupling_option.get("continuousVibratoryTorque"),
        "permissiblePowerLoss": coupling_option.get("permissiblePowerLoss"),
        "permissibleAxialDisplacement": coupling_option.get("permissibleAxialDisplacement"),
        "axialStiffness": coupling_option.get("axialStiffness"),
        "permissibleRadialDisplacement": coupling_option.get("permissibleRadialDisplacement"),
        "radialStiffness": coupling_option.get("radialStiffness"),
        "permissibleAngularDisplacement": coupling_option.get("permissibleAngularDisplacement"),
        "angularStiffness": coupling_option.get("angularStiffness")
    }
    
    # Hub2 is a Disk at Node 2, and it's also the pinion
    hub2_disk = {
        "name": "Coupling_Hub2",
        "type": "Disk",
        "index": 2,
        "inertia": inertia_2 if inertia_2 is not None else 0.0,
        "damping": 0.0,
        "stiffness": 0.0,
        "mass": mass_2,
        "gear_radius": 25.0,  # The pinion radius
        "maxSpeed": max_speed
    }
    
    # Add these three elements to the component's 'elements' list
    coupling_component['elements'] = [hub1_disk, flexible_element, hub2_disk]
    
    # 5. Add the new shaft connections to the 'structure' list

    # Motor (0) --motor-shaft--> Hub1 (1)
    motor_shaft = {
      "name": "Motor_Shaft",
      "nodes": [ 0, 1 ],
      "type": "ShaftFromElement",
      "element_name": "Motor_Component.Motor_Shaft_Element"
    }
    
    # Hub1 (1) --flex-element--> Hub2 (2)
    flex_shaft = {
      "name": "Coupling_FlexElement_Shaft",
      "nodes": [ 1, 2 ],
      "type": "ShaftFromElement",
      "element_name": "Coupling_Component.Coupling_FlexElement"
    }
    
    # The Propeller_Shaft (3->4) is already in the template.
    # We just need to prepend the two new shafts.
    # This correctly creates the full chain: 0-1 (Motor), 1-2 (Flex), 3-4 (Propeller)
    # The 2-3 connection is a Gear, defined by parent_gear_node
    template['structure'] = [
        motor_shaft,
        flex_shaft
    ] + template['structure']

    # 6. Save the final, assembled system JSON
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Successfully created assembled system file at: {output_path}")


# --- Example of how to use this function ---
if __name__ == '__main__':
    # Define file paths
    # *** USE THE NEW TEMPLATE FILE ***
    TEMPLATE_FILE = './placeHolder.json'
    COUPLING_DATA_FILE = './centa_coupling.json'
    OUTPUT_DIR = './analysis_cases'

    # Create an output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load the coupling data containing all options
    with open(COUPLING_DATA_FILE, 'r') as f:
        coupling_data = json.load(f)

    # --- This is the logic that will go inside your OPTIMIZER LOOP ---
    
    # Get the list of all available coupling options
    all_options = coupling_data.get('options', [])
    if not all_options:
        print("No options found in the coupling data file.")
    else:
        # Get the selected option name
        selected_option_name = coupling_data.get('selectedOption')
        
        if selected_option_name is None:
            print("No selectedOption specified in the coupling data file.")
        else:
            # Find the matching option in the options list
            selected_option = None
            for option in all_options:
                if option.get('name') == selected_option_name:
                    selected_option = option
                    break
            
            if selected_option is None:
                print(f"Could not find option '{selected_option_name}' in the options list.")
            else:
                # Define a unique output file name based on the option name
                output_filename = os.path.join(OUTPUT_DIR, f"system_{selected_option['name']}.json")
                
                # Call the assembler function
                assemble_system(
                    template_path=TEMPLATE_FILE,
                    coupling_data=coupling_data,
                    coupling_option=selected_option,
                    output_path=output_filename
                )