import numpy as np
import opentorsion as ot


def build_drivetrain():
    motor = ot.Disk(0, I=2.5, c=15.0)
    coupling = ot.Disk(1, I=0.4, c=6.0)
    gear_output = ot.Disk(2, I=0.8, c=8.0)
    prop = ot.Disk(3, I=18.0, c=45.0)

    flex = ot.Shaft(0, 1, k=2.2e6, c=1200.0)
    tail = ot.Shaft(2, 3, k=9.0e5, c=950.0)

    pinion = ot.Gear(1, I=0.05, R=25.0)
    bullgear = ot.Gear(2, I=0.07, R=75.0, parent=pinion)

    return ot.Assembly(
        shaft_elements=[flex, tail],
        disk_elements=[motor, coupling, gear_output, prop],
        gear_elements=[pinion, bullgear],
    )


def map_physical_to_reduced(assembly, U_phys):
    if assembly.gear_elements is None:
        T = np.eye(assembly.dofs)
    else:
        T = assembly.T(assembly.E())
    return T.T @ U_phys


def frequency_sweep():
    assembly = build_drivetrain()

    rpm = np.linspace(300, 1500, 25)
    omegas = 2 * np.pi * rpm / 60.0

    excitation = ot.PeriodicExcitation(assembly.M.shape[0], omegas)

    U_phys = np.zeros((assembly.dofs, len(omegas)), dtype=complex)
    torque_amp = 600.0 * np.ones_like(omegas)
    U_phys[0, :] = torque_amp  # motor side torque

    excitation.U = map_physical_to_reduced(assembly, U_phys)

    _, torque_sum = assembly.vibratory_torque(excitation)
    return rpm, torque_sum


if __name__ == "__main__":
    drivetrain = build_drivetrain()
    wn, _, _ = drivetrain.modal_analysis()
    print("First natural frequencies (Hz):", (wn / (2 * np.pi))[:6])

    rpm, torque = frequency_sweep()
    print("Peak shaft torques (Nm):", np.abs(torque))
