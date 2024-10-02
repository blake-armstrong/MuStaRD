from mustard import Mustard
from mustard.coupling import BaseCoupling

import numpy as np

DIST_CUTOFF = 3.5


def main():
    lmp_coord_file = "coord.lmp"
    force_field_file = "ff.lmp"
    temperature = 300
    timestep = 1e-3
    r1, r2 = np.random.randint(1, 99999), np.random.randint(1, 99999)

    commands = [
        "units metal",
        "atom_style full",
        "boundary p p p",
        f"read_data {lmp_coord_file}",
        # f"read_data {lmp_coord_file} extra/dihedral/per/atom 2 extra/special/per/atom 10",
        f"change_box all triclinic",
        f"include {force_field_file}",
        "fix md all nve",
        f"fix tst all temp/csvr {temperature} {temperature} 0.1 {r1}",
        f"timestep {timestep}",
        f"velocity all create {temperature} {r2} mom yes dist gaussian",
        "fix com all momentum 100 linear 1 1 1",
    ]

    coupling = BaseCoupling()

    inputs = {
        "temperature": temperature,
        "constant_volume": True,
        "atom_types": {"Li": 1, "Co2": 2, "Co3": 3, "Co4": 4, "O": 5},
        "bond_types": {},
        "angle_types": {},
        "improper_types": {},
        "proper_types": {},
        "type_charges": {
            "Li": 0.600000,
            "Co2": 1.200000,
            "Co3": 1.800000,
            "Co4": 2.400000,
            "O": -1.200000,
        },
        "lammps_unit_system": "metal",
        "neighbour_list_update": 10,
        "shells": 1,
        "fermi_mixing": True,
        "reactions": [
            {
                "reaction": (None, "Co4", "Co3"),
                "type_changes0": {
                    None: None,
                    "Co4": "Co3",
                    "Co3": "Co4",
                },
                "type_changes1": {},
                "cutoffs": {
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_function": coupling,
            }
        ],
    }
    mpi_list = np.ones(10) * 1
    msevb = Mustard(
        commands,
        inputs,
        # debug=True,
        mpi_list=mpi_list,
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=1)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(filename=None, write_frequency=100)
    msevb.add_output(filename="mustard.log", write_frequency=10)
    # msevb.finite_differences(index_array=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    # msevb.minimise(bound=0.2, full=False)
    # msevb.step(250)


if __name__ == "__main__":
    main()
