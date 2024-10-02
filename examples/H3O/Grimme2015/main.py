from mustard import Mustard
from mustard.coupling import Grimme2015
import random
import numpy as np


def main():
    lmp_coord_file = "../coord.lmp"
    force_field_file = "ff.lmp"
    temperature = 300
    timestep = 1e-3
    r1, r2 = random.randint(1, 99999), random.randint(1, 99999)

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

    a = 0.6
    b = 0.3
    coupling = Grimme2015(a=a, b=b)
    dist_cutoff = 2.5  # A
    dist_taper = 2.0  # A
    coupling.add_taper(dist_cutoff, dist_taper)

    inputs = {
        "temperature": temperature,
        "constant_volume": True,
        "atom_types": {"O2": 1, "H2": 2, "O3": 3, "H3": 4},
        "bond_types": {"O2-H2": 1, "O3-H3": 2},
        "angle_types": {"H2-O2-H2": 1, "H3-O3-H3": 2},
        "improper_types": {},
        "proper_types": {},
        "type_charges": {
            "O2": -0.820000,
            "H2": 0.410000,
            "O3": -0.500000,
            "H3": 0.500000,
        },
        "lammps_unit_system": "metal",
        "computes": None,
        "shells": 3,
        "neighbour_list_update": 5,
        "topology_update": 1,
        "pbc": True,
        "reactions": [
            {
                "reaction": ("O3", "H3", "O2"),
                "type_changes0": {
                    "O2": "O3",
                    "O3": "O2",
                    "H3": "H3",
                },
                "type_changes1": {
                    "O2": "O3",
                    "H2": "H3",
                    "O3": "O2",
                    "H3": "H2",
                },
                "cutoffs": {
                    "distance": dist_cutoff,
                    "angle": None,
                },
                "coupling_function": coupling,
            }
        ],
    }

    mpi_list = list(np.ones(48) * 1)
    msevb = Mustard(
        commands,
        inputs,
        # debug=True,
        mpi_list=mpi_list,
    )

    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=100)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(filename=None, write_frequency=100)
    msevb.add_output(filename="mustard.log", write_frequency=100)
    msevb.run_for_seconds(25)
    # msevb.lmp.command(f"write_restart lammps.{msevb.universe.rank.color}.restart")


if __name__ == "__main__":
    main()
