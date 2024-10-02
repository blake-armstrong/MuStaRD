from mustard import Mustard
from mustard.coupling import Raiteri2011
import random


def main():
    lmp_coord_file = "coord.lmp"
    force_field_file = "ff.lmp"
    temperature = 1500  # Kelvin
    timestep = 1e-3  # ps
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

    coupling = Raiteri2011()
    dist_cutoff = 1.8  # Angstroms
    dist_taper = 1.7  # Angstroms
    coupling.add_taper(cutoff=dist_cutoff, taper=dist_taper)

    inputs = {
        "temperature": temperature,
        "constant_volume": True,
        "atom_types": {"Ba": 1, "Zr": 2, "Y": 3, "O1": 4, "O2": 5, "O3": 6, "H1": 7},
        "bond_types": {"O1-H1": 1},
        "type_charges": {
            "Ba": 2.000000,
            "Zr": 4.000000,
            "Y": 3.000000,
            "O1": -1.308698,
            "O2": -2.000000,
            "O3": -2.000000,
            "H1": 0.308698,
        },
        "neighbour_list_update": 10,
        "topology_update": 1,
        "lammps_unit_system": "metal",
        "reactions": [
            {
                "reaction": ("O1", "H1", "O2"),
                "type_changes0": {
                    "O1": "O2",
                    "H1": "H1",
                    "O2": "O1",
                },
                "type_changes1": {
                    "O1": "O2",
                    "H1": "H1",
                    "O2": "O1",
                },
                "cutoffs": {
                    "distance": dist_cutoff,
                    "angle": None,
                },
                "coupling_function": coupling,
            }
        ],
    }
    mpi_list = [4, 4]
    msevb = Mustard(
        commands,
        inputs,
        mpi_list=mpi_list,
        # debug=True,
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=100)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(
        filename=None, properties=["temp", "pe", "ke"], write_frequency=100
    )
    msevb.add_output(
        filename="mustard.log",
        properties=["temp", "pe", "ke"],
        write_frequency=100,
    )
    # msevb.minimise()
    # msevb.finite_differences(file="fd.out", delta=1e-3, index_array=[0, 1, 2, 3])
    msevb.step(1000)
    # msevb.lmp.command(f"write_restart lammps.{msevb.universe.rank.color}.restart")


if __name__ == "__main__":
    main()
