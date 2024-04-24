import numpy as np

from mustard import Mustard
from mustard.coupling import Raiteri2011

LMB = 0.7998
ZETA = 16

DIST_CUTOFF = 1.8
DIST_TAPER = 1.7

coupling = Raiteri2011(
    lmb=LMB, zeta=ZETA, dist_cutoff=DIST_CUTOFF, dist_taper=DIST_TAPER
)


def main():
    lmp_coord_file = "8.lmp"
    force_field_file = "ff.lmp"
    temperature = 1500
    timestep = 1e-3
    r1 = np.random.randint(1, 99999)
    r2 = np.random.randint(1, 99999)

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

    inputs = {
        "temperature": temperature,
        "constant_volume": True,
        "atom_types": {
            "Ba": 1,
            "Zr": 2,
            "Y": 3,
            "O1": 4,
            "O2": 5,
            "O3": 6,
            "H1": 7,
            "O4": 8,
        },
        "bond_types": {"O1-H1": 1, "O4-H1": 2},
        "type_charges": {
            "Ba": 2.000000,
            "Zr": 4.000000,
            "Y": 3.000000,
            "O1": -1.308698,
            "O2": -2.000000,
            "O3": -2.000000,
            "H1": 0.308698,
            "O4": -1.308698,
        },
        "neighbour_list_update": 10,
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
                    # "O1": "O2",
                    # "H1": "H1",
                    # "O2": "O1",
                },
                "cutoffs": {
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_function": coupling,
            },
            {
                "reaction": ("O1", "H1", "O3"),
                "type_changes0": {
                    "O1": "O2",
                    "H1": "H1",
                    "O3": "O4",
                },
                "type_changes1": {
                    # "O1": "O2",
                    # "H1": "H1",
                    # "O3": "O4",
                },
                "cutoffs": {
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_function": coupling,
            },
            {
                "reaction": ("O4", "H1", "O2"),
                "type_changes0": {
                    "O4": "O3",
                    "H1": "H1",
                    "O2": "O1",
                },
                "type_changes1": {
                    # "O1": "O2",
                    # "H1": "H1",
                    # "O3": "O4",
                },
                "cutoffs": {
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_function": coupling,
            },
            {
                "reaction": ("O4", "H1", "O3"),
                "type_changes0": {
                    "O4": "O3",
                    "H1": "H1",
                    "O3": "O4",
                },
                "type_changes1": {
                    # "O1": "O2",
                    # "H1": "H1",
                    # "O3": "O4",
                },
                "cutoffs": {
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_function": coupling,
            },
        ],
    }
    mpi_list = [5, 5]
    msevb = Mustard(
        commands,
        inputs,
        debug=True,
        mpi_list=mpi_list,
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=1000)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=1000, rxn=True)
    msevb.add_output(
        filename=None, properties=["temp", "pe", "ke"], write_frequency=1000
    )
    msevb.add_output(
        filename="mustard.log",
        properties=["temp", "pe", "ke"],
        write_frequency=1000,
    )
    # msevb.minimise()
    msevb.finite_differences(file="tmp_fd.out", delta=1e-3, index_array=[0, 1, 2, 3])
    # msevb.step(10000)


if __name__ == "__main__":
    main()
