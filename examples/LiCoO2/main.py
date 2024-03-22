import numpy as np

from mustard import Mustard, utils

LMB = 20
GAMMA = 0.1

DIST_CUTOFF = 3.0


def coupling(Q, lmb=LMB, gamma=GAMMA):
    return lmb * np.exp(-gamma * Q**2)


def coupling_value_function(rxn_ids, snapshot):
    snapshot.h_idx = snapshot.atoms[rxn_ids["H"]].idx
    h_pos = snapshot.frame.pos[snapshot.h_idx]
    snapshot.y_idx = snapshot.atoms[rxn_ids["Y"]].idx
    y_pos = snapshot.frame.pos[snapshot.y_idx]
    snapshot.dQpos = utils.get_distance_xyz(h_pos, y_pos, snapshot.frame.box_vectors)
    snapshot.Q = utils.get_distances(snapshot.dQpos)
    snapshot.cpl = coupling(snapshot.Q)
    return snapshot.cpl


def coupling_forces_function(rxn_ids, snapshot, initial_forces, new_forces):
    cpl_forces = np.zeros(new_forces.shape)
    dcpl_dQ = -GAMMA * 2 * snapshot.Q * snapshot.cpl
    dcpl_dR = dcpl_dQ / snapshot.Q
    fcpl_R = -dcpl_dR * snapshot.dQpos
    cpl_forces[snapshot.h_idx] += fcpl_R
    cpl_forces[snapshot.y_idx] -= fcpl_R
    return cpl_forces


def main():
    lmp_coord_file = "coord.lmp"
    force_field_file = "ff.lmp"
    temperature = 300
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
        # "fermi_mixing": True,
        "shells": 3,
        "reactions": [
            {
                "reaction": (None, "Co2", "Co3"),
                "type_changes0": {
                    None: None,
                    "Co2": "Co3",
                    "Co3": "Co2",
                },
                "type_changes1": {},
                "cutoffs": {
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_value_function": coupling_value_function,
                "coupling_forces_function": coupling_forces_function,
            }
        ],
    }
    mpi_list = np.ones(32) * 1
    msevb = Mustard(
        commands,
        inputs,
        mpi_list=mpi_list,
        debug=True,
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=1)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(filename=None, write_frequency=100)
    msevb.add_output(filename="mustard.log", write_frequency=10)
    # msevb.finite_differences(index_array=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    # msevb.minimise(bound=0.1)
    # msevb.step(250)


if __name__ == "__main__":
    main()
