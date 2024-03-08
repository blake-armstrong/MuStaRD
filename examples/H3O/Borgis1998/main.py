import numpy as np

from mustard import Mustard, utils

V12 = 3.156906788803108
ALPHA = 0.85
GAMMA = 0
# V12 = 4.436163134
# ALPHA = 0.80
# GAMMA = 0.80
DIST_CUTOFF = 1.9


#def calc_coupling_dists(H_pos, X_pos, Y_pos, xyz_pbc):
#    # O-O distance
#    dQpos = X_pos - Y_pos
#    dQpos -= xyz_pbc * (dQpos / xyz_pbc).round()
#    Q = np.linalg.norm(dQpos, axis=-1)
#    # distance between the position of the transferring proton and the middle of the O–O distance
#    dqpos = H_pos - ((dQpos / 2) + Y_pos)
#    dqpos -= xyz_pbc * (dqpos / xyz_pbc).round()
#    q = np.linalg.norm(dqpos, axis=-1)
#    return Q, q
#
#
#def get_dist(pos1, pos2, pbc):
#    d_pos = pos1 - pos2
#    d_pos -= pbc * (d_pos / pbc).round()
#    dist = np.linalg.norm(d_pos, axis=-1)
#    return dist


def Vuilleumier1998_coupling(Q, q, v12=V12, alpha=ALPHA, gamma=GAMMA):
    # https://doi.org/10.1016/S0009-2614(97)01365-1
    return v12 * np.exp(-alpha * Q - gamma * q**2)


def coupling_value_function(
    rxn_ids, snapshot
):
    snapshot.h_idx = snapshot.atoms[rxn_ids["H"]].idx
    snapshot.x_idx = snapshot.atoms[rxn_ids["X"]].idx
    snapshot.y_idx = snapshot.atoms[rxn_ids["Y"]].idx
    h_pos = snapshot.frame.pos[snapshot.h_idx]
    x_pos = snapshot.frame.pos[snapshot.x_idx]
    y_pos = snapshot.frame.pos[snapshot.y_idx]
    snapshot.dQpos = utils.get_distance_xyz(x_pos, y_pos, snapshot.frame.box_vectors)
    snapshot.Q = utils.get_distances(snapshot.dQpos)
    centrexy_pos = 0.5 * (x_pos + y_pos)
    snapshot.dqpos = utils.get_distance_xyz(h_pos, centrexy_pos, snapshot.frame.box_vectors)
    q = utils.get_distances(snapshot.dqpos)
    snapshot.cpl = Vuilleumier1998_coupling(snapshot.Q, q)
    return snapshot.cpl


def coupling_forces_function(rxn_ids, snapshot, new_forces, initial_forces):
    cpl_forces = np.zeros(shape=initial_forces.shape)
    derivOO = -ALPHA * snapshot.cpl * snapshot.dQpos.flatten() / snapshot.Q
    derivHOO = -GAMMA * 2 * snapshot.cpl * snapshot.dqpos.flatten()
    fOO = -derivOO
    fHOO = derivHOO
    cpl_forces[snapshot.x_idx] += fOO
    cpl_forces[snapshot.y_idx] -= fOO
    cpl_forces[snapshot.h_idx] -= fHOO
    cpl_forces[snapshot.x_idx] += 0.5 * fHOO
    cpl_forces[snapshot.y_idx] += 0.5 * fHOO
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
        "temperature": 300,
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
        "shells": 1,
        "neighbour_list_update": 4,
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
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_value_function": coupling_value_function,
                "coupling_forces_function": coupling_forces_function,
            }
        ],
    }

    mpi_list = [1, 1, 1, 1, 1]
    # mpi_list = [4, 4, 4, 4, 4]
    msevb = Mustard(
        commands,
        inputs,
        debug=True,
        mpi_list=mpi_list,
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=100)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(filename=None, write_frequency=100)
    msevb.add_output(filename="mustard.log", write_frequency=10)
    msevb.minimise()
    # msevb.finite_differences(file="new_fd.out", delta=1e-3)
    # msevb.finite_differences(file="new_fd.out", delta=1e-3, index_array=[0, 1, 2, 3, 4, 5, 6, 7])

    # msevb.step(10000)


if __name__ == "__main__":
    main()
