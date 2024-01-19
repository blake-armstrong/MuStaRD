import numpy as np

from mustard import Mustard

V12 = 3.156906788803108
# V12 = 4.436163134
ALPHA = 0.85
# ALPHA = 0.80
GAMMA = 0
# GAMMA = 0.80
DIST_CUTOFF = 1.9


def calc_coupling_dists(H_pos, X_pos, Y_pos, xyz_pbc):
    # O-O distance
    dQpos = X_pos - Y_pos
    dQpos -= xyz_pbc * (dQpos / xyz_pbc).round()
    Q = np.linalg.norm(dQpos, axis=-1)
    # distance between the position of the transferring proton and the middle of the O–O distance
    dqpos = H_pos - ((dQpos / 2) + Y_pos)
    dqpos -= xyz_pbc * (dqpos / xyz_pbc).round()
    q = np.linalg.norm(dqpos, axis=-1)
    return Q, q


def get_dist(pos1, pos2, pbc):
    d_pos = pos1 - pos2
    d_pos -= pbc * (d_pos / pbc).round()
    dist = np.linalg.norm(d_pos, axis=-1)
    return dist


def Vuilleumier1998_coupling(Q, q, v12=V12, alpha=ALPHA, gamma=GAMMA):
    # https://doi.org/10.1016/S0009-2614(97)01365-1
    return v12 * np.exp(-alpha * Q - gamma * q**2)


def coupling_value_function(
    rxn_ids, snapshot, new_pe, initial_pe, new_compute, initial_compute
):
    h_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["h_id"]].idx]
    x_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["x_id"]].idx]
    y_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["y_id"]].idx]
    xyz_pbc = snapshot.frame.xyz_pbc
    dQpos = x_pos - y_pos
    dQpos -= xyz_pbc * (dQpos / xyz_pbc).round()
    Q = np.linalg.norm(dQpos, axis=-1)
    # distance between the position of the transferring proton and the middle of the O–O distance
    dqpos = h_pos - 0.5 * (x_pos + y_pos)
    dqpos -= xyz_pbc * (dqpos / xyz_pbc).round()
    q = np.linalg.norm(dqpos, axis=-1)

    return Vuilleumier1998_coupling(Q, q)


def coupling_forces_function(rxn_ids, snapshot, computes, forces):
    cpl_forces = np.zeros(shape=forces.shape)
    h_idx = snapshot.atoms[rxn_ids["h_id"]].idx
    h_pos = snapshot.frame.pos[h_idx]
    x_idx = snapshot.atoms[rxn_ids["x_id"]].idx
    x_pos = snapshot.frame.pos[x_idx]
    y_idx = snapshot.atoms[rxn_ids["y_id"]].idx
    y_pos = snapshot.frame.pos[y_idx]
    xyz_pbc = snapshot.frame.xyz_pbc
    dQpos = x_pos - y_pos
    dQpos -= xyz_pbc * (dQpos / xyz_pbc).round()
    Q = np.linalg.norm(dQpos, axis=-1)
    dqpos = h_pos - 0.5 * (x_pos + y_pos)
    dqpos -= xyz_pbc * (dqpos / xyz_pbc).round()
    q = np.linalg.norm(dqpos, axis=-1)
    cpl = Vuilleumier1998_coupling(Q, q)
    derivOO = -ALPHA * cpl * dQpos.flatten() / Q
    derivHOO = -GAMMA * 2 * cpl * dqpos.flatten()
    fOO = -derivOO
    fHOO = derivHOO
    cpl_forces[x_idx] += fOO
    cpl_forces[y_idx] -= fOO
    cpl_forces[h_idx] -= fHOO
    cpl_forces[x_idx] += 0.5 * fHOO
    cpl_forces[y_idx] += 0.5 * fHOO
    return cpl_forces


def main():
    lmp_coord_file = "coord.lmp"
    force_field_file = "ff.lmp"
    header = ["units metal", "atom_style full", "boundary p p p"]

    TEMPERATURE = 300
    TIMESTEP = 1e-3
    r1 = 38472
    r2 = 2837
    commands = [
        "fix md all nve",
        f"fix tst all temp/csvr {TEMPERATURE} {TEMPERATURE} 0.1 {r1}",
        f"timestep {TIMESTEP}",
        f"velocity all create {TEMPERATURE} {r2} mom yes dist gaussian",
        "fix com all momentum 100 linear 1 1 1",
    ]

    INPUTS = {
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

    # msevb = mustard.Mustard(lmp_coord_file, force_field_file, coupling_function, header, commands, INPUTS, msevb_mpi_ranks=[11,11,11,11])
    # mpi_list = list(np.ones(32))
    mpi_list = [4, 4, 4, 4, 4]
    msevb = Mustard(
        lmp_coord_file,
        force_field_file,
        header,
        commands,
        INPUTS,
        debug=True,
        mpi_list=mpi_list,
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=100)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(filename=None, write_frequency=100)
    msevb.add_output(filename="mustard.log", write_frequency=10)
    # msevb.minimise()
    # msevb.finite_differences(file="new_fd.out", delta=1e-3)
    msevb.finite_differences(file="new_fd.out", delta=1e-3, index_array=[0, 1, 2, 3])

    # msevb.step(10000)


if __name__ == "__main__":
    main()
