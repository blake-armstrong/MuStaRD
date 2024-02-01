import numpy as np

from mustard import Mustard, utils

VCONST = -1.00549507  # eV
GAMMA = 1.8302895  # A^-2
P = 0.2327260  # dimless
K = 9.562153  # A^-2
DOO = 2.94  # A
BETA = 6.0179066  # A^-1
ALPHA = 10.0380922  # A^-1
ROO0 = 3.1  # A
PP = 10.8831327  # dimless (paper says A^-1, which doesn't make sense)
ROO02 = 1.8136426  # A

DIST_CUTOFF = 2.5  # A
CF = 14.399645  # qq/r (e^2/A) -> eV


def Voth2007_coupling(V_ex, A_val, V_const=VCONST):
    return (V_const + V_ex) * A_val


def coupling_value_function(
    rxn_ids, snapshot, new_pe, initial_pe, new_compute, initial_compute
):
    h_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["h_id"]].idx]
    x_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["x_id"]].idx]
    y_idx = snapshot.atoms[rxn_ids["y_id"]].idx
    y_pos = snapshot.frame.pos[y_idx]
    xyz_pbc = snapshot.frame.xyz_pbc
    snapshot.dRoopos = x_pos - y_pos
    snapshot.dRoopos -= xyz_pbc * (snapshot.dRoopos / xyz_pbc).round()
    snapshot.Roo = np.linalg.norm(snapshot.dRoopos, axis=-1)
    snapshot.dqpos = 0.5 * (x_pos + y_pos) - h_pos
    snapshot.dqpos -= xyz_pbc * (snapshot.dqpos / xyz_pbc).round()
    snapshot.q = np.linalg.norm(snapshot.dqpos, axis=-1)
    group1_ids = np.concatenate(
        [
            snapshot.residues[snapshot.atoms[rxn_ids["h_id"]].molecule],
            snapshot.residues[snapshot.atoms[rxn_ids["y_id"]].molecule],
        ]
    )
    snapshot.group1_idxs = np.array([snapshot.atoms[eyed].idx for eyed in group1_ids])
    snapshot.group2_idxs = np.delete(
        np.arange(len(snapshot.frame.pos)), snapshot.group1_idxs
    )
    group1_pos = snapshot.frame.pos[snapshot.group1_idxs]
    group1_qs = snapshot.qs[snapshot.group1_idxs]
    group2_pos = snapshot.frame.pos[snapshot.group2_idxs]
    group2_qs = snapshot.qs[snapshot.group2_idxs]
    dists, dxs = utils.get_distances(group1_pos, group2_pos, xyz_pbc, dx=True)
    dxr = dxs / dists[:, :, None]
    q_prod = group1_qs[:, None] * group2_qs[None, :]
    q_prod_r = q_prod / dists
    V_ex = np.sum(q_prod_r) * CF
    snapshot.ds = dxr * -(q_prod_r / dists)[:, :, None]
    snapshot.V_ex = V_ex

    snapshot.term1 = np.exp(-GAMMA * snapshot.q**2)
    snapshot.term2 = P * np.exp(-K * (snapshot.Roo - DOO) ** 2)
    snapshot.term3 = BETA * (snapshot.Roo - ROO0)
    snapshot.term4 = PP * np.exp(-ALPHA * (snapshot.Roo - ROO02))
    snapshot.A_val = (
        snapshot.term1
        * (1 + snapshot.term2)
        * (0.5 * (1 - np.tanh(snapshot.term3)) + snapshot.term4)
    )
    snapshot.c = Voth2007_coupling(V_ex, snapshot.A_val)
    return snapshot.c


def sech2(x):
    return 1 - (np.tanh(x)) ** 2


def coupling_forces_function(rxn_ids, snapshot, computes, forces):
    dA_dq = -2 * GAMMA * snapshot.A_val
    dA_dq_H = -1 * dA_dq * snapshot.dqpos
    dA_dq_O = 0.5 * dA_dq * snapshot.dqpos
    dA_dr_term1 = -2 * K * (snapshot.Roo - DOO) * snapshot.term2
    dA_dr_term2 = (
        -0.5 * BETA * sech2(BETA * (snapshot.Roo - ROO0)) - ALPHA * snapshot.term4
    )
    dA_dr = snapshot.term1 * (
        (dA_dr_term1 * (0.5 * (1 - np.tanh(snapshot.term3)) + snapshot.term4))
        + (dA_dr_term2 * (1 + snapshot.term2))
    )
    dA_dx = dA_dr * snapshot.dRoopos / snapshot.Roo
    group1_deriv = np.sum(snapshot.ds, axis=1) * CF
    group2_deriv = -1 * np.sum(snapshot.ds, axis=0) * CF
    # V_ex derivs
    V_ex_derivs = np.zeros(shape=snapshot.frame.pos.shape)
    V_ex_derivs[snapshot.group1_idxs] += group1_deriv
    V_ex_derivs[snapshot.group2_idxs] += group2_deriv

    # q derivs
    q_derivs = np.zeros(shape=snapshot.frame.pos.shape)
    q_derivs[snapshot.atoms[rxn_ids["h_id"]].idx] += dA_dq_H
    q_derivs[snapshot.atoms[rxn_ids["y_id"]].idx] += dA_dq_O
    q_derivs[snapshot.atoms[rxn_ids["x_id"]].idx] += dA_dq_O
    # Roo derivs
    Roo_derivs = np.zeros(shape=snapshot.frame.pos.shape)
    Roo_derivs[snapshot.atoms[rxn_ids["y_id"]].idx] -= dA_dx
    Roo_derivs[snapshot.atoms[rxn_ids["x_id"]].idx] += dA_dx

    full_derivs = (
        VCONST * (q_derivs + Roo_derivs)
        + V_ex_derivs * snapshot.A_val
        + snapshot.V_ex * (q_derivs + Roo_derivs)
    )
    return full_derivs * -1


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
        "temperature": TEMPERATURE,
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
        "fermi_mixing": True,
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

    # mpi_list = list(np.ones(24))
    mpi_list = [1, 1, 1, 1]
    msevb = Mustard(
        lmp_coord_file,
        force_field_file,
        header,
        commands,
        INPUTS,
        mpi_list=mpi_list,
        debug=True,
    )

    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=100)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=100, rxn=True)
    msevb.add_output(filename=None, write_frequency=100)
    msevb.add_output(filename="mustard.log", write_frequency=10)
    # msevb.msevb_minimise()
    # msevb.finite_differences(
    #     file="voth_finite_differences_multishell.dat",
    #     delta=1e-3,
    # )

    msevb.step(1)


if __name__ == "__main__":
    main()
