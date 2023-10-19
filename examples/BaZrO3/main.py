import numpy as np

from mustard import Mustard
from mustard.utils import MDF, dMDF

LMB = 0.7998
ZETA = 16

DIST_CUTOFF = 1.8
DIST_TAPER = 1.7


def get_dist(pos1, pos2, pbc):
    d_pos = pos1 - pos2
    d_pos -= pbc * (d_pos / pbc).round()
    dist = np.linalg.norm(d_pos, axis=-1)
    return dist


def Raiteri2011_coupling(Q):
    # 10.1088/0953-8984/23/33/334213
    return LMB * np.exp(-ZETA * Q**2)


def coupling_value_function(
    rxn_ids, snapshot, new_pe, initial_pe, new_compute, initial_compute
):
    h_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["h_id"]].idx]
    x_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["x_id"]].idx]
    y_pos = snapshot.frame.pos[snapshot.atoms[rxn_ids["y_id"]].idx]
    rHY = get_dist(h_pos, y_pos, snapshot.frame.xyz_pbc)
    rHX = get_dist(h_pos, x_pos, snapshot.frame.xyz_pbc)
    Q = abs(rHY - rHX)
    return Raiteri2011_coupling(Q) * MDF(rHY, DIST_TAPER, DIST_CUTOFF)


def coupling_forces_function(rxn_ids, snapshot, computes, forces):
    cpl_forces = np.zeros(shape=forces.shape)
    h_idx = snapshot.atoms[rxn_ids["h_id"]].idx
    h_pos = snapshot.frame.pos[h_idx]
    x_idx = snapshot.atoms[rxn_ids["x_id"]].idx
    x_pos = snapshot.frame.pos[x_idx]
    y_idx = snapshot.atoms[rxn_ids["y_id"]].idx
    y_pos = snapshot.frame.pos[y_idx]
    dHY = h_pos - y_pos
    dHY -= snapshot.frame.xyz_pbc * (dHY / snapshot.frame.xyz_pbc).round()
    rHY = np.linalg.norm(dHY, axis=-1)
    dHX = h_pos - x_pos
    dHX -= snapshot.frame.xyz_pbc * (dHX / snapshot.frame.xyz_pbc).round()
    rHX = np.linalg.norm(dHX, axis=-1)
    drHYHX = rHY - rHX
    Q = abs(drHYHX)
    cpl = Raiteri2011_coupling(Q)
    taper = MDF(rHY, DIST_TAPER, DIST_CUTOFF)
    taper_derivative = 0
    if rHY < DIST_CUTOFF and rHY > DIST_TAPER:
        _dHY = dHY.flatten()
        taper_derivative = dMDF(_dHY[0], _dHY[1], _dHY[2], DIST_TAPER, DIST_CUTOFF)
    prefactor = -2 * ZETA * cpl * drHYHX
    derivHY = prefactor * (dHY.flatten() / rHY) * taper + taper_derivative * cpl
    derivHX = prefactor * -(dHX.flatten() / rHX) * taper
    fHY = -derivHY
    fHX = -derivHX
    cpl_forces[h_idx] += fHY
    cpl_forces[y_idx] -= fHY
    cpl_forces[h_idx] += fHX
    cpl_forces[x_idx] -= fHX
    return cpl_forces


def main():
    lmp_coord_file = "1500.lmp"
    force_field_file = "ff.lmp"
    header = ["units metal", "atom_style full", "boundary p p p"]

    temperature = 1500
    timestep = 1e-3
    r1 = np.random.randint(1, 99999)
    r2 = np.random.randint(1, 99999)
    commands = [
        "fix md all nve",
        f"fix tst all temp/csvr {temperature} {temperature} 0.1 {r1}",
        f"timestep {timestep}",
        # f"velocity all create {temperature} {r2} mom yes dist gaussian",
        "fix com all momentum 100 linear 1 1 1",
    ]
    #    commands = ["fix md all nph iso 1 1 1 tchain 5 pchain 5 mtk yes", f"fix tst all temp/csvr {TEMPERATURE} {TEMPERATURE} 0.1 20384", f"timestep {TIMESTEP}", f"velocity all create {TEMPERATURE} 30094 mom yes dist gaussian", "fix com all momentum 100 linear 1 1 1"]
    minimise = [
        "min_style cg",
        "min_modify line quadratic",
        "minimize 1e-6 1e-6 100 100",
        "reset_timestep 0",
    ]

    INPUTS = {
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
        "neighbour_list_update": 1,
        # "type_charges": {
        #     "Ba": 0.000000,
        #     "Zr": 0.000000,
        #     "Y": 0.0,
        #     "O1": 0.000,
        #     "O2": 0.0000,
        #     "O3": 0.00000,
        #     "H1": 0.0000,
        # },
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
                    "distance": DIST_CUTOFF,
                    "angle": None,
                },
                "coupling_value_function": coupling_value_function,
                "coupling_forces_function": coupling_forces_function,
            }
        ],
    }

    # msevb = mustard.Mustard(lmp_coord_file, force_field_file, coupling_function, header, commands, INPUTS, msevb_mpi_ranks=[11,11,11,11])
    msevb = Mustard(
        lmp_coord_file,
        force_field_file,
        header,
        commands,
        INPUTS,
        debug=True,
        # msevb_mpi_ranks=[1, 1],
    )
    msevb.add_trajectory(filename="trajectory.dcd", write_frequency=50)
    msevb.add_trajectory(filename="reaction.xyz", write_frequency=50, rxn=True)
    msevb.add_output(filename=None, properties=["temp", "pe", "ke"], write_frequency=50)
    msevb.add_output(
        filename="mustard.log",
        properties=["temp", "pe", "ke"],
        write_frequency=50,
    )
    # msevb.msevb_minimise()
    # msevb.finite_differences(
    #     file="finite_differences_e3.out", delta=1e-3
    # )
    # msevb.step(0)
    msevb.step(2000000)
    # msevb.step(50000)


if __name__ == "__main__":
    main()
