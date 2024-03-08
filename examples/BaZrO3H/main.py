import numpy as np

from mustard import Mustard, utils

LMB = 0.7998
ZETA = 16

DIST_CUTOFF = 1.8
DIST_TAPER = 1.7


def Raiteri2011_coupling(Q):
    # 10.1088/0953-8984/23/33/334213
    return LMB * np.exp(-ZETA * Q**2)


def coupling_value_function(
    rxn_ids, snapshot
):
    snapshot.h_idx = snapshot.atoms[rxn_ids["H"]].idx
    snapshot.x_idx = snapshot.atoms[rxn_ids["X"]].idx
    snapshot.y_idx = snapshot.atoms[rxn_ids["Y"]].idx
    snapshot.h_pos = snapshot.frame.pos[snapshot.h_idx]
    snapshot.x_pos = snapshot.frame.pos[snapshot.x_idx]
    snapshot.y_pos = snapshot.frame.pos[snapshot.y_idx]
    snapshot.dHX = utils.get_distance_xyz(snapshot.h_pos, snapshot.x_pos, snapshot.frame.box_vectors)
    snapshot.rHX = utils.get_distances(snapshot.dHX)
    snapshot.dHY = utils.get_distance_xyz(snapshot.h_pos, snapshot.y_pos, snapshot.frame.box_vectors)
    snapshot.rHY = utils.get_distances(snapshot.dHY)
    snapshot._Q = snapshot.rHY - snapshot.rHX
    snapshot.cpl = Raiteri2011_coupling(abs(snapshot._Q))
    snapshot.taper = utils.MDF(snapshot.rHY, DIST_TAPER, DIST_CUTOFF)
    return  snapshot.cpl * snapshot.taper 


def coupling_forces_function(rxn_ids, snapshot, new_forces, initial_forces):
    cpl_forces = np.zeros(shape=initial_forces.shape)
    taper_derivative = 0
    if snapshot.rHY < DIST_CUTOFF and snapshot.rHY > DIST_TAPER:
        _dHY = snapshot.dHY.flatten()
        taper_derivative = utils.dMDF(_dHY[0], _dHY[1], _dHY[2], DIST_TAPER, DIST_CUTOFF)
    prefactor = -2 * ZETA * snapshot.cpl * snapshot._Q
    derivHY = prefactor * (snapshot.dHY.flatten() / snapshot.rHY) * snapshot.taper + taper_derivative * snapshot.cpl
    derivHX = prefactor * -(snapshot.dHX.flatten() / snapshot.rHX) * snapshot.taper
    fHY = -derivHY
    fHX = -derivHX
    cpl_forces[snapshot.h_idx] += fHY
    cpl_forces[snapshot.y_idx] -= fHY
    cpl_forces[snapshot.h_idx] += fHX
    cpl_forces[snapshot.x_idx] -= fHX
    return cpl_forces


def main():
    lmp_coord_file = "coord.lmp"
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
    mpi_list = [1,1]
    msevb = Mustard(
        commands,
        inputs,
        debug=True,
        mpi_list=mpi_list,
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
    # msevb.finite_differences(file="tmp_fd.out", delta=1e-3, index_array=[0,1])
    msevb.step(1000)


if __name__ == "__main__":
    main()
