import numpy as np
from ctypes import c_int, c_double
import math
from collections import defaultdict


def get_distances(pos_arr_1, pos_arr_2, xyz_pbc):
    # pass two numpy position arrays with pbc
    # returns array with distances accounting for pbc
    # calc x2-x1, y2-y1, z2-z1
    pos_matrix = pos_arr_1[:, None, :] - pos_arr_2[None, :, :]
    # take into account pbc
    pos_matrix.T[0] -= xyz_pbc[0] * ((pos_matrix.T[0]) / xyz_pbc[0]).round()
    pos_matrix.T[1] -= xyz_pbc[1] * ((pos_matrix.T[1]) / xyz_pbc[1]).round()
    pos_matrix.T[2] -= xyz_pbc[2] * ((pos_matrix.T[2]) / xyz_pbc[2]).round()
    # calc distances
    distances = np.linalg.norm(pos_matrix, axis=-1)

    return distances


def get_angles(pos_arr_1, pos_arr_2, pos_arr_3, xyz_pbc):
    # take in three position np arrays of equal shape
    # return array containing angles between each point
    ba = pos_arr_1 - pos_arr_2
    ba.T[0] -= xyz_pbc[0] * (ba.T[0] / xyz_pbc[0]).round()
    ba.T[1] -= xyz_pbc[1] * (ba.T[1] / xyz_pbc[1]).round()
    ba.T[2] -= xyz_pbc[2] * (ba.T[2] / xyz_pbc[2]).round()
    bc = pos_arr_3 - pos_arr_2
    bc.T[0] -= xyz_pbc[0] * (bc.T[0] / xyz_pbc[0]).round()
    bc.T[1] -= xyz_pbc[1] * (bc.T[1] / xyz_pbc[1]).round()
    bc.T[2] -= xyz_pbc[2] * (bc.T[2] / xyz_pbc[2]).round()
    cos_ang = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    )
    angles = np.arccos(cos_ang) * 180 / np.pi

    return angles


def MDF(r, rm, rc):
    if r < rm:
        return 1.0
    if r > rc:
        return 0.0
    x = (r - rm) / (rc - rm)
    return (1 - x) ** 3 * (1 + 3 * x + 6 * x**2)


def _dMDF(d, dx, dy, dz, r, rm, rc):
    return (
        -3
        * d
        * (
            rc**2
            - 5 * rm * rc
            + 3 * r * rc
            + 10 * rm**2
            + 6 * dx**2
            + 6 * dy**2
            + 6 * dz**2
            - 15 * rm * r
        )
        * (-r + rc) ** 2
        + (-r + rc) ** 3 * (12 * d * r + 3 * rc * d - 15 * rm * d)
    ) / ((-rm + rc) ** 5 * r)


def dMDF(dx, dy, dz, rm, rc):
    r = (dx**2 + dy**2 + dz**2) ** 0.5
    if r < rm or r > rc:
        return np.array([0, 0, 0])
    ddx = _dMDF(dx, dx, dy, dz, r, rm, rc)
    ddy = _dMDF(dy, dx, dy, dz, r, rm, rc)
    ddz = _dMDF(dz, dx, dy, dz, r, rm, rc)
    return np.array([ddx, ddy, ddz])


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


# def get_pairs(positions, xyz_pbc, cutoffs, X_idxs, H_idxs, Y_idxs, Topology):
#     dist_cut, ang_cut = (
#         cutoffs["distance"],
#         cutoffs["angle"],
#     )
#     if H_idxs.size == 0 or Y_idxs.size == 0:
#         return None
#     H_pos = positions[H_idxs]
#     Y_pos = positions[Y_idxs]
#     dists = get_distances(H_pos, Y_pos, xyz_pbc)
#     dists_bool = dists < dist_cut
#     if not dists_bool.any():
#         return None
#     pair_dists = dists[dists_bool.nonzero()[0], dists_bool.nonzero()[1]]
#     sort = np.argsort(pair_dists)
#     # grab indexes of pairs that meet dist cutoff
#     H_ready_idx = H_idxs[(dists_bool).nonzero()[0]]
#     Y_ready_idx = Y_idxs[(dists_bool).nonzero()[1]]
#     rxn_pairs = np.array([Topology.ids[H_ready_idx], Topology.ids[Y_ready_idx]]).T
#     if ang_cut is None or not X_idxs.any():
#         return rxn_pairs[sort], pair_dists[sort], [None] * len(rxn_pairs)
#     X_ready_idx = [
#         Topology.atoms[id].idx
#         for idx in H_ready_idx
#         for id in Topology.residues[Topology.atoms[Topology.ids[idx]].molecule]
#         if Topology.atoms[id].idx in X_idxs
#     ]
#     # check angles work as well.
#     # positions of atoms in angle
#     H_pos = positions[H_ready_idx]
#     X_pos = positions[X_ready_idx]
#     Y_pos = positions[Y_ready_idx]
#     # get angles..
#     angles = get_angles(H_pos, X_pos, Y_pos, xyz_pbc)
#     angle_bool = angles < ang_cut
#     if not angle_bool.any():
#         return None
#     hxy_angles = angles[angle_bool]
#     rxn_pairs = rxn_pairs[angle_bool]
#     rxn_pairs = rxn_pairs[np.argsort(rxn_pairs[:, 0])]
#     return rxn_pairs[sort], pair_dists[sort], hxy_angles[sort]


def gather_atoms(lmp, *args):
    result = lmp.gather_atoms(*args)
    if result is None:
        raise ValueError(f"None returned for gather_atoms({args[0]})")
    return list(result)


def convert_to_c_type(array, c_type):
    _array = array.flatten()
    return (len(_array) * c_type)(*_array)


def extract_box(box_data):
    boxlo, boxhi, xy, yz, xz, _, _ = box_data
    lx, ly, lz = np.array(boxhi) - np.array(boxlo)
    abc = [lx, 0, 0, xy, ly, 0, xz, yz, lz]
    abcabc = get_abcabc(abc)
    return abcabc, abc


def get_abcabc(abc):
    lx, _, _, xy, ly, _, xz, yz, lz = abc
    a = lx
    b = (ly**2 + xy**2) ** 0.5
    c = (lz**2 + xz**2 + yz**2) ** 0.5
    alpha = math.acos((xy * xz + ly * yz) / (b * c)) * 180 / math.pi
    beta = math.acos(xz / c) * 180 / math.pi
    gamma = math.acos(xy / b) * 180 / math.pi
    return np.array([a, b, c, alpha, beta, gamma])


def get_box_data(lmp):
    return lmp.extract_box()


def set_box_data(lmp, box_data):
    """
    Assumes box is already triclinic
    """
    lmp.command(
        (
            "change_box all "
            f"x final {box_data[0][0]} {box_data[1][0]} "
            f"y final {box_data[0][1]} {box_data[1][1]} "
            f"z final {box_data[0][2]} {box_data[1][2]} "
            f"xy final {box_data[2]} xz final {box_data[4]} yz final {box_data[3]}"
        )
    )


def get_images(lmp):
    return np.array(gather_atoms(lmp, "image", 0, 3)).reshape(-1, 3)


def get_velocities(lmp):
    return np.array(gather_atoms(lmp, "v", 1, 3)).reshape(-1, 3)


def get_positions(lmp):
    return np.array(gather_atoms(lmp, "x", 1, 3)).reshape(-1, 3)


def set_images(lmp, images):
    lmp.scatter_atoms("image", 0, 3, convert_to_c_type(images, c_int))


def set_velocities(lmp, velocities):
    lmp.scatter_atoms("v", 1, 3, convert_to_c_type(velocities, c_double))


def set_positions(lmp, positions):
    lmp.scatter_atoms("x", 1, 3, convert_to_c_type(positions, c_double))


def get_forces(lmp):
    return np.array(gather_atoms(lmp, "f", 1, 3)).reshape(-1, 3)


def get_virial(lmp, pr2vir, vol=None):
    if vol is None:
        vol = lmp.get_thermo("vol")
    p_vir = lmp.numpy.extract_compute("pre_vir", 0, 1)
    if p_vir is None:
        raise ValueError("Could not extract virial")
    vir = p_vir / pr2vir * vol
    return vir


def set_virial(lmp, virial_diff):
    lmp.fix_external_set_virial_global("ext", list(virial_diff))
