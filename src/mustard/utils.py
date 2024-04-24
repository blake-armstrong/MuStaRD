import numpy as np
import math
from ctypes import c_int, c_double
from collections import defaultdict
from typing import Tuple, Union


def get_distance_xyz(
    pos_arr_1: np.ndarray, pos_arr_2: np.ndarray, box_vectors: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ----------
    pos_arr_1 : ndarray, shape (3,)
        The position of particle 1.
    pos_arr_2 : ndarray, shape (3,)
        The position of particle 2.
    box_vectors : ndarray, shape (9,)
        Flattened 3x3 vector matrix of the periodic box.
        Function accesses the bottom diagonal of the matrix.

    Returns
    -------
    ndarray, shape (3,)
        The minimum distance of the x, y and z components between pos_arr_1
        and pos_arr_2 after accounting for the periodic box.
    """
    dx = pos_arr_1 - pos_arr_2
    scale2 = (dx[2] / box_vectors[8]).round()
    dx -= scale2 * box_vectors[6:9]
    scale1 = (dx[1] / box_vectors[4]).round()
    dx[:2] -= scale1 * box_vectors[3:5]
    scale0 = (dx[0] / box_vectors[0]).round()
    dx[0] -= scale0 * box_vectors[0]
    return dx


def get_distances_xyz(
    pos_arr_1: np.ndarray, pos_arr_2: np.ndarray, box_vectors: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ----------
    pos_arr_1 : ndarray, shape (N,3)
        The positions of N particles.
    pos_arr_2 : ndarray, shape (N,3)
        The positions of N particles.
    box_vectors : ndarray, shape (9,)
        Flattened 3x3 vector matrix of the periodic box.
        Function accesses the bottom diagonal of the matrix.

    Returns
    -------
    ndarray, shape (N,3)
        The minimum distance of the x, y and z components between a given particle in
        pos_arr_1 and the particle at the same positions in pos_arr_2 after accounting
        for the periodic box.
    """
    dx = pos_arr_1 - pos_arr_2
    scale2 = (dx[:, 2] / box_vectors[8]).round()
    dx -= scale2[:, None] * box_vectors[6:9]
    scale1 = (dx[:, 1] / box_vectors[4]).round()
    dx[:, :2] -= scale1[:, None] * box_vectors[3:5]
    scale0 = (dx[:, 0] / box_vectors[0]).round()
    dx[:, 0] -= scale0 * box_vectors[0]
    return dx


def get_distances_comb_xyz(
    pos_arr_1: np.ndarray, pos_arr_2: np.ndarray, box_vectors: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ----------
    pos_arr_1 : ndarray, shape (N,3)
        The positions of N particles.
    pos_arr_2 : ndarray, shape (M,3)
        The positions of M particles.
    box_vectors : ndarray, shape (9,)
        Flattened 3x3 vector matrix of the periodic box.
        Function accesses the bottom diagonal of the matrix.

    Returns
    -------
    ndarray, shape (N,M,3)
        The minimum distance of the x, y and z components between a given particle in
        pos_arr_1 and the particle at the same index in pos_arr_2 after accounting
        for the periodic box.
    """
    dx = (pos_arr_1[:, None, :] - pos_arr_2[None, :, :]).T
    scale2 = (dx[2] / box_vectors[8]).round()
    dx -= scale2[None, :, :] * box_vectors[6:9][:, None, None]
    scale1 = (dx[1] / box_vectors[4]).round()
    dx[:2] -= scale1[None, :, :] * box_vectors[3:5][:, None, None]
    scale0 = (dx[0] / box_vectors[0]).round()
    dx[0] -= scale0 * box_vectors[0]
    return dx.T


def get_distances(distances_xyz: np.ndarray) -> Union[np.ndarray, float]:
    """
    Parameters
    ----------
    distances_xyz : ndarray, shape (3,) | (N,3) | (N,M,3)
        Array containing dx, dy, dz between particles.

    Returns
    -------
    float | ndarray, shape of distances_xyz.shape reduced by one dimension.
        The euclidean distances.
    """
    return np.linalg.norm(distances_xyz, axis=-1)


def get_angles_comb(
    pos_arr_1: np.ndarray, pos_arr_2: np.ndarray, pos_arr_3, box_vectors: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ----------
    pos_arr_1 : ndarray, shape (N,3)
        The positions of N particles.
    pos_arr_2 : ndarray, shape (N,3)
        The positions of N particles.
    pos_arr_3 : ndarray, shape (N,3)
        The positions of N particles.
    box_vectors : ndarray, shape (9,)
        Flattened 3x3 vector matrix of the periodic box.
        Function accesses the bottom diagonal of the matrix.

    Returns
    -------
    ndarray, shape (N,)
        The angles between each particle in the three position
        arrays of the same index after accounting for the periodic box.
    """
    ba_dx = get_distances_xyz(pos_arr_1, pos_arr_2, box_vectors)
    ba_dists = get_distances(ba_dx)
    bc_dx = get_distances_xyz(pos_arr_3, pos_arr_2, box_vectors)
    bc_dists = get_distances(bc_dx)
    cos_ang = np.sum(ba_dx * bc_dx, axis=1) / (ba_dists * bc_dists)
    angles = np.arccos(cos_ang) * 180 / np.pi

    return angles


def get_fractional_coords(
    pos_arr: np.ndarray, inv_box_matrix: np.ndarray
) -> np.ndarray:
    return np.dot(inv_box_matrix, pos_arr.T).T


def get_periodic_images(pos_arr: np.ndarray, inv_box_matrix: np.ndarray) -> np.ndarray:
    frac_pos_arr = get_fractional_coords(pos_arr, inv_box_matrix)
    return np.floor(frac_pos_arr).astype(int)


def wrap_coordinates(
    pos_arr: np.ndarray,
    box_matrix: np.ndarray,
    inv_box_matrix: np.ndarray,
    periodic_images: np.ndarray,
) -> np.ndarray:
    frac_pos_arr = get_fractional_coords(pos_arr, inv_box_matrix)
    wrapped_frac_pos_arr = frac_pos_arr - periodic_images
    return np.dot(box_matrix, wrapped_frac_pos_arr.T).T


def unwrap_coordinates(
    pos_arr: np.ndarray,
    box_matrix: np.ndarray,
    inv_box_matrix: np.ndarray,
    periodic_images: np.ndarray,
) -> np.ndarray:
    frac_pos_arr = get_fractional_coords(pos_arr, inv_box_matrix)
    unwrapped_frac_pos_arr = frac_pos_arr + periodic_images
    return np.dot(box_matrix, unwrapped_frac_pos_arr.T).T


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


def gather_atoms(lmp, *args):
    result = lmp.gather_atoms(*args)
    if result is None:
        raise ValueError(f"None returned for gather_atoms({args[0]})")
    return list(result)


def convert_to_c_type(array, c_type):
    _array = array.flatten()
    return (len(_array) * c_type)(*_array)


def extract_box(box_data: tuple) -> Tuple[np.ndarray, np.ndarray]:
    boxlo, boxhi, xy, yz, xz, _, _ = box_data
    lx, ly, lz = np.array(boxhi) - np.array(boxlo)
    abc = np.array([lx, 0, 0, xy, ly, 0, xz, yz, lz])
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


def get_X(id_h, bonds: dict):
    try:
        # NOTE assumes transferring atom is only bonded to one other atom
        return bonds[id_h][0]
    except KeyError:
        return None
