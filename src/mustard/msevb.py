import numpy as np
import time

from dataclasses import dataclass
from collections import defaultdict
from sympy import Matrix
from mpi4py import MPI
from typing import Callable, Tuple, Dict, Union
from lammps import lammps

from .topology import Topology, Frame, Site, System
from .io import SystemInfo
from .mpi import Universe
from . import utils


@dataclass
class EigenSolution:
    index: int
    parent: Union[None, int]
    energy: float
    forces: np.ndarray
    amplitudes: np.ndarray


class MSEVB:
    NONE = 0
    MAIN = 1
    UPDATE = 2

    def __init__(
        self,
        universe: Universe,
        topology: Topology,
        SI: SystemInfo,
    ):
        self.universe = universe
        self.log = self.universe.log
        self.topology = topology
        self.SI = SI
        self.get_eig_vecs = self._get_eig_vecs()
        self.get_atom_coeffs = self._get_mix_scf_forces()
        self.mix_states_multi = self.mix_states_multi2
        self.frame = self.topology.frame
        self.run: int = MSEVB.NONE
        self.min_eval: float = 0.0
        self.min_system: System = self.topology.EMPTY_SYSTEM
        self.step_count: int = 0
        self.forces_shape = utils.get_forces(self.topology.lmp).shape
        self.ntimestep = 0
        self.current_mixed_forces = np.zeros(shape=self.forces_shape)
        self.sync = True

    def __call__(self, lmp, ntimestep, nlocal, tag, x, f):
        self.ntimestep = ntimestep
        if self.run == MSEVB.NONE:
            callback = self.callback_none
        elif self.run == MSEVB.MAIN:
            callback = self.callback_main
        elif self.run == MSEVB.UPDATE:
            callback = self.callback_update
        else:
            raise RuntimeError("Undefined run command")
        callback(lmp, ntimestep, nlocal, tag, x, f)

    def callback_update(self, *args):
        pass

    def callback_none(self, lmp, ntimestep, nlocal, tag, x, f):
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = utils.get_forces(lmp)
        mixed_forces = self.universe.global_comm.bcast(current_forces, root=0)
        idxs = self.topology.id_to_idx(tag)
        total_x, total_v = self.sync_x_v(lmp, x, idxs)
        self.frame(lmp, pos=total_x, vel=total_v, forces=current_forces)
        pe = lmp.extract_compute("get_pe", 0, 0)
        if self.universe.sub_rank == 0:
            self.log(
                f"Potential energy for color {self.universe.rank.color}: {pe}",
                level=Universe.DEBUG,
                rank=-1,
            )
        self.min_eval = pe
        self.log(f"Final energy: {self.min_eval}", level=Universe.DEBUG)
        if len(idxs) > 0:
            new_forces[:, :] = mixed_forces[idxs] - current_forces[idxs]
        with np.printoptions(precision=8, suppress=True, linewidth=10000):
            self.log(f"Mixed forces:\n{mixed_forces}", level=Universe.DEBUG)
        self.current_mixed_forces = mixed_forces

    def callback_main(self, lmp, ntimestep, nlocal, tag, x, f):
        color = self.universe.rank.color
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = utils.get_forces(lmp)
        idxs = self.topology.id_to_idx(tag)
        if len(idxs) < 1:
            raise ValueError("No idxs.")
        total_x, total_v = self.sync_x_v(lmp, x, idxs)
        pe = lmp.extract_compute("get_pe", 0, 0)
        if self.universe.sub_rank == 0:
            self.log(
                f"Potential energy for color {color}: {pe}",
                level=Universe.DEBUG,
                rank=-1,
            )
        sendpes = np.zeros(self.universe.total_colors, dtype="d")
        pes = np.zeros(self.universe.total_colors, dtype="d")
        if self.universe.sub_rank == 0:
            sendpes[color] = pe
        self.universe.global_comm.Allreduce(sendpes, pes, op=MPI.SUM)
        pes = pes[: self.topology.num_systems]
        self.frame(lmp, pos=total_x, vel=total_v, forces=current_forces)
        min_eval, min_system, mixed_forces = self.mix_states(
            pes,
            self.frame,
        )
        self.min_system = min_system
        self.min_eval = min_eval
        new_forces[:, :] = mixed_forces[idxs] - current_forces[idxs]
        self.current_mixed_forces = mixed_forces
        self.log(f"Final energy: {self.min_eval}", level=Universe.DEBUG)
        with np.printoptions(precision=8, suppress=True, linewidth=10000):
            self.log(f"Mixed forces:\n{mixed_forces}", level=Universe.DEBUG)
        # TODO: deal with virial/pressure later

    def _get_eig_vecs(self) -> Callable:
        if self.SI.eig_solver == "SYMPY":
            return self.get_eig_vecs_sympy
        return self.get_eig_vecs_numpy

    def hellmann_feynman(self, matrix: np.ndarray, evec: np.ndarray) -> np.ndarray:
        return np.einsum("ijkl,i,j->kl", matrix, evec, evec)

    def sync_x_v(
        self, lmp: lammps, pos: np.ndarray, idxs: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        if not self.sync:
            return None, None
        nx = utils.get_positions(lmp)
        nx = self.universe.global_comm.bcast(nx, root=0)
        vel = utils.get_velocities(lmp)
        vel = self.universe.global_comm.bcast(vel, root=0)
        utils.set_velocities(lmp, vel)
        if len(idxs) > 0:
            pos[:, :] = nx[idxs]
        return nx, vel

    def get_computes(self, lmp):
        if self.SI.computes is None:
            return None
        cs = np.zeros(len(self.SI.computes))
        for n, c_id in enumerate(self.SI.computes):
            c = lmp.extract_compute(c_id, 0, 0)
            if c is None:
                self.log(
                    f"Extracted compute {c_id} returned None. Setting to 0.0",
                    level=Universe.WARN,
                    rank=-1,
                )
                c = 0.0
            try:
                c = float(c)
            except TypeError:
                raise TypeError(
                    f"Could not convert compute {c_id} to type float. Returned type: {type(c)}"
                )
            cs[n] = c
        return cs

    def get_min_EVB_state(
        self, matrix: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        System Hamiltonian represented as a Numpy matrix is diagonalised using
        either Numpy or Sympy (see self.get_eig_vecs()).

        The occupancies for each Eigen value are determined, see get_occupancies()
        in io.py. Occupancies are used to calculate and return the minimum
        Eigen value (min_eval), minimum Eigen vector coefficients (min_evec_coeffs),
        and the squares of the minimum Eigen vector coefficients (ampltiudes).

        Also deals with logging the output for Debug=True.
        """
        with np.printoptions(precision=6, suppress=True, linewidth=10000):
            self.log(f"System matrix:\n{matrix}", level=Universe.DEBUG)
        eig_vals, eig_vecs = self.get_eig_vecs(matrix)
        with np.printoptions(precision=6, suppress=True, linewidth=10000):
            self.log(f"Eigen values:\n{eig_vals}", level=Universe.DEBUG)
            self.log(f"Eigen vectors:\n{eig_vecs}", level=Universe.DEBUG)
        occupancies = self.SI.get_occupancies(eig_vals, self.SI)
        min_evec_coeffs = np.sum(occupancies * eig_vecs, axis=1)
        amplitudes = min_evec_coeffs**2
        min_eval = float(np.sum(occupancies * eig_vals))
        with np.printoptions(precision=8, suppress=True, linewidth=10000):
            self.log(f"Occupancies:\n{occupancies}", level=Universe.DEBUG)
            self.log(f"Eigen Vector:\n{min_evec_coeffs}", level=Universe.DEBUG)
            self.log(f"Amplitudes:\n{amplitudes}", level=Universe.DEBUG)
            self.log(f"Minimum Eigenvalue: {min_eval}", level=Universe.DEBUG)
        return min_eval, min_evec_coeffs, amplitudes

    def get_eig_vecs_numpy(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalises square matrix (matrix) using Numpy and returns the
        Eigen values and Eigen vectors as Numpy arrays, respectively.
        See https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html.
        """
        eig_vals, eig_vecs = np.linalg.eigh(matrix)
        return eig_vals, eig_vecs

    def get_eig_vecs_sympy(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalises square matrix (matrix) using Sympy and returns the
        Eigen values and Eigen vectors as Numpy arrays, respectively.
        See https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html.
        """
        matrixobj = Matrix(list(matrix))
        _eig_vecs = matrixobj.eigenvects()
        eig_vecs = []
        eig_vals = []
        for val, _, vec in _eig_vecs:
            eig_vals.append(val)
            if len(vec) > 1:
                raise RuntimeError("More than one eigenvector for an eigenvalue.")
            eig_vecs.append(list(vec[0].normalized()))
        eig_vecs = np.array(eig_vecs, dtype=float).T
        eig_vals = np.array(eig_vals, dtype=float)
        return eig_vals, eig_vecs

    def get_coupling(
        self,
        site: Site,
        frame: Frame,
        energies: Dict[str, float],
        init_forces: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        The coupling value (cpl_val) and coupling forces (cpl_forces) are
        evaluated for a given Site object using the user-input for the
        'coupling_function' input and returned.
        """
        computes = None
        forces = {"new": frame.forces, "initial": init_forces}
        self.topology.update_snapshot(
            frame,
            self.step_count,
            site=site,
            energies=energies,
            computes=computes,
            forces=forces,
        )
        if site.rxn_num is None:
            raise ValueError("site.rxn_num is None.")
        cpl_val, cpl_forces = self.SI.coupling[site.rxn_num](self.topology.snapshot)
        cpl_val = float(cpl_val)
        if cpl_forces.shape != frame.forces.shape:
            raise ValueError(
                (
                    "Returned coupling forces shape {cpl_forces.shape} "
                    "should be the same as forces shape {forces.shape}"
                )
            )
        return cpl_val, cpl_forces

    def mix_states(self, *args) -> Tuple[float, System, np.ndarray]:
        num_sites = self.topology.num_sites
        if num_sites < 1:
            raise RuntimeError("Number of sites is < 1. Should not have happened")
        if num_sites == 1:
            return self.mix_states_single(*args)
        return self.mix_states_multi(*args)

    def _mix_states(
        self, pes: np.ndarray, frame: Frame
    ) -> Tuple[np.ndarray, np.ndarray]:
        matrix_size = self.topology.num_systems
        matrix = np.zeros(shape=(matrix_size, matrix_size))
        dmatrix = np.zeros(
            shape=(
                matrix_size,
                matrix_size,
                *frame.forces.shape,
            )
        )
        if self.universe.rank.color >= matrix_size:
            return matrix, dmatrix
        my_pe = pes[self.universe.rank.color]
        if self.universe.sub_rank == 0:
            matrix[self.universe.rank.color, self.universe.rank.color] = my_pe
            dmatrix[self.universe.rank.color, self.universe.rank.color][
                :, :
            ] = frame.forces
        return matrix, dmatrix

    def _mix_states_single(
        self,
        pes: np.ndarray,
        frame: Frame,
        all_forces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        matrix, dmatrix = self._mix_states(pes, frame)
        if self.universe.rank.color == 0:
            return matrix, dmatrix
        if self.universe.rank.color >= self.topology.num_systems:
            return matrix, dmatrix
        system = self.topology.current_system
        site = system.sites[0]
        if site is None:
            raise ValueError("site is None")
        parent = site.parent
        energies = {"new": pes[self.universe.rank.color], "initial": pes[parent]}
        init_forces = all_forces[parent]
        cpl_val, cpl_forces = self.get_coupling(
            site,
            frame,
            energies,
            init_forces,
        )
        if self.universe.sub_rank != 0:
            return matrix, dmatrix
        matrix[self.universe.rank.color, parent] = cpl_val
        matrix[parent, self.universe.rank.color] = cpl_val
        dmatrix[self.universe.rank.color, parent][:, :] = cpl_forces
        dmatrix[parent, self.universe.rank.color][:, :] = cpl_forces
        return matrix, dmatrix

    def mix_states_single(
        self,
        pes: np.ndarray,
        frame: Frame,
    ) -> Tuple[float, System, np.ndarray]:
        all_forces = self.get_all_forces(frame)
        matrix, dmatrix = self._mix_states_single(pes, frame, all_forces)
        mbuff = np.empty_like(matrix)
        self.universe.global_comm.Allreduce(matrix, mbuff, op=MPI.SUM)
        min_eval, min_evec_coeffs, amplitudes = self.get_min_EVB_state(mbuff)
        dmbuff = np.empty_like(dmatrix)
        self.universe.global_comm.Allreduce(dmatrix, dmbuff, op=MPI.SUM)
        mixed_forces = self.hellmann_feynman(dmbuff, min_evec_coeffs)
        min_system = self.topology.systems[np.argmax(amplitudes)]
        return min_eval, min_system, mixed_forces

    def _get_all_forces(
        self,
        frame: Frame,
    ) -> np.ndarray:
        all_forces = np.zeros(shape=(self.topology.num_systems, *frame.forces.shape))
        if self.universe.rank.color >= self.topology.num_systems:
            return all_forces
        if self.universe.sub_rank != 0:
            return all_forces
        all_forces[self.universe.rank.color][:, :] = frame.forces
        return all_forces

    def get_all_forces(self, frame: Frame) -> np.ndarray:
        all_forces = self._get_all_forces(frame)
        all_forces_buff = np.empty_like(all_forces)
        self.universe.global_comm.Allreduce(all_forces, all_forces_buff, op=MPI.SUM)
        return all_forces_buff

    def mix_scf_forces_average(
        self, atom_coeffs: np.ndarray, num_sites: int
    ) -> np.ndarray:
        return np.full(atom_coeffs.shape, 1 / num_sites)

    def mix_scf_forces_weighted(
        self, atom_coeffs: np.ndarray, num_sites: int
    ) -> np.ndarray:
        full_coeffs = np.sum(atom_coeffs, axis=0)
        mask = full_coeffs > 0.0
        for n in range(num_sites):
            atom_coeffs[n][mask] /= full_coeffs[mask]
            atom_coeffs[n][~mask] = 1 / num_sites
        return atom_coeffs

    def _get_mix_scf_forces(self) -> Callable:
        if self.SI.scf_mix_method == "AVERAGE":
            return self.mix_scf_forces_average
        return self.mix_scf_forces_weighted

    def mix_states_multi1(
        self,
        pes: np.ndarray,
        frame: Frame,
    ) -> Tuple[float, System, np.ndarray]:
        """Exact multi-site solution via multiple mini matrices"""
        all_forces = self.get_all_forces(frame)
        system_coefficients = np.ones(shape=self.topology.num_systems)
        systems_idxs = self.topology.systems_idxs
        d = defaultdict(dict)
        for n in range(self.topology.num_sites):
            site_init = systems_idxs[0, 0]
            init_n_sites = np.delete(
                self.topology.systems_idxs[
                    np.where(systems_idxs[:, 0] == site_init)[0]
                ],
                np.arange(n + 1),
                axis=1,
            )
            non_n_sites = np.delete(systems_idxs, 0, axis=1)
            new_systems_idxs, new_pes, new_forces = [], [], []
            if not init_n_sites.any():
                if n != self.topology.num_sites - 1:
                    raise RuntimeError("Multi Site didn't work")
                init_n_sites = [[0]]
                non_n_sites = np.zeros(shape=(len(pes), 1))
            for m, state_idxs in enumerate(init_n_sites):
                which_systems = np.argwhere(
                    np.all(non_n_sites == state_idxs, axis=1)
                ).flatten()
                d[n][m] = which_systems
                mat_len = len(which_systems)
                matrix = np.zeros(shape=(mat_len, mat_len))
                dmatrix = np.zeros(shape=(mat_len, mat_len, *frame.forces.shape))
                for i, state_idx in enumerate(which_systems):
                    matrix[i, i] = pes[state_idx]
                    dmatrix[i, i][:, :] = all_forces[state_idx]
                    if i == 0:
                        continue
                    energies = {"new": matrix[i, i], "initial": matrix[0, 0]}
                    init_forces = dmatrix[0, 0]
                    # TODO: Also mix relevant site properties (charges...)
                    site = self.topology.sites[self.topology.systems_idxs[state_idx, n]]
                    frame.setattr("forces", all_forces[state_idx])
                    cpl_val, cpl_forces = self.get_coupling(
                        site,
                        frame,
                        energies,
                        init_forces,
                    )
                    matrix[0, i] = cpl_val
                    matrix[i, 0] = cpl_val
                    dmatrix[0, i][:, :] = cpl_forces
                    dmatrix[i, 0][:, :] = cpl_forces

                min_eval, min_evec_coeffs, amplitudes = self.get_min_EVB_state(matrix)
                mixed_forces = self.hellmann_feynman(dmatrix, min_evec_coeffs)
                new_systems_idxs.append(state_idxs)
                new_pes.append(min_eval)
                new_forces.append(mixed_forces)
                if n == 0:
                    system_coefficients[which_systems] *= amplitudes
                    continue
                for j, k in enumerate(which_systems):
                    amp = amplitudes[j]
                    system_idxs = self.find_parents(d, k, n)
                    system_coefficients[system_idxs] *= amp

            systems_idxs = np.array(new_systems_idxs)
            pes = np.array(new_pes)
            all_forces = np.array(new_forces)
        if len(pes) > 1 or len(all_forces) > 1:
            raise RuntimeError("Multi Site didn't work")
        min_eval = pes[0]
        mixed_forces = all_forces[0]
        if abs(1.0 - np.sum(system_coefficients)) > 1e-8:
            raise RuntimeError("Multi Site didn't work")
        min_system = self.topology.systems[np.argmax(system_coefficients)]
        return min_eval, min_system, mixed_forces

    @staticmethod
    def find_parents(
        dictionary: Dict[int, Dict[int, np.ndarray]], key: int, nsite: int
    ) -> np.ndarray:
        system_idxs = []
        prev_site = nsite - 1
        if prev_site < 0:
            return np.array(system_idxs)
        prev_idxs = list(dictionary[prev_site][key])
        system_idxs += prev_idxs
        prev_site -= 1
        while prev_site != -1:
            new_system_idxs = []
            for idx in system_idxs:
                prev_idxs = list(dictionary[prev_site][idx])
                new_system_idxs += prev_idxs
            system_idxs = new_system_idxs
            prev_site -= 1
        return np.array(system_idxs)

    def _mix_states_multi2(
        self,
        pes: np.ndarray,
        frame: Frame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        matrix, dmatrix = self._mix_states(pes, frame)
        all_forces = self.get_all_forces(frame)
        system = self.topology.current_system
        if system.parents is None:
            return matrix, dmatrix
        if self.universe.rank.color >= self.topology.num_systems:
            return matrix, dmatrix
        for parent in system.parents:
            reference_pe = pes[parent]
            energies = {"new": pes[self.universe.rank.color], "initial": reference_pe}
            if system.site_parents is None:
                raise RuntimeError("Shouldn't be None.")
            site = system.sites[system.site_parents[parent]]
            reference_forces = all_forces[parent]
            cpl_val, cpl_forces = self.get_coupling(
                site,
                frame,
                energies,
                reference_forces,
            )
            if self.universe.sub_rank == 0:
                matrix[self.universe.rank.color, parent] = cpl_val
                matrix[parent, self.universe.rank.color] = cpl_val
                dmatrix[self.universe.rank.color, parent][:, :] = cpl_forces
                dmatrix[parent, self.universe.rank.color][:, :] = cpl_forces
        return matrix, dmatrix

    def mix_states_multi2(
        self,
        pes: np.ndarray,
        frame: Frame,
    ) -> Tuple[float, System, np.ndarray]:
        """Exact multi-site solution via one large matrix"""
        matrix, dmatrix = self._mix_states_multi2(pes, frame)
        mbuff = np.empty_like(matrix)
        self.universe.global_comm.Allreduce(matrix, mbuff, op=MPI.SUM)
        min_eval, min_evec_coeffs, amplitudes = self.get_min_EVB_state(mbuff)
        dmbuff = np.empty_like(dmatrix)
        self.universe.global_comm.Allreduce(dmatrix, dmbuff, op=MPI.SUM)
        mixed_forces = self.hellmann_feynman(dmbuff, min_evec_coeffs)
        min_system = self.topology.systems[np.argmax(amplitudes)]
        return min_eval, min_system, mixed_forces
