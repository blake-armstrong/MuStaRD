import numpy as np
from sympy import Matrix
from mpi4py import MPI
from typing import Union, Tuple, Dict
from .topology import Topology, Frame, Site
from .io import SystemInfo
from .mpi import Universe
from . import utils


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
        self.frame = self.topology.frame
        self.run: int = MSEVB.NONE
        self.min_eval: float = 0.0
        self.min_state_idx: np.intp = np.intp(0)
        self.step_count: int = 0
        self.forces_shape = utils.get_forces(self.topology.lmp).shape
        self.ntimestep = 0
        self.current_mixed_forces = np.zeros(shape=self.forces_shape)
        self.sync = True

    def __call__(self, lmp, ntimestep, nlocal, tag, x, f):
        self.ntimestep = ntimestep
        callback = self.callback_main
        if self.run == MSEVB.NONE:
            callback = self.callback_none
        if self.run == MSEVB.UPDATE:
            callback = self.callback_update
        return callback(lmp, ntimestep, nlocal, tag, x, f)

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
                level="debug",
                rank=-1,
            )
        self.min_eval = pe
        if len(idxs) > 0:
            new_forces[:, :] = mixed_forces[idxs] - current_forces[idxs]
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
                level="debug",
                rank=-1,
            )
        sendpes = np.zeros(self.universe.total_colors, dtype="d")
        pes = np.zeros(self.universe.total_colors, dtype="d")
        if self.universe.sub_rank == 0:
            sendpes[color] = pe
        self.universe.global_comm.Allreduce(sendpes, pes, op=MPI.SUM)
        pes = pes[: self.topology.num_systems]
        self.frame(lmp, pos=total_x, vel=total_v, forces=current_forces)
        root_forces = self.universe.global_comm.bcast(current_forces, root=0)
        (
            min_eval,
            min_evec_coeffs,
            amplitudes,
            dmatrix,
        ) = self.mix_states(
            pes,
            self.frame,
            root_forces,
        )
        mixed_forces = self.hellmann_feynman(dmatrix, min_evec_coeffs)
        self.min_state_idx = np.argmax(amplitudes)
        self.min_eval = float(min_eval)
        new_forces[:, :] = mixed_forces[idxs] - current_forces[idxs]
        self.current_mixed_forces = mixed_forces
        with np.printoptions(precision=8, suppress=True, linewidth=10000):
            self.log(f"Mixed forces:\n{mixed_forces}", level="debug")
        # TODO: deal with virial/pressure later

    def _get_eig_vecs(self):
        if self.SI.eig_solver == "SYMPY":
            return self.get_eig_vecs_sympy
        return self.get_eig_vecs_numpy

    def hellmann_feynman(self, matrix, evec):
        return np.einsum("ijkl,i,j->kl", matrix, evec, evec)

    def sync_x_v(self, lmp, pos, idxs):
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
                    level="warn",
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
            self.log(f"System matrix:\n{matrix}", level="debug")
        eig_vals, eig_vecs = self.get_eig_vecs(matrix)
        with np.printoptions(precision=6, suppress=True, linewidth=10000):
            self.log(f"Eigen values:\n{eig_vals}", level="debug")
            self.log(f"Eigen vectors:\n{eig_vecs}", level="debug")
        occupancies = self.SI.get_occupancies(eig_vals, self.SI)
        min_evec_coeffs = np.sum(occupancies * eig_vecs, axis=1)
        amplitudes = min_evec_coeffs**2
        min_eval = float(np.sum(occupancies * eig_vals))
        with np.printoptions(precision=8, suppress=True, linewidth=10000):
            self.log(f"Occupancies:\n{occupancies}", level="debug")
            self.log(f"Eigen Vector:\n{min_evec_coeffs}", level="debug")
            self.log(f"Amplitudes:\n{amplitudes}", level="debug")
            self.log(f"Minimum Eigenvalue: {min_eval}", level="debug")
        return min_eval, min_evec_coeffs, amplitudes

    def get_eig_vecs_numpy(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalises square matrix (matrix) using Numpy and returns the
        Eigen values and Eigen vectors as Numpy arrays, respectively.
        See https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html.
        """
        eig_vals, eig_vecs = np.linalg.eig(matrix)
        eig_vals = eig_vals.real
        eig_vecs = eig_vecs.real
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
        if site.xhy is None:
            raise ValueError("site.xhy is None.")
        x, h, y = site.xhy
        computes = {"new": 0.0, "initial": 0.0}
        forces = {"new": frame.forces, "initial": init_forces}
        self.topology.update_snapshot(
            frame,
            self.step_count,
            site=site,
            energies=energies,
            computes=computes,
            forces=forces,
        )
        rxn_ids = {"X": x, "H": h, "Y": y}
        if site.rxn_num is None:
            raise ValueError("site.rxn_num is None.")
        cpl_val, cpl_forces = self.SI.coupling[site.rxn_num](
            rxn_ids, self.topology.snapshot
        )
        cpl_val = float(cpl_val)
        if cpl_forces.shape != frame.forces.shape:
            raise ValueError(
                (
                    "Returned coupling forces shape {cpl_forces.shape} "
                    "should be the same as forces shape {forces.shape}"
                )
            )
        return cpl_val, cpl_forces

    def mix_states(self, *args) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        num_sites = self.topology.num_sites
        if num_sites < 1:
            raise RuntimeError("should not have happened")
        if num_sites == 1:
            return self.mix_states_single(*args)
        return self.mix_states_scf(*args)

    def _mix_states_single(
        self,
        pes: np.ndarray,
        frame: Frame,
        init_forces: np.ndarray,
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
        cpl_forces = np.zeros(shape=frame.forces.shape)
        if self.universe.rank.color >= matrix_size:
            return matrix, dmatrix
        color = self.universe.rank.color
        my_pe = pes[color]
        if self.universe.sub_rank == 0:
            matrix[color, color] = my_pe
            dmatrix[color, color][:, :] = frame.forces
        if color == 0:
            return matrix, dmatrix
        system = self.topology.current_system
        site = system.sites[0]
        if site is None:
            raise ValueError("site is None")
        parent = site.parent
        energies = {"new": my_pe, "initial": pes[0]}
        cpl_val, cpl_forces = self.get_coupling(
            site,
            frame,
            energies,
            init_forces,
        )
        if self.universe.sub_rank != 0:
            return matrix, dmatrix
        matrix[color, parent] = cpl_val
        matrix[parent, color] = cpl_val
        dmatrix[color, parent][:, :] = cpl_forces
        dmatrix[parent, color][:, :] = cpl_forces
        return matrix, dmatrix

    def mix_states_single(
        self,
        pes: np.ndarray,
        frame: Frame,
        init_forces: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        matrix, dmatrix = self._mix_states_single(pes, frame, init_forces)
        mbuff = np.empty_like(matrix)
        self.universe.global_comm.Allreduce(matrix, mbuff, op=MPI.SUM)
        dmbuff = np.empty_like(dmatrix)
        self.universe.global_comm.Allreduce(dmatrix, dmbuff, op=MPI.SUM)
        return *self.get_min_EVB_state(mbuff), dmbuff

    def mix_states_scf(self, pes, frame, forces):
        raise RuntimeError("multi site not ready yet")
        # TODO:
        min_eval = 0.0
        min_evec_coeffs = np.array([0.0, 0.0])
        cpl_forces = np.array([[0.0, 0.0, 0.0]])
        return min_eval, min_evec_coeffs, cpl_forces

    # def mix_states_scf(
    #    self,
    #    pes,
    #    computes,
    #    frame,
    #    forces,
    # ):
    #    # rxn_pairs, systems_idxs, pairs_idxs, _ = pair_info
    #    # num_sites = len(pairs_idxs)
    #    us = defaultdict(dict)
    #    init_idx = pes.argmin()
    #    matrices = {}
    #    mixed_computes = {}
    #    cs = 1
    #    evals = np.zeros(self.topology.num_sites)
    #    new_forces = np.zeros(shape=frame.forces.shape)
    #    if computes is not None:
    #        cs = computes.shape[1]
    #    for n, site in enumerate(self.topology.pairs_idxs):
    #        us[n].update({state: 0.0 for state in site})
    #        us[n][self.topology.systems_idxs[init_idx][n]] = 1.0
    #        num_states = len(site)
    #        matrices[n] = np.zeros(shape=(num_states, num_states))
    #        mixed_computes[n] = np.zeros(shape=(num_states, cs))
    #    get_us = np.vectorize(lambda site, state: us[site][state])
    #    ref_energy = pes[init_idx]
    #    for _ in range(self.SI.scf_max_iter):
    #        if self.universe.me == 0:
    #            self.log(f"initial energy: {pes[init_idx]}", level="debug")
    #            self.log(f"initial us: {us}", level="debug")
    #        ref_evals = copy(evals)
    #        for n, site in enumerate(self.topology.pairs_idxs):
    #            matrix = matrices[n]
    #            mixed_compute = mixed_computes[n]
    #            for m, state in enumerate(site):
    #                state_idxs = np.where(self.topology.systems_idxs[:, n] == state)[0]
    #                state_systems = self.topology.systems_idxs[state_idxs]
    #                state_pes = pes[state_idxs]
    #                _sites = (
    #                    np.ones(shape=state_systems.shape) * np.arange(self.topology.num_sites)
    #                ).astype(int)
    #                mix = get_us(
    #                    np.delete(_sites, n, axis=1),
    #                    np.delete(state_systems, n, axis=1),
    #                ).prod(axis=1)
    #                # FIXME: COME BACK TO THIS AND TEST
    #                matrix[m, m] = np.sum(state_pes * mix)
    #                _mix_compute = None
    #                if computes is not None:
    #                    _mix_compute = np.sum(computes[pes] * mix[:, None], axis=0)
    #                mixed_compute[m] = _mix_compute
    #                if m == 0:
    #                    continue
    #                # TODO: parallelise cpl calculation
    #                cpl = self.get_coupling(
    #                    rxn_pairs[state],
    #                    frame,
    #                    matrix[0, 0],
    #                    matrix[m, m],
    #                    mixed_compute[0],
    #                    mixed_compute[m],
    #                )
    #                matrix[0, m] = cpl
    #                matrix[m, 0] = cpl
    #            min_eval, min_evec_coeffs = None, None
    #            if self.universe.me == 0:
    #                min_eval, min_evec_coeffs = self.get_min_EVB_state(matrix)
    #            min_eval, min_evec_coeffs = self.universe.global_comm.bcast(
    #                (min_eval, min_evec_coeffs), root=0
    #            )
    #            amplitudes = min_evec_coeffs**2
    #            self.universe.global_comm.Barrier()
    #            us[n].update({state: amplitudes[n] for n, state in enumerate(site)})
    #            evals[n] = min_eval
    #        if self.universe.me == 0:
    #            self.logger.debug("cycle %s energy: %s", cycle, evals)
    #            self.logger.debug("cycle %s us: %s", cycle, us)
    #        if (abs(evals - ref_evals) < self.SI.scf_tol).all():
    #            cpl_forces = None
    #            return min_eval, min_evec_coeffs, cpl_forces
    #    raise RuntimeError(
    #        f"Could not converge multi-site problem using SCF within {self.SI.scf_max_iter} cycles."
    #    )

    def _get_mix_properties(self, systems_idxs, get_us, properties):
        amps = self._get_amps(systems_idxs, get_us)
        return properties * amps

    def _make_mix_properties(self, systems_idxs, get_us, amps=None):
        if amps is None:
            amps = self._get_amps(systems_idxs, get_us)

        def get_mix_properties(properties):
            if len(properties) != len(amps):
                raise RuntimeError("Need all potential energies")
            return properties * amps

        return get_mix_properties

    def _get_amps(self, systems_idxs, get_us):
        sites = (
            np.ones(shape=systems_idxs.shape) * np.arange(systems_idxs.shape[1])
        ).astype(int)
        return get_us(sites, systems_idxs).prod(axis=1)
