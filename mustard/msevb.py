import numpy as np
from mpi4py import MPI
from .topology import Topology
from .io import SystemInfo
from .mpi import Universe
from . import utils
from copy import copy
from time import time


class MSEVB:
    def __init__(
        self,
        universe: Universe,
        topology: Topology,
        SI: SystemInfo,
        frame: Topology.Frame,
    ):
        self.universe = universe
        self.log = self.universe.log
        self.topology = topology
        self.SI = SI
        self.frame = frame
        self.run: int = 0
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
        if self.run == 0:
            callback = self.callback_none
        if self.run == 2:
            callback = self.callback_update
        # if self.run == 99:
        #     callback = self.callback_minimise
        return callback(lmp, ntimestep, nlocal, tag, x, f)

    def callback_update(self, lmp, ntimestep, nlocal, tag, x, f):
        pass

    def callback_none(self, lmp, ntimestep, nlocal, tag, x, f):
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = utils.get_forces(lmp)
        mixed_forces = None
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
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = utils.get_forces(lmp)
        idxs = self.topology.id_to_idx(tag)
        total_x, total_v = self.sync_x_v(lmp, x, idxs)
        pe = lmp.extract_compute("get_pe", 0, 0)
        if self.universe.sub_rank == 0:
            self.log(
                f"Potential energy for color {self.universe.rank.color}: {pe}",
                level="debug",
                rank=-1,
            )
        # virial = None
        # if self.SI.scale_box:
        #     virial = utils.get_virial(lmp, pr2vir=self.SI.units["pr2vir"])
        pes = np.zeros(self.topology.num_systems, dtype="d")
        cs = self.get_computes(lmp)
        computes = None
        if cs is not None:
            computes = np.zeros((self.topology.num_systems, len(cs)), dtype="d")
        if self.universe.me == 0:
            source = MPI.ANY_SOURCE
            pes[0] = pe
            if computes is not None:
                computes[0, :] = cs
            for system in range(1, self.topology.num_systems):
                pes[system] = self.universe.global_comm.recv(source=source, tag=system)
                if computes is not None:
                    computes[system, :] = self.universe.global_comm.recv(
                        source=source, tag=system + 400
                    )
        if (
            self.universe.rank.color != 0
            and self.universe.rank.color < self.topology.num_systems
            and self.universe.sub_rank == 0
        ):
            self.universe.global_comm.send(pe, dest=0, tag=self.universe.rank.color)
            if cs is not None:
                self.universe.global_comm.send(
                    cs, dest=0, tag=self.universe.rank.color + 400
                )
        self.frame(lmp, pos=total_x, vel=total_v, forces=current_forces)
        mix_states = self.mix_states()
        min_eval, min_evec_coeffs, cpl_forces = mix_states(
            pes,
            computes,
            self.frame,
        )
        self.log(f"Minimum Eigenvalue: {min_eval}", level="debug")
        min_evec_coeffs = self.universe.global_comm.bcast(min_evec_coeffs, root=0)
        amplitudes = min_evec_coeffs**2
        self.log(f"Amplitudes: {amplitudes}", level="debug")
        min_state_idx = np.argmax(amplitudes)
        # min_evec_coeff = 0
        # if (
        #     self.universe.rank.color < self.topology.num_systems
        #     and self.universe.sub_rank == 0
        # ):
        #     min_evec_coeff = min_evec_coeffs[self.universe.rank.color]
        # ondiag_forces = current_forces * min_evec_coeff**2
        # offdiag_forces = cpl_forces * 2 * min_evec_coeffs[0] * min_evec_coeff
        # mixed_forces = ondiag_forces + offdiag_forces
        # force_commbuff = np.empty_like(mixed_forces)
        # self.universe.global_comm.Allreduce(mixed_forces, force_commbuff, op=MPI.SUM)
        m = np.zeros(
            shape=(
                self.topology.num_systems,
                self.topology.num_systems,
                *current_forces.shape,
            )
        )
        if (
            self.universe.rank.color < self.topology.num_systems
            and self.universe.sub_rank == 0
        ):
            m[self.universe.rank.color, self.universe.rank.color][:, :] = current_forces
            if self.universe.rank.color != 0:
                # HERE
                m[0, self.universe.rank.color][:, :] = cpl_forces
                m[self.universe.rank.color, 0][:, :] = cpl_forces
        mbuff = np.empty_like(m)
        self.universe.global_comm.Allreduce(m, mbuff, op=MPI.SUM)
        mixed_forces = self.hellmann_feynman(mbuff, min_evec_coeffs)
        self.min_state_idx = min_state_idx
        self.min_eval = float(min_eval)
        if len(idxs) > 0:
            new_forces[:, :] = mixed_forces[idxs] - current_forces[idxs]
        self.current_mixed_forces = mixed_forces
        # self.log(f"Mixed forces: {force_commbuff}", level="debug")
        self.log(f"Mixed forces: {mixed_forces}", level="debug")
        # TODO: deal with virial/pressure later
        # mixed_virial = None
        # if virials is not None:
        #     mixed_virial = np.einsum("ij,i->j", virials, amplitudes)

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

    def get_min_EVB_state(self, matrix):
        # matrix diagonalisation
        eig_vals, eig_vecs = np.linalg.eig(matrix)
        eig_vals = eig_vals.real
        occupancies = self.SI.get_occupancies(eig_vals, self.SI)
        min_evec_coeffs = np.sum(occupancies * eig_vecs, axis=1)
        self.log("System matrix: ", level="debug")
        self.log(matrix, level="debug")
        self.log("Eigen values: ", level="debug")
        self.log(eig_vals, level="debug")
        self.log("Occupancies: ", level="debug")
        self.log(occupancies, level="debug")
        return np.min(eig_vals), min_evec_coeffs

    def get_coupling(self, pair, frame, init_pe, new_pe, init_cmp=None, new_cmp=None):
        pair = tuple(pair)
        h, y = pair
        try:
            x = self.topology.bonds[h][
                0
            ]  # NOTE assumes transferring atom is only bonded to one other atom
        except KeyError:
            x = None
        if (
            self.SI.computes is not None
            and init_cmp is not None
            and new_cmp is not None
        ):
            new_cmp = dict(zip(self.SI.computes, new_cmp))
            init_cmp = dict(zip(self.SI.computes, init_cmp))
        rxn_num = self.topology.rxn_pair_info[pair]["num"]
        snapshot = self.topology._get_snapshot(frame, self.step_count)
        rxn_ids = {"x_id": x, "h_id": h, "y_id": y}
        cpl_val = self.SI.coupling_value_functions[rxn_num](
            rxn_ids,
            snapshot,
            new_pe,
            init_pe,
            new_cmp,
            init_cmp,
        )
        cpl_forces = self.SI.coupling_forces_functions[rxn_num](
            rxn_ids,
            snapshot,
            new_cmp,
            frame.forces,
        )
        if cpl_forces.shape != frame.forces.shape:
            raise ValueError(
                "Returned coupling forces shape {cpl_forces.shape} should be the same as forces shape {forces.shape}"
            )
        # taper = self.SI.taper_functions[self.topology.rxn_nums_dict[pair]](
        #    *self.topology.rxn_taper_info[pair]
        # )
        return cpl_val, cpl_forces

    def mix_states(self):
        num_sites = self.topology.num_sites
        if num_sites == 1:
            return self.mix_states_single
        return self.mix_states_scf

    def mix_states_single(self, pes, computes, frame):
        matrix = np.zeros(shape=(self.topology.num_systems, self.topology.num_systems))
        cpl_forces = np.zeros(shape=frame.forces.shape)
        if self.universe.rank.color == 0:
            if self.universe.me == 0:
                matrix[0, 0] = pes[0]
                for state in range(1, self.topology.num_systems):
                    cpl_val = self.universe.global_comm.recv(
                        source=MPI.ANY_SOURCE, tag=state
                    )
                    matrix[state, state] = pes[state]
                    loc = self.topology.systems[state].sites[0].parents
                    matrix[state, loc] = cpl_val
                    matrix[loc, state] = cpl_val
        elif self.universe.rank.color < self.topology.num_systems:
            init_compute, new_compute = None, None
            system = self.topology.current_system
            if computes is not None:
                init_compute = computes[0]
                new_compute = computes[self.universe.rank.color]
            cpl_val, cpl_forces = self.get_coupling(
                system.pairs[0],
                frame,
                pes[0],
                pes[self.universe.rank.color],
                init_compute,
                new_compute,
            )
            if self.universe.sub_rank == 0:
                self.universe.global_comm.send(
                    cpl_val, dest=0, tag=self.universe.rank.color
                )
        min_eval = 0.0
        min_evec_coeffs = np.zeros(shape=self.topology.num_systems)
        if self.universe.me == 0:
            min_eval, min_evec_coeffs = self.get_min_EVB_state(matrix)
        return min_eval, min_evec_coeffs, cpl_forces

    def mix_states_scf(self, pes, computes, frame):
        raise RuntimeError("multi site not ready yet")
        # TODO:
        min_eval = 0.0
        min_evec_coeffs = np.array([0.0, 0.0])
        cpl_forces = np.array([[0.0, 0.0, 0.0]])
        return min_eval, min_evec_coeffs, cpl_forces

    # def _mix_states_scf(
    #     self,
    #     pes,
    #     computes,
    #     frame,
    #     pair_info,
    # ):
    #     rxn_pairs, systems_idxs, pairs_idxs, _ = pair_info
    #     num_sites = len(pairs_idxs)
    #     us = defaultdict(dict)
    #     init_idx = pes.argmin()
    #     matrices = {}
    #     mixed_computes = {}
    #     cs = 1
    #     evals = np.zeros(num_sites)
    #     new_forces = np.zeros(shape=frame.forces.shape)
    #     if computes is not None:
    #         cs = computes.shape[1]
    #     for n, site in enumerate(pairs_idxs):
    #         us[n].update({state: 0.0 for state in site})
    #         us[n][systems_idxs[init_idx][n]] = 1.0
    #         num_states = len(site)
    #         matrices[n] = np.zeros(shape=(num_states, num_states))
    #         mixed_computes[n] = np.zeros(shape=(num_states, cs))
    #     get_us = np.vectorize(lambda site, state: us[site][state])
    #     ref_energy = pes[init_idx]
    #     cycle = 0
    #     while True:
    #         if self.universe.me == 0:
    #             self.logger.debug("initial energy: %s", pes[init_idx])
    #             self.logger.debug("initial us: %s", us)
    #         ref_evals = copy(evals)
    #         for n, site in enumerate(pairs_idxs):
    #             matrix = matrices[n]
    #             mixed_compute = mixed_computes[n]
    #             for m, state in enumerate(site):
    #                 state_idxs = np.where(systems_idxs[:, n] == state)[0]
    #                 state_systems = systems_idxs[state_idxs]
    #                 state_pes = pes[state_idxs]
    #                 _sites = (
    #                     np.ones(shape=state_systems.shape) * np.arange(num_sites)
    #                 ).astype(int)
    #                 mix = get_us(
    #                     np.delete(_sites, n, axis=1),
    #                     np.delete(state_systems, n, axis=1),
    #                 ).prod(axis=1)
    #                 # FIXME: COME BACK TO THIS AND TEST
    #                 matrix[m, m] = np.sum(state_pes * mix)
    #                 _mix_compute = None
    #                 if computes is not None:
    #                     _mix_compute = np.sum(computes[pes] * mix[:, None], axis=0)
    #                 mixed_compute[m] = _mix_compute
    #                 if m == 0:
    #                     continue
    #                 # TODO: parallelise cpl calculation
    #                 cpl = self._get_coupling(
    #                     rxn_pairs[state],
    #                     frame,
    #                     matrix[0, 0],
    #                     matrix[m, m],
    #                     mixed_compute[0],
    #                     mixed_compute[m],
    #                 )
    #                 matrix[0, m] = cpl
    #                 matrix[m, 0] = cpl
    #             min_eval, min_evec_coeffs = None, None
    #             if self.universe.me == 0:
    #                 min_eval, min_evec_coeffs = self._get_min_EVB_state(matrix)
    #             min_eval, min_evec_coeffs = self.universe.global_comm.bcast(
    #                 (min_eval, min_evec_coeffs), root=0
    #             )
    #             amplitudes = min_evec_coeffs**2
    #             self.universe.global_comm.Barrier()
    #             us[n].update({state: amplitudes[n] for n, state in enumerate(site)})
    #             evals[n] = min_eval
    #         if self.universe.me == 0:
    #             self.logger.debug("cycle %s energy: %s", cycle, evals)
    #             self.logger.debug("cycle %s us: %s", cycle, us)
    #         if (abs(evals - ref_evals) < self.SI.scf_tol).all():
    #             break
    #         cycle += 1
    #         if cycle > self.SI.scf_max_iter:
    #             raise RuntimeError(
    #                 f"Could not converge multi-site problem using SCF within {self.SI.scf_max_iter} cycles."
    #             )
    #     cpl_forces = None
    #     return min_eval, min_evec_coeffs, cpl_forces

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
