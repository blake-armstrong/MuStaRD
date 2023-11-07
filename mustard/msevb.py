import numpy as np
from mpi4py import MPI
from .topology import Topology
from .io import SystemInfo
from .mpi import Universe
from . import utils
from copy import copy


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

    def __call__(self, lmp, ntimestep, nlocal, tag, x, f):
        if self.run == 0:
            return self.callback_none(lmp, ntimestep, nlocal, tag, x, f)
        if self.run == 2:
            return self.callback_update(lmp, ntimestep, nlocal, tag, x, f)
        return self.callback(lmp, ntimestep, nlocal, tag, x, f)

    def callback_update(self, lmp, ntimestep, nlocal, tag, x, f):
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = lmp.numpy.extract_atom("f")
        if len(current_forces) > 0:
            new_forces[:, :] = current_forces

    def callback_none(self, lmp, ntimestep, nlocal, tag, x, f):
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = utils.get_forces(lmp)
        current_forces[:, :] = self.universe.global_comm.bcast(current_forces, root=0)
        idxs = self.topology.id_to_idx(tag)
        total_x, total_v = self.sync_x_v(lmp, current_forces.shape, x, idxs)
        self.frame(lmp, pos=total_x, vel=total_v, forces=current_forces)
        pe = lmp.extract_compute("get_pe", 0, 0)
        self.min_eval = pe
        if len(idxs) > 0:
            new_forces[:, :] = current_forces[idxs]

    def callback(self, lmp, ntimestep, nlocal, tag, x, f):
        # grab the numpified fexternal array
        tag1 = copy(tag)
        new_forces = lmp.numpy.fix_external_get_force("ext")
        current_forces = utils.get_forces(lmp)
        idxs = self.topology.id_to_idx(tag)
        total_x, total_v = self.sync_x_v(lmp, current_forces.shape, x, idxs)
        pe = lmp.extract_compute("get_pe", 0, 0)
        if self.universe.sub_rank == 0:
            # self.log(f"Forces: {current_forces}", level="debug", rank=-1)
            self.log(
                f"Potential energy for color {self.universe.rank.color}: {pe}",
                level="debug",
                rank=-1,
            )
        virial = None
        if self.SI.scale_box:
            virial = utils.get_virial(lmp, pr2vir=self.SI.units["pr2vir"])
        cs = self.get_computes(lmp)
        pes = np.zeros(self.topology.num_systems, dtype="d")
        forces = np.zeros(
            shape=(self.topology.num_systems, *current_forces.shape), dtype="d"
        )
        virials = None
        if self.SI.scale_box:
            virials = np.zeros((self.topology.num_systems, 6), dtype="d")
        computes = None
        if self.SI.computes is not None:
            computes = np.zeros(
                (self.topology.num_systems, len(self.SI.computes)), dtype="d"
            )
        if self.universe.me == 0:
            source = MPI.ANY_SOURCE
            pes[0] = pe
            forces[0, :, :] = current_forces
            if virials is not None:
                virials[0, :] = virial
            if computes is not None:
                computes[0, :] = cs
            for system in range(1, self.topology.num_systems):
                pes[system] = self.universe.global_comm.recv(source=source, tag=system)
                forces[system, :, :] = self.universe.global_comm.recv(
                    source=source, tag=system + 200
                )
                if virials is not None:
                    virials[system, :] = self.universe.global_comm.recv(
                        source=source, tag=system + 300
                    )
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
            self.universe.global_comm.send(
                current_forces, dest=0, tag=self.universe.rank.color + 200
            )
            if virial is not None:
                self.universe.global_comm.send(
                    virial, dest=0, tag=self.universe.rank.color + 300
                )
            if cs is not None:
                self.universe.global_comm.send(
                    cs, dest=0, tag=self.universe.rank.color + 400
                )
        pes = self.universe.global_comm.bcast(pes, root=0)
        forces = self.universe.global_comm.bcast(forces, root=0)
        if self.SI.scale_box:
            virials = self.universe.global_comm.bcast(virials, root=0)
        if self.SI.computes is not None:
            computes = self.universe.global_comm.bcast(computes, root=0)

        self.frame(lmp, pos=total_x, vel=total_v, forces=current_forces)
        mix_states = self.mix_states()
        min_eval, min_evec_coeffs, cpl_forces = mix_states(
            pes,
            computes,
            self.frame,
        )
        self.log(f"Minimum Eigenvalue: {min_eval}", level="debug")
        amplitudes = min_evec_coeffs**2
        min_state_idx = np.argmax(amplitudes)
        self.log(f"Amplitudes: {amplitudes}", level="debug")
        ondiag_forces = np.einsum("ijk,i->jk", forces, amplitudes)
        offdiag_forces = np.einsum(
            "ijk,i->jk", cpl_forces, 2 * min_evec_coeffs[0] * min_evec_coeffs[1:]
        )
        mixed_forces = ondiag_forces + offdiag_forces
        mixed_virial = None
        if virials is not None:
            mixed_virial = np.einsum("ij,i->j", virials, amplitudes)
        self.min_state_idx = min_state_idx
        self.min_eval = min_eval
        ids = lmp.numpy.extract_atom("id")
        if len(idxs) > 0:
            new_forces[:, :] = mixed_forces[idxs]
        # TODO: deal with virial/pressure later

    def sync_x_v(self, lmp, shape, pos, idxs):
        nx = np.zeros(shape=shape)
        vel = utils.get_velocities(lmp)
        if self.universe.rank.color == 0:
            if len(idxs) > 0:
                nx[idxs] = pos
        distributed_total_x = np.empty_like(nx)
        self.universe.global_comm.Allreduce(nx, distributed_total_x, op=MPI.SUM)
        if len(idxs) > 0:
            pos[:, :] = distributed_total_x[idxs]
        vel = self.universe.global_comm.bcast(vel, root=0)
        utils.set_velocities(lmp, vel)
        return distributed_total_x, vel

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
        rxn_num = self.topology.rxn_nums_dict[pair]
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
        num_sites = len(self.topology.pairs_idxs)
        if num_sites == 1:
            return self.mix_states_single
        return self.mix_states_scf

    def mix_states_single(self, pes, computes, frame):
        # rxn_pairs, _, pairs_idxs, _ = pair_info
        states = self.topology.pairs_idxs[0]
        num_states = len(states)
        matrix = np.zeros(shape=(num_states, num_states))
        cpl_forces = np.zeros(shape=(num_states - 1, *frame.forces.shape))
        if self.universe.rank.color == 0:
            if self.universe.me == 0:
                matrix[0, 0] = pes[0]
                for state in range(1, num_states):
                    cpl_val = self.universe.global_comm.recv(
                        source=MPI.ANY_SOURCE, tag=state
                    )
                    matrix[state, state] = pes[state]
                    matrix[state, 0] = cpl_val
                    matrix[0, state] = cpl_val
                    cpl_forces[state - 1, :, :] = self.universe.global_comm.recv(
                        source=MPI.ANY_SOURCE, tag=state + 100
                    )
        elif self.universe.rank.color < num_states:
            init_compute, new_compute = None, None
            if computes is not None:
                init_compute = computes[0]
                new_compute = computes[self.universe.rank.color]
            cpl_val, cpl_forces = self.get_coupling(
                self.topology.rxn_pairs[states[self.universe.rank.color]],
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
                self.universe.global_comm.send(
                    cpl_forces, dest=0, tag=self.universe.rank.color + 100
                )
        min_eval, min_evec_coeffs = None, None
        if self.universe.me == 0:
            min_eval, min_evec_coeffs = self.get_min_EVB_state(matrix)
        min_eval, min_evec_coeffs, cpl_forces = self.universe.global_comm.bcast(
            (min_eval, min_evec_coeffs, cpl_forces), root=0
        )

        return min_eval, min_evec_coeffs, cpl_forces

    def mix_states_scf(self, pes, computes, frame):
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
