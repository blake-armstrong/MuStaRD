from lammps import lammps
from mpi4py import MPI
from ctypes import c_int, c_double
from scipy.optimize import minimize
from collections import defaultdict
import numpy as np
import os
import uuid

from copy import copy
import pandas as pd
from .topology import Topology
from .io import SystemInfo, Trajectory, Output
from .mpi import synchronize_args, Universe
from .utils import get_pairs, convert_to_c_type, gather_atoms, extract_box
from .mixing import get_FD_occupancies


class Forces:
    def __init__(self, current_forces, current_virial):
        self.current_forces = current_forces
        self.mixed_forces = current_forces
        self.current_virial = current_virial
        self.mixed_virial = current_virial
        self.step = 1
        self.prev_step = 1
        self.empty_forces = np.zeros(shape=self.current_forces.shape)
        self.empty_virial = np.zeros(6)
        self.force_diff = copy(self.empty_forces)
        self.virial_diff = copy(self.empty_virial)

    def set_mixed_forces(self, mixed_forces):
        if self.prev_step == 0:
            self.mixed_forces = self.current_forces
        else:
            self.mixed_forces = mixed_forces

    def set_mixed_virial(self, mixed_virial):
        if self.prev_step == 0:
            self.mixed_virial = self.current_virial
        else:
            self.mixed_virial = mixed_virial

    def _set_step(self, step):
        self.prev_step = self.step
        self.step = step

    def _calc_force_diff(self):
        if self.prev_step == 0:
            return self.force_diff
        if self.mixed_forces is None:
            return self.empty_forces
        if np.isnan(self.mixed_forces).any():
            raise ValueError("One or more forces are NaN")
        self.force_diff = self.mixed_forces - self.current_forces
        return self.force_diff

    def _calc_virial_diff(self):
        if self.prev_step == 0:
            return self.virial_diff
        if self.mixed_virial is None:
            return self.empty_virial
        if np.isnan(self.mixed_virial).any():
            raise ValueError("One or more virial components are NaN")
        self.virial_diff = self.mixed_virial - self.current_virial
        return self.virial_diff


class Mustard:
    @synchronize_args
    def __init__(
        self,
        lmp_coord_file,
        force_field_file,
        header,
        commands,
        # once_off_cmds,
        reaction_parameters,
        # read_restart=None,  # TODO
        mpi_list=None,
        # write_restart=None
        debug=False,
    ):
        # SYSTEM INFO
        self.SI = SystemInfo(reaction_parameters)
        self.get_occupancies = self._set_get_occupancies()
        self.prev_system = np.array([[None, None]])
        self.safe = False
        self.num_colors = 1
        self.step_count = 0
        self.rebuild = True
        self.universe = Universe(mpi_list, debug)
        self.log = self.universe.log

        f = None
        if self.universe.me == 0:
            f = str(uuid.uuid4().hex)
        f = "mustard." + self.universe.global_comm.bcast(f, root=0)
        self.SI.file = f
        self.universe.global_comm.Barrier()
        self.log("MuStaRD")
        self.log(f"Running with {self.universe.num_procs} available processors")
        self.universe._initialise_mpi_distribution(mpi_list)
        self.cmds = {
            "header": header,
            "ff": [f"include {force_field_file}"],
            "user": commands,
            "virial": [
                "compute pre_vir all pressure NULL virial",
            ],
        }
        cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        if self.universe.debug:
            cmdargs[-1] = "main.log"
        self.lmp = lammps(
            name="",
            cmdargs=cmdargs,
            comm=self.universe.lmp_comm,
        )
        self.lmp.commands_list(
            header
            + [
                f"read_data {lmp_coord_file}",
                "change_box all triclinic",
                f"include {force_field_file}",
            ]
        )
        virial = "compute pre_vir all pressure NULL virial"
        if self.SI.scale_box:
            self.lmp.command(virial)
        if self.universe.rank.color == 0:
            self.lmp.commands_list(commands)
            fixes = [
                "fix ux all store/state 1 xu yu zu",
                "fix ext all external pf/array 1",
                "fix_modify ext energy no",
                "fix_modify ext virial no",
            ]
            if self.SI.scale_box:
                fixes[-1] = fixes[-1].replace("no", "yes")
            self.lmp.commands_list(fixes)
        self.lmp.command("run 0 post no")
        self.restart_commands = header + [
            f"read_data /tmp/{self.SI.file}",
            "change_box all triclinic",
            f"include {force_field_file}",
        ]
        if self.SI.scale_box:
            self.restart_commands.append(virial)
        self.topology = Topology(self.lmp, self.SI)
        frame = self._get_frame()
        rxn_pairs, systems_idxs, _ = self._get_systems(frame.pos, frame.xyz_pbc)
        num_systems = 1
        if rxn_pairs:
            num_systems = len(systems_idxs)
        self._redistribute_EVB_states(num_systems, frame)
        self.forces = self._initialise_forces_and_virial()
        self.Trajectory = Trajectory()
        self.Output = Output()

    def _initialise_forces_and_virial(self):
        forces = None
        if self.universe.rank.color == 0:
            current_forces = self.get_forces()
            current_virial = None
            if self.SI.scale_box:
                current_virial = self.get_virial()
            if current_forces is None:
                raise ValueError("Could not get forces")
            forces = Forces(current_forces, current_virial)
        forces = self.universe.global_comm.bcast(forces, root=0)
        return forces

    def _set_get_occupancies(self):
        if self.SI.FM:
            # fermi mixing

            def get_occupancies_FM(eig_vals, SI):
                return get_FD_occupancies(eig_vals, SI.temperature, SI.fd_tols, SI.RT)

            return get_occupancies_FM

        else:

            def get_occupancies_NO_FM(eig_vals, _):
                occupancies = np.zeros(len(eig_vals))
                occupancies[np.argmin(eig_vals)] = 1.0
                return occupancies

            return get_occupancies_NO_FM

    def _get_min_EVB_state(self, matrix):
        # matrix diagonalisation
        eig_vals, eig_vecs = np.linalg.eig(matrix)
        eig_vals = eig_vals.real
        occupancies = self.get_occupancies(eig_vals, self.SI)
        min_evec_coeffs = np.sum(occupancies * eig_vecs, axis=1)
        # sqrd_amplitudes = non_sqrd_amplitudes**2
        # min_EVB_state = np.argmax(sqrd_amplitudes)
        self.log("System matrix: ", level="debug")
        self.log(matrix, level="debug")
        self.log("Eigen values: ", level="debug")
        self.log(eig_vals, level="debug")
        self.log("Occupancies: ", level="debug")
        self.log(occupancies, level="debug")

        return np.min(eig_vals), min_evec_coeffs

    def _reset_lmp_topology(self, frame):
        if self.universe.rank.color == 0:
            raise RuntimeError("dont do this")
        self.lmp.commands_list(["clear"] + self.restart_commands)
        self.set_positions(frame.pos)
        self.set_images(frame.images)
        self.set_box_data(frame.box_data)
        self.lmp.command("run 0 pre yes post no")

    def _reset_forces_and_virial(self):
        nloc = self.lmp.extract_setting("nlocal")
        if nloc is None:
            raise RuntimeError(
                f"Could not get number of atoms on processor {self.universe.me}"
            )
        force = self.lmp.numpy.fix_external_get_force("ext")
        if force is None:
            raise ValueError("Something went wrong with fix external")
        force[:, :] = np.zeros(shape=force.shape)
        if self.SI.scale_box:
            self.lmp.fix_external_set_virial_global(
                "ext", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )

    def _redistribute_EVB_states(self, num_total_colors, frame):
        if self.universe.total_colors == num_total_colors:
            return
        self.log("REDISTRIBUTE CALLED")
        self.log(f"num_total_colors {num_total_colors}")
        self.lmp.command(f"run 0 pre yes post no")
        if self.universe.rank.color == 0:
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        self.universe.global_comm.Barrier()
        # vel = self.universe.global_comm.bcast(vel, root=0)
        if self.universe.rank.modify:
            self.lmp.close()
        self.universe.available_ranks_to_colors(num_total_colors)
        # self.MPI_info = distribute_mpi_ranks(colors, num_colors, self.universe.me)
        self.universe.global_comm.Barrier()
        if not self.universe.rank.modify:
            return
        cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        if self.universe.debug:
            cmdargs[-1] = f"{self.universe.rank.color}.log"
        self.lmp = lammps(
            name="",
            cmdargs=cmdargs,
            comm=self.universe.lmp_comm,
        )
        self.topology.set_lmp(self.lmp)
        self.lmp.commands_list(self.restart_commands)
        self.set_positions(frame.pos)
        self.set_box_data(frame.box_data)
        self.set_images(frame.images)
        self.lmp.command("run 0 pre yes post no")

    def _get_frame(self, pos=None):
        frame = None
        if self.universe.rank.color == 0:
            if pos is None:
                pos = self.get_positions()
            box_data = self.get_box_data()
            images = self.get_images()
            xyz_pbc = np.array(box_data[1]) - np.array(box_data[0])
            vel = self.get_velocities()
            frame = Topology.Frame(
                pos=pos, box_data=box_data, images=images, xyz_pbc=xyz_pbc, vel=vel
            )
        frame = self.universe.global_comm.bcast(frame, root=0)
        return frame

    def _get_systems(self, pos, xyz_pbc):
        rxn_pairs, rxn_nums, pair_dists, hxy_angles, systems_idxs, pairs_idxs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for rxn_num, rxn in enumerate(self.SI.reactions):
            rxn_info = get_pairs(
                pos,
                xyz_pbc,
                rxn.cutoffs,
                rxn.X,
                self.topology.ST[rxn_num].H_idxs,
                self.topology.ST[rxn_num].Y_idxs,
                self.topology,
            )
            if rxn_info is not None:
                rxn_pairs.append(rxn_info[0])
                rxn_nums.append([rxn_num] * len(rxn_info[0]))
                pair_dists.append(rxn_info[1])
                hxy_angles.append(rxn_info[2])
        if rxn_pairs:
            rxn_pairs = np.concatenate(rxn_pairs, axis=0)
            rxn_nums = np.concatenate(rxn_nums, axis=0)
            pair_dists = np.concatenate(pair_dists, axis=0)
            hxy_angles = np.concatenate(hxy_angles, axis=0)
            systems_idxs, pairs_idxs = self.topology.rxn_pairs_to_systems(
                rxn_pairs,
                rxn_nums,
                pair_dists,
                hxy_angles,
            )
        return list(rxn_pairs), systems_idxs, pairs_idxs

    def _calc_color_pe(self, color, system_idxs, frame, rxn_pairs):
        system = tuple(
            rxn_pairs[pair_idxs] for pair_idxs in system_idxs if pair_idxs is not None
        )
        change_topology = True
        if np.array_equal(self.prev_system, system) and self.safe:
            change_topology = False
        if change_topology:
            self._reset_lmp_topology(frame)
            self.safe = True
        self.set_positions(frame.pos)
        self.set_velocities(frame.vel)
        self.set_images(frame.images)
        if self.SI.scale_box:
            self.set_box_data(frame.box_data)
        if change_topology:
            self.topology.change_topology_to_system(system, frame)
        run = "run 0 pre yes post no"  # FIXME: dont know why but this needs to happen every step
        # if step_count % 1 != 0 and not change_topology:
        #    run = "run 0 pre no post no"
        self.lmp.command(run)  # re-calculate pe w/ new topology
        pe = self.lmp.get_thermo("pe")
        cs = self._get_computes()
        _forces = self.get_forces()
        _virial = None
        if self.SI.scale_box:
            _virial = self.get_virial()
        if self.universe.sub_rank == 0:
            self.log(f"Potential energy for color {color}: {pe}", level="debug")
            self.universe.global_comm.send(pe, dest=0, tag=color)
            self.universe.global_comm.send(_forces, dest=0, tag=color + 200)
            if _virial is not None:
                self.universe.global_comm.send(_virial, dest=0, tag=color + 300)
            if self.SI.computes is not None:
                self.universe.global_comm.send(cs, dest=0, tag=color + 400)
        self.prev_system = system

    def _get_color_info(self, rxn_pairs, systems_idxs, num_systems, frame):
        pes = np.zeros(num_systems)
        forces = np.zeros(shape=(num_systems, len(frame.pos), len(frame.pos[0])))
        virials = None
        computes = None
        if self.SI.scale_box:
            virials = np.zeros((num_systems, 6))
        if self.SI.computes is not None:
            computes = np.zeros((num_systems, len(self.SI.computes)))
        if self.universe.rank.color == 0:
            init_pe = self.lmp.get_thermo("pe")
            init_forces = self.get_forces()
            init_virial = None
            init_computes = None
            if virials is not None:
                init_virial = self.get_virial()
            if computes is not None:
                init_computes = self._get_computes()
            if self.universe.me == 0:
                pes[0] = init_pe
                self.log(f"Potential energy for color 0: {init_pe}", level="debug")
                forces[0] = init_forces
                if virials is not None and init_virial is not None:
                    virials[0, :] = init_virial
                if computes is not None and init_computes is not None:
                    computes[0, :] = init_computes
                for i in range(1, num_systems):
                    source = MPI.ANY_SOURCE
                    pes[i] = self.universe.global_comm.recv(source=source, tag=i)
                    forces[i, :, :] = self.universe.global_comm.recv(
                        source=source, tag=i + 200
                    )
                    if virials is not None:
                        virials[i, :] = self.universe.global_comm.recv(
                            source=source, tag=i + 300
                        )
                    if computes is not None:
                        computes[i, :] = self.universe.global_comm.recv(
                            source=source, tag=i + 400
                        )
        elif self.universe.colors == (None, None):
            self._calc_color_pe(
                self.universe.rank.color,
                systems_idxs[self.universe.rank.color],
                frame,
                rxn_pairs,
            )
        else:
            for color in self.universe.colors:
                self._calc_color_pe(
                    color,
                    systems_idxs[color],
                    frame,
                    rxn_pairs,
                )
        pes = self.universe.global_comm.bcast(pes, root=0)
        forces = self.universe.global_comm.bcast(forces, root=0)
        if self.SI.scale_box:
            virials = self.universe.global_comm.bcast(virials, root=0)
        if self.SI.computes is not None:
            computes = self.universe.global_comm.bcast(computes, root=0)
        return pes, forces, computes, virials

    def _get_coupling(
        self, pair, frame, init_pe, new_pe, forces, init_cmp=None, new_cmp=None
    ):
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
        snapshot = self.topology._get_snapshot(frame)
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
            forces,
        )
        if cpl_forces.shape != forces.shape:
            raise ValueError(
                "Returned coupling forces shape {cpl_forces.shape} should be the same as forces shape {forces.shape}"
            )
        # taper = self.SI.taper_functions[self.topology.rxn_nums_dict[pair]](
        #    *self.topology.rxn_taper_info[pair]
        # )
        return cpl_val, cpl_forces

    def _mix_states_single(self, pes, computes, states, frame, rxn_pairs, forces):
        matrix = np.zeros(shape=(len(pes), len(pes)))
        cpl_forces = np.zeros(shape=(len(pes) - 1, len(frame.pos), len(frame.pos[0])))
        if self.universe.rank.color == 0:
            if self.universe.me == 0:
                matrix[0, 0] = pes[0]
                for i in range(1, len(states)):
                    _cpl_val = self.universe.global_comm.recv(
                        source=MPI.ANY_SOURCE, tag=i
                    )
                    matrix[i, i] = pes[i]
                    matrix[i, 0] = _cpl_val
                    matrix[0, i] = _cpl_val
                    cpl_forces[i - 1, :, :] = self.universe.global_comm.recv(
                        source=MPI.ANY_SOURCE, tag=i + 100
                    )
        elif self.universe.colors == (None, None):
            init_compute, new_compute = None, None
            if computes is not None:
                init_compute = computes[0]
                new_compute = computes[self.universe.rank.color]
            _cpl_val, _cpl_forces = self._get_coupling(
                rxn_pairs[states[self.universe.rank.color]],
                frame,
                pes[0],
                pes[self.universe.rank.color],
                forces[self.universe.rank.color],
                init_compute,
                new_compute,
            )
            if self.universe.sub_rank == 0:
                print("cpl_forces", _cpl_forces)
                self.universe.global_comm.send(
                    _cpl_val, dest=0, tag=self.universe.rank.color
                )
                self.universe.global_comm.send(
                    _cpl_forces, dest=0, tag=self.universe.rank.color + 100
                )
        else:
            for color in self.universe.colors:
                if color is None:
                    raise ValueError("Color is None")
                init_compute, new_compute = None, None
                if computes is not None:
                    init_compute = computes[0]
                    new_compute = computes[color]
                _cpl_val, _cpl_forces = self._get_coupling(
                    rxn_pairs[states[color]],
                    frame,
                    pes[0],
                    pes[color],
                    forces[color],
                    init_compute,
                    new_compute,
                )
                if self.universe.sub_rank == 0:
                    print("cpl_forces", _cpl_forces)
                    self.universe.global_comm.send(_cpl_val, dest=0, tag=color)
                    self.universe.global_comm.send(_cpl_forces, dest=0, tag=color + 100)
        min_eval, min_evec_coeffs = None, None
        if self.universe.me == 0:
            min_eval, min_evec_coeffs = self._get_min_EVB_state(matrix)
        min_eval = self.universe.global_comm.bcast(min_eval, root=0)
        min_evec_coeffs = self.universe.global_comm.bcast(min_evec_coeffs, root=0)

        return min_eval, min_evec_coeffs, cpl_forces

    def _mix_states_scf(
        self,
        pairs_idxs,
        pes,
        computes,
        frame,
        systems_idxs,
        rxn_pairs,
        num_sites,
        forces,
    ):
        us = defaultdict(dict)
        init_idx = pes.argmin()
        matrices = {}
        mixed_computes = {}
        cs = 1
        evals = np.zeros(num_sites)
        new_forces = np.zeros(shape=forces.shape)
        if computes is not None:
            cs = computes.shape[1]
        for n, site in enumerate(pairs_idxs):
            us[n].update({state: 0.0 for state in site})
            us[n][systems_idxs[init_idx][n]] = 1.0
            num_states = len(site)
            matrices[n] = np.zeros(shape=(num_states, num_states))
            mixed_computes[n] = np.zeros(shape=(num_states, cs))
        get_us = np.vectorize(lambda site, state: us[site][state])
        ref_energy = pes[init_idx]
        cycle = 0
        while True:
            if self.universe.me == 0:
                self.logger.debug("initial energy: %s", pes[init_idx])
                self.logger.debug("initial us: %s", us)
            ref_evals = copy(evals)
            for n, site in enumerate(pairs_idxs):
                matrix = matrices[n]
                mixed_compute = mixed_computes[n]
                for m, state in enumerate(site):
                    state_idxs = np.where(systems_idxs[:, n] == state)[0]
                    state_systems = systems_idxs[state_idxs]
                    state_pes = pes[state_idxs]
                    _sites = (
                        np.ones(shape=state_systems.shape) * np.arange(num_sites)
                    ).astype(int)
                    mix = get_us(
                        np.delete(_sites, n, axis=1),
                        np.delete(state_systems, n, axis=1),
                    ).prod(axis=1)
                    # FIXME: COME BACK TO THIS AND TEST
                    matrix[m, m] = np.sum(state_pes * mix)
                    _mix_compute = None
                    if computes is not None:
                        _mix_compute = np.sum(computes[pes] * mix[:, None], axis=0)
                    mixed_compute[m] = _mix_compute
                    if m == 0:
                        continue
                    # TODO: parallelise cpl calculation
                    cpl = self._get_coupling(
                        rxn_pairs[state],
                        frame,
                        matrix[0, 0],
                        matrix[m, m],
                        forces,
                        mixed_compute[0],
                        mixed_compute[m],
                    )
                    matrix[0, m] = cpl
                    matrix[m, 0] = cpl
                min_eval, min_evec_coeffs = None, None
                if self.universe.me == 0:
                    min_eval, min_evec_coeffs = self._get_min_EVB_state(matrix)
                min_eval, min_evec_coeffs = self.universe.global_comm.bcast(
                    (min_eval, min_evec_coeffs), root=0
                )
                amplitudes = min_evec_coeffs**2
                self.universe.global_comm.Barrier()
                us[n].update({state: amplitudes[n] for n, state in enumerate(site)})
                evals[n] = min_eval
            if self.universe.me == 0:
                self.logger.debug("cycle %s energy: %s", cycle, evals)
                self.logger.debug("cycle %s us: %s", cycle, us)
            if (abs(evals - ref_evals) < self.SI.scf_tol).all():
                break
            cycle += 1
            if cycle > self.SI.scf_max_iter:
                raise RuntimeError(
                    f"Could not converge multi-site problem using SCF within {self.SI.scf_max_iter} cycles."
                )
        cpl_forces = None
        return min_eval, min_evec_coeffs, cpl_forces

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

    def _mix_all_states(
        self, pairs_idxs, pes, computes, frame, systems_idxs, rxn_pairs, forces, virials
    ):
        num_sites = len(pairs_idxs)
        if num_sites == 1:
            min_eval, min_evec_coeffs, cpl_forces = self._mix_states_single(
                pes, computes, pairs_idxs[0], frame, rxn_pairs, forces
            )
        else:
            # SCF
            min_eval, min_evec_coeffs, cpl_forces = self._mix_states_scf(
                pairs_idxs,
                pes,
                computes,
                frame,
                systems_idxs,
                rxn_pairs,
                num_sites,
                forces,
            )
        amplitudes = min_evec_coeffs**2
        if self.universe.me == 0:
            self.log("Amplitudes: ", level="debug")
            self.log(amplitudes, level="debug")
        _forces = np.einsum("ijk,i->jk", forces, amplitudes)
        _cpl_forces = np.einsum(
            "ijk,i->jk", cpl_forces, 2 * min_evec_coeffs[0] * min_evec_coeffs[1:]
        )
        if self.universe.me == 0:
            # print(self.universe.rank.color, "forces", forces[self.universe.rank.color])
            print("forces0", forces[0])
            print("forces1", forces[1])
            print("_forces", _forces)
            print("_cpl_forces", _cpl_forces)
        mixed_forces = _forces + _cpl_forces
        mixed_virial = None
        if virials is not None:
            mixed_virial = np.einsum("ij,i->j", virials, amplitudes)
        return min_eval, np.argmax(amplitudes), mixed_forces, mixed_virial

    def _get_new_imgs(self, h, y, frame):
        yids = self.topology.residues[self.topology.atoms[y].molecule]
        yidxs = [self.topology.atoms[ID].idx for ID in yids]
        hpos = frame.pos[self.topology.atoms[h].idx]
        himg = frame.images[self.topology.atoms[h].idx]
        ypos = frame.pos[yidxs]
        disp = ypos - hpos
        disp -= frame.xyz_pbc * (disp / frame.xyz_pbc).round()
        abcabc, _ = extract_box(frame.box_data)
        uhpos = himg * abcabc[:3] + hpos
        uypos = uhpos + disp
        new_imgs = np.floor(uypos / abcabc[:3]).astype(int)
        return new_imgs, yids

    def _out(self, frame, pe):
        if self.universe.rank.color == 0:
            self.Trajectory.write(
                self.step_count,
                self.lmp,
                frame.box_data,
                self.universe,
                self.topology,
            )
            if self.universe.me == 0:
                self.Output.write(self.step_count, pe, self.lmp)

    def _get_mixed_properties(
        self, rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    ):
        pes, init_forces, computes, virials = self._get_color_info(
            rxn_pairs, systems_idxs, num_systems, frame
        )
        self.forces.current_forces = init_forces
        self.forces.current_virial = virials
        return self._mix_all_states(
            pairs_idxs,
            pes,
            computes,
            frame,
            systems_idxs,
            rxn_pairs,
            init_forces,
            virials,
        )

    def finite_differences(
        self, file="finite_differences.out", delta=1e-3, index_array=None
    ):
        if self.universe.me == 0:
            self.Output.log(
                f"Running finite differences calculating with delta {delta} to file {file}"
            )
        if self.universe.rank.color == 0:
            self.lmp.command("run 0 pre yes post no")
        frame = self._get_frame()
        rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
            frame.pos, frame.xyz_pbc
        )
        if not rxn_pairs:
            self.log(
                (
                    "Could not complete finite differences as no "
                    "possible reactions were detected with starting configuration."
                ),
                level="warn",
            )
            return False
        num_systems = len(systems_idxs)
        if self.universe.me == 0:
            self.log(f"Reaction systems: {systems_idxs}", level="debug")
            self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
            self.log(
                f"Starting total colors: {self.universe.total_colors}", level="debug"
            )
            self.log(f"Number of systems: {num_systems}", level="debug")
        self._redistribute_EVB_states(num_systems, frame)
        if self.universe.me == 0:
            self.log(f"New total colors: {self.universe.total_colors}", level="debug")
        _, _, ref_mixed_forces, _ = self._get_mixed_properties(
            rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
        )
        pos_orig = None
        if self.universe.rank.color == 0:
            pos_orig = self.get_positions()
        pos_orig = self.universe.global_comm.bcast(pos_orig, root=0)
        check_forces = np.zeros(shape=ref_mixed_forces.shape)
        if index_array is None:
            index_array = range(len(pos_orig))
        if len(index_array) > len(pos_orig):
            raise ValueError("Index array length is greater than number of particles")

        for particle in index_array:
            for coord in range(3):
                self.Output.log(
                    f"calculating force {particle*3 + coord + 1} / {len(pos_orig) * 3}"
                )
                _pos = copy(pos_orig)
                _pos[particle][coord] = pos_orig[particle][coord] + delta
                self.set_positions(_pos)
                self.lmp.command("run 0 post no")
                frame = self._get_frame(pos=_pos)
                pos_m_eval, _, _, _ = self._get_mixed_properties(
                    rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
                )
                _pos = copy(pos_orig)
                _pos[particle][coord] = pos_orig[particle][coord] - delta
                self.set_positions(_pos)
                self.lmp.command("run 0 post no")
                frame = self._get_frame(pos=_pos)
                neg_m_eval, _, _, _ = self._get_mixed_properties(
                    rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
                )
                check_forces[particle][coord] = -(pos_m_eval - neg_m_eval) / (2 * delta)

        if self.universe.me == 0:
            diff = ref_mixed_forces - check_forces
            norm_diff = abs(diff) / abs(ref_mixed_forces)
            df = pd.DataFrame(
                {
                    "Analytic": ref_mixed_forces.flatten(),
                    "Finite_differences": check_forces.flatten(),
                    "Diff": diff.flatten(),
                    "Abs_diff": abs(diff).flatten(),
                    "Norm_diff": norm_diff.flatten(),
                }
            )
            df.to_csv(file, sep="\t", index=False)
            self.Output.log(f"Finite differences written to {file}")
        self.universe.global_comm.Barrier()
        return True

    def _force_step(self):
        if self.universe.rank.color == 0:
            self._reset_forces_and_virial()
        frame = self._get_frame()
        rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
            frame.pos, frame.xyz_pbc
        )
        if not rxn_pairs:
            self._redistribute_EVB_states(1, frame)
            pe = self.lmp.get_thermo("pe")
            self._out(frame, pe)
            self.num_colors = 1
            self.rebuild = False
            self.forces.set_mixed_forces(None)
            self.forces.set_mixed_virial(None)
            return None
        num_systems = len(systems_idxs)
        self.log(f"Reaction systems: {systems_idxs}", level="debug")
        self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
        self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
        self.log(f"Number of systems: {num_systems}", level="debug")
        self._redistribute_EVB_states(num_systems, frame)
        self.log(f"New total colors: {self.universe.total_colors}", level="debug")
        (
            min_eval,
            min_state_idx,
            mixed_forces,
            mixed_virial,
        ) = self._get_mixed_properties(
            rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
        )
        self.forces.set_mixed_forces(mixed_forces)
        self.forces.current_forces = self.forces.current_forces[min_state_idx]
        if self.SI.scale_box:
            self.forces.set_mixed_virial(mixed_virial)
            self.forces.current_virial = self.forces.current_virial[min_state_idx]
        self.log("Minimum eigen value: ", level="debug")
        self.log(min_eval, level="debug")
        self.log("Minimum state: ", level="debug")
        self.log(min_state_idx, level="debug")
        self._out(frame, min_eval)
        if min_state_idx == 0:
            self.universe.global_comm.Barrier()
            self.rebuild = False
            return None

        # reaction has occured - update topology
        min_system_idxs = systems_idxs[min_state_idx]
        min_system = tuple(
            rxn_pairs[pair_idxs]
            for pair_idxs in min_system_idxs
            if pair_idxs is not None
        )
        self.log(f"min system: {min_system}", level="debug")
        new_imgs, yids = [], []
        for h, y in min_system:
            _new_imgs, _yids = self._get_new_imgs(h, y, frame)
            new_imgs += list(_new_imgs)
            yids += list(_yids)
        self.lmp.commands_list(["reset_atoms mol all single yes", "run 0 post no"])
        if self.universe.rank.color == 0:
            self.topology.change_topology_to_system(min_system, frame)
            self.lmp.commands_list(
                [
                    f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
                    for ID, imgs in zip(yids, new_imgs)
                ]
                + [
                    "reset_atoms mol all single yes",
                    "run 0 post no",
                ]
            )
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        self.universe.global_comm.Barrier()
        if self.universe.rank.color != 0:
            self._reset_lmp_topology(frame)
        self.universe.global_comm.Barrier()
        self.topology.build_topology()
        self.safe = False
        self.prev_system = np.array([[None, None]])
        self.rebuild = True
        self.universe.global_comm.Barrier()
        return min_system

    def _step(self, n_step=1, nl_update=None, mini=False):
        self.forces._set_step(n_step)
        rxn = self._force_step()
        if rxn is not None and self.universe.me == 0 and not mini:
            for pair in rxn:
                self.Output.log(
                    f"reaction occured at step {self.step_count} between IDs {pair[0]} and {pair[1]}"
                )
        if self.universe.rank.color == 0:
            self._set_forces()
            if self.SI.scale_box:
                self._set_virial()
            ## Integrate equations of motion
            # pre yes computes neighlist
            run = f"run {n_step} pre yes post no"
            if nl_update is None:
                nl_update = self.SI.nl_update
            # if self.step_count % nl_update != 0:
            if self.step_count % nl_update != 0 and not self.rebuild:
                # pre no stops computing neighlist
                run = f"run {n_step} pre no post no"
            self.lmp.command(run)
        self.universe.global_comm.Barrier()
        return rxn

    def add_output(
        self, filename=None, properties=["temp", "pe", "vol"], write_frequency=1000
    ):
        self.Output.add_output(
            Output.output(
                self.lmp,
                self.universe.me,
                filename,
                properties,
                write_frequency,
            )
        )

    def add_trajectory(
        self, filename="trajectory.dcd", write_frequency=1000, rxn=False
    ):
        self.Trajectory.add_trajectory(
            Trajectory.trajectory(filename, write_frequency, rxn)
        )

    def set_box_data(self, box_data):
        """
        Assumes box is already triclinic
        """
        self.lmp.command(
            (
                "change_box all "
                f"x final {box_data[0][0]} {box_data[1][0]} "
                f"y final {box_data[0][1]} {box_data[1][1]} "
                f"z final {box_data[0][2]} {box_data[1][2]} "
                f"xy final {box_data[2]} xz final {box_data[4]} yz final {box_data[3]}"
            )
        )

    def get_box_data(self):
        return self.lmp.extract_box()

    def set_images(self, images):
        self.lmp.scatter_atoms("image", 0, 3, convert_to_c_type(images, c_int))

    def get_images(self):
        return np.array(gather_atoms(self.lmp, "image", 0, 3)).reshape(-1, 3)

    def set_velocities(self, velocities):
        self.lmp.scatter_atoms("v", 1, 3, convert_to_c_type(velocities, c_double))

    def get_velocities(self):
        return np.array(gather_atoms(self.lmp, "v", 1, 3)).reshape(-1, 3)

    def set_positions(self, positions):
        self.lmp.scatter_atoms("x", 1, 3, convert_to_c_type(positions, c_double))

    def get_positions(self):
        return np.array(gather_atoms(self.lmp, "x", 1, 3)).reshape(-1, 3)

    def _set_forces(self):
        force = self.lmp.numpy.fix_external_get_force("ext")
        if force is None:
            raise ValueError("Something went wrong with fix external")
        ids = self.lmp.numpy.extract_atom("id")
        if ids is None:
            raise RuntimeError("ids is None")
        idxs = self.topology.id_to_idx(ids)
        force_diff = self.forces._calc_force_diff()
        if self.universe.me == 0:
            print("mixed forces")
            print(self.forces.mixed_forces)
        force[:, :] = force_diff[idxs]

    def get_forces(self):
        return np.array(gather_atoms(self.lmp, "f", 1, 3)).reshape(-1, 3)

    def _set_virial(self):
        if self.forces.mixed_virial is None:
            raise RuntimeError(
                "Call was made to set virial but it is None. This is probably unintentional."
            )
        virial_diff = self.forces._calc_virial_diff()
        self.lmp.fix_external_set_virial_global("ext", list(virial_diff))

    def get_virial(self, vol=None):
        if vol is None:
            vol = self.lmp.get_thermo("vol")
        p_vir = self.lmp.numpy.extract_compute("pre_vir", 0, 1)
        if p_vir is None:
            raise ValueError("Could not extract virial")
        vir = p_vir / self.SI.units["pr2vir"] * vol  # type: ignore
        return vir

    def _get_computes(self):
        if self.SI.computes is not None:
            cs = np.zeros(len(self.SI.computes))
            for n, c_id in enumerate(self.SI.computes):
                c = self.lmp.extract_compute(c_id, 0, 0)
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
        return None

    def minimise(self, cmd_list):
        if self.universe.rank.color == 0:
            self.log("Regular energy minimisation called...")
            self.lmp.commands_list(cmd_list)
            self.log("...system minimised.")
        self.universe.global_comm.Barrier()

    def msevb_minimise(self, traj=True, scale_box=False):
        self.step_count = 0.2
        if scale_box:
            raise ValueError(
                "MSEVB minimisation with fluctuating box not implemented yet. Flag is there to remind me :)"
            )
        if traj:
            self.add_trajectory(filename="minimise.dcd", write_frequency=1)
        frame = self._get_frame()
        u_frame_pos = frame.images * frame.xyz_pbc + frame.pos

        def objective(coords):
            upos = coords.reshape(len(frame.pos), len(frame.pos[0]))
            diff = upos - frame.pos
            diff -= frame.xyz_pbc * (diff / frame.xyz_pbc).round()
            nupos = u_frame_pos - diff
            new_imgs = np.floor(nupos / frame.xyz_pbc).astype(int)
            pos = nupos - frame.xyz_pbc * new_imgs
            new_frame = Topology.Frame(
                pos=pos,
                box_data=frame.box_data,
                images=new_imgs,
                xyz_pbc=frame.xyz_pbc,
                vel=frame.vel,
            )
            self.set_positions(pos)
            self.set_images(new_imgs)
            self.lmp.command("run 0 pre yes post no")
            if self.universe.rank.color == 0 and traj:
                self.Trajectory.trajs[-1].write(
                    1,
                    self.lmp,
                    frame.box_data,
                    self.universe,
                    self.topology,
                    pos=nupos,
                )
            rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(pos, frame.xyz_pbc)
            if rxn_pairs:
                num_systems = len(systems_idxs)
                self.log(f"Reaction systems: {systems_idxs}", level="debug")
                self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
                self.log(
                    f"Starting total colors: {self.universe.total_colors}",
                    level="debug",
                )
                self.log(f"Number of systems: {num_systems}", level="debug")
                self._redistribute_EVB_states(num_systems, frame)
                self.log(
                    f"New total colors: {self.universe.total_colors}", level="debug"
                )
                min_eval, _, mixed_forces, _ = self._get_mixed_properties(
                    rxn_pairs, systems_idxs, num_systems, new_frame, pairs_idxs
                )
                return min_eval, mixed_forces.flatten()
            else:
                pe = self.lmp.get_thermo("pe")
                forces = self.get_forces()
                return pe, forces.flatten()

        init_pe, _ = objective(frame.pos)
        self.log("Running minimisation...")
        if self.universe.me == 0:
            reg_pe = self.lmp.get_thermo("pe")
            self.log(f"starting non-mixed potential energy for minimisation: {reg_pe} ")
        self.log(
            f"starting mixed potential energy for minimisation: {init_pe} ",
        )
        self.log(" updating initial topology...")
        cycle = 0
        while True:
            self.log(f"  cycle: {cycle}")
            reaction = self._step(n_step=0, nl_update=1, mini=True)
            if not reaction:
                self.log(" ...no change in topology")
                break
            self.log("    topology updated")
            cycle += 1
            if cycle > 100:
                break

        objective_values = []

        def callback(xk):
            objective_values.append(objective(xk))

        result = minimize(
            objective,
            frame.pos.flatten(),
            method="L-BFGS-B",
            jac=True,
            tol=1e-6,
            callback=callback,
        )
        self.log(result, level="debug")
        # self.logger.info("Objective function values at each step:")
        for i, value in enumerate(objective_values):
            self.log(f"  step {i}: {value[0]:.4f}")
        self.log(f"...minimised MSEVB potential energy: {result.fun}")

        upos = result.x.reshape(len(frame.pos), len(frame.pos[0]))
        diff = upos - frame.pos
        diff -= frame.xyz_pbc * (diff / frame.xyz_pbc).round()
        nupos = u_frame_pos - diff
        new_imgs = np.floor(nupos / frame.xyz_pbc).astype(int)
        pos = nupos - frame.xyz_pbc * new_imgs

        self.set_positions(pos)
        self.set_images(new_imgs)
        self.lmp.command("run 0 pre yes post no")
        self.Trajectory.trajs.pop()
        self.step_count = 0

    def step(self, steps):
        if self.universe.me == 0:
            self.Output._header()

        if steps == 0:
            self._step(n_step=0)
            return

        for _ in range(steps):
            self._step()
            self.step_count += 1

    def __del__(self):
        if hasattr(self, "data_io"):
            try:
                os.remove(f"/tmp/{self.SI.file}")
            except OSError:
                pass
