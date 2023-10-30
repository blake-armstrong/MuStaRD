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
from .mpi import Universe
from .utils import get_pairs, convert_to_c_type, gather_atoms, extract_box
from .mixing import get_FD_occupancies


# dt = self.lmp.extract_global("dt")
# if type(dt) != float:
#     raise ValueError
# ftm2v = self.lmp.extract_global("ftm2v")
# if type(ftm2v) != float:
#     raise ValueError
# self.dtv = dt
# self.dtf = 0.5 * dt * ftm2v
# self.dtfm = self.dtf / self.topology.masses
# current_forces = self._get_forces()
# vel = frame.vel + self.dtfm[:, None] * current_forces
# pos_next = frame.pos + self.dtv * vel




# class Forces:
#     def __init__(self, current_forces, current_virial):
#         self.current_forces = current_forces
#         self.mixed_forces = current_forces
#         self.current_virial = current_virial
#         self.mixed_virial = current_virial
#         self.has_run_0_been_called = False
#         self.step = 1
#         self.prev_step = 1
#         self.empty_forces = np.zeros(shape=self.current_forces.shape)
#         self.empty_virial = np.zeros(6)
#         self.force_diff = copy(self.empty_forces)
#         self.virial_diff = copy(self.empty_virial)
#         self.dtf = 1.0
#         self.dftm = np.zeros(shape=len(self.current_forces))
#         self.dtv = 1.0
#
#     def set_mixed_forces(self, mixed_forces):
#         if self.prev_step == 0:
#             self.mixed_forces = self.current_forces
#         else:
#             self.mixed_forces = mixed_forces
#
#     def set_mixed_virial(self, mixed_virial):
#         if self.prev_step == 0:
#             self.mixed_virial = self.current_virial
#         else:
#             self.mixed_virial = mixed_virial
#
#     def _set_step(self, step):
#         self.prev_step = self.step
#         self.step = step
#
#     def _calc_force_diff(self):
#         if self.prev_step == 0:
#             return self.force_diff
#         if self.mixed_forces is None:
#             return self.empty_forces
#         if np.isnan(self.mixed_forces).any():
#             raise ValueError("One or more forces are NaN")
#         self.force_diff = self.mixed_forces - self.current_forces
#         return self.force_diff
#
#     def _calc_virial_diff(self):
#         if self.prev_step == 0:
#             return self.virial_diff
#         if self.mixed_virial is None:
#             return self.empty_virial
#         if np.isnan(self.mixed_virial).any():
#             raise ValueError("One or more virial components are NaN")
#         self.virial_diff = self.mixed_virial - self.current_virial
#         return self.virial_diff


class Mustard:
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
        self.universe = Universe(mpi_list, debug)
        self.log = self.universe.log
        # List of parameters to broadcast
        params_to_broadcast = (
            lmp_coord_file,
            force_field_file,
            header,
            commands,
            reaction_parameters,
            mpi_list,
            debug,
        )
        # Broadcast all parameters from rank 0
        broadcasted_params = self.universe.global_comm.bcast(
            params_to_broadcast, root=0
        )
        # Unpack the broadcasted parameters
        (
            lmp_coord_file,
            force_field_file,
            header,
            commands,
            reaction_parameters,
            mpi_list,
            debug,
        ) = broadcasted_params
        # SYSTEM INFO
        self.SI = SystemInfo(reaction_parameters)
        self.get_occupancies = self._set_get_occupancies()
        self.prev_system = np.array([[None, None]])
        self.safe = False
        self.num_colors = 1
        self.step_count = 0
        self.rebuild = True
        self.SI.file = "mustard." + self.universe.global_comm.bcast(
            str(uuid.uuid4().hex), root=0
        )
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
            if self.universe.rank.color == 0:
                cmdargs[-1] = "state_main.log"
            else:
                cmdargs[-1] = f"state_{self.universe.rank.color}.log"
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
                "thermo_style custom etotal epair ebond",
            ]
        )
        virial = "compute pre_vir all pressure NULL virial"
        if self.SI.scale_box:
            self.lmp.command(virial)
        # initial_forces = None
        # if self.universe.rank.color == 0:
        self.lmp.commands_list(commands)
        self.lmp.command("run 0 post no")
        self.box_data = self.get_box_data(self.lmp)
        self.xyz_pbc = np.array(self.box_data[1]) - np.array(self.box_data[0])

        if self.universe.rank.color == 0:
            self.lmp.command("variable min_state_idx string 0")
            self.lmp.command("variable system string [[None]]")
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")

        fixes = [
            "fix ux all store/state 1 xu yu zu",
            "fix ext all external pf/callback 1 1",
            # "fix ext all external pf/array 1",
            "fix_modify ext energy no",
            "fix_modify ext virial no",
        ]
        if self.SI.scale_box:
            fixes[-1] = fixes[-1].replace("no", "yes")
        # if self.universe.rank.color == 0:
        self.lmp.commands_list(fixes)
        self.restart_commands = {
            "header": header,
            "read_data": [f"read_data /tmp/{self.SI.file}"],
            "change_box": ["change_box all triclinic"],
            "force_field": [f"include {force_field_file}"],
            "fixes": fixes,
            "virial": [""],
        }
        if self.SI.scale_box:
            self.restart_commands["virial"][0] = virial
        self.topology = Topology(self.lmp, self.SI)
        callback = self.generate_callback()
        self.lmp.set_fix_external_callback("ext", callback, self.lmp)
        self.Trajectory = Trajectory()
        self.Output = Output()
        self.lmp.command("run 1 post no")
        print("pos", self.get_positions(self.lmp))
        # self._reset_forces_and_virial()
        # print("init_forces", initial_forces)
        # frame = self._get_frame()
        # self.log(f"pos {frame.pos}")
        # rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
            # frame.pos, frame.xyz_pbc
        # )
        # if self.universe.rank.color == 1:
        #     system_idxs = systems_idxs[1]
        #     system = tuple(
        #         rxn_pairs[pair_idxs] for pair_idxs in system_idxs if pair_idxs is not None
        #     )
        #     self.topology.change_topology_to_system(system, frame)
        # self.lmp.command("variable previous_system string 0,1")
        # if self.universe.rank.color == 0:
        # self.lmp.command("run 1 post no")
        #     # self.lmp.command("run 1 post no")
        #         # print("break")
        # if self.universe.rank.color == 0:
        #     post_forces = self._get_forces()
        #     print("post_forces", post_forces)
        # self.lmp.command("run 1 post no")
        # print(self.lmp.extract_variable("previous_system"))

        #        self.restart_commands = {
        #            "header": header,
        #            "read_data": [f"read_data /tmp/{self.SI.file}"],
        #            "change_box": ["change_box all triclinic"],
        #            "force_field": [f"include {force_field_file}"],
        #            "virial": [""],
        #        }
        #        if self.SI.scale_box:
        #            self.restart_commands["virial"][0] = virial
        #        self.topology = Topology(self.lmp, self.SI)
        #        pos = self.get_positions()
        #        print("pos", pos)
        #        frame = self._get_frame(pos=pos, init=True)
        #        rxn_pairs, systems_idxs, _ = self._get_systems(frame.pos, frame.xyz_pbc)
        #        num_systems = 1
        #        if rxn_pairs:
        #            num_systems = len(systems_idxs)
        #        self._redistribute_EVB_states(num_systems, frame)
        #        dt, ftm2v = None, None
        #        if self.universe.rank.color == 0:
        #            # positions = np.loadtxt("pos.txt")
        #            # velocities = np.loadtxt("vel.txt") * 0
        #            # velocities[0] = np.array([1000, 1000, 1000])
        #            # positions = np.trunc(positions * 10**3) / (10**3)
        #            # print(positions)
        #            # self.set_positions(positions)
        #            # self.set_velocities(velocities)
        #            dt = self.lmp.extract_global("dt")
        #            ftm2v = self.lmp.extract_global("ftm2v")
        #            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        #        dt, ftm2v = self.universe.global_comm.bcast((dt, ftm2v), root=0)
        #        self.lmp.command("run 1 post no")
        #        # self.forces.has_run_0_been_called = True
        #        self.forces.dtv = dt
        #        self.forces.dtf = 0.5 * dt * ftm2v
        #        self.forces.dtfm = self.forces.dtf / self.topology.masses
        #        print("forces", self._get_forces())

    def _initialise_forces_and_virial(self, initial_forces):
        forces = None
        if self.universe.rank.color == 0:
            current_virial = None
            if self.SI.scale_box:
                current_virial = self.get_virial()
            if initial_forces is None:
                raise ValueError("Initial forces is None")
            forces = Forces(initial_forces, current_virial)
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
        self.lmp.commands_list(
            ["clear"]
            + self.restart_commands["header"]
            + self.restart_commands["read_data"]
        )
        self.set_positions(self.lmp, frame.pos)
        self.set_images(self.lmp, frame.images)
        self.set_velocities(self.lmp, frame.vel)
        self.lmp.commands_list(
            self.restart_commands["change_box"]
            + self.restart_commands["force_field"]
            + self.restart_commands["virial"]
            + ["thermo_style custom etotal epair ebond"]
        )
        self.set_box_data(self.lmp, frame.box_data)
        # self.lmp.commands_list(self.restart_commands["fixes"])
        self.lmp.command("run 0 pre yes post no")

    # def _reset_forces_and_virial(self):
    #     nloc = self.lmp.extract_setting("nlocal")
    #     if nloc is None:
    #         raise RuntimeError(
    #             f"Could not get number of atoms on processor {self.universe.me}"
    #         )
    #     force = self.lmp.numpy.fix_external_get_force("ext")
    #     if force is None:
    #         raise ValueError("Something went wrong with fix external")
    #     force[:, :] = np.zeros(shape=force.shape)
    #     if self.SI.scale_box:
    #         self.lmp.fix_external_set_virial_global(
    #             "ext", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #         )

    def _redistribute_EVB_states(self, num_total_colors, frame):
        if num_total_colors <= self.universe.total_colors:
            return
        self.log("REDISTRIBUTE CALLED")
        self.log(f"num_total_colors {num_total_colors}")
        # if self.universe.rank.color != 0:
            # self.lmp.command(f"run 0 pre yes post no")
        if self.universe.rank.color == 0:
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        self.universe.global_comm.Barrier()
        if self.universe.rank.modify:
            self.lmp.close()
        self.universe.available_ranks_to_colors(num_total_colors)
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
        # self.lmp.commands_list(self.restart_commands)
        self.lmp.commands_list(
            self.restart_commands["header"] + self.restart_commands["read_data"]
        )
        self.set_positions(self.lmp, frame.pos)
        self.set_images(self.lmp, frame.images)
        self.lmp.commands_list(
            self.restart_commands["change_box"]
            + self.restart_commands["force_field"]
            + self.restart_commands["virial"]
        )
        self.set_box_data(self.lmp, frame.box_data)
        self.lmp.commands_list(self.restart_commands["fixes"])
        self.lmp.command("run 0 pre yes post no")

    # def _get_frame(self, pos=None, init=False):
    #     frame = None
    #     if self.universe.rank.color == 0:
    #         vel = self.get_velocities()
    #         pos = self.get_positions()
    #         # if pos is None:
    #         #     vel += self.forces.dtfm * self.forces.current_forces
    #         #     pos = pos0 + self.forces.dtv * vel
    #         # if not init:
    #         #     if self.forces.step == 0:
    #         #         pos = pos0
    #         box_data = self.get_box_data()
    #         images = self.get_images()
    #         xyz_pbc = np.array(box_data[1]) - np.array(box_data[0])
    #         vel = self.get_velocities()
    #         frame = Topology.Frame(
    #             pos=pos, box_data=box_data, images=images, xyz_pbc=xyz_pbc, vel=vel
    #         )
    #     frame = self.universe.global_comm.bcast(frame, root=0)
    #     return frame

    def _get_systems(self, pos, xyz_pbc):
        rxn_pairs = []
        rxn_nums = []
        pair_dists = []
        hxy_angles = []
        systems_idxs = []
        pairs_idxs = []
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
        eb = self.lmp.get_thermo("ebond")
        ep = self.lmp.get_thermo("epair")
        if self.universe.sub_rank == 0:
            print("ebond 1", self.step_count, eb)
            print("epair 1", self.step_count, ep)
        pe = self.lmp.get_thermo("pe")
        cs = self._get_computes()
        _forces = self.get_forces()
        _virial = None
        if self.SI.scale_box:
            _virial = self.get_virial()
        if self.universe.sub_rank == 0:
            self.log(
                f"Potential energy for color {color}: {pe}", level="debug", rank=-1
            )
            self.universe.global_comm.send(pe, dest=0, tag=color)
            self.universe.global_comm.send(_forces, dest=0, tag=color + 200)
            if _virial is not None:
                self.universe.global_comm.send(_virial, dest=0, tag=color + 300)
            if self.SI.computes is not None:
                self.universe.global_comm.send(cs, dest=0, tag=color + 400)
        self.prev_system = system

    # def _get_color_info(self, rxn_pairs, systems_idxs, num_systems, frame):
    #     pes = np.zeros(num_systems)
    #     forces = np.zeros(shape=(num_systems, len(frame.pos), len(frame.pos[0])))
    #     virials = None
    #     computes = None
    #     if self.SI.scale_box:
    #         virials = np.zeros((num_systems, 6))
    #     if self.SI.computes is not None:
    #         computes = np.zeros((num_systems, len(self.SI.computes)))
    #     if self.universe.rank.color == 0:
    #         init_pe = self.lmp.get_thermo("pe")
    #         # init_forces = self.get_forces()
    #         init_virial = None
    #         init_computes = None
    #         if virials is not None:
    #             init_virial = self.get_virial()
    #         if computes is not None:
    #             init_computes = self._get_computes()
    #         if self.universe.me == 0:
    #             pes[0] = init_pe
    #             self.log(f"Potential energy for color 0: {init_pe}", level="debug")
    #             forces[0] = init_forces
    #             if virials is not None and init_virial is not None:
    #                 virials[0, :] = init_virial
    #             if computes is not None and init_computes is not None:
    #                 computes[0, :] = init_computes
    #             for i in range(1, num_systems):
    #                 source = MPI.ANY_SOURCE
    #                 pes[i] = self.universe.global_comm.recv(source=source, tag=i)
    #                 forces[i, :, :] = self.universe.global_comm.recv(
    #                     source=source, tag=i + 200
    #                 )
    #                 if virials is not None:
    #                     virials[i, :] = self.universe.global_comm.recv(
    #                         source=source, tag=i + 300
    #                     )
    #                 if computes is not None:
    #                     computes[i, :] = self.universe.global_comm.recv(
    #                         source=source, tag=i + 400
    #                     )
    #     elif self.universe.colors == (None, None):
    #         if self.universe.rank.color < len(systems_idxs):
    #             self._calc_color_pe(
    #                 self.universe.rank.color,
    #                 systems_idxs[self.universe.rank.color],
    #                 frame,
    #                 rxn_pairs,
    #             )
    #     else:
    #         for color in self.universe.colors:
    #             self._calc_color_pe(
    #                 color,
    #                 systems_idxs[color],
    #                 frame,
    #                 rxn_pairs,
    #             )
    #     pes = self.universe.global_comm.bcast(pes, root=0)
    #     forces = self.universe.global_comm.bcast(forces, root=0)
    #     if self.SI.scale_box:
    #         virials = self.universe.global_comm.bcast(virials, root=0)
    #     if self.SI.computes is not None:
    #         computes = self.universe.global_comm.bcast(computes, root=0)
    #     return pes, forces, computes, virials

    def _get_coupling(
        self, pair, frame, init_pe, new_pe, init_cmp=None, new_cmp=None
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
        snapshot = self.topology._get_snapshot(frame, self.step_count)
        rxn_ids = {"x_id": x, "h_id": h, "y_id": y}
        # self.rxn_ids = rxn_ids
        print("xhy", self.step_count, x, h, y)
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


    def _mix_states_single(self, pes, computes, frame, system_info):
        rxn_pairs, _, pairs_idxs = system_info
        states = pairs_idxs[0]
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
            cpl_val, cpl_forces = self._get_coupling(
                rxn_pairs[states[self.universe.rank.color]],
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
            min_eval, min_evec_coeffs = self._get_min_EVB_state(matrix)
        min_eval, min_evec_coeffs, cpl_forces = self.universe.global_comm.bcast((min_eval, min_evec_coeffs, cpl_forces), root=0)

        return min_eval, min_evec_coeffs, cpl_forces

    def _mix_states_scf(
        self,
        pes,
        computes,
        frame,
        system_info,
    ):
        rxn_pairs, systems_idxs, pairs_idxs = system_info
        num_sites = len(pairs_idxs)
        us = defaultdict(dict)
        init_idx = pes.argmin()
        matrices = {}
        mixed_computes = {}
        cs = 1
        evals = np.zeros(num_sites)
        new_forces = np.zeros(shape=frame.forces.shape)
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
                pes, computes, pairs_idxs, frame, rxn_pairs
            )
        else:
            # SCF
            min_eval, min_evec_coeffs, cpl_forces = self._mix_states_scf(
                pes,
                computes,
                pairs_idxs,
                frame,
                rxn_pairs,
                systems_idxs,
            )
        amplitudes = min_evec_coeffs**2
        if self.universe.me == 0:
            self.log("Amplitudes: ", level="debug")
            self.log(amplitudes, level="debug")
        h_idxs = self.topology.atoms[self.rxn_ids["h_id"]].idx
        x_idxs = self.topology.atoms[self.rxn_ids["x_id"]].idx
        y_idxs = self.topology.atoms[self.rxn_ids["y_id"]].idx
        self.log(f"forces0 h {forces[0][h_idxs]}")
        self.log(f"forces0 x {forces[0][x_idxs]}")
        self.log(f"forces0 y {forces[0][y_idxs]}")
        self.log(f"forces1 h {forces[1][h_idxs]}")
        self.log(f"forces1 x {forces[1][x_idxs]}")
        self.log(f"forces1 y {forces[1][y_idxs]}")
        # forces[0]
        # new_forces = forces[0] - self.forces.force_diff
        # self.log(f"new forces 0 {new_forces}")
        # self.lmp.command("run 0 pre yes")
        # self.forces.has_run_0_been_called = True
        # f = self.get_forces()
        # self.log(f"new forces run 0 h {f[0]}")
        # self.log(f"new forces run 0 x {f[1]}")
        # self.log(f"new forces run 0 y {f[12]}")
        # f = self.lmp.numpy.extract_atom("f")
        # ids = self.lmp.numpy.extract_atom("id")
        # idxs = self.topology.id_to_idx(ids)
        # f = f[idxs]
        # self.log(f"new extracted forces 0 {f}")
        _forces = np.einsum("ijk,i->jk", forces, amplitudes)
        _cpl_forces = np.einsum(
            "ijk,i->jk", cpl_forces, 2 * min_evec_coeffs[0] * min_evec_coeffs[1:]
        )
        if self.universe.me == 0:
            print("pos h ", frame.pos[h_idxs])
            print("pos x ", frame.pos[x_idxs])
            print("pos y ", frame.pos[y_idxs])
            print(" mixed_cpl_forces: h", _cpl_forces[h_idxs])
            print(" mixed_cpl_forces: x", _cpl_forces[x_idxs])
            print(" mixed_cpl_forces: y", _cpl_forces[y_idxs])
        mixed_forces = _forces + _cpl_forces
        # for i in range(20):
        self.log(f"mixed forces h {mixed_forces[h_idxs]}")
        self.log(f"mixed forces x {mixed_forces[x_idxs]}")
        self.log(f"mixed forces y {mixed_forces[y_idxs]}")
        # self.log(f"pos {frame.pos}")
        mixed_virial = None
        self.log("min eval", min_eval)
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

    def _out(self, lmp, frame):
        if self.universe.rank.color == 0:
            pe = lmp.get_termo("pe")
            self.Trajectory.write(
                self.step_count,
                lmp,
                frame.box_data,
                self.universe,
                self.topology,
            )
            if self.universe.me == 0:
                self.Output.write(self.step_count, pe, lmp)

    def _get_mixed_properties(
        self, rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    ):
        pes, init_forces, computes, virials = self._get_color_info(
            rxn_pairs, systems_idxs, num_systems, frame
        )
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
        self.log("START")
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

    # def _force_step(self):
    #     self.prev_system = np.array([[None, None]])
    #     if self.universe.rank.color == 0:
    #         self._reset_forces_and_virial()
    #     frame = self._get_frame()
    #     self.log(f"pos {frame.pos}")
    #     rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
    #         frame.pos, frame.xyz_pbc
    #     )
    #     if not rxn_pairs:
    #         self._redistribute_EVB_states(1, frame)
    #         pe = self.lmp.get_thermo("pe")
    #         self._out(frame, pe)
    #         self.num_colors = 1
    #         self.rebuild = False
    #         self.forces.set_mixed_forces(None)
    #         self.forces.set_mixed_virial(None)
    #         # self._set_forces()
    #         return None
    #     num_systems = len(systems_idxs)
    #     self.log(f"Reaction systems: {systems_idxs}", level="debug")
    #     self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
    #     self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
    #     self.log(f"Number of systems: {num_systems}", level="debug")
    #     self._redistribute_EVB_states(num_systems, frame)
    #     self.log(f"New total colors: {self.universe.total_colors}", level="debug")
    #     (
    #         min_eval,
    #         min_state_idx,
    #         mixed_forces,
    #         mixed_virial,
    #     ) = self._get_mixed_properties(
    #         rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    #     )
    #     self.forces.set_mixed_forces(mixed_forces)
    #     # if min_state_idx != 0:
    #     #     self.Output.log("reaction would have occurred")
    #     # min_state_idx = 0
    #     self.forces.current_forces = self.forces.current_forces[min_state_idx]
    #     if self.SI.scale_box:
    #         self.forces.set_mixed_virial(mixed_virial)
    #         self.forces.current_virial = self.forces.current_virial[min_state_idx]
    #     self.log("Minimum eigen value: ", level="debug")
    #     self.log(min_eval, level="debug")
    #     self.log("Minimum state: ", level="debug")
    #     self.log(min_state_idx, level="debug")
    #     self._out(frame, min_eval)
    #     # min_state_idx = 0
    #     eb = self.lmp.get_thermo("ebond")
    #     ep = self.lmp.get_thermo("epair")
    #     self.log(f"ebond 0 {self.step_count} {eb}")
    #     self.log(f"epair 0 {self.step_count} {ep}")
    #     if min_state_idx == 0:
    #         self.universe.global_comm.Barrier()
    #         self.rebuild = False
    #         return None
    #
    #     # reaction has occured - update topology
    #     min_system_idxs = systems_idxs[min_state_idx]
    #     min_system = tuple(
    #         rxn_pairs[pair_idxs]
    #         for pair_idxs in min_system_idxs
    #         if pair_idxs is not None
    #     )
    #     self.log(f"min system: {min_system}", level="debug")
    #     new_imgs, yids = [], []
    #     for h, y in min_system:
    #         _new_imgs, _yids = self._get_new_imgs(h, y, frame)
    #         new_imgs += list(_new_imgs)
    #         yids += list(_yids)
    #     ke = self.lmp.get_thermo("ke")
    #     self.log(f"ke before {ke}")
    #     self.lmp.commands_list(["reset_atoms mol all single yes", "run 0 post no"])
    #     ke = self.lmp.get_thermo("ke")
    #     self.log(f"ke after {ke}")
    #     if self.universe.rank.color == 0:
    #         self.topology.change_topology_to_system(min_system, frame)
    #         self.lmp.commands_list(
    #             [
    #                 f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
    #                 for ID, imgs in zip(yids, new_imgs)
    #             ]
    #             + [
    #                 "reset_atoms mol all single yes",
    #                 "run 0 post no",
    #             ]
    #         )
    #         self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
    #     self.universe.global_comm.Barrier()
    #     if self.universe.rank.color != 0:
    #         self._reset_lmp_topology(frame)
    #     self.universe.global_comm.Barrier()
    #     self.topology.build_topology()
    #     if self.universe.rank.color == 0:
    #         self.lmp.command("run 0 ")
    #         ke = self.lmp.get_thermo("ke")
    #         self.log(f"ke after after {ke}")
    #         print("post react forces")
    #         f = self.get_forces()
    #         h_idxs = self.topology.atoms[self.rxn_ids["h_id"]].idx
    #         x_idxs = self.topology.atoms[self.rxn_ids["x_id"]].idx
    #         y_idxs = self.topology.atoms[self.rxn_ids["y_id"]].idx
    #         self.log(f"forces0 h {f[h_idxs]}")
    #         self.log(f"forces0 x {f[x_idxs]}")
    #         self.log(f"forces0 y {f[y_idxs]}")
    #     #     for i in range(20):
    #     #         self.log(f"init forces post reaction {i} {f[i]}")
    #     #     force_diff = self.forces._calc_force_diff()
    #     #     for i in range(20):
    #     #         self.log(f"force diff {i} {force_diff[i]}")
    #     self.safe = False
    #     print("YES")
    #     self.prev_system = np.array([[None, None]])
    #     self.rebuild = True
    #     self.universe.global_comm.Barrier()
    #     return min_system
    
    def generate_callback(self, system_info):
        rxn_pairs, systems_idxs, pairs_idxs = system_info
        num_systems = len(systems_idxs)
        # self.prev_system = np.array([[None, None]])
        # frame = None
        # if self.universe.rank.color == 0:
        #     frame = self.get_frame(self.lmp)
        # frame = self.universe.global_comm.bcast(frame, root=0)
        # rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
        #     frame.pos, frame.xyz_pbc
        # )
        # if not rxn_pairs:
        #     self._redistribute_EVB_states(1, frame)
        #     pe = self.lmp.get_thermo("pe")
        #     self._out(frame, pe)
        #     self.num_colors = 1
        #     self.rebuild = False
        #     def callback(lmp, ntimestep, nlocal, tag, x, f):
        #         new_forces = lmp.numpy.fix_external_get_force("ext")
        #         current_forces = lmp.numpy.extract_atom("f")
        #         new_forces[:, :] = current_forces
        #
        #     return callback
        #
        # self.log(f"Reaction systems: {systems_idxs}", level="debug")
        # self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
        # self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
        # self.log(f"Number of systems: {num_systems}", level="debug")
        # self._redistribute_EVB_states(num_systems, frame)
        # self.log(f"New total colors: {self.universe.total_colors}", level="debug")
        #
        # system_idxs = systems_idxs[self.universe.rank.color]
        # system = tuple(
        #     rxn_pairs[pair_idxs] for pair_idxs in system_idxs if pair_idxs is not None
        # )
        # if self.universe.rank.color != 0:
        #     change_topology = True
        #     if np.array_equal(self.prev_system, system) and self.safe:
        #         change_topology = False
        #     if change_topology:
        #         self._reset_lmp_topology(frame)
        #         self.safe = True
        #     self.set_positions(self.lmp, frame.pos)
        #     self.set_velocities(self.lmp, frame.vel)
        #     self.set_images(self.lmp, frame.images)
        #     if self.SI.scale_box:
        #         self.set_box_data(self.lmp, frame.box_data)
        #     if change_topology:
        #         self.topology.change_topology_to_system(self.lmp, system, frame)
        #         self.lmp.commands_list(self.restart_commands["fixes"])
        #
        # self.prev_system = system
        # color = self.universe.rank.color
        num_sites = len(pairs_idxs)
        self.mix_states = self._mix_states_single
        if num_sites > 1:
            mix_states = self._mix_states_scf

        def callback(lmp, ntimestep, nlocal, tag, x, f):
            # grab the numpified fexternal array
            new_forces = lmp.numpy.fix_external_get_force("ext")
            current_forces = self.get_forces(lmp)
            # synchronise positions across all lammps objects
            # this method does not require all lammps objects 
            # to have the same number of processors
            nx = np.zeros(shape=current_forces.shape)
            idxs = self.topology.id_to_idx(tag)
            if self.universe.rank.color == 0:
                nx[idxs] = x
            distributed_total_x = np.empty_like(nx)
            self.universe.global_comm.Allreduce(nx, distributed_total_x, op=MPI.SUM)
            x[:, :] = distributed_total_x[idxs]
            pe = lmp.get_thermo("pe")
            if self.universe.sub_rank == 0:
                self.log(
                    f"Potential energy for color {self.universe.rank.color}: {pe}", level="debug", rank=-1
                )
            virial = self.get_virial(lmp)
            cs = self.get_computes(lmp)
            pes = np.zeros(num_systems, dtype='d')
            forces = np.zeros(shape=(num_systems, *current_forces.shape), dtype='d')
            virials = None
            if self.SI.scale_box:
                virials = np.zeros((num_systems, 6), dtype='d')
            computes = None
            if self.SI.computes is not None:
                computes = np.zeros((num_systems, len(self.SI.computes)), dtype='d')
            if self.universe.me == 0:
                source = MPI.ANY_SOURCE
                pes[0] = pe
                forces[0, :, :] = current_forces
                if virials is not None:
                    virials[0, :] = virial
                if computes is not None:
                    computes[0, :] = cs
                for system in range(1, num_systems):
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
            if self.universe.rank.color != 0 and self.universe.sub_rank == 0:
                self.universe.global_comm.send(pe, dest=0, tag=self.universe.rank.color)
                self.universe.global_comm.send(current_forces, dest=0, tag=self.universe.rank.color + 200)
                if virial is not None:
                    self.universe.global_comm.send(virial, dest=0, tag=self.universe.rank.color + 300)
                if cs is not None:
                    self.universe.global_comm.send(cs, dest=0, tag=self.universe.rank.color + 400)
            pes = self.universe.global_comm.bcast(pes, root=0)
            forces = self.universe.global_comm.bcast(forces, root=0)
            if self.SI.scale_box:
                virials = self.universe.global_comm.bcast(virials, root=0)
            if self.SI.computes is not None:
                computes = self.universe.global_comm.bcast(computes, root=0)

            frame = self.get_frame(lmp, pos=distributed_total_x, forces=current_forces)
            min_eval, min_evec_coeffs, cpl_forces = self.mix_states(pes, computes, frame, (rxn_pairs, systems_idxs, pairs_idxs))
            print("forces", forces.shape)
            print("cpl_forces", cpl_forces.shape)
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
            if self.universe.rank.color == 0 and min_state_idx != 0:
                lmp.set_variable("min_state_idx", f"{min_state_idx}")
            new_forces[:, :] = mixed_forces[idxs]
            # deal with virial/pressure later

        return callback

    def before_callback(self):
        self.prev_system = np.array([[None, None]])
        frame = None
        if self.universe.rank.color == 0:
            frame = self.get_frame(self.lmp)
        frame = self.universe.global_comm.bcast(frame, root=0)
        rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
            frame.pos, frame.xyz_pbc
        )
        if not rxn_pairs:
            self._redistribute_EVB_states(1, frame)
            pe = self.lmp.get_thermo("pe")
            self._out(frame, pe)
            self.num_colors = 1
            self.rebuild = False
            def callback(lmp, ntimestep, nlocal, tag, x, f):
                new_forces = lmp.numpy.fix_external_get_force("ext")
                current_forces = lmp.numpy.extract_atom("f")
                new_forces[:, :] = current_forces

            return callback

        self.log(f"Reaction systems: {systems_idxs}", level="debug")
        self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
        self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
        self.log(f"Number of systems: {num_systems}", level="debug")
        self._redistribute_EVB_states(num_systems, frame)
        self.log(f"New total colors: {self.universe.total_colors}", level="debug")

        system_idxs = systems_idxs[self.universe.rank.color]
        system = tuple(
            rxn_pairs[pair_idxs] for pair_idxs in system_idxs if pair_idxs is not None
        )
        if self.universe.rank.color != 0:
            change_topology = True
            if np.array_equal(self.prev_system, system) and self.safe:
                change_topology = False
            if change_topology:
                self._reset_lmp_topology(frame)
                self.safe = True
            self.set_positions(self.lmp, frame.pos)
            self.set_velocities(self.lmp, frame.vel)
            self.set_images(self.lmp, frame.images)
            if self.SI.scale_box:
                self.set_box_data(self.lmp, frame.box_data)
            if change_topology:
                self.topology.change_topology_to_system(self.lmp, system, frame)
                self.lmp.commands_list(self.restart_commands["fixes"])

        self.prev_system = system
        color = self.universe.rank.color


    #     def callback(lmp, ntimestep, nlocal, tag, x, f0, f):
    #         frame = self._get_frame()
    #         self.log(f"pos {frame.pos}")
    #         rxn_pairs, systems_idxs, pairs_idxs = self._get_systems(
    #             frame.pos, frame.xyz_pbc
    #         )
    #         if not rxn_pairs:
    #             self._redistribute_EVB_states(1, frame)
    #             pe = self.lmp.get_thermo("pe")
    #             self._out(frame, pe)
    #             self.num_colors = 1
    #             self.rebuild = False
    #             self.forces.set_mixed_forces(None)
    #             self.forces.set_mixed_virial(None)
    #             # self._set_forces()
    #             return None
    #         num_systems = len(systems_idxs)
    #         self.log(f"Reaction systems: {systems_idxs}", level="debug")
    #         self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
    #         self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
    #         self.log(f"Number of systems: {num_systems}", level="debug")
    #         self._redistribute_EVB_states(num_systems, frame)
    #         self.log(f"New total colors: {self.universe.total_colors}", level="debug")
    #         (
    #             min_eval,
    #             min_state_idx,
    #             mixed_forces,
    #             mixed_virial,
    #         ) = self._get_mixed_properties(
    #             rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    #         )
    #         self.forces.set_mixed_forces(mixed_forces)
    #         # if min_state_idx != 0:
    #         #     self.Output.log("reaction would have occurred")
    #         # min_state_idx = 0
    #         self.forces.current_forces = self.forces.current_forces[min_state_idx]
    #         if self.SI.scale_box:
    #             self.forces.set_mixed_virial(mixed_virial)
    #             self.forces.current_virial = self.forces.current_virial[min_state_idx]
    #         self.log("Minimum eigen value: ", level="debug")
    #         self.log(min_eval, level="debug")
    #         self.log("Minimum state: ", level="debug")
    #         self.log(min_state_idx, level="debug")
    #         self._out(frame, min_eval)
    #         # min_state_idx = 0
    #         eb = self.lmp.get_thermo("ebond")
    #         ep = self.lmp.get_thermo("epair")
    #         self.log(f"ebond 0 {self.step_count} {eb}")
    #         self.log(f"epair 0 {self.step_count} {ep}")
    #         if min_state_idx == 0:
    #             self.universe.global_comm.Barrier()
    #             self.rebuild = False
    #             return None
    #
    #         # reaction has occured - update topology
    #         min_system_idxs = systems_idxs[min_state_idx]
    #         min_system = tuple(
    #             rxn_pairs[pair_idxs]
    #             for pair_idxs in min_system_idxs
    #             if pair_idxs is not None
    #         )
    #         self.log(f"min system: {min_system}", level="debug")
    #         new_imgs, yids = [], []
    #         for h, y in min_system:
    #             _new_imgs, _yids = self._get_new_imgs(h, y, frame)
    #             new_imgs += list(_new_imgs)
    #             yids += list(_yids)
    #         ke = self.lmp.get_thermo("ke")
    #         self.log(f"ke before {ke}")
    #         self.lmp.commands_list(["reset_atoms mol all single yes", "run 0 post no"])
    #         ke = self.lmp.get_thermo("ke")
    #         self.log(f"ke after {ke}")
    #         if self.universe.rank.color == 0:
    #             self.topology.change_topology_to_system(min_system, frame)
    #             self.lmp.commands_list(
    #                 [
    #                     f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
    #                     for ID, imgs in zip(yids, new_imgs)
    #                 ]
    #                 + [
    #                     "reset_atoms mol all single yes",
    #                     "run 0 post no",
    #                 ]
    #             )
    #             self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
    #         self.universe.global_comm.Barrier()
    #         if self.universe.rank.color != 0:
    #             self._reset_lmp_topology(frame)
    #         self.universe.global_comm.Barrier()
    #         self.topology.build_topology()
    #         if self.universe.rank.color == 0:
    #             self.lmp.command("run 0 ")
    #             ke = self.lmp.get_thermo("ke")
    #             self.log(f"ke after after {ke}")
    #             print("post react forces")
    #             f = self.get_forces()
    #             h_idxs = self.topology.atoms[self.rxn_ids["h_id"]].idx
    #             x_idxs = self.topology.atoms[self.rxn_ids["x_id"]].idx
    #             y_idxs = self.topology.atoms[self.rxn_ids["y_id"]].idx
    #             self.log(f"forces0 h {f[h_idxs]}")
    #             self.log(f"forces0 x {f[x_idxs]}")
    #             self.log(f"forces0 y {f[y_idxs]}")
    #         #     for i in range(20):
    #         #         self.log(f"init forces post reaction {i} {f[i]}")
    #         #     force_diff = self.forces._calc_force_diff()
    #         #     for i in range(20):
    #         #         self.log(f"force diff {i} {force_diff[i]}")
    #         self.safe = False
    #         print("YES")
    #         self.prev_system = np.array([[None, None]])
    #         self.rebuild = True
    #         self.universe.global_comm.Barrier()
    #         return min_system
    #
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
        self.forces.has_run_0_been_called = False
        if n_step == 0:
            self.forces.has_run_0_been_called = True
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

    def set_box_data(self, lmp, box_data):
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

    def get_box_data(self, lmp):
        return lmp.extract_box()
    
    def get_images(self, lmp):
        return np.array(gather_atoms(lmp, "image", 0, 3)).reshape(-1, 3)
    
    def get_velocities(self, lmp):
        return np.array(gather_atoms(lmp, "v", 1, 3)).reshape(-1, 3)
    
    def get_positions(self, lmp):
        return np.array(gather_atoms(lmp, "x", 1, 3)).reshape(-1, 3)
    
    def get_frame(self, lmp, pos=None, forces=None):
        if pos is None:
            pos = self.get_positions(lmp)
        vel = self.get_velocities(lmp)
        box_data = self.box_data
        xyz_pbc = self.xyz_pbc
        if self.SI.scale_box:
            box_data = self.get_box_data(lmp)
            xyz_pbc = np.array(box_data[1]) - np.array(box_data[0])
        images = self.get_images(lmp)
        if forces is None:
            forces = self.get_forces(lmp)
        frame = Topology.Frame(
            pos=pos, box_data=box_data, images=images, xyz_pbc=xyz_pbc, vel=vel, forces=forces
        )
        return frame

    # def get_box_data(self):
    #     return self.lmp.extract_box()

    def set_images(self, lmp, images):
        lmp.scatter_atoms("image", 0, 3, convert_to_c_type(images, c_int))

    # def get_images(self):
    #     return np.array(gather_atoms(self.lmp, "image", 0, 3)).reshape(-1, 3)

    def set_velocities(self, lmp, velocities):
        lmp.scatter_atoms("v", 1, 3, convert_to_c_type(velocities, c_double))

    # def get_velocities(self):
    #     return np.array(gather_atoms(self.lmp, "v", 1, 3)).reshape(-1, 3)

    def set_positions(self, lmp, positions):
        lmp.scatter_atoms("x", 1, 3, convert_to_c_type(positions, c_double))

    # def get_positions(self):
    #     return np.array(gather_atoms(self.lmp, "x", 1, 3)).reshape(-1, 3)

    def _set_forces(self):
        force = self.lmp.numpy.fix_external_get_force("ext")
        if force is None:
            raise ValueError("Something went wrong with fix external")
        ids = self.lmp.numpy.extract_atom("id")
        print("ids", ids)
        if ids is None:
            raise RuntimeError("ids is None")
        idxs = self.topology.id_to_idx(ids)
        print("mixed_forces", self.forces.mixed_forces)
        # force_diff = self.forces._calc_force_diff()
        # print("force_diff", force_diff[idxs])
        # f = self._get_forces()
        # force[:, :] = force_diff[idxs]
        force[:, :] = self.forces.mixed_forces[idxs]
        # force[:, :] = f[idxs] * -1

    def get_forces(self, lmp):
        return np.array(gather_atoms(lmp, "f", 1, 3)).reshape(-1, 3)

    # def get_forces(self):
    #     # if self.universe.rank.color == 0:
    #     #     self.lmp.command("run 0")
    #     # f = self._get_forces()
    #     # return f
    #     f = self._get_forces()
    #     if self.universe.rank.color == 0:
    #         if not self.forces.has_run_0_been_called:
    #             return f - self.forces.force_diff
    #     return f

    def _set_virial(self):
        if self.forces.mixed_virial is None:
            raise RuntimeError(
                "Call was made to set virial but it is None. This is probably unintentional."
            )
        virial_diff = self.forces._calc_virial_diff()
        self.lmp.fix_external_set_virial_global("ext", list(virial_diff))

    def get_virial(self, lmp, vol=None):
        if not self.SI.scale_box:
            return None
        if vol is None:
            vol = lmp.get_thermo("vol")
        p_vir = lmp.numpy.extract_compute("pre_vir", 0, 1)
        if p_vir is None:
            raise ValueError("Could not extract virial")
        vir = p_vir / self.SI.units["pr2vir"] * vol  # type: ignore
        return vir

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
            self.log(f"Step {self.step_count}")
            self._step()
            self.step_count += 1

    def __del__(self):
        if hasattr(self, "data_io"):
            try:
                os.remove(f"/tmp/{self.SI.file}")
            except OSError:
                pass
