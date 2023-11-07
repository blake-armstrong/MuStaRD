from lammps import lammps
from scipy.optimize import minimize
from collections import defaultdict
import numpy as np
import os
import uuid

from copy import copy
import pandas as pd
from .msevb import MSEVB
from .topology import Topology
from .io import SystemInfo, Trajectory, Output
from .mpi import Universe
from . import utils


class Mustard:
    def __init__(
        self,
        lmp_coord_file,
        force_field_file,
        header,
        commands,
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
            ]
        )
        virial = "compute pre_vir all pressure NULL virial"
        if self.SI.scale_box:
            self.lmp.command(virial)

        self.lmp.commands_list(commands)
        self.lmp.command("run 0 post no")
        shape = utils.get_positions(self.lmp).shape
        if self.universe.rank.color == 0:
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        self.universe.global_comm.Barrier()

        fixes = [
            "fix ux all store/state 1 xu yu zu",
            "fix ext all external pf/callback 1 1",
            "fix_modify ext energy no",
            "fix_modify ext virial no",
            "compute get_pe all pe",
        ]
        if self.SI.scale_box:
            fixes[-2] = fixes[-2].replace("no", "yes")
        self.lmp.commands_list(fixes)
        self.restart_commands = {
            "header": header,
            "read_data": [f"read_data /tmp/{self.SI.file}"],
            "change_box": ["change_box all triclinic"],
            "force_field": [f"include {force_field_file}"],
            "user": commands,
            "fixes": fixes,
            "virial": [""],
        }
        if self.SI.scale_box:
            self.restart_commands["virial"][0] = virial
        self.topology = Topology(self.lmp, self.SI)
        frame = Topology.Frame(self.lmp, *shape, scale_box=self.SI.scale_box)
        self.msevb = MSEVB(
            universe=self.universe,
            topology=self.topology,
            SI=self.SI,
            frame=frame,
        )
        self.lmp.set_fix_external_callback("ext", self.msevb, self.lmp)
        self.msevb.run = 0
        self.lmp.command("run 0 post no")
        self.Trajectory = Trajectory()
        self.Output = Output()
        self.prev_system = np.array(tuple())
        self.safe = False
        self.rebuild = True
        self._run_step(n_step=0, nl_update=1)
        self.msevb.step_count = 0

    def _reset_lmp_topology(self, lmp, frame):
        self.log("reset lmp called", rank=-1)
        if self.universe.rank.color == 0:
            raise RuntimeError("dont do this")
        lmp.commands_list(
            ["clear"]
            + self.restart_commands["header"]
            + self.restart_commands["read_data"]
        )
        utils.set_positions(lmp, frame.pos)
        utils.set_images(lmp, frame.images)
        lmp.commands_list(
            self.restart_commands["change_box"]
            + self.restart_commands["force_field"]
            + self.restart_commands["virial"]
            + self.restart_commands["user"]
        )
        utils.set_velocities(lmp, frame.vel)
        utils.set_box_data(lmp, frame.box_data)
        self.lmp.commands_list(self.restart_commands["fixes"])
        self.lmp.set_fix_external_callback("ext", self.msevb, self.lmp)
        self.msevb.run = 2
        lmp.command("run 0 pre yes post no")

    def _redistribute_EVB_states(self, num_total_colors, frame):
        if num_total_colors <= self.universe.total_colors:
            return
        self.log("REDISTRIBUTE CALLED")
        self.log(f"num_total_colors {num_total_colors}")
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
        self.lmp.commands_list(
            self.restart_commands["header"] + self.restart_commands["read_data"]
        )
        utils.set_positions(self.lmp, frame.pos)
        utils.set_images(self.lmp, frame.images)
        self.lmp.commands_list(
            self.restart_commands["change_box"]
            + self.restart_commands["force_field"]
            + self.restart_commands["virial"]
            + self.restart_commands["user"]
        )
        utils.set_box_data(self.lmp, frame.box_data)
        utils.set_box_data(self.lmp, frame.vel)
        self.lmp.commands_list(self.restart_commands["fixes"])
        self.msevb.run = 2
        self.lmp.command("run 0 pre yes post no")

    def identify_pairs(self):
        # self.prev_system = np.array([[None, None]])
        self.msevb.frame._update_vel(utils.get_velocities(self.lmp))
        self.topology.get_pairs(self.msevb.frame.pos, self.msevb.frame.xyz_pbc)
        if not self.topology.rxn_pairs:
            self._redistribute_EVB_states(1, self.msevb.frame)
            self.num_colors = 1
            self.rebuild = False
            self.prev_system = np.array(tuple())
            return False

        self.log(f"Reaction systems: {self.topology.systems_idxs}", level="debug")
        self.log(f"rxn_pairs: {self.topology.rxn_pairs}", level="debug")
        self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
        self.log(f"Number of systems: {self.topology.num_systems}", level="debug")
        self._redistribute_EVB_states(self.topology.num_systems, self.msevb.frame)
        self.log(f"New total colors: {self.universe.total_colors}", level="debug")

        system = self.topology.grab_system(np.intp(self.universe.rank.color))
        self.log(f"{self.universe.rank.color} system: {system}", rank=-1)
        self.log(f"{self.universe.rank.color} prev_system: {self.prev_system}", rank=-1)
        if len(system) != 0:
            change_topology = True
            if np.array_equal(self.prev_system, system) and self.safe:
                change_topology = False
            if change_topology:
                self._reset_lmp_topology(self.lmp, self.msevb.frame)
                self.safe = True
            utils.set_positions(self.lmp, self.msevb.frame.pos)
            utils.set_velocities(self.lmp, self.msevb.frame.vel)
            utils.set_images(self.lmp, self.msevb.frame.images)
            if self.SI.scale_box:
                utils.set_box_data(self.lmp, self.msevb.frame.box_data)
            if change_topology:
                self.topology.change_topology_to_system(
                    self.lmp, system, self.msevb.frame
                )
                self.msevb.run = 2
                self.lmp.command("run 0 pre yes post no")
        self.prev_system = system
        return True

    def update_topology(self, system):
        new_imgs, yids = [], []
        for h, y in system:
            _new_imgs, _yids = self.topology.get_new_imgs(h, y, self.msevb.frame)
            new_imgs += list(_new_imgs)
            yids += list(_yids)
        self.msevb.run = 2
        self.lmp.command(f"run 0 pre yes post no")
        if self.universe.rank.color == 0:
            self.topology.change_topology_to_system(self.lmp, system, self.msevb.frame)
            self.lmp.commands_list(
                [
                    f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
                    for ID, imgs in zip(yids, new_imgs)
                ]
                + [
                    "reset_atoms mol all single yes",
                ]
            )
        self.universe.global_comm.Barrier()
        self.msevb.run = 2
        self.lmp.command(f"run 0 pre yes post no")
        if self.universe.rank.color == 0:
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        self.universe.global_comm.Barrier()
        if self.universe.rank.color != 0:
            self._reset_lmp_topology(self.lmp, self.msevb.frame)
        self.universe.global_comm.Barrier()
        self.msevb.run = 2
        self.lmp.command(f"run 0 pre yes post no")
        self.universe.global_comm.Barrier()
        self.topology.build_topology()
        self.safe = False
        self.prev_system = np.array(tuple())
        self.rebuild = True
        self.universe.global_comm.Barrier()

    def _run_step(self, n_step=1, nl_update=None):
        any_pairs = self.identify_pairs()
        run = f"run {n_step} pre yes post no"
        if nl_update is None:
            nl_update = self.SI.nl_update
        if self.msevb.step_count % nl_update != 0 and not self.rebuild:
            # pre no stops computing neighlist
            # run = f"run {n_step} pre no post no"
            run.replace("pre yes", "pre no")
        self.log("RUN 1 HAS BEEN CALLED")
        self.msevb.run = int(any_pairs)
        self.lmp.command(run)

    def _step(self, n_step=1, nl_update=None):
        self.log(f"CALL TO STEP {self.msevb.step_count}")
        if self.universe.rank.color == 0:
            if self.universe.me == 0:
                self.Output.write(
                    int(self.msevb.step_count), self.msevb.min_eval, self.lmp
                )
            self.Trajectory.write(
                self.msevb.step_count,
                self.lmp,
                self.msevb.frame.box_data,
                self.universe,
                self.topology,
            )
        self._run_step(n_step, nl_update)
        if self.msevb.min_state_idx == 0:
            return None
        # reaction has occured - update topology
        min_system = self.topology.grab_system(self.msevb.min_state_idx)
        for pair in min_system:
            self.Output.log(
                f"reaction occured at step {self.msevb.step_count} between IDs {pair[0]} and {pair[1]}"
            )
        self.update_topology(min_system)
        return min_system

    def step(self, steps):
        if self.universe.me == 0:
            self.Output._header()

        if steps == 0:
            self._step(n_step=0)
            return

        for _ in range(steps):
            self._step()
            self.msevb.step_count += 1

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

    # def finite_differences(
    #     self, file="finite_differences.out", delta=1e-3, index_array=None
    # ):
    #     if self.universe.me == 0:
    #         self.Output.log(
    #             f"Running finite differences calculating with delta {delta} to file {file}"
    #         )
    #     if self.universe.rank.color == 0:
    #         self.lmp.command("run 0 pre yes post no")
    #     frame = self._get_frame()
    #     rxn_pairs, systems_idxs, pairs_idxs = self._get_pairinfo(
    #         frame.pos, frame.xyz_pbc
    #     )
    #     if not rxn_pairs:
    #         self.log(
    #             (
    #                 "Could not complete finite differences as no "
    #                 "possible reactions were detected with starting configuration."
    #             ),
    #             level="warn",
    #         )
    #         return False
    #     num_systems = len(systems_idxs)
    #     if self.universe.me == 0:
    #         self.log(f"Reaction systems: {systems_idxs}", level="debug")
    #         self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
    #         self.log(
    #             f"Starting total colors: {self.universe.total_colors}", level="debug"
    #         )
    #         self.log(f"Number of systems: {num_systems}", level="debug")
    #     self._redistribute_EVB_states(num_systems, frame)
    #     if self.universe.me == 0:
    #         self.log(f"New total colors: {self.universe.total_colors}", level="debug")
    #     self.log("START")
    #     _, _, ref_mixed_forces, _ = self._get_mixed_properties(
    #         rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    #     )
    #     pos_orig = None
    #     if self.universe.rank.color == 0:
    #         pos_orig = self.get_positions()
    #     pos_orig = self.universe.global_comm.bcast(pos_orig, root=0)
    #     check_forces = np.zeros(shape=ref_mixed_forces.shape)
    #     if index_array is None:
    #         index_array = range(len(pos_orig))
    #     if len(index_array) > len(pos_orig):
    #         raise ValueError("Index array length is greater than number of particles")
    #
    #     for particle in index_array:
    #         for coord in range(3):
    #             self.Output.log(
    #                 f"calculating force {particle*3 + coord + 1} / {len(pos_orig) * 3}"
    #             )
    #             _pos = copy(pos_orig)
    #             _pos[particle][coord] = pos_orig[particle][coord] + delta
    #             self.set_positions(_pos)
    #             self.lmp.command("run 0 post no")
    #             frame = self._get_frame(pos=_pos)
    #             pos_m_eval, _, _, _ = self._get_mixed_properties(
    #                 rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    #             )
    #             _pos = copy(pos_orig)
    #             _pos[particle][coord] = pos_orig[particle][coord] - delta
    #             self.set_positions(_pos)
    #             self.lmp.command("run 0 post no")
    #             frame = self._get_frame(pos=_pos)
    #             neg_m_eval, _, _, _ = self._get_mixed_properties(
    #                 rxn_pairs, systems_idxs, num_systems, frame, pairs_idxs
    #             )
    #             check_forces[particle][coord] = -(pos_m_eval - neg_m_eval) / (2 * delta)
    #
    #     if self.universe.me == 0:
    #         diff = ref_mixed_forces - check_forces
    #         norm_diff = abs(diff) / abs(ref_mixed_forces)
    #         df = pd.DataFrame(
    #             {
    #                 "Analytic": ref_mixed_forces.flatten(),
    #                 "Finite_differences": check_forces.flatten(),
    #                 "Diff": diff.flatten(),
    #                 "Abs_diff": abs(diff).flatten(),
    #                 "Norm_diff": norm_diff.flatten(),
    #             }
    #         )
    #         df.to_csv(file, sep="\t", index=False)
    #         self.Output.log(f"Finite differences written to {file}")
    #     self.universe.global_comm.Barrier()
    #     return True

    # def minimise(self, cmd_list):
    #     if self.universe.rank.color == 0:
    #         self.log("Regular energy minimisation called...")
    #         self.lmp.commands_list(cmd_list)
    #         self.log("...system minimised.")
    #     self.universe.global_comm.Barrier()

    # def msevb_minimise(self, traj=True, scale_box=False):
    #     self.step_count = 0.2
    #     if scale_box:
    #         raise ValueError(
    #             "MSEVB minimisation with fluctuating box not implemented yet. Flag is there to remind me :)"
    #         )
    #     if traj:
    #         self.add_trajectory(filename="minimise.dcd", write_frequency=1)
    #     frame = self._get_frame()
    #     u_frame_pos = frame.images * frame.xyz_pbc + frame.pos
    #
    #     def objective(coords):
    #         upos = coords.reshape(len(frame.pos), len(frame.pos[0]))
    #         diff = upos - frame.pos
    #         diff -= frame.xyz_pbc * (diff / frame.xyz_pbc).round()
    #         nupos = u_frame_pos - diff
    #         new_imgs = np.floor(nupos / frame.xyz_pbc).astype(int)
    #         pos = nupos - frame.xyz_pbc * new_imgs
    #         new_frame = Topology.Frame(
    #             pos=pos,
    #             box_data=frame.box_data,
    #             images=new_imgs,
    #             xyz_pbc=frame.xyz_pbc,
    #             vel=frame.vel,
    #         )
    #         self.set_positions(pos)
    #         self.set_images(new_imgs)
    #         self.lmp.command("run 0 pre yes post no")
    #         if self.universe.rank.color == 0 and traj:
    #             self.Trajectory.trajs[-1].write(
    #                 1,
    #                 self.lmp,
    #                 frame.box_data,
    #                 self.universe,
    #                 self.topology,
    #                 pos=nupos,
    #             )
    #         rxn_pairs, systems_idxs, pairs_idxs = self._get_pairinfo(pos, frame.xyz_pbc)
    #         if rxn_pairs:
    #             num_systems = len(systems_idxs)
    #             self.log(f"Reaction systems: {systems_idxs}", level="debug")
    #             self.log(f"rxn_pairs: {rxn_pairs}", level="debug")
    #             self.log(
    #                 f"Starting total colors: {self.universe.total_colors}",
    #                 level="debug",
    #             )
    #             self.log(f"Number of systems: {num_systems}", level="debug")
    #             self._redistribute_EVB_states(num_systems, frame)
    #             self.log(
    #                 f"New total colors: {self.universe.total_colors}", level="debug"
    #             )
    #             min_eval, _, mixed_forces, _ = self._get_mixed_properties(
    #                 rxn_pairs, systems_idxs, num_systems, new_frame, pairs_idxs
    #             )
    #             return min_eval, mixed_forces.flatten()
    #         else:
    #             pe = self.lmp.get_thermo("pe")
    #             forces = self.get_forces()
    #             return pe, forces.flatten()
    #
    #     init_pe, _ = objective(frame.pos)
    #     self.log("Running minimisation...")
    #     if self.universe.me == 0:
    #         reg_pe = self.lmp.get_thermo("pe")
    #         self.log(f"starting non-mixed potential energy for minimisation: {reg_pe} ")
    #     self.log(
    #         f"starting mixed potential energy for minimisation: {init_pe} ",
    #     )
    #     self.log(" updating initial topology...")
    #     cycle = 0
    #     while True:
    #         self.log(f"  cycle: {cycle}")
    #         reaction = self._step(n_step=0, nl_update=1, mini=True)
    #         if not reaction:
    #             self.log(" ...no change in topology")
    #             break
    #         self.log("    topology updated")
    #         cycle += 1
    #         if cycle > 100:
    #             break
    #
    #     objective_values = []
    #
    #     def callback(xk):
    #         objective_values.append(objective(xk))
    #
    #     result = minimize(
    #         objective,
    #         frame.pos.flatten(),
    #         method="L-BFGS-B",
    #         jac=True,
    #         tol=1e-6,
    #         callback=callback,
    #     )
    #     self.log(result, level="debug")
    #     # self.logger.info("Objective function values at each step:")
    #     for i, value in enumerate(objective_values):
    #         self.log(f"  step {i}: {value[0]:.4f}")
    #     self.log(f"...minimised MSEVB potential energy: {result.fun}")
    #
    #     upos = result.x.reshape(len(frame.pos), len(frame.pos[0]))
    #     diff = upos - frame.pos
    #     diff -= frame.xyz_pbc * (diff / frame.xyz_pbc).round()
    #     nupos = u_frame_pos - diff
    #     new_imgs = np.floor(nupos / frame.xyz_pbc).astype(int)
    #     pos = nupos - frame.xyz_pbc * new_imgs
    #
    #     self.set_positions(pos)
    #     self.set_images(new_imgs)
    #     self.lmp.command("run 0 pre yes post no")
    #     self.Trajectory.trajs.pop()
    #     self.step_count = 0

    def __del__(self):
        if hasattr(self, "data_io"):
            try:
                os.remove(f"/tmp/{self.SI.file}")
            except OSError:
                pass
