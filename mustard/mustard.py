from lammps import lammps
from scipy.optimize import minimize
from copy import copy
import numpy as np
import uuid
import warnings

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
            "read": [f"read_data {lmp_coord_file}"],
            "ff": [f"include {force_field_file}"],
            "user": commands,
            "virial": [
                "compute pre_vir all pressure NULL virial",
            ],
        }
        cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        if self.universe.debug:
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
        self.msevb.run = 2
        self.lmp.command("run 0 post no")
        self.Trajectory = Trajectory()
        self.Output = Output()
        self.prev_system = self.topology.current_system
        self.safe = False
        self.rebuild = True
        self.identify_pairs()
        self.log("Systems", level="debug")
        [self.log(f"{system}", level="debug") for system in self.topology.systems]
        self.msevb.run = 3
        self.lmp.command("run 0 pre yes post no")
        self.msevb.step_count = 0

    def _redistribute_EVB_states(self, num_total_colors, frame):
        # NOTE: This is from an old version of the code. Delete soon.
        # self.log(f"rxn_pairs: {self.topology.rxn_pairs}", level="debug")
        # self.log(f"Starting total colors: {self.universe.total_colors}", level="debug")
        # self.log(f"Number of systems: {self.topology.num_systems}", level="debug")
        # self.log(f"New total colors: {self.universe.total_colors}", level="debug")
        if num_total_colors <= self.universe.total_colors:
            return
        if self.topology.num_systems > self.universe.num_fixed_colors:
            self.log(
                (
                    f"{self.topology.num_systems} states were identified but "
                    f"only {self.universe.num_fixed_colors} systems are available. "
                    f"This means only the first {self.universe.num_fixed_colors} "
                    "will be evaluated. "
                ),
                level="warn",
            )
            self.topology.num_systems = self.universe.num_fixed_colors
            return
        # self.log("REDISTRIBUTE CALLED")
        # self.log(f"num_total_colors {num_total_colors}")
        # if self.universe.rank.color == 0:
        #     self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        # self.universe.global_comm.Barrier()
        # if self.universe.rank.modify:
        #     self.lmp.close()
        # self.universe.available_ranks_to_colors(num_total_colors)
        # self.universe.global_comm.Barrier()
        # if not self.universe.rank.modify:
        #     return
        # cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        # if self.universe.debug:
        #     cmdargs[-1] = f"{self.universe.rank.color}.log"
        # self.lmp = lammps(
        #     name="",
        #     cmdargs=cmdargs,
        #     comm=self.universe.lmp_comm,
        # )
        # self.topology.set_lmp(self.lmp)
        # self.lmp.commands_list(
        #     self.restart_commands["header"] + self.restart_commands["read_data"]
        # )
        # utils.set_positions(self.lmp, frame.pos)
        # utils.set_images(self.lmp, frame.images)
        # self.lmp.commands_list(
        #     self.restart_commands["change_box"]
        #     + self.restart_commands["force_field"]
        #     + self.restart_commands["virial"]
        #     + self.restart_commands["user"]
        # )
        # utils.set_box_data(self.lmp, frame.box_data)
        # utils.set_velocities(self.lmp, frame.vel)
        # self.lmp.commands_list(self.restart_commands["fixes"])
        # self.msevb.run = 2
        # self.lmp.command("run 0 pre yes post no")

    def identify_pairs(self):
        self.msevb.frame._update_vel(utils.get_velocities(self.lmp))
        any_reactions = self.topology.get_systems(
            self.msevb.frame.pos, self.msevb.frame.xyz_pbc
        )

        if not any_reactions:
            self.prev_system = self.topology.systems[0]
            return False

        for system in self.topology.systems:
            self.log(f"System {system.index}: {system}", level="debug")

        self._redistribute_EVB_states(self.topology.num_systems, self.msevb.frame)

        system = self.topology.grab_system(self.universe.rank.color)
        change_topology = True
        if system.index == 0:
            self.prev_system = system
            return True

        if np.array_equal(self.prev_system.pairs, system.pairs) and self.safe:
            change_topology = False
        if change_topology:
            self.topology.reset_lmp_topology()
            self.safe = True
        utils.set_positions(self.lmp, self.msevb.frame.pos)
        utils.set_velocities(self.lmp, self.msevb.frame.vel)
        utils.set_images(self.lmp, self.msevb.frame.images)
        if self.SI.scale_box:
            utils.set_box_data(self.lmp, self.msevb.frame.box_data)
        if change_topology:
            self.topology.change_topology_to_system(self.lmp, system, self.msevb.frame)
            self.msevb.run = 2
            self.lmp.command("run 0 pre yes post no")

        self.prev_system = system
        return True

    def update_topology(self, system):
        new_imgs, yids = [], []
        for site in system.sites:
            h, y = site.pair
            _new_imgs, _yids = self.topology.get_new_imgs(h, y, self.msevb.frame, site)
            new_imgs += list(_new_imgs)
            yids += list(_yids)
        self.msevb.run = 2
        self.lmp.command(f"run 0 pre yes post no")
        if self.universe.rank.color != 0:
            self.topology.reset_lmp_topology()
        self.topology.change_topology_to_system(self.lmp, system, self.msevb.frame)
        self.topology.current_system = self.topology.empty_system()
        update = [
            f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
            for ID, imgs in zip(yids, new_imgs)
        ] + ["reset_atoms mol all single yes"]
        self.lmp.commands_list(update)
        self.universe.global_comm.Barrier()
        self.msevb.run = 2
        self.lmp.command(f"run 0 pre yes post no")
        self.universe.global_comm.Barrier()
        self.topology.build_topology()
        self.safe = False
        self.prev_system = self.topology.empty_system()
        self.rebuild = True
        self.universe.global_comm.Barrier()

    def _step(self, n_step=1, out=True):
        if self.universe.rank.color == 0 and out:
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
        any_pairs = self.identify_pairs()
        self.log(
            f"Number of identified systems: {self.topology.num_systems}", level="debug"
        )
        self.msevb.ntimestep = self.universe.global_comm.bcast(
            self.msevb.ntimestep, root=0
        )
        self.msevb.run = int(any_pairs)
        pre = "no"
        if self.msevb.step_count % self.SI.nl_update == 0 or self.rebuild:
            # pre = yes recomputes neighlist
            pre = "yes"
            self.rebuild = False
        self.lmp.command(f"run {n_step} pre {pre} post no update no")
        if self.msevb.min_state_idx == 0:
            return None
        # reaction has occured - update topology
        min_system = self.topology.grab_system(int(self.msevb.min_state_idx))
        for pair in min_system.pairs:
            h, y = pair
            try:
                # NOTE assumes transferring atom is only bonded to one other atom
                x = self.topology.bonds[h][0]
            except KeyError:
                x = None
            if out:
                self.Output.log(
                    f"reaction occured at step {self.msevb.step_count} between IDs(xhy) {x} {h} {y}"
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

    def finite_differences(
        self, file="finite_differences.out", delta=1e-3, index_array=None
    ):
        import pandas as pd

        self.log("Beginning finite differences...")
        if self.universe.me == 0:
            self.Output.log(
                f"Running finite differences calculating with delta {delta} to file {file}"
            )

        any_pairs = self.identify_pairs()
        if not any_pairs:
            self.log(
                (
                    "No possible reactions were identified "
                    "finite differences forces should therefore be the "
                    "same as those without any msevb effects."
                ),
                level="warn",
            )
        self.msevb.run = int(any_pairs)
        self.lmp.command("run 0 pre yes post no update yes")
        ref_mixed_forces = copy(self.msevb.current_mixed_forces)
        self.msevb.sync = False

        starting_positions = utils.get_positions(self.lmp)
        starting_positions = self.universe.global_comm.bcast(starting_positions, root=0)
        check_forces = np.zeros(shape=self.msevb.forces_shape)
        num_particles = self.msevb.forces_shape[0]
        if index_array is None:
            index_array = range(num_particles)
        if len(index_array) > num_particles:
            raise ValueError("Index array length is greater than number of particles")
        for particle in index_array:
            for coord in range(3):
                self.Output.log(
                    f"calculating force {particle*3 + coord + 1} / {num_particles * 3}"
                )
                evals = []
                for d in (delta, -delta):
                    pos_copy = copy(starting_positions)
                    pos_copy[particle][coord] = starting_positions[particle][coord] + d
                    utils.set_positions(self.lmp, pos_copy)
                    self.lmp.command("run 0 pre yes post no update yes")
                    evals.append(copy(self.msevb.min_eval))
                check_forces[particle][coord] = -(evals[0] - evals[1]) / (2 * delta)

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

    def minimise(self, traj=True, file="minimised.pdb"):
        if traj:
            self.add_trajectory(filename="minimise.dcd", write_frequency=1)
        u_frame_pos = (
            self.msevb.frame.images * self.msevb.frame.xyz_pbc + self.msevb.frame.pos
        )
        frame = copy(self.msevb.frame)
        self.cycle = 0

        def objective(coords):
            if self.cycle % 2 == 0:
                self.log(f" step {self.cycle // 2}: {self.msevb.min_eval}")
            self.cycle += 1
            upos = coords.reshape(len(frame.pos), len(frame.pos[0]))
            diff = upos - frame.pos
            diff -= frame.xyz_pbc * (diff / frame.xyz_pbc).round()
            nupos = u_frame_pos - diff
            new_imgs = np.floor(nupos / frame.xyz_pbc).astype(int)
            pos = nupos - frame.xyz_pbc * new_imgs
            utils.set_positions(self.lmp, pos)
            utils.set_images(self.lmp, new_imgs)
            self.msevb.frame(self.lmp, pos=pos, imgs=new_imgs)
            any_pairs = self.identify_pairs()
            self.msevb.run = int(any_pairs)
            self.lmp.command("run 0 pre yes post no update yes")
            if self.universe.rank.color == 0 and traj:
                self.Trajectory.trajs[-1].write(
                    1,
                    self.lmp,
                    frame.box_data,
                    self.universe,
                    self.topology,
                    pos=nupos,
                )
            self.universe.global_comm.Barrier()
            self.msevb.min_eval = self.universe.global_comm.bcast(
                self.msevb.min_eval, root=0
            )
            return self.msevb.min_eval, self.msevb.current_mixed_forces.flatten()

        # init_pe, _ = objective(frame.pos)
        self.log("Running minimisation...")
        if self.universe.me == 0:
            reg_pe = self.lmp.get_thermo("pe")
            self.log(f"starting non-mixed potential energy for minimisation: {reg_pe} ")
        self.log(
            f"starting mixed potential energy for minimisation: {self.msevb.min_eval} ",
        )
        self.log(" updating initial topology...")
        cycle = 0
        while True:
            self.log(f"  cycle: {cycle}")
            nlup = copy(self.SI.nl_update)
            self.SI.nl_update = 1
            reaction = self._step(n_step=0, out=False)
            if reaction is None:
                self.log(" ...no change in topology")
                break
            self.log("    topology updated")
            cycle += 1
            if cycle > 100:
                break
        self.SI.nl_update = nlup
        any_pairs = self.identify_pairs()
        self.msevb.run = int(any_pairs)
        self.lmp.command("run 0 pre yes post no update yes")

        objective_values = []

        def callback(xk):
            e, _ = objective(xk)
            objective_values.append(e)

        self.log("Minimum eigenvalue at each step:")
        result = minimize(
            objective,
            frame.pos.flatten(),
            method="L-BFGS-B",
            jac=True,
            tol=1e-6,
            callback=callback,
        )
        self.log(f"...final minimum eigenvalue: {result.fun}")
        self.log(result, level="debug")

        upos = result.x.reshape(len(frame.pos), len(frame.pos[0]))
        diff = upos - frame.pos
        diff -= frame.xyz_pbc * (diff / frame.xyz_pbc).round()
        nupos = u_frame_pos - diff
        new_imgs = np.floor(nupos / frame.xyz_pbc).astype(int)
        pos = nupos - frame.xyz_pbc * new_imgs

        utils.set_positions(self.lmp, pos)
        utils.set_images(self.lmp, new_imgs)
        self.universe.global_comm.Barrier()
        self.msevb.frame(self.lmp, pos=pos, imgs=new_imgs)
        any_pairs = self.identify_pairs()
        self.msevb.run = int(any_pairs)
        self.lmp.command("run 0 pre yes post no update yes")
        if self.universe.me == 0:
            Trajectory.save_file(
                pos=self.msevb.frame.pos,
                filename_save=file,
                mass=self.topology.masses,
                types=self.topology.xyz_types,
            )
        self.log(f"Minised structure written to {file}")
        if traj:
            self.Trajectory.trajs.pop()
        del self.cycle

    def _extend_forces(self, frame):
        if self.universe.rank.color == 0:
            self.lmp.command(f"write_data /tmp/{self.SI.file} nocoeff")
        self.elec_lmp.command(f"clear")
        self.elec_lmp.commands_list(
            self.cmds["header"]
            + [
                f"read_data /tmp/{self.SI.file}",
                "change_box all triclinic",
            ]
            + self.cmds["elec_ff"]
        )
        utils.set_positions(self.elec_lmp, frame.pos)
        self.elec_lmp.command("run 0 pre yes post no update yes")
        elec_forces = utils.get_forces(self.elec_lmp)
        self.ml_data["R"].append(frame.pos)
        self.ml_data["F"].append(self.msevb.current_mixed_forces - elec_forces)
        self.ml_data["E"].append(self.msevb.min_eval)
        cell, _ = utils.extract_box(utils.get_box_data(self.lmp))
        self.ml_data["cell"].append(cell)

    def _rerun(self, trajectory, skip=1, do_something=None):
        if do_something is None:
            do_something = self.do_something
        self.log(f"Beginning rerun")
        frames = Trajectory.read_xyz(trajectory, skip=skip)
        for frame in frames:
            self.log(f"Processing frame {frame.frame}")
            self.topology.set_traj_frame(frame)
            utils.set_positions(self.lmp, frame.pos)
            utils.set_images(self.lmp, frame.imgs)
            # self.msevb.frame(self.lmp, pos=frame.pos)
            self.msevb.run = 2
            self.lmp.command("run 0 pre yes post no update yes")
            self.universe.global_comm.Barrier()
            self.topology.build_topology()
            self.universe.global_comm.Barrier()
            #    any_pairs = self.identify_pairs()
            self.topology.get_pairs(self.msevb.frame.pos, self.msevb.frame.xyz_pbc)
            self._redistribute_EVB_states(self.topology.num_systems, self.msevb.frame)
            any_pairs = 1
            if not self.topology.rxn_pairs:
                any_pairs = 0
            if any_pairs == 1:
                system = self.topology.grab_system(np.intp(self.universe.rank.color))
                if len(system) != 0:
                    self.topology.change_topology_to_system(
                        self.lmp, system, self.msevb.frame
                    )
                    self.msevb.run = 2
                    self.lmp.command("run 0 pre yes post no")
            self.msevb.ntimestep = self.universe.global_comm.bcast(
                self.msevb.ntimestep, root=0
            )
            self.msevb.run = any_pairs
            self.lmp.command("run 0 pre yes post no update yes")
            do_something(frame)

    def do_something(self, _):
        pass

    def forces_for_ml(self, trajectory, elec_ff, skip=1):
        cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        self.elec_lmp = lammps(
            name="",
            cmdargs=cmdargs,
            comm=self.universe.global_comm,
        )
        self.cmds["elec_ff"] = [f"include {elec_ff}"]
        self.ml_data = {
            "R": [],
            "E": [],
            "z": Trajectory.get_z(self.topology.masses),
            "F": [],
            "pbc": [1, 1, 1],
            "cell": [],
        }
        self._rerun(trajectory, do_something=self._extend_forces, skip=skip)

        if self.universe.me == 0:
            np.savez(
                "data.npz",
                R=self.ml_data["R"],
                E=self.ml_data["E"],
                z=self.ml_data["z"],
                F=self.ml_data["F"],
                pbc=self.ml_data["pbc"],
                cell=self.ml_data["cell"],
            )

    def __del__(self):
        if hasattr(self, "SI"):
            import os

            try:
                os.remove(f"/tmp/{self.SI.file}")
            except OSError:
                pass
