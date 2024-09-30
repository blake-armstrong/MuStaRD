import numpy as np

from time import time
from copy import copy
from typing import Union
from lammps import lammps
from scipy.optimize import minimize
from mpi4py import MPI

import mustard.io as MIO
from mustard.msevb import MSEVB
from mustard.topology import Topology, System
from mustard.mpi import Universe, synchronize
from mustard import utils


@synchronize
class Mustard:
    """
    Hello
    """

    def __init__(
        self,
        user_commands: list,
        reaction_parameters: dict,
        mpi_list: Union[None, list, tuple, np.ndarray] = None,
        debug: bool = False,
    ):
        self.universe = Universe(mpi_list, debug)
        self.log = self.universe.log
        self.system_info = MIO.SystemInfo(reaction_parameters)
        self.log(str(self.system_info))
        self.lmp = self._set_lmp()
        self._init_lmp(user_commands)
        self.topology = Topology(self.lmp, self.system_info)
        self.msevb = MSEVB(
            universe=self.universe,
            topology=self.topology,
            SI=self.system_info,
        )
        self._set_lmp_callback()
        self.trajectory = MIO.Trajectorys()
        self.output = MIO.Outputs()
        self.prev_system = self.topology.current_system
        self.safe = False
        self.rebuild = True
        self._init_msevb()

    def _set_lmp(self) -> lammps:
        cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        if self.universe.debug:
            cmdargs[-1] = f"state_{self.universe.rank.color}.log"
        return lammps(
            name="",
            cmdargs=cmdargs,
            comm=self.universe.lmp_comm,
        )

    def _init_lmp(self, user_commands: list):
        self.lmp.commands_list(user_commands)
        self.lmp.command("run 0 post no")
        self.universe.global_comm.Barrier()
        fix_virial = "fix_modify ext virial no"
        if self.system_info.scale_box:
            fix_virial = fix_virial.replace("no", "yes")
        fixes = [
            "fix ux all store/state 1 xu yu zu",
            "fix ext all external pf/callback 1 1",
            "fix_modify ext energy no",
            fix_virial,
            "compute get_pe all pe",
        ]
        self.lmp.commands_list(fixes)

    def _set_lmp_callback(self):
        self.lmp.set_fix_external_callback("ext", self.msevb, self.lmp)
        self.msevb.run = MSEVB.UPDATE
        self.lmp.command("run 0 post no")

    def _init_msevb(self):
        self.msevb.run = self.identify_pairs()
        self.log("Systems", level=Universe.DEBUG)
        for system in self.topology.systems:
            self.log(f"{system}", level=Universe.DEBUG)
        self.lmp.command("run 0 pre yes post no")
        self.msevb.step_count = 0

    def compare_systems_with_colors(self):
        if self.topology.num_systems <= self.universe.total_colors:
            return
        self.log(
            (
                f"{self.topology.num_systems} states were identified but "
                f"only {self.universe.num_fixed_colors} systems are available. "
                f"This means only the first {self.universe.num_fixed_colors} "
                "will be evaluated. "
            ),
            level=Universe.WARN,
        )
        if self.topology.num_sites == 1:
            self.topology.num_systems = self.universe.num_fixed_colors
            if self.topology.num_systems > 1:
                self.topology.systems = self.topology.systems[
                    : self.topology.num_systems
                ]
            return
        while True:
            remove_idx = None
            for i in self.topology.grouped_site_idxs:
                if len(i) > 2:
                    remove_idx = i[-1]
                    break
            if remove_idx is None:
                sites = [
                    self.topology.sites[site_idx]
                    for site_idx in self.topology.grouped_site_idxs[0]
                ]
            else:
                sites = (
                    self.topology.sites[:remove_idx]
                    + self.topology.sites[remove_idx + 1 :]
                )
            for n, site in enumerate(sites):
                site.index = n

            self.topology.sites_to_systems(sites, self.msevb.frame)
            if self.topology.num_systems <= self.universe.total_colors:
                return

    def identify_pairs(self) -> int:
        self.msevb.frame.setattr("vel", utils.get_velocities(self.lmp))
        any_reactions = self.topology.get_systems(self.msevb.frame)
        if not any_reactions:
            self.prev_system = self.topology.systems[0]
            return MSEVB.NONE

        self.compare_systems_with_colors()

        for system in self.topology.systems:
            self.log(f"{system}", level=Universe.DEBUG)

        system = self.topology.grab_system(self.universe.rank.color)
        change_topology = True
        if self.universe.rank.color == 0:
            self.prev_system = system
            return MSEVB.MAIN

        utils.set_positions(self.lmp, self.msevb.frame.pos)
        utils.set_velocities(self.lmp, self.msevb.frame.vel)
        utils.set_images(self.lmp, self.msevb.frame.images)
        if self.system_info.scale_box:
            utils.set_box_data(self.lmp, self.msevb.frame.box_data)

        if system.index == 0:
            if self.prev_system.index != 0:
                self.msevb.run = MSEVB.UPDATE
                self.topology.reset_lmp_topology()
                self.rebuild = True
                # self.lmp.command("run 0 pre yes post no")
            self.prev_system = system
            self.safe = True
            return MSEVB.MAIN

        if np.array_equal(self.prev_system.pairs, system.pairs) and self.safe:
            change_topology = False

        if change_topology:
            self.topology.reset_lmp_topology()
            self.msevb.run = MSEVB.UPDATE
            # self.lmp.command("run 0 pre yes post no")
            self.safe = True
            self.topology.change_topology_to_system(self.lmp, system)
            self.rebuild = True
            # self.lmp.command("run 0 pre yes post no")

        self.prev_system = system
        return MSEVB.MAIN

    def update_topology(self, system: System) -> None:
        new_imgs, yids = [], []
        for site in system.sites:
            if site.pair is None:
                continue
            h, y = site.pair
            _new_imgs, _yids = self.topology.get_new_imgs(h, y, self.msevb.frame, site)
            if not self.system_info.pbc:
                _new_imgs *= 0
            new_imgs += list(_new_imgs)
            yids += list(_yids)
        self.msevb.run = MSEVB.UPDATE
        self.lmp.command("run 0 pre yes post no")
        if self.universe.rank.color != 0:
            self.topology.reset_lmp_topology()
        self.topology.change_topology_to_system(self.lmp, system)
        self.topology.current_system = self.topology.empty_system()
        update = [
            f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
            for ID, imgs in zip(yids, new_imgs)
        ] + ["reset_atoms mol all single yes"]
        self.lmp.commands_list(update)
        self.universe.global_comm.Barrier()
        self.msevb.run = MSEVB.UPDATE
        self.lmp.command("run 0 pre yes post no")
        self.universe.global_comm.Barrier()
        self.topology.build_topology()
        self.safe = False
        self.prev_system = self.topology.empty_system()
        self.rebuild = True
        self.universe.global_comm.Barrier()

    def _basic_step(self, n_step: int) -> System:
        self.msevb.run = MSEVB.UPDATE
        self.msevb.min_system = Topology.EMPTY_SYSTEM
        self.topology.update = False
        pre = "no"
        if self.msevb.step_count % self.system_info.top_update == 0 or not self.safe:
            self.topology.update = True  # recalculates possible EVB states
        self.msevb.run = self.identify_pairs()
        self.rebuild = self.universe.global_comm.allreduce(self.rebuild, op=MPI.LOR)
        if self.msevb.step_count % self.system_info.nl_update == 0 or self.rebuild:
            pre = "yes"  # pre = yes recomputes neighlist
            self.rebuild = False
        self.log(
            f"Number of identified systems: {self.topology.num_systems}",
            level=Universe.DEBUG,
        )
        self.msevb.ntimestep = self.universe.global_comm.bcast(
            self.msevb.ntimestep, root=0
        )
        self.lmp.command(f"run {n_step} pre {pre} post no")
        # self.lmp.command(f"run {n_step} pre {pre} post no update {pre}")
        if self.msevb.min_system.index != 0:
            # reaction has occured - update topology
            self.update_topology(self.msevb.min_system)
        return self.msevb.min_system

    def _full_step(self, n_step: int = 1):
        self.output.write(self.msevb.step_count, self.msevb.min_eval)
        self.trajectory.write(
            self.msevb.step_count,
            self.msevb.frame,
            self.topology,
        )
        system = self._basic_step(n_step)
        if self.msevb.min_system.index == 0:
            return
        for site in system.sites:
            if site.xhy is None:
                continue
            x, h, y = site.xhy
            self.output.log(
                f"Reaction of type {site.rxn_num} in shell {site.shell} occured at step {self.msevb.step_count} between IDs(xhy) {x} {h} {y}"
            )

    def step(self, steps: int):
        self.output.header()
        if steps == 0:
            self._full_step(n_step=0)
            return
        for step in range(steps):
            self.log(f"Step {step}", level=Universe.DEBUG)
            self._full_step()
            self.log("\n", level=Universe.DEBUG)
            self.msevb.step_count += 1

    def run_for_seconds(self, seconds: int):
        self.output.header()
        t0 = t1 = time()
        while t1 - t0 < seconds:
            self.log(f"Step {self.msevb.step_count}", level=Universe.DEBUG)
            self._full_step()
            self.log("\n", level=Universe.DEBUG)
            self.msevb.step_count += 1
            t1 = time()

    def add_output(self, filename=None, properties=None, write_frequency=1000):
        if properties is None:
            properties = ["temp", "pe", "vol"]
        self.output.add_output(
            MIO.Output(
                self.lmp,
                self.universe,
                filename,
                properties,
                write_frequency,
            )
        )

    def add_trajectory(
        self, filename="trajectory.dcd", write_frequency=1000, rxn=False
    ):
        self.trajectory.add_trajectory(
            MIO.Trajectory(self.lmp, self.universe, filename, write_frequency, rxn)
        )

    def finite_differences(
        self, file="finite_differences.out", delta=1e-3, index_array=None
    ):
        import pandas as pd

        self.log("Beginning finite differences...")
        if self.universe.me == 0:
            self.output.log(
                f"Running finite differences calculating with delta {delta} to file {file}"
            )

        self.msevb.run = self.identify_pairs()
        if self.msevb.run == MSEVB.NONE:
            self.log(
                (
                    "No possible reactions were identified "
                    "finite differences forces should therefore be the "
                    "same as those without any msevb effects."
                ),
                level=Universe.WARN,
            )
        self.lmp.command("run 0 pre yes post no update yes")
        ref_mixed_forces = copy(self.msevb.current_mixed_forces)
        self.msevb.sync = False

        starting_positions = utils.get_positions(self.lmp)
        starting_positions = self.universe.global_comm.bcast(starting_positions, root=0)
        u_starting_pos = utils.unwrap_coordinates(
            self.msevb.frame.pos,
            self.msevb.frame.box_matrix,
            self.msevb.frame.inv_box_matrix,
            self.msevb.frame.images,
        )
        frame = copy(self.msevb.frame)
        check_forces = np.zeros(shape=self.msevb.forces_shape)
        num_particles = self.msevb.forces_shape[0]
        if index_array is None:
            index_array = range(num_particles)
        if len(index_array) > num_particles:
            raise ValueError("Index array length is greater than number of particles")
        ndofs = len(index_array) * 3
        for particle in index_array:
            for coord in range(3):
                self.output.log(f"calculating force {particle*3 + coord + 1} / {ndofs}")
                evals = []
                for d in (delta, -delta):
                    pos_copy = copy(u_starting_pos)
                    pos_copy[particle][coord] += d
                    new_imgs = utils.get_periodic_images(pos_copy, frame.inv_box_matrix)
                    pos = utils.wrap_coordinates(
                        pos_copy, frame.box_matrix, frame.inv_box_matrix, new_imgs
                    )
                    utils.set_positions(self.lmp, pos)
                    utils.set_images(self.lmp, new_imgs)
                    self.msevb.frame(self.lmp, pos=pos, imgs=new_imgs)
                    self.msevb.run = self.identify_pairs()
                    self.lmp.command("run 0 pre yes post no update yes")
                    evals.append(copy(self.msevb.min_eval))
                check_forces[particle][coord] = -(evals[0] - evals[1]) / (2 * delta)

        if self.universe.me != 0:
            return
        diff = ref_mixed_forces - check_forces
        norm_diff = abs(diff) / abs(ref_mixed_forces)
        index_array = np.array(list(index_array))
        ids = (
            np.arange(1, len(diff.flatten()) + 1)
            .reshape(diff.shape)[index_array]
            .flatten()
        )
        df = pd.DataFrame(
            {
                "ID": ids,
                "Analytic": ref_mixed_forces[index_array].flatten(),
                "Finite_differences": check_forces[index_array].flatten(),
                "Diff": diff[index_array].flatten(),
                "Abs_diff": abs(diff)[index_array].flatten(),
                "Norm_diff": norm_diff[index_array].flatten(),
            }
        )
        df.to_csv(file, sep="\t", index=False)
        self.output.log(f"Finite differences written to {file}")
        # self.universe.global_comm.Barrier()

    def minimise(
        self,
        traj=True,
        file="minimised.pdb",
        fix=None,
        bound: float = 1.0,
        full=True,
        tol=1e-8,
    ):
        if traj:
            self.add_trajectory(filename="minimise.dcd", write_frequency=1)
        frame = copy(self.msevb.frame)
        unwrapped_starting_pos = frame.pos
        if self.system_info.pbc:
            unwrapped_starting_pos = utils.unwrap_coordinates(
                frame.pos, frame.box_matrix, frame.inv_box_matrix, frame.images
            )
        coords_shape = unwrapped_starting_pos.shape

        nlup = copy(self.system_info.nl_update)
        self.system_info.nl_update = 1

        if fix is not None:
            if type(fix) not in (list, tuple, np.ndarray):
                raise ValueError("fix should be a list")
            fix = np.array(list(fix))

        def objective(coords):
            current_coords = coords.reshape(coords_shape)
            if fix is not None:
                current_coords[fix] = unwrapped_starting_pos[fix]
            current_images = np.zeros(shape=coords_shape, dtype=int)
            current_wrapped_coords = current_coords
            if self.system_info.pbc:
                current_images = utils.get_periodic_images(
                    current_coords, frame.inv_box_matrix
                )
                current_wrapped_coords = utils.wrap_coordinates(
                    current_coords,
                    frame.box_matrix,
                    frame.inv_box_matrix,
                    current_images,
                )
                utils.set_images(self.lmp, current_images)
            utils.set_positions(self.lmp, current_wrapped_coords)
            self.msevb.frame(self.lmp, pos=current_wrapped_coords, imgs=current_images)
            self.msevb.run = self.identify_pairs()
            self.lmp.command("run 0 pre yes post no")
            self.msevb.min_eval = self.universe.global_comm.bcast(
                self.msevb.min_eval, root=0
            )
            if fix is not None:
                self.msevb.current_mixed_forces[fix] = 0.0
            return self.msevb.min_eval, self.msevb.current_mixed_forces.flatten() * -1

        self.output.log("Running minimisation...")
        if self.universe.me == 0:
            reg_pe = self.lmp.get_thermo("pe")
            self.output.log(
                f"starting non-mixed potential energy for minimisation: {reg_pe} "
            )
        self.output.log(
            f"starting mixed potential energy for minimisation: {self.msevb.min_eval} ",
        )
        self.output.log(" updating initial topology...")
        cycle = 0
        while True:
            self.output.log(f"  cycle: {cycle}")
            min_system = self._basic_step(n_step=0)
            if self.msevb.min_system.index == 0:
                self.output.log(" ...no change in topology")
                break

            for site in min_system.sites:
                if site.xhy is None:
                    continue
                x, h, y = site.xhy
                self.output.log(
                    f"Reaction of type {site.rxn_num} in shell {site.shell} occured at cycle {cycle} between IDs(xhy) {x} {h} {y}"
                )
            cycle += 1
            if cycle > 100:
                break
        self.msevb.run = self.identify_pairs()
        self.lmp.command("run 0 pre yes post no")
        if not full:
            exit()
        self.cycle = 0

        def callback(xk):
            e, _ = objective(xk)
            self.output.log(f" step {self.cycle:>8}: {e:20.20f}")
            self.cycle += 1
            if not traj:
                return
            frame(self.lmp, pos=xk.reshape(coords_shape))
            self.trajectory.write(
                self.msevb.step_count,
                frame,
                self.topology,
                pos=frame.pos,
            )

        bounds = [(p - bound, p + bound) for p in unwrapped_starting_pos.flatten()]

        self.output.log("Minimum eigenvalue at each step:")
        result = minimize(
            objective,
            unwrapped_starting_pos.flatten(),
            method="L-BFGS-B",
            jac=True,
            tol=tol,
            callback=callback,
            bounds=bounds,
            # options={"disp":True},
        )
        self.output.log(f"...final minimum eigenvalue: {result.fun}")
        self.output.log(result)

        minimised_coords = result.x.reshape(coords_shape)
        if fix is not None:
            minimised_coords[fix] = unwrapped_starting_pos[fix]
        minimised_wrapped_coords = minimised_coords
        minimised_images = np.zeros(shape=coords_shape, dtype=int)
        if self.system_info.pbc:
            minimised_images = utils.get_periodic_images(
                minimised_coords, frame.inv_box_matrix
            )
            minimised_wrapped_coords = utils.wrap_coordinates(
                minimised_coords,
                frame.box_matrix,
                frame.inv_box_matrix,
                minimised_images,
            )
            utils.set_images(self.lmp, minimised_images)
        utils.set_positions(self.lmp, minimised_wrapped_coords)
        self.msevb.frame(self.lmp, pos=minimised_wrapped_coords, imgs=minimised_images)
        self.msevb.run = self.identify_pairs()
        self.lmp.command("run 0 pre yes post no")
        if self.universe.me == 0:
            MIO.Trajectorys.save_file(
                pos=self.msevb.frame.pos,
                filename_save=file,
                mass=self.topology.masses,
                types=self.topology.xyz_types,
            )
        self.output.log(f"Minised structure written to {file}")
        if traj:
            self.trajectory.trajs.pop()
        self.system_info.nl_update = nlup
        del self.cycle

    def __del__(self):
        if hasattr(self, "universe"):
            self.universe.lmp_comm.Free()
