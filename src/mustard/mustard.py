from lammps import lammps
from scipy.optimize import minimize
from copy import copy
import numpy as np

from mustard.msevb import MSEVB
from mustard.topology import Topology
from mustard.io import SystemInfo, Trajectory, Output
from mustard.mpi import Universe
from mustard import utils


class Mustard:
    def __init__(
        self,
        user_commands,
        reaction_parameters,
        # read_restart=None,  # TODO
        # write_restart=None, # TODO 
        mpi_list=None,
        debug=False,
    ):
        self.universe = Universe(mpi_list, debug)
        self.log = self.universe.log
        (
            user_commands,
            reaction_parameters,
            mpi_list,
            debug,
        ) = self.universe.global_comm.bcast(
        (
            user_commands,
            reaction_parameters,
            mpi_list,
            debug,
        ), root=0)
        self.SI = SystemInfo(reaction_parameters)
        cmdargs = ["-nocite", "-screen", "none", "-log", "none"]
        if self.universe.debug:
            cmdargs[-1] = f"state_{self.universe.rank.color}.log"
        self.lmp = lammps(
            name="",
            cmdargs=cmdargs,
            comm=self.universe.lmp_comm,
        )
        self.lmp.commands_list(user_commands)
        self.lmp.command("run 0 post no")
        self.universe.global_comm.Barrier()
        fix_virial = "fix_modify ext virial no"
        if self.SI.scale_box:
            fix_virial = fix_virial.replace("no", "yes")
        fixes = [
            "fix ux all store/state 1 xu yu zu",
            "fix ext all external pf/callback 1 1",
            "fix_modify ext energy no",
            fix_virial,
            "compute get_pe all pe",
        ]
        self.lmp.commands_list(fixes)
        self.topology = Topology(self.lmp, self.SI)
        self.msevb = MSEVB(
            universe=self.universe,
            topology=self.topology,
            SI=self.SI,
        )
        self.lmp.set_fix_external_callback("ext", self.msevb, self.lmp)
        self.msevb.run = 2
        self.lmp.command("run 0 post no")
        self.Trajectory = Trajectory()
        self.Output = Output()
        self.prev_system = self.topology.current_system
        self.safe = False
        self.rebuild = True
        any_pairs = int(self.identify_pairs())
        self.log("Systems", level="debug")
        [self.log(f"{system}", level="debug") for system in self.topology.systems]
        self.msevb.run = any_pairs
        self.lmp.command("run 0 pre yes post no")
        self.msevb.step_count = 0

    def _redistribute_EVB_states(self, num_total_colors):
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
            if self.topology.num_systems > 1:
                self.topology.systems = [self.topology.systems[idx] for idx in self.topology.dist_sort if idx < self.topology.num_systems]
     
    def identify_pairs(self):
        self.msevb.frame.setattr("vel", utils.get_velocities(self.lmp))
        any_reactions = self.topology.get_systems(
            self.msevb.frame.pos, self.msevb.frame.box_vectors
        )
        if not any_reactions:
            self.prev_system = self.topology.systems[0]
            return False

        for system in self.topology.systems:
            self.log(f"{system}", level="debug")

        self._redistribute_EVB_states(self.topology.num_systems)

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
            if not self.SI.pbc:
                _new_imgs *= 0
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
        self.msevb.min_state_idx = np.intp(0)
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
        self.topology.update = False
        pre = "no"
        if self.msevb.step_count % self.SI.nl_update == 0 or self.rebuild:
            # pre = yes recomputes neighlist
            pre = "yes"
            self.rebuild = False
            self.topology.update = True
        any_pairs = self.identify_pairs()
        self.log(
            f"Number of identified systems: {self.topology.num_systems}", level="debug"
        )
        self.msevb.ntimestep = self.universe.global_comm.bcast(
            self.msevb.ntimestep, root=0
        )
        self.msevb.run = int(any_pairs)
        self.lmp.command(f"run {n_step} pre {pre} post no update {pre}")
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
        u_starting_pos = utils.unwrap_coordinates(self.msevb.frame.pos, self.msevb.frame.box_matrix, self.msevb.frame.inv_box_matrix, self.msevb.frame.images)
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
                self.Output.log(f"calculating force {particle*3 + coord + 1} / {ndofs}")
                evals = []
                for d in (delta, -delta):
                    pos_copy = copy(u_starting_pos)
                    pos_copy[particle][coord] += d
                    new_imgs = utils.get_periodic_images(pos_copy, frame.inv_box_matrix)
                    pos = utils.wrap_coordinates(pos_copy, frame.box_matrix, frame.inv_box_matrix, new_imgs)
                    utils.set_positions(self.lmp, pos)
                    utils.set_images(self.lmp, new_imgs)
                    self.msevb.frame(self.lmp, pos=pos, imgs=new_imgs)
                    any_pairs = self.identify_pairs()
                    self.msevb.run = int(any_pairs)
                    self.lmp.command("run 0 pre yes post no update yes")
                    evals.append(copy(self.msevb.min_eval))
                check_forces[particle][coord] = -(evals[0] - evals[1]) / (2 * delta)

        if self.universe.me == 0:
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
        self.Output.log(f"Finite differences written to {file}")
        self.universe.global_comm.Barrier()

    def minimise(self, traj=True, file="minimised.pdb", fix=None, bound: float = 1.0):
        if traj:
            self.add_trajectory(filename="minimise.dcd", write_frequency=1)
        frame = copy(self.msevb.frame)
        unwrapped_starting_pos = frame.pos
        if self.SI.pbc:
            unwrapped_starting_pos = utils.unwrap_coordinates(frame.pos, frame.box_matrix, frame.inv_box_matrix, frame.images)
        coords_shape = unwrapped_starting_pos.shape

        if fix is not None:
            if type(fix) not in (list, tuple, np.ndarray):
                raise ValueError("fix should be a list")
            fix = np.array(list(fix))

        def objective(coords):
            current_coords = coords.reshape(coords_shape)
            if fix is not None:
                current_coords[fix] = frame.pos[fix]
            current_images = np.zeros(shape=coords_shape, dtype=int)
            current_wrapped_coords = current_coords
            if self.SI.pbc:
                current_images = utils.get_periodic_images(current_coords, frame.inv_box_matrix)
                current_wrapped_coords = utils.wrap_coordinates(current_coords, frame.box_matrix, frame.inv_box_matrix, current_images)
                utils.set_images(self.lmp, current_images)
            utils.set_positions(self.lmp, current_wrapped_coords)
            self.msevb.frame(self.lmp, pos=current_wrapped_coords, imgs=current_images)
            any_pairs = self.identify_pairs()
            self.msevb.run = int(any_pairs)
            self.lmp.command("run 0 pre yes post no update yes")
            self.universe.global_comm.Barrier()
            self.msevb.min_eval = self.universe.global_comm.bcast(
                self.msevb.min_eval, root=0
            )
            return self.msevb.min_eval, self.msevb.current_mixed_forces.flatten() * -1

        self.Output.log("Running minimisation...")
        if self.universe.me == 0:
            reg_pe = self.lmp.get_thermo("pe")
            self.Output.log(f"starting non-mixed potential energy for minimisation: {reg_pe} ")
        self.Output.log(
            f"starting mixed potential energy for minimisation: {self.msevb.min_eval} ",
        )
        self.Output.log(" updating initial topology...")
        cycle = 0
        while True:
            self.Output.log(f"  cycle: {cycle}")
            nlup = copy(self.SI.nl_update)
            self.SI.nl_update = 1
            reaction = self._step(n_step=0, out=False)
            if reaction is None:
                self.Output.log(" ...no change in topology")
                break
            self.Output.log("    topology updated")
            cycle += 1
            if cycle > 100:
                break
        self.SI.nl_update = nlup
        any_pairs = self.identify_pairs()
        self.msevb.run = int(any_pairs)
        self.lmp.command("run 0 pre yes post no update yes")

        self.cycle = 0
        def callback(xk):
            e, _ = objective(xk)
            self.Output.log(f" step {self.cycle:>8}: {e:20.6f}")
            if self.universe.rank.color == 0 and traj:
                self.Trajectory.trajs[-1].write(
                    1,
                    self.lmp,
                    frame.box_data,
                    self.universe,
                    self.topology,
                    pos=xk.reshape(coords_shape),
                )
            self.cycle += 1

        bounds = [(p - bound, p + bound) for p in unwrapped_starting_pos.flatten()]

        self.Output.log("Minimum eigenvalue at each step:")
        result = minimize(
            objective,
            unwrapped_starting_pos.flatten(),
            method="L-BFGS-B",
            jac=True,
            tol=1e-6,
            callback=callback,
            bounds=bounds,
            # options={"disp":True},
        )
        self.Output.log(f"...final minimum eigenvalue: {result.fun}")
        self.Output.log(result)

        minimised_coords = result.x.reshape(coords_shape)
        minimised_wrapped_coords = minimised_coords
        minimised_images = np.zeros(shape=coords_shape, dtype=int)
        if self.SI.pbc:
            minimised_images = utils.get_periodic_images(minimised_coords, frame.inv_box_matrix)
            minimised_wrapped_coords = utils.wrap_coordinates(minimised_coords, frame.box_matrix, frame.inv_box_matrix, minimised_images)
            utils.set_images(self.lmp, minimised_images)
        utils.set_positions(self.lmp, minimised_wrapped_coords)
        self.universe.global_comm.Barrier()
        self.msevb.frame(self.lmp, pos=minimised_wrapped_coords, imgs=minimised_images)
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
        self.Output.log(f"Minised structure written to {file}")
        if traj:
            self.Trajectory.trajs.pop()
        del self.cycle

    def __del__(self):
        if hasattr(self.universe, "lmp_comm"):
            self.universe.lmp_comm.Free()
