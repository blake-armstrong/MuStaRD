from mpi4py import MPI
import numpy as np
from typing import NamedTuple
import logging
from .io import logger


class Rank:
    def __init__(self, color, modify):
        self.color = color
        self.modify = modify


class Universe:
    def __init__(self, mpi_list, debug):
        self.global_comm = MPI.COMM_WORLD
        self.num_procs = self.global_comm.Get_size()
        self.me = self.global_comm.Get_rank()
        self.debug = debug
        level = logging.INFO
        if self.debug:
            level = logging.DEBUG
        self.logger = logger("internal", filename=None, level=level)
        self.log("MuStaRD")
        self.log(f"Running with {self.num_procs} available processor(s)")
        self.global_comm.Barrier()
        self._initialise_mpi_distribution(mpi_list)

    def _initialise_mpi_distribution(self, mpi_list):
        color_list = []
        num_fixed_colors = 1
        if mpi_list is None:
            self.log("No arguments passed to mpi_list.")
            self.log("Using half of the available processors for main LAMMPS object")
            self.log("and dynamic MPI redistribution for the other half.")
            ranks0 = self.num_procs // 2
            num_fixed_ranks = ranks0
        else:
            if type(mpi_list) not in (list, np.ndarray, tuple):
                raise ValueError(
                    f"mpi_list should be a list/tuple/array of number of processors for each state"
                )
            mpi_list = np.array(mpi_list, dtype=int)
            if np.sum(mpi_list) > self.num_procs:
                raise ValueError("mpi_list described more processors than requested.")
            ranks0 = mpi_list[0]
            self.log(f"Using {ranks0} processor(s) for main LAMMPS object.")
            num_fixed_ranks = np.sum(mpi_list)
            color_list = mpi_list[1:]
            num_fixed_colors += len(color_list)
        self.global_comm.Barrier()
        color = 0
        modify = False
        if self.me >= ranks0:
            ranks_start = ranks0
            color = len(color_list) + 1
            modify = True
            for n, num_ranks in enumerate(color_list):
                ranks_start += num_ranks
                if self.me < ranks_start:
                    modify = False
                    color = 1 + n
                    self.log(
                        f"State {color} using {num_ranks} processor(s)",
                        rank=ranks_start - num_ranks,
                    )
                    break
        self.rank = Rank(color, modify)
        self.sub_comm = self.global_comm.Split(self.rank.color, self.me)
        self.lmp_comm = self.sub_comm
        if self.rank.modify:
            self.subsub_comm = self.sub_comm.Split(self.rank.color, self.me)
            self.lmp_comm = self.subsub_comm
        # self.total_colors = total_colors
        # print("total colors", self.total_colors)
        self.num_fixed_colors = num_fixed_colors
        self.num_fixed_ranks = num_fixed_ranks
        self.num_free_ranks = self.num_procs - self.num_fixed_ranks
        self.sub_rank = self.lmp_comm.Get_rank()
        self.sub_size = self.lmp_comm.Get_size()
        self.colors = (None, None)
        self.total_colors = self.global_comm.allreduce(self.rank.color, op=MPI.MAX) + 1

    def _available_ranks_to_colors(self, num_total_colors):
        if num_total_colors <= self.total_colors:
            return
        if not self.rank.modify:
            return
        # available ranks can now be modified
        new_colors = num_total_colors - self.num_fixed_colors
        color = ((self.me - self.num_free_ranks) % new_colors) + new_colors + 1
        # self.colors = (None, None)
        if new_colors > self.num_free_ranks:
            # overlapping_colors = new_colors - self.num_free_ranks
            # color = (self.me - self.num_free_ranks) + new_colors + 1
            # if self.me == self.num_procs - 1:
            #     self.colors = tuple(
            #         range(
            #             num_total_colors - 1,
            #             num_total_colors - 1 - (overlapping_colors + 1),
            #             -1,
            #         )
            #     )
            error = (
                "More states identified than available processors."
                "Request more processors/virtual processors."
            )
            raise RuntimeError(error)
        self.rank.color = color
        self.subsub_comm.Free()
        self.subsub_comm = self.sub_comm.Split(self.rank.color, self.me)
        self.lmp_comm = self.subsub_comm
        self.sub_size = self.lmp_comm.Get_size()
        self.sub_rank = self.lmp_comm.Get_rank()

    def available_ranks_to_colors(self, num_total_colors):
        self._available_ranks_to_colors(num_total_colors)
        self.global_comm.Barrier()
        self.total_colors = self.global_comm.allreduce(self.rank.color, op=MPI.MAX) + 1

    def log(self, msg, rank=0, level="info"):
        log = self.logger.info
        if level == "debug":
            log = self.logger.debug
        if level == "warn":
            log = self.logger.warn
        if self.me == rank:
            log(msg)
            return
        if rank == -1:
            log(msg)


# def synchronize_args(cls):
#     def _sync(*args):
#         return [MPI.COMM_WORLD.bcast(arg, root=0) for arg in args]
#
#     original_init = cls.__init__
#
#     def new_init(self, *args, **kwargs):
#         synchronized_args = _sync(*args)
#         original_init(self, *synchronized_args, **kwargs)
#
#     cls.__init__ = new_init
#     MPI.COMM_WORLD.Barrier()
#     return cls
