from mpi4py import MPI
from dataclasses import dataclass
from typing import Union
import numpy as np
import logging
import sys


@dataclass
class Rank:
    color: int
    modify: bool


class Universe:
    DEBUG = 0
    INFO = 1
    WARN = 2

    def __init__(self, mpi_list: Union[None, list, np.ndarray, tuple], debug: bool):
        # MPI.File.Set_atomicity(False)
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

    def log(self, msg: str, rank: int = 0, level: int = INFO):
        if level == Universe.DEBUG:
            logger = self.logger.debug
        elif level == Universe.INFO:
            logger = self.logger.info
        elif level == Universe.WARN:
            logger = self.logger.warning
            msg = f"\033[91m{msg}\033[0m"
        else:
            logger = self.logger.info
        if rank == -1:
            logger(msg)
            return
        if self.me == rank:
            logger(msg)


def logger(
    name,
    filename=None,
    level=logging.DEBUG,
    fmt="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
):
    handler = logging.StreamHandler(sys.stdout)
    if filename:
        handler = logging.FileHandler(filename, mode="w")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler.setLevel(level)

    class CustomFormatter(logging.Formatter):
        white = "\x1b[1;39m"
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        # fmts = {
        #     logging.DEBUG: grey + fmt + reset,
        #     logging.INFO: white + fmt + reset,
        #     logging.WARNING: yellow + fmt + reset,
        #     logging.ERROR: red + fmt + reset,
        #     logging.CRITICAL: bold_red + fmt + reset,
        # }
        fmts = {
            logging.DEBUG: fmt,
            logging.INFO: fmt,
            logging.WARNING: fmt,
            logging.ERROR: fmt,
            logging.CRITICAL: fmt,
        }

        def format(self, record):
            log_fmt = self.fmts.get(record.levelno)
            formatter = logging.Formatter(log_fmt, datefmt=datefmt)
            return formatter.format(record)

    formatter = logging.Formatter(fmt, datefmt=datefmt)
    if filename is None:
        formatter = CustomFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def synchronize(cls):
    def _sync_args(args):
        return [MPI.COMM_WORLD.bcast(arg, root=0) for arg in args]

    def _sync_kwargs(kwargs):
        return {k: MPI.COMM_WORLD.bcast(v, root=0) for k, v in kwargs.items()}

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        synchronized_args = _sync_args(args)
        synchronized_kwargs = _sync_kwargs(kwargs)
        original_init(self, *synchronized_args, **synchronized_kwargs)

    cls.__init__ = new_init
    MPI.COMM_WORLD.Barrier()
    return cls
