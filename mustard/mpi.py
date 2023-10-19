from mpi4py import MPI
import numpy as np
from typing import NamedTuple


class MPI_info(NamedTuple):
    color: int
    colors: np.ndarray
    color_locs: list
    color_rank_0: int
    total_ranks: np.ndarray
    total_colors: int
    comm: MPI.Comm


def ranks_to_colors(total_ranks, num_colors):
    return sorted(np.arange(total_ranks) % num_colors), num_colors


def distribute_mpi_ranks(color_list, total_colors, rank):
    color = color_list[rank]
    color_locs = [np.argwhere(np.array(color_list) == c)[0, 0] for c in set(color_list)]
    colors = np.array([color])
    if len(color_list) < total_colors:
        ol = np.arange(total_colors)
        _colors = ol[1:] % (len(color_list) - 1)
        tmp = [
            ol[1:][np.argwhere(_colors == i).flatten()]
            for i in range((len(color_list) - 1))
        ]
        colors = ([np.array([0])] + tmp)[rank]
    return MPI_info(
        color=color,
        colors=colors,
        color_locs=color_locs,
        color_rank_0=np.argwhere(np.array(color_list) == color)[0, 0],
        total_ranks=np.argwhere(np.array(color_list) == color).flatten(),
        total_colors=total_colors,
        comm=MPI.COMM_WORLD.Split(color, rank),
    )


def synchronize_args(cls):
    def _sync(*args):
        return [MPI.COMM_WORLD.bcast(arg, root=0) for arg in args]

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        synchronized_args = _sync(*args)
        original_init(self, *synchronized_args, **kwargs)

    cls.__init__ = new_init
    return cls
