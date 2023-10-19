import logging
import sys
import mdtraj
import numpy as np
import inspect
from .constants import UNITS
from time import time
from typing import NamedTuple
from .utils import extract_box

_DEFAULTS = {
    "atom_types": {},
    "bond_types": {},
    "angle_types": {},
    "proper_types": {},
    "improper_types": {},
    "type_charges": {},
    "reactions": [
        {
            "reaction": (None, None, None),
            "type_changes0": {},
            "type_changes1": {},
            "cutoffs": {
                "distance": None,
                "angle": None,
            },
            "coupling_value_function": None,
            "coupling_forces_function": None,
        }
    ],
    "temperature": None,
    "fd_tol_1": 1e-6,
    "fd_tol_2": 1e-6,
    "constant_volume": False,
    "lammps_unit_system": None,
    "computes": None,
    "fermi_mixing": False,
    "neighbour_list_update": 2,
    "scf_tol": 1e-4,
    "scf_max_iter": 100,
}


class SystemInfo:
    class Reaction(NamedTuple):
        X: str
        H: str
        Y: str
        type_changes0: dict
        type_changes1: dict
        cutoffs: dict

    def __init__(self, input_params):
        params = _DEFAULTS.copy()
        params.update(input_params)
        self._set_temperature(params.pop("temperature"))
        fdt2 = self._set_fd_tols(params.pop("fd_tol_1"))
        fdt3 = self._set_fd_tols(params.pop("fd_tol_2"))
        self.fd_tols = (fdt2, fdt3)
        self._set_const_vol(params.pop("constant_volume"))
        self.atom_types, self.reverse_atom_types = self._set_types(
            params.pop("atom_types")
        )
        self.bond_types = self._set_potential_types(params.pop("bond_types"))
        self.angle_types = self._set_potential_types(params.pop("angle_types"))
        self.proper_types = self._set_potential_types(params.pop("proper_types"))
        self.improper_types = self._set_potential_types(params.pop("improper_types"))
        self.type_charges = self._set_type_charges(params.pop("type_charges"))
        self.num_types = (
            len(self.atom_types),
            len(self.bond_types),
            len(self.angle_types),
            len(self.proper_types),
            len(self.improper_types),
        )
        self._set_reaction_species(params.pop("reactions"))
        self.computes = self._set_computes(params.pop("computes"))
        self.units = self._set_unit_system(params.pop("lammps_unit_system"))
        self.FM = self._set_fermi_mixing(params.pop("fermi_mixing"))
        self.nl_update = self._set_nl_update(params.pop("neighbour_list_update"))
        self.scf_tol = float(params.pop("scf_tol"))
        self.scf_max_iter = int(params.pop("scf_max_iter"))
        self.set_RT(self.temperature * self.units["boltz"])

        if params:
            raise ValueError(f"Unknown keys in reaction parameters: {params.keys()}")

    def _set_types(self, _types):
        if type(_types) != dict:
            raise ValueError(f"Unrecognised type for input {_types}. Expected dict")
        types = {}
        reverse_types = {}
        for k, v in _types.items():
            key = str(k)
            value_int = int(v)
            types[key] = value_int
            reverse_types[value_int] = key
        types[None] = None
        reverse_types[None] = None
        return types, reverse_types

    def _set_potential_types(self, _types):
        types, _ = self._set_types(_types)
        new_types = {}
        for k, v in types.items():
            if k is None:
                continue
            ptypes = [str(self.atom_types[p]) for p in k.split("-")]
            new_types["-".join(ptypes)] = v
        return new_types

    def _set_type_charges(self, _charges):
        if type(_charges) != dict:
            raise ValueError(f"Unrecognised type for input {_charges}. Expected dict")
        charges = {}
        for k, v in _charges.items():
            charges[self.atom_types[k]] = float(v)
        return charges

    def _set_reaction_species(self, _reactions):
        if type(_reactions) not in (list, tuple, dict):
            raise ValueError(
                f"Unrecognised type for input {_reactions}. Expected a list/tuple of dictionaries or a single dictionary"
            )
        if type(_reactions) == dict:
            _reactions = [_reactions]
        self.reactions = []
        self.coupling_value_functions = []
        self.coupling_forces_functions = []
        for __reaction in _reactions:
            if type(__reaction) != dict:
                raise ValueError(
                    f"Unrecognised type for reaction input {__reaction}. Expected dict"
                )
            _reaction = __reaction.pop("reaction")
            if len(_reaction) != 3:
                raise ValueError("Must provide 3 reaction species")
            reaction = []
            for k in _reaction:
                if k is None:
                    reaction.append(None)
                    continue
                if k not in self.atom_types.keys():
                    raise ValueError(f"key {k} not found in atom types")
                reaction.append(str(k))
            _type_changes0 = __reaction.pop("type_changes0")
            if type(_type_changes0) != dict:
                raise ValueError(
                    f"Unrecognised type for reaction input {_type_changes0}. Expected dict"
                )
            type_changes0 = {}
            for k, v in _type_changes0.items():
                if k not in self.atom_types.keys():
                    raise ValueError(f"type {k} not found in atom types keys")
                if v not in self.atom_types.keys():
                    raise ValueError(f"type {v} not found in atom types keys")
                if k is not None:
                    k = int(self.atom_types[str(k)])
                if v is not None:
                    v = int(self.atom_types[str(v)])
                type_changes0[k] = v

            _type_changes1 = __reaction.pop("type_changes1")
            if type(_type_changes1) != dict:
                raise ValueError(
                    f"Unrecognised type for reaction input {_type_changes1}. Expected dict"
                )
            type_changes1 = {}
            for k, v in _type_changes1.items():
                if k not in self.atom_types.keys():
                    raise ValueError(f"type {k} not found in atom types keys")
                if v not in self.atom_types.keys():
                    raise ValueError(f"type {v} not found in atom types keys")
                type_changes1[int(self.atom_types[str(k)])] = int(
                    self.atom_types[str(v)]
                )

            _coupling_value_function = __reaction.pop("coupling_value_function")
            if not callable(_coupling_value_function):
                raise ValueError(
                    f"argument passed to coupling_value_function not callable"
                )
            args = len(inspect.signature(_coupling_value_function).parameters)
            nargs = 6
            if args != nargs:
                raise ValueError(
                    f"coupling_function should have {nargs} arguments (rxn_ids, snapshot, new_pe, initial_pe, new_computes, initial_computes), found {args}"
                )
            self.coupling_value_functions.append(_coupling_value_function)

            _coupling_forces_function = __reaction.pop("coupling_forces_function")
            if not callable(_coupling_forces_function):
                raise ValueError(
                    f"argument passed to coupling_forces_function not callable"
                )
            args = len(inspect.signature(_coupling_forces_function).parameters)
            nargs = 4
            if args != nargs:
                raise ValueError(
                    f"coupling_function should have {nargs} arguments (rxn_ids, snapshot, computes, forces), found {args}"
                )
            self.coupling_forces_functions.append(_coupling_forces_function)

            _cutoffs = __reaction.pop("cutoffs")
            cutoffs = {}
            cutoffs["distance"] = float(_cutoffs.pop("distance"))
            ang = _cutoffs.pop("angle")
            if ang is None:
                cutoffs["angle"] = None
            else:
                cutoffs["angle"] = float(ang)
            if _cutoffs:
                raise ValueError(
                    f"Unknown keys in reaction parameters: {_cutoffs.keys()}"
                )

            self.reactions.append(
                SystemInfo.Reaction(
                    X=self.atom_types[reaction[0]],
                    H=self.atom_types[reaction[1]],
                    Y=self.atom_types[reaction[2]],
                    type_changes0=type_changes0,
                    type_changes1=type_changes1,
                    cutoffs=cutoffs,
                )
            )

    def _set_temperature(self, _temp):
        self.temperature = float(_temp)

    def _set_fd_tols(self, _fdt):
        return float(_fdt)

    def _set_const_vol(self, _cv):
        self.scale_box = not bool(_cv)

    def _set_unit_system(self, unit):
        if UNITS is None:
            raise ValueError(f"Unit system required. Pick one of {UNITS.keys()}")
        if unit not in UNITS.keys():
            raise ValueError(
                f"Unit system {unit} not part of the LAMMPS unit systems {UNITS.keys()}"
            )
        return UNITS[unit]

    def _set_computes(self, computes):
        if computes is None:
            return None
        if type(computes) == str:
            return [computes]
        if type(computes) not in (list, tuple):
            raise ValueError(
                f"Unexpected type for computes {computes}. Expect a str or list of strs"
            )
        return [str(compute) for compute in computes]

    def _set_fermi_mixing(self, fm):
        return bool(fm)

    def _set_nl_update(self, nl_update):
        return int(nl_update)

    def set_RT(self, RT):
        self.RT = RT


class Output:
    class output:
        def __init__(
            self,
            lmp,
            rank,
            fname=None,
            properties=["temp", "pe", "vol"],
            write_frequency=1000,
        ):
            self.write_frequency = write_frequency
            self.properties = properties
            self.rank = rank
            if self.rank == 0:
                self.logger = logger(str(id(fname)), filename=fname, fmt="%(message)s")
            self.info = "{:>10} {:>19.10f} {:>19.10f}"
            self.header = "      Step          Pe(mixed)          E_total"
            for title in self.properties:
                self.header += f"{title.title():>20}"
                self.info += " {:>19.10f}"
            self.header += "   Speed(ns/day)"
            self.info += " {:>15.8f}"
            # FIXME: timestep units
            self.timestep = lmp.extract_global("dt")
            self.t0 = None

        def _header(self):
            self.logger.info(self.header)

        def _write(self, lmp, step, speed, pe):
            props = [lmp.get_thermo(prop) for prop in self.properties]
            ke = lmp.get_thermo("ke")
            self.logger.info(self.info.format(step, pe, pe + ke, *props, speed))

        def write(self, step, pe, lmp):
            if step % self.write_frequency == 0:
                t1 = time()
                speed = 0
                if self.t0:
                    speed = (
                        (86400 / (t1 - self.t0))
                        * (self.write_frequency * self.timestep)
                    ) / 1000
                self._write(lmp, step, speed, pe)
                self.t0 = t1

        def log(self, info):
            if self.rank == 0:
                self.logger.info(info)

    def __init__(self):
        self.outputs = []

    def add_output(self, output):
        self.outputs.append(output)

    def _header(self):
        for output in self.outputs:
            output._header()

    def write(self, *args):
        for output in self.outputs:
            output.write(*args)

    def log(self, *args):
        for output in self.outputs:
            output.log(*args)


class Trajectory:
    class trajectory:
        def __init__(self, fname=None, write_frequency=1e10, xyz=False):
            self.fname = fname
            self.write_frequency = write_frequency
            self.xyz = xyz
            if self.fname is not None:
                # if self.rxn:
                if self.xyz:
                    self.xyz_file = open(str(fname), "w")
                else:
                    self.dcd_file = mdtraj.open(fname, "w")

        def _write(self, lmp, box_data, MPI_info, rank, id_to_idx, xyz_types, pos=None):
            abcabc, abc = extract_box(box_data)
            na = lmp.extract_global("natoms")
            z = np.zeros((na, 3))  # type: ignore
            xu = lmp.numpy.extract_fix("ux", 1, 2)
            ids = lmp.numpy.extract_atom("id")
            if ids.size != 0:
                z[id_to_idx(ids)] = xu
            if rank == 0:
                for i in MPI_info.total_ranks[1:]:
                    _z = MPI_info.comm.recv(source=i, tag=i)
                    z += _z
            else:
                MPI_info.comm.send(z, dest=0, tag=rank)
            if rank == 0:
                unwrapped_pos = z
                if pos is not None:
                    unwrapped_pos = pos
                if self.xyz:
                    self.xyz_file.write(f"{len(unwrapped_pos)}\n")
                    # self.xyz_file.write("Step\n")
                    self.xyz_file.write(
                        'Lattice="{0[0]:.3f} {0[1]:.3f} {0[2]:.3f} {0[3]:.3f} {0[4]:.3f} {0[5]:.3f} {0[6]:.3f} {0[7]:.3f} {0[8]:.3f}"\n'.format(
                            abc
                        )
                    )
                    self.xyz_file.writelines(
                        [
                            "{0} {1[0]:.3f} {1[1]:.3f} {1[2]:.3f}\n".format(typ, pos)
                            for typ, pos in zip(xyz_types, unwrapped_pos)
                        ]
                    )
                    self.xyz_file.flush()
                else:
                    self.dcd_file.write(
                        unwrapped_pos.astype(np.float32),
                        cell_lengths=list(abcabc[:3]),
                        cell_angles=abcabc[3:],
                    )

        def write(self, step, *args, **kwargs):
            if self.fname:
                if step % self.write_frequency == 0:
                    self._write(*args, **kwargs)
            else:
                raise ValueError("No trajectory was added but write called.")

        def close(self):
            if self.fname:
                if self.xyz:
                    self.xyz_file.close()
                else:
                    self.dcd_file.close()

    def __init__(self):
        self.trajs = []

    def add_trajectory(self, trajectory):
        self.trajs.append(trajectory)

    def write(self, *args):
        for traj in self.trajs:
            traj.write(*args)

    def close(self):
        for traj in self.trajs:
            traj.close()

    @staticmethod
    def slice(pos, filename_save, mass, types):
        topology = mdtraj.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("RXN", chain)
        for m, t in zip(mass, types):
            topology.add_atom(str(t), mdtraj.element.Element.getByMass(m), residue)
        _f = mdtraj.open(filename_save, "w")
        _f.write(pos, topology=topology)
        _f.close()


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
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
