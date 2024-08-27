import mdtraj
import numpy as np
import inspect

from time import time
from dataclasses import dataclass
from lammps import lammps
from typing import Union, IO, Callable
from itertools import permutations

from .topology import Topology, Frame, TrajectoryFrame
from .mpi import Universe, logger
from .constants import UNITS
from . import utils
from .mixing import get_FD_occupancies

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
            "coupling_function": None,
        }
    ],
    "temperature": None,
    "fermi_tolerance_1": 1e-10,
    "fermi_tolerance_2": 1e-12,
    "constant_volume": True,
    "lammps_unit_system": None,
    "computes": None,
    "fermi_mixing": False,
    "neighbour_list_update": 4,
    "topology_update": 1,
    "scf_tol": 1e-8,
    "scf_max_iter": 100,
    "scf_mix_method": "average",
    "shells": 1,
    "pbc": True,
    "eig_solver": "numpy",
}


class SystemInfo:
    def __init__(self, input_params: dict):
        params = _DEFAULTS.copy()
        params.update(input_params)
        self.repr = dict()
        self.units = self._set_unit_system(key := "lammps_unit_system", params.pop(key))
        self.temperature = self._set_temperature(
            key := "temperature", _temp=params.pop(key)
        )
        self.scale_box = self._set_const_vol(key := "constant_volume", params.pop(key))
        self.set_RT(self.temperature * self.units["boltz"])
        self.nl_update = self._set_nl_update(
            key := "neighbour_list_update", params.pop(key)
        )
        self.shells = self._set_shells(key := "shells", params.pop(key))
        self.pbc = self._set_pbc(key := "pbc", params.pop(key))
        self.eig_solver = self._set_eig_solver(key := "eig_solver", params.pop(key))
        self.scf_tol = self._set_scf_tol(key := "scf_tol", params.pop(key))
        self.scf_max_iter = self._set_scf_max_iter(
            key := "scf_max_iter", params.pop(key)
        )
        self.scf_mix_method = self._set_scf_mix_method(
            key := "scf_mix_method", params.pop(key)
        )
        self.top_update = self._set_top_update(
            key := "topology_update", params.pop(key)
        )
        self.atom_types, self.reverse_atom_types = self._set_types(
            key := "atom_types", params.pop(key)
        )
        self.bond_types = self._set_potential_types(
            key := "bond_types", params.pop(key), "Bond types"
        )
        self.angle_types = self._set_potential_types(
            key := "angle_types", params.pop(key), "Angle types"
        )
        self.proper_types = self._set_potential_types(
            key := "proper_types", params.pop(key), "Proper types"
        )
        self.improper_types = self._set_potential_types(
            key := "improper_types", params.pop(key), "Improper types"
        )
        self.type_charges = self._set_type_charges(
            key := "type_charges", params.pop("type_charges")
        )
        self.num_types = self._set_num_types()
        self.FM = self._set_fermi_mixing(key := "fermi_mixing", params.pop(key))
        self.fdt1 = self._set_fd_tols(key := "fermi_tolerance_1", params.pop(key), "1")
        self.fdt2 = self._set_fd_tols(key := "fermi_tolerance_2", params.pop(key), "2")
        self.computes = self._set_computes(key := "computes", params.pop(key))
        self.reactions, self.coupling = self._set_reaction_species(
            key := "reactions", params.pop(key)
        )
        self.get_occupancies = self._set_get_occupancies()

        if params:
            raise ValueError(
                (
                    f"Unknown keys in reaction parameters: {params.keys()}\n"
                    f"Available keys are: {_DEFAULTS.keys()}"
                )
            )

    def __str__(self):
        repr = "MuStaRD Parameters:\n"
        for k, v in self.repr.items():
            if k is None:
                continue
            if not v:
                continue
            repr += f"{v}\n"
        return repr

    def __repr__(self):
        return self.__str__()

    def _set_types(self, key, _types):
        if type(_types) != dict:
            raise ValueError(f"Unrecognised type for input {_types}. Expected dict")
        types = {}
        reverse_types = {}
        for k, v in _types.items():
            _key = str(k)
            value_int = int(v)
            types[_key] = value_int
            reverse_types[value_int] = _key
        types[None] = None
        reverse_types[None] = None
        if key is not None:
            repr = f"Atom types ({key})"
            _repr = f"{repr:>40}:\n"
            _ws = len(repr) - 1
            inc = 40 - _ws + 4
            inc = 40 - 4
            for k, v in types.items():
                if k is None:
                    continue
                _repr += f"{str(k):>{inc}}: {str(v):<{inc}}\n"
            # _repr += f"{'          ':>40}"
            li = _repr.rsplit("\n", 1)
            _repr = "".join(li)
            self.repr[key] = _repr
        return types, reverse_types

    def _set_potential_types(self, key, _types, repr: str):
        types, _ = self._set_types(None, _types)
        new_types = {}
        for k, v in types.items():
            if k is None:
                continue
            ptypes = [str(self.atom_types[p]) for p in k.split("-")]
            if key == "improper_types":
                for comb in permutations(ptypes[1:]):
                    new_types[f"{ptypes[0]}-{'-'.join(comb)}"] = v
            elif key == "proper_types" or key == "bond_types":
                new_types["-".join(ptypes)] = v
                new_types["-".join(list(reversed(ptypes)))] = v
            elif key == "angle_types":
                for comb in permutations((ptypes[0], ptypes[-1])):
                    new_types[f"{comb[0]}-{ptypes[1]}-{comb[-1]}"] = v
            else:
                new_types["-".join(ptypes)] = v

        if new_types:
            _repr = f"{repr:>40}: {' '}\n"
            inc = 40 - 4
            for k, v in new_types.items():
                _repr += f"{k:>{inc}}: {v:<{inc}}\n"
            li = _repr.rsplit("\n", 1)
            _repr = "".join(li)
        else:
            _repr = f"{repr:>40}: {'None'}"
        self.repr[key] = _repr
        return new_types

    def _set_type_charges(self, key, _charges):
        if type(_charges) != dict:
            raise ValueError(f"Unrecognised type for input {_charges}. Expected dict")
        if len(_charges) != len(self.atom_types) - 1:
            raise ValueError(
                "Atom type dictionary should be the same length as type charges dictionary"
            )
        charges = {}
        for k, v in _charges.items():
            charges[self.atom_types[k]] = float(v)
        _str = f"Type charges ({key})"
        _repr = f"{_str:>40}: {' '}\n"
        _ws = len(_str) - 1
        inc = 40 - _ws + 4
        inc = 40 - 4
        for k, v in charges.items():
            _repr += f"{k:>{inc}}: {v:>{9}.6f}\n"
        li = _repr.rsplit("\n", 1)
        _repr = "".join(li)
        self.repr[key] = _repr
        return charges

    def _set_num_types(self):
        num_types = (
            len(self.atom_types),
            len(self.bond_types),
            len(self.angle_types),
            len(self.proper_types),
            len(self.improper_types),
        )
        _repr = f"{'No. Types':>40}:\n"
        _repr += f"{'Atom types':>36}: {num_types[0]:<36}\n"
        _repr += f"{'Bond types':>36}: {num_types[1]:<36}\n"
        _repr += f"{'Angle types':>36}: {num_types[2]:<36}\n"
        _repr += f"{'Proper types':>36}: {num_types[3]:<36}\n"
        _repr += f"{'Improper types':>36}: {num_types[4]:<36}"
        self.repr["num_types"] = _repr
        return num_types

    def _set_reaction_species(self, key, _reactions):
        if type(_reactions) not in (list, tuple, dict):
            raise ValueError(
                f"Unrecognised type for input {_reactions}. Expected a list/tuple of dictionaries or a single dictionary"
            )
        if type(_reactions) == dict:
            _reactions = [_reactions]
        reactions = []
        coupling = []
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

            _coupling_function = __reaction.pop("coupling_function")
            if not callable(_coupling_function):
                raise ValueError(
                    f"argument passed to coupling_value_function not callable"
                )
            args = len(inspect.signature(_coupling_function).parameters)
            nargs = 1
            if args != nargs:
                raise ValueError(
                    f"coupling_function should have {nargs} arguments (snapshot), found {args}"
                )
            coupling.append(_coupling_function)

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

            reactions.append(
                Reaction(
                    X=self.atom_types[reaction[0]],
                    H=self.atom_types[reaction[1]],
                    Y=self.atom_types[reaction[2]],
                    type_changes0=type_changes0,
                    type_changes1=type_changes1,
                    cutoffs=cutoffs,
                )
            )
        _repr = "\n"
        for n, r in enumerate(reactions):
            _repr += f"{f'Reaction {n}':>40} \n"
            _repr += f"{r}"
            _repr += f"{'Coupling Function':>40}:\n"
            _repr += f"{str(coupling[n]):>40}\n"

        self.repr[key] = _repr
        return reactions, coupling

    def _set_temperature(self, key, _temp):
        temp = float(_temp)
        self.repr[key] = f"{'Temperature':>40}: {temp:<40}"
        return temp

    def _set_fd_tols(self, key, _fdt, n):
        fdt = float(_fdt)
        _repr = ""
        if self.FM:
            _repr = f"{'Fermi-Dirac Tolerance':>40} {n}: {fdt:<40}"
        self.repr[key] = _repr
        return fdt

    def _set_const_vol(self, key, _cv):
        scale_box = not bool(_cv)
        if scale_box:
            raise ValueError("Only constant volume right now.")
        self.repr[key] = f"{'Variable Box Size':>40}: {str(scale_box):<40}"
        return scale_box

    def _set_unit_system(self, key, unit):
        if UNITS is None:
            raise ValueError(f"Unit system required. Pick one of {UNITS.keys()}")
        if unit not in UNITS.keys():
            raise ValueError(
                f"Unit system {unit} not part of the LAMMPS unit systems {UNITS.keys()}"
            )
        self.repr[key] = f"{'LAMMPS Unit System':>40}: {unit:<40}"
        return UNITS[unit]

    def _set_computes(self, key, computes):
        if computes is None:
            return None
        if type(computes) == str:
            return [computes]
        if type(computes) not in (list, tuple):
            raise ValueError(
                f"Unexpected type for computes {computes}. Expect a str or list of strs"
            )
        cps = [str(compute) for compute in computes]
        self.repr[key] = f"{'Computes':>40}: {str(cps):<40}"
        return cps

    def _set_shells(self, key, _shells):
        shells = int(_shells)
        self.repr[key] = f"{'Shells':>40}: {shells:<40}"
        return shells

    def _set_fermi_mixing(self, key, _fm):
        fm = bool(_fm)
        self.repr[key] = f"{'Use Fermi Mixing':>40}: {str(fm):<40}"
        return fm

    def _set_get_occupancies(self):
        if self.FM and self.temperature > 1e-6:

            def get_occupancies_FM(eig_vals, SI):
                return get_FD_occupancies(
                    eig_vals, SI.RT, tol1=self.fdt1, tol2=self.fdt2
                )

            return get_occupancies_FM

        else:

            def get_occupancies_NO_FM(eig_vals, _):
                occupancies = np.zeros(len(eig_vals))
                occupancies[np.argmin(eig_vals)] = 1.0
                return occupancies

            return get_occupancies_NO_FM

    def _set_nl_update(self, key, _nl_update):
        nl_update = int(_nl_update)
        self.repr[key] = f"{'Neighbour List Rebuild Frequency':>40}: {nl_update:<40}"
        return nl_update

    def _set_top_update(self, key, _topology_update):
        topology_update = int(_topology_update)
        self.repr[key] = f"{'EVB States Rebuild Frequency':>40}: {topology_update:<40}"
        return topology_update

    def _set_eig_solver(self, key, eig_solver):
        SOLVERS = ("NUMPY", "SYMPY")
        eig_solver = str(eig_solver).upper()
        if eig_solver not in SOLVERS:
            raise ValueError(
                f"Eigen value solver {eig_solver} not in available solvers: {SOLVERS}"
            )
        self.repr[key] = f"{'Eigen Solver':>40}: {eig_solver:<40}"
        return eig_solver

    def _set_scf_tol(self, key, _scf_tol):
        scf_tol = float(_scf_tol)
        self.repr[key] = f"{'SCF Tolerance':>40}: {scf_tol:<40}"
        return scf_tol

    def _set_scf_max_iter(self, key, _scf_max_iter):
        scf_max_iter = int(_scf_max_iter)
        self.repr[key] = f"{'SCF Max Iterations':>40}: {scf_max_iter:<40}"
        return scf_max_iter

    def _set_scf_mix_method(self, key, _scf_mix_method):
        MIX_METHODS = ("AVERAGE", "WEIGHTED")
        mix_method = str(_scf_mix_method).upper()
        if mix_method not in MIX_METHODS:
            raise ValueError(
                f"SCF force mixing method {mix_method} not in available mixing methods: {MIX_METHODS}"
            )
        self.repr[key] = f"{'SCF force mixing method':>40}: {mix_method:<40}"
        return mix_method

    def _set_pbc(self, key, _pbc):
        pbc = bool(_pbc)
        self.repr[key] = f"{'PBC':>40}: {str(pbc):<40}"
        return pbc

    def set_RT(self, RT):
        self.RT = RT


class Output:
    def __init__(
        self,
        lmp: lammps,
        universe: Universe,
        fname=None,
        properties=["temp", "pe", "vol"],
        write_frequency=1000,
    ):
        self.lmp = lmp
        self.universe = universe
        self.write_frequency = write_frequency
        self.properties = properties
        self.logger = logger(str(id(fname)), filename=fname, fmt="%(message)s")
        self.timestep = float(str(self.lmp.extract_global("dt")))
        self.t0 = None
        self.write_header = True

    def header(self):
        if not self.write_header:
            return
        header = f"{'Step':>10} {'Pe(mixed)':>19} {'E_total':>19} {'E_conserve':>19}"
        self.info = "{:>10} {:>19.10f} {:>19.10f} {:>19.10f}"
        for title in self.properties:
            header += f" {title.title():>19}"
            self.info += " {:>19.10f}"
        header += f" {'Speed(ns/day)':>15}"
        self.info += " {:>15.8f}"
        self.log(header)
        self.write_header = False
        self.universe.global_comm.Barrier()

    def _write(self, step: int, speed: float, pe: float, ke: float, ecpl: float):
        props = [self.lmp.get_thermo(prop) for prop in self.properties]
        if self.universe.me != 0:
            return

        self.log(self.info.format(step, pe, pe + ke, pe + ke + ecpl, *props, speed))

    def write(self, step: int, pe: float):
        if step % self.write_frequency != 0:
            return
        t1 = time()
        speed = 0
        if self.t0:
            speed = (
                (86400 / (t1 - self.t0)) * (self.write_frequency * self.timestep)
            ) / 1000
        ke = float(str(self.lmp.get_thermo("ke")))
        ecpl = float(str(self.lmp.get_thermo("ecouple")))
        self._write(step, speed, pe, ke, ecpl)
        self.t0 = t1

    def log(self, info: str):
        if self.universe.me != 0:
            return
        self.logger.info(info)


class Outputs:
    def __init__(self):
        self.outputs = []

    def add_output(self, output: Output):
        self.outputs.append(output)

    def header(self):
        for output in self.outputs:
            output.header()

    def write(self, *args):
        for output in self.outputs:
            output.write(*args)

    def log(self, *args):
        for output in self.outputs:
            output.log(*args)


class Trajectory:
    def __init__(
        self,
        lmp: lammps,
        universe: Universe,
        file_name: Union[str, None] = None,
        write_frequency=1e10,
        xyz=False,
    ):
        self.lmp = lmp
        self.universe = universe
        self.write_frequency = int(write_frequency)
        self.xyz = bool(xyz)
        self.io = self._set_output(file_name)
        self._write = self._set_write()

    def _set_output(self, fname: Union[str, None]) -> IO:
        if fname is None:
            file_type = "dcd"
            if self.xyz:
                file_type = "xyz"
            fname = f"trajectory.{file_type}"
        fname = str(fname)
        if self.xyz:
            return open(fname, "w")
        else:
            return mdtraj.open(fname, "w")

    def _unwrap_positions(
        self, topology: Topology, pos=None
    ) -> Union[None, np.ndarray]:
        if pos is not None:
            return pos
        na = int(str(self.lmp.extract_global("natoms")))
        z = np.zeros((na, 3))
        xu = self.lmp.numpy.extract_fix("ux", 1, 2)
        ids = self.lmp.numpy.extract_atom("id")
        if ids is None:
            raise RuntimeError("ids is None")
        if ids.size != 0:
            z[topology.id_to_idx(ids)] = xu
        if self.universe.me == 0:
            for i in range(1, self.universe.sub_size):
                _z = self.universe.global_comm.recv(source=i, tag=i)
                z += _z
        else:
            self.universe.global_comm.send(z, dest=0, tag=self.universe.me)
        if self.universe.me != 0:
            return
        unwrapped_pos = z
        return unwrapped_pos

    def _set_write(self) -> Callable:
        if self.xyz:

            def _write_xyz(unwrapped_pos: np.ndarray, frame: Frame, topology: Topology):
                self.io.write(f"{len(unwrapped_pos)}\n")
                self.io.write(
                    'Lattice="{0[0]:.3f} {0[1]:.3f} {0[2]:.3f} {0[3]:.3f} {0[4]:.3f} {0[5]:.3f} {0[6]:.3f} {0[7]:.3f} {0[8]:.3f}"\n'.format(
                        frame.box_vectors
                    )
                )
                self.io.writelines(
                    [
                        "{0} {1[0]:.8f} {1[1]:.8f} {1[2]:.8f}\n".format(typ, pos)
                        for typ, pos in zip(topology.xyz_types, unwrapped_pos)
                    ]
                )
                self.io.write(f"Bonds {self._fmt(topology.top_ref['bonds'])}\n")
                self.io.write(f"Angles {self._fmt(topology.top_ref['angles'])}\n")
                self.io.write(f"Impropers {self._fmt(topology.top_ref['impropers'])}\n")
                self.io.write(f"Dihedrals {self._fmt(topology.top_ref['dihedrals'])}\n")
                self.io.flush()

            return _write_xyz
        else:

            def _write_dcd(unwrapped_pos: np.ndarray, frame: Frame, *args):
                self.io.write(  # type: ignore
                    unwrapped_pos.astype(np.float32),
                    cell_lengths=list(frame.box_lengths[:3]),
                    cell_angles=frame.box_lengths[3:],
                )

            return _write_dcd

    def _fmt(self, arr):
        return np.array2string(arr, separator=",", formatter={"int": "{}".format})

    def write(self, step: int, frame: Frame, topology: Topology, pos=None):
        if step % self.write_frequency != 0:
            return
        if self.universe.rank.color != 0:
            return
        unwrapped_pos = self._unwrap_positions(topology, pos)
        if self.universe.me != 0:
            return
        self._write(unwrapped_pos, frame, topology)

    def close(self):
        self.io.close()


class Trajectorys:
    def __init__(self):
        self.trajs = []

    def add_trajectory(self, trajectory: Trajectory):
        self.trajs.append(trajectory)

    def write(self, *args, **kwargs):
        for traj in self.trajs:
            traj.write(*args, **kwargs)

    def close(self):
        for traj in self.trajs:
            traj.close()

    @staticmethod
    def get_z(mass):
        z = []
        for m in mass:
            element = mdtraj.element.Element.getByMass(m)
            if element is None:
                raise RuntimeError("Could not determine element.")
            z.append(element.atomic_number)
        return z

    @staticmethod
    def save_file(pos, filename_save, mass, types):
        topology = mdtraj.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("RXN", chain)
        for m, t in zip(mass, types):
            topology.add_atom(str(t), mdtraj.element.Element.getByMass(m), residue)
        _f = mdtraj.open(filename_save, "w")
        _f.write(pos, topology=topology)
        _f.close()

    @staticmethod
    def read_xyz(trajectory, skip=1):
        if trajectory.split(".")[-1] != "xyz":
            raise ValueError("Need reactive xyz file")
        num_frames = 0
        frames = []
        with open(trajectory, "r") as open_traj:
            while True:
                na = open_traj.readline()
                if na == "":
                    break
                na = int(na)
                if num_frames % skip != 0:
                    open_traj.readline()
                    for _ in range(na):
                        open_traj.readline()
                    for _ in range(4):
                        open_traj.readline()
                    num_frames += 1
                    continue
                _pbc = open_traj.readline()
                _pbc = _pbc.replace("Lattice=", "")
                _pbc = _pbc.replace('"', "")
                split_pbc = _pbc.split()
                pbc = np.array(split_pbc, dtype=float)
                abcabc = utils.get_abcabc(pbc)
                xyz = np.empty(shape=(na, 3))
                types = list(np.empty(shape=na, dtype=str))
                for particle in range(na):
                    line = open_traj.readline()
                    splitline = line.split()
                    types[particle] = splitline[0]
                    xyz[particle] = [float(x) for x in splitline[1:4]]
                imgs = np.floor(xyz / abcabc[:3]).astype(int)
                xyz -= abcabc[:3] * (xyz / abcabc[:3]).round()
                _bonds = open_traj.readline()
                split_bonds = _bonds.split()
                bonds = eval(f"np.array({split_bonds[-1]}, dtype=int)")
                _angles = open_traj.readline()
                split_angles = _angles.split()
                angles = eval(f"np.array({split_angles[-1]}, dtype=int)")
                _impropers = open_traj.readline()
                split_impropers = _impropers.split()
                impropers = eval(f"np.array({split_impropers[-1]}, dtype=int)")
                _dihedrals = open_traj.readline()
                split_dihedrals = _dihedrals.split()
                dihedrals = eval(f"np.array({split_dihedrals[-1]}, dtype=int)")
                frames.append(
                    TrajectoryFrame(
                        frame=num_frames,
                        natoms=na,
                        pbc=abcabc,
                        pos=xyz,
                        imgs=imgs,
                        types=np.array(types),
                        bonds=bonds,
                        angles=angles,
                        impropers=impropers,
                        dihedrals=dihedrals,
                    )
                )
                num_frames += 1
        return frames


@dataclass
class Reaction:
    X: int
    H: int
    Y: int
    type_changes0: dict
    type_changes1: dict
    cutoffs: dict

    def __str__(self):
        _repr1 = "Type Changes [X-H--Y]"
        _repr2 = f"{self.X}-{self.H}--{self.Y}"
        _repr2 += " --> "
        _repr2 += f"{self.type_changes0[self.X]}-{self.type_changes0[self.H]}--{self.type_changes0[self.Y]}"
        repr = f"{_repr1:>40}: {_repr2:<40}\n"
        _repr = "Type Changes [non XHY]"
        repr += f"{_repr:>40}:\n"
        for k, v in self.type_changes1.items():
            repr += "{:>42}{} --> {:<40}\n".format("", str(k), str(v))
        _repr = "Cutoffs"
        repr += f"{_repr:>40}\n"
        for k, v in self.cutoffs.items():
            repr += "{:>40}: {:<40}\n".format(str(k).title(), str(v))
        return repr
