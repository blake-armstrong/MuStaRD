import logging
import sys
import mdtraj
import numpy as np
import inspect
from .constants import UNITS
from time import time
from dataclasses import dataclass
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
            "coupling_value_function": None,
            "coupling_forces_function": None,
        }
    ],
    "temperature": None,
    "fd_tol_1": 1e-6,
    "fd_tol_2": 1e-6,
    "constant_volume": True,
    "lammps_unit_system": None,
    "computes": None,
    "fermi_mixing": False,
    "neighbour_list_update": 4,
    "scf_tol": 1e-4,
    "scf_max_iter": 100,
    "shells": 1,
    "pbc": True,
}


class SystemInfo:
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
        self.get_occupancies = self._set_get_occupancies()
        self.nl_update = self._set_nl_update(params.pop("neighbour_list_update"))
        self.shells = self._set_shells(params.pop("shells"))
        self.scf_tol = float(params.pop("scf_tol"))
        self.scf_max_iter = int(params.pop("scf_max_iter"))
        self.set_RT(self.temperature * self.units["boltz"])
        self.pbc = bool(params.pop("pbc"))

        if params:
            raise ValueError(
                (
                    f"Unknown keys in reaction parameters: {params.keys()}\n"
                    f"Available keys are: {_DEFAULTS.keys()}"
                )
            )

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
        if len(_charges) != len(self.atom_types) - 1:
            raise ValueError("Atom type dictionary should be the same length as type charges dictionary")
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
            nargs = 2
            if args != nargs:
                raise ValueError(
                    f"coupling_function should have {nargs} arguments (rxn_ids, snapshot), found {args}"
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
                    f"coupling_function should have {nargs} arguments (rxn_ids, snapshot, new_forces, initial_forces), found {args}"
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
                Reaction(
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
        if self.scale_box:
            raise ValueError("Only constant volume right now.")

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

    def _set_shells(self, shells):
        return int(shells)

    def _set_fermi_mixing(self, fm):
        return bool(fm)

    def _set_get_occupancies(self):
        if self.FM and self.temperature > 1e-6:
            # fermi mixing

            def get_occupancies_FM(eig_vals, SI):
                return get_FD_occupancies(eig_vals, SI.RT)

            return get_occupancies_FM

        else:

            def get_occupancies_NO_FM(eig_vals, _):
                occupancies = np.zeros(len(eig_vals))
                occupancies[np.argmin(eig_vals)] = 1.0
                return occupancies

            return get_occupancies_NO_FM

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

        def _write(self, lmp, step, speed, pe, ke):
            props = [lmp.get_thermo(prop) for prop in self.properties]
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
                ke = lmp.get_thermo("ke")
                self._write(lmp, step, speed, pe, ke)
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

        def _write(self, lmp, box_data, universe, topology, pos=None):
            abcabc, abc = utils.extract_box(box_data)
            na = lmp.extract_global("natoms")
            z = np.zeros((na, 3))  # type: ignore
            xu = lmp.numpy.extract_fix("ux", 1, 2)
            ids = lmp.numpy.extract_atom("id")
            if ids.size != 0:
                z[topology.id_to_idx(ids)] = xu
            if universe.me == 0:
                for i in range(1, universe.sub_size):
                    _z = universe.global_comm.recv(source=i, tag=i)
                    z += _z
            else:
                universe.global_comm.send(z, dest=0, tag=universe.me)
            if universe.me == 0:
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
                            "{0} {1[0]:} {1[1]:} {1[2]:}\n".format(typ, pos)
                            for typ, pos in zip(topology.xyz_types, unwrapped_pos)
                        ]
                    )
                    self.xyz_file.write(
                        f"Bonds {self._fmt(topology.top_ref['bonds'])}\n"
                    )
                    self.xyz_file.write(
                        f"Angles {self._fmt(topology.top_ref['angles'])}\n"
                    )
                    self.xyz_file.write(
                        f"Impropers {self._fmt(topology.top_ref['impropers'])}\n"
                    )
                    self.xyz_file.write(
                        f"Dihedrals {self._fmt(topology.top_ref['dihedrals'])}\n"
                    )
                    self.xyz_file.flush()
                else:
                    self.dcd_file.write(
                        unwrapped_pos.astype(np.float32),
                        cell_lengths=list(abcabc[:3]),
                        cell_angles=abcabc[3:],
                    )

        def _fmt(self, arr):
            return np.array2string(arr, separator=",", formatter={"int": "{}".format})

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
    def get_z(mass):
        z = []
        for m in mass:
            z.append(mdtraj.element.Element.getByMass(m).atomic_number)  # type: ignore
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
                    Frame(
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
class Frame:
    frame: int
    natoms: int
    pbc: np.ndarray
    pos: np.ndarray
    imgs: np.ndarray
    types: np.ndarray
    bonds: np.ndarray
    angles: np.ndarray
    impropers: np.ndarray
    dihedrals: np.ndarray


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

        fmts = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: white + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset,
        }

        def format(self, record):
            log_fmt = self.fmts.get(record.levelno)
            formatter = logging.Formatter(log_fmt, datefmt=datefmt)
            return formatter.format(record)

    # formatter = logging.Formatter(fmt, datefmt=datefmt)
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    if filename is None:
        formatter = CustomFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

@dataclass
class Reaction:
    X: str
    H: str
    Y: str
    type_changes0: dict
    type_changes1: dict
    cutoffs: dict
