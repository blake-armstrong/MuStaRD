import mdtraj
import numpy as np
import inspect
from time import time
from dataclasses import dataclass
from lammps import lammps
from typing import Union, IO, Callable
from .topology import Topology, Frame
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
    "fd_tol_1": 1e-6,
    "fd_tol_2": 1e-6,
    "constant_volume": True,
    "lammps_unit_system": None,
    "computes": None,
    "fermi_mixing": False,
    "neighbour_list_update": 4,
    "topology_update": 1,
    "scf_tol": 1e-4,
    "scf_max_iter": 100,
    "shells": 1,
    "pbc": True,
    "eig_solver": "numpy",
}


class SystemInfo:
    def __init__(self, input_params: dict):
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
        self.top_update = self._set_top_update(params.pop("topology_update"))
        self.shells = self._set_shells(params.pop("shells"))
        self.scf_tol = float(params.pop("scf_tol"))
        self.scf_max_iter = int(params.pop("scf_max_iter"))
        self.set_RT(self.temperature * self.units["boltz"])
        self.pbc = bool(params.pop("pbc"))
        self.eig_solver = self._set_eig_solver(params.pop("eig_solver"))

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
            raise ValueError(
                "Atom type dictionary should be the same length as type charges dictionary"
            )
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
        self.coupling = []
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
            nargs = 2
            if args != nargs:
                raise ValueError(
                    f"coupling_function should have {nargs} arguments (rxn_ids, snapshot), found {args}"
                )
            self.coupling.append(_coupling_function)

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

    def _set_top_update(self, topology_update):
        return int(topology_update)

    def _set_eig_solver(self, eig_solver):
        SOLVERS = ("NUMPY", "SYMPY")
        eig_solver = str(eig_solver).upper()
        if eig_solver not in SOLVERS:
            raise ValueError(
                f"Eigen value solver {eig_solver} not in available solvers: {SOLVERS}"
            )
        return eig_solver

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
        header = "      Step          Pe(mixed)          E_total            E_conserve"
        self.info = "{:>10} {:>19.10f} {:>19.10f} {:>19.10f}"
        for title in self.properties:
            header += f"{title.title():>20}"
            self.info += " {:>19.10f}"
        header += "      Speed(ns/day)"
        self.info += " {:>15.8f}"
        self.log(header)
        self.write_header = False

    def _write(self, step, speed, pe, ke, ecpl):
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
        ke = self.lmp.get_thermo("ke")
        ecpl = self.lmp.get_thermo("ecouple")
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
        # abcabc, abc = utils.extract_box(box_data)
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
                        "{0} {1[0]:} {1[1]:} {1[2]:}\n".format(typ, pos)
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
class TrajectoryFrame:
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


@dataclass
class Reaction:
    X: int
    H: int
    Y: int
    type_changes0: dict
    type_changes1: dict
    cutoffs: dict
