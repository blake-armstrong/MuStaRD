import numpy as np
import warnings

from typing import Union, Tuple, Dict, ClassVar
from itertools import combinations, product
from dataclasses import dataclass, field
from copy import copy
from collections import defaultdict

from . import utils
from lammps import lammps


@dataclass
class SpecificTopology:
    H_ids: np.ndarray
    H_idxs: np.ndarray
    X_ids: np.ndarray
    X_idxs: np.ndarray
    Y_ids: np.ndarray
    Y_idxs: np.ndarray


@dataclass
class Atom:
    idx: int
    type: int
    charge: float
    molecule: int
    mass: float
    image: np.ndarray


@dataclass  # partially immutable
class Frame:
    PRIVATE_VARIABLES: ClassVar[tuple] = (
        "pos",
        "images",
        "box_data",
        "box_vectors",
        "box_lengths",
        "box_matrix",
        "inv_box_matrix",
        "vel",
        "forces",
    )
    INIT: ClassVar[bool] = False
    lmp: lammps
    natoms: int
    dim: int
    scale_box: bool = False

    def __post_init__(self):
        self.pos = np.zeros((self.natoms, self.dim))
        self.images = np.zeros((self.natoms, self.dim))
        self.box_data = (None, None, None, None, None, None, None)
        self.box_vectors = np.zeros(9)
        self.box_lengths = np.zeros(3)
        self.box_matrix = np.zeros(shape=(3, 3))
        self.inv_box_matrix = np.zeros(shape=(3, 3))
        self.vel = np.zeros((self.natoms, self.dim))
        self.forces = np.zeros((self.natoms, self.dim))
        self._call = self._generate_call()
        self._set_one(self.lmp)
        self._set_two(self.lmp)
        Frame.INIT = True

    def __call__(self, lmp, pos=None, vel=None, forces=None, imgs=None):
        return self._call(self, lmp, pos=pos, vel=vel, forces=forces, imgs=imgs)

    def _generate_call(self):
        if self.scale_box:

            def call(self, lmp, pos=None, vel=None, forces=None, imgs=None):
                self._set_one(lmp, pos=pos, vel=vel, forces=forces, imgs=imgs)
                self._set_two(lmp)

            return call

        def call(self, lmp, pos=None, vel=None, forces=None, imgs=None):
            self._set_one(lmp, pos=pos, vel=vel, forces=forces, imgs=imgs)

        return call

    def _set_one(self, lmp, pos=None, vel=None, forces=None, imgs=None):
        if pos is None:
            pos = utils.get_positions(lmp)
        self.setattr("pos", pos)
        if vel is None:
            vel = utils.get_velocities(lmp)
        self.setattr("vel", vel)
        if imgs is None:
            imgs = utils.get_images(lmp)
        self.setattr("images", imgs)
        if forces is None:
            forces = utils.get_forces(lmp)
        self.setattr("forces", forces)

    def _set_two(self, lmp):
        box_data = utils.get_box_data(lmp)
        box_lengths, box_vectors = utils.extract_box(box_data)
        box_matrix = box_vectors.reshape(3, 3).T
        inv_box_matrix = np.linalg.inv(box_matrix)
        self.setattr("box_data", box_data)
        self.setattr("box_vectors", box_vectors)
        self.setattr("box_lengths", box_lengths)
        self.setattr("box_matrix", box_matrix)
        self.setattr("inv_box_matrix", inv_box_matrix)

    def __setattr__(self, prop, val):
        if Frame.INIT:
            if prop in Frame.PRIVATE_VARIABLES:
                raise RuntimeError(
                    (
                        f"Variable {prop} in Frame not intended for external modification by user. "
                        # f"To do this anyway call frame.setattr(prop={prop}, val={val})"
                    )
                )
        super().__setattr__(prop, val)

    def setattr(self, prop, val):
        super().__setattr__(prop, val)


@dataclass  # partially immutable
class Snapshot:
    """
    Class for storing information accessible to coupling function calls.
    """

    PRIVATE_VARIABLES: ClassVar[tuple] = (
        "frame",
        "ids",
        "types",
        "atoms",
        "residues",
        "bonds",
        "qs",
        "step",
        "energies",
        "forces",
        "computes",
    )
    INIT: ClassVar[bool] = False
    frame: Frame
    ids: np.ndarray
    types: np.ndarray
    atoms: dict
    residues: dict
    bonds: dict
    qs: np.ndarray
    step: int
    energies: Dict[str, float]
    forces: Dict[str, np.ndarray]
    computes: Dict[str, dict]

    def __post_init__(self):
        Snapshot.INIT = True

    def __setattr__(self, prop, val):
        if Snapshot.INIT:
            if prop in Snapshot.PRIVATE_VARIABLES:
                raise RuntimeError(
                    (
                        f"Variable {prop} in Snapshot not intended for external modification by user. "
                        f"To do this anyway call snapshot.setattr(prop={prop}, val={val})"
                    )
                )
        super().__setattr__(prop, val)

    def setattr(self, prop, val):
        super().__setattr__(prop, val)


@dataclass(frozen=True)
class Site:
    pair: Union[np.ndarray, None]
    xhy: Union[Tuple[Union[None, int], int, int], None]
    rxn_num: Union[int, None]
    index: int = 0
    dist: float = 0.0
    atoms: dict = field(default_factory=dict)
    bonds: dict = field(default_factory=dict)
    residues: dict = field(default_factory=dict)
    qs: np.ndarray = np.array([])
    types: np.ndarray = np.array([])
    shell: int = 1
    parent: int = 0
    site: int = 0

    def __repr__(self):
        return (
            "Site("
            f"index={self.index}, "
            f"pair={self.pair}, "
            f"rxn_num={self.rxn_num}, "
            f"site={self.site}, "
            f"distance={self.dist:.2f}, "
            f"shell={self.shell}, "
            f"parent={self.parent}"
            ")"
        )


@dataclass
class System:
    index: int
    sites: list[Site]
    atom_changes: dict = field(default_factory=dict)
    bond_changes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pairs = np.array([site.pair for site in self.sites])
        self.distances = np.array([site.dist for site in self.sites])

    def __repr__(self):
        return "System(" f"index={self.index}, " f"sites={self.sites}" ")"


class Topology:
    def __init__(self, lmp_obj, system_info):
        self.SI = system_info
        self.lmp = lmp_obj
        self.build_topology()
        self.systems: list
        self.num_systems: int
        self.num_sites: int
        self.current_system = self.empty_system()
        self.rxn_pair_info = dict()
        self.distsort: np.ndarray
        self.prev_rxn_pairs: np.ndarray = np.array([])
        self.update = True
        shape = utils.get_positions(self.lmp).shape
        self.frame = Frame(self.lmp, *shape, scale_box=self.SI.scale_box)
        self.snapshot = self.init_snapshot()

    def build_topology(self):
        (
            self.ids,
            self.types,
            self.atoms,
            self.residues,
            self.bonds,
            self.qs,
        ) = self.generate_generic_topology(self.lmp)
        self.ST = self.generate_specific_topology(self.atoms)
        self.xyz_types = np.vectorize(lambda x: self.SI.reverse_atom_types[x])(
            self.types
        )

        self._id_to_idx = np.vectorize(lambda x: self.atoms[x].idx)
        self.id_to_mass = np.vectorize(lambda x: self.atoms[x].mass)
        self.masses = self.id_to_mass(self.ids)

    def id_to_idx(self, id_arr):
        if len(id_arr) == 0:
            return np.array([])
        return self._id_to_idx(id_arr)

    def generate_generic_topology(self, lmp):
        ids = np.array(utils.gather_atoms(lmp, "id", 0, 1))
        types = np.array(utils.gather_atoms(lmp, "type", 0, 1))
        qs = np.array(utils.gather_atoms(lmp, "q", 1, 1))
        mols = np.array(utils.gather_atoms(lmp, "molecule", 0, 1))
        imgs = utils.get_images(lmp)
        masses = lmp.numpy.extract_atom("mass")[types]

        # dict stores per-ID info
        atom_info = {}
        for idx, ID in enumerate(ids):
            atom_info[ID] = Atom(
                idx=idx,
                type=types[idx],
                charge=qs[idx],
                molecule=mols[idx],
                mass=masses[idx],
                image=imgs[idx],
            )

        # dict converts residue ID to list of IDs
        mol_to_ids = defaultdict(list)
        # for mol in set(mols):
        # mol_to_ids[mol] = []
        for ID, atom in atom_info.items():
            mol_to_ids[atom.molecule].append(ID)

        with warnings.catch_warnings(record=True):
            self.top_ref = {
                "bonds": lmp.numpy.gather_bonds().astype(int),
                "angles": lmp.numpy.gather_angles().astype(int),
                "impropers": lmp.numpy.gather_impropers().astype(int),
                "dihedrals": lmp.numpy.gather_dihedrals().astype(int),
            }
            bonds = self.top_ref["bonds"][:, 1:]
        bonds_dict = {}
        for bond in bonds:
            a, b = bond
            if a not in bonds_dict:
                bonds_dict[a] = []
            if b not in bonds_dict:
                bonds_dict[b] = []
            bonds_dict[a].append(b)
            bonds_dict[b].append(a)

        return ids, types, atom_info, mol_to_ids, bonds_dict, qs

    def _generate_specific_topology(self, X, H, Y, atoms):
        H_ids, H_idxs = self._get_ids_from_type(H, atoms)
        Y_ids, Y_idxs = self._get_ids_from_type(Y, atoms)
        X_ids, X_idxs = self._get_ids_from_type(X, atoms)
        return SpecificTopology(
            H_ids=H_ids,
            H_idxs=H_idxs,
            X_ids=X_ids,
            X_idxs=X_idxs,
            Y_ids=Y_ids,
            Y_idxs=Y_idxs,
        )

    def generate_specific_topology(self, atoms):
        return [
            self._generate_specific_topology(rxn.X, rxn.H, rxn.Y, atoms)
            for rxn in self.SI.reactions
        ]

    def init_snapshot(self):
        return Snapshot(
            frame=self.frame,
            ids=self.ids,
            types=self.types,
            atoms=self.atoms,
            residues=self.residues,
            bonds=self.bonds,
            qs=self.qs,
            step=0,
            computes={},
            energies={},
            forces={},
        )

    def update_snapshot(
        self, frame, step, site=None, energies=None, computes=None, forces=None
    ):
        self.snapshot.setattr("frame", frame)
        self.snapshot.setattr("step", step)
        self.snapshot.setattr("ids", self.ids)
        if site is None:
            self.snapshot.setattr("types", self.types)
            self.snapshot.setattr("atoms", self.atoms)
            self.snapshot.setattr("residues", self.residues)
            self.snapshot.setattr("bonds", self.bonds)
            self.snapshot.setattr("qs", self.qs)
        else:
            self.snapshot.setattr("types", site.types)
            self.snapshot.setattr("atoms", site.atoms)
            self.snapshot.setattr("residues", site.residues)
            self.snapshot.setattr("bonds", site.bonds)
            self.snapshot.setattr("qs", site.qs)
        if computes is not None:
            self.snapshot.setattr("computes", computes)
        if energies is not None:
            self.snapshot.setattr("energies", energies)
        if forces is not None:
            self.snapshot.setattr("forces", forces)

    def full_get_pairs(
        self,
        positions,
        box_vectors,
        cutoffs,
        X_idxs,
        H_idxs,
        Y_idxs,
        rxn_num,
        residues,
        atoms,
        bonds,
    ):
        dist_cut, ang_cut = (
            cutoffs["distance"],
            cutoffs["angle"],
        )
        if H_idxs.size == 0 or Y_idxs.size == 0:
            return None
        H_pos = positions[H_idxs]
        Y_pos = positions[Y_idxs]
        dists = utils.get_distances(
            utils.get_distances_comb_xyz(H_pos, Y_pos, box_vectors)
        )
        if self.update:
            dists_bool = dists < dist_cut
        else:
            dists_bool = np.ones(shape=dists.shape) == 1  # type: ignore
        if not dists_bool.any():
            return None
        pair_dists = dists[dists_bool.nonzero()[0], dists_bool.nonzero()[1]]  # type: ignore
        # grab indexes of pairs that meet dist cutoff
        H_ready_idx = H_idxs[(dists_bool).nonzero()[0]]
        Y_ready_idx = Y_idxs[(dists_bool).nonzero()[1]]
        rxn_pairs = np.array([self.ids[H_ready_idx], self.ids[Y_ready_idx]]).T
        X_ready_idx = [
            atoms[eyed].idx
            for idx in H_ready_idx
            for eyed in residues[atoms[self.ids[idx]].molecule]
            if atoms[eyed].idx in X_idxs
        ]
        hid = [utils.get_X(self.ids[idx], bonds) for idx in H_ready_idx]
        mask = [
            (
                atoms[eyed].type == self.SI.reactions[rxn_num].X
                if eyed is not None
                else True
            )
            for eyed in hid
        ]
        rxn_pairs = rxn_pairs[mask]
        pair_dists = pair_dists[mask]
        if not X_idxs.any():
            return rxn_pairs, pair_dists, [None] * len(rxn_pairs)
        if ang_cut is None:
            return rxn_pairs, pair_dists, [None] * len(rxn_pairs)
        # check angles work as well.
        # positions of atoms in angle
        H_pos = positions[np.array(H_ready_idx)[mask]]
        X_pos = positions[np.array(X_ready_idx)[mask]]
        Y_pos = positions[np.array(Y_ready_idx)[mask]]
        # get angles..
        angles = utils.get_angles_comb(H_pos, X_pos, Y_pos, box_vectors)
        angle_bool = angles < ang_cut
        if not angle_bool.any():
            return None
        hxy_angles = angles[angle_bool]
        rxn_pairs = rxn_pairs[angle_bool]
        rxn_pairs = rxn_pairs[np.argsort(rxn_pairs[:, 0])]
        return rxn_pairs, pair_dists, hxy_angles

    def get_pairs(
        self,
        positions,
        box_vectors,
        cutoffs,
        X_idxs,
        H_idxs,
        Y_idxs,
        rxn_num,
        residues=None,
        atoms=None,
        bonds=None,
    ):
        if residues is None:
            residues = self.residues
        if atoms is None:
            atoms = self.atoms
        if bonds is None:
            bonds = self.bonds
        if not self.update:
            if not self.prev_rxn_pairs.any():
                return None
            H_idx = self.id_to_idx(self.prev_rxn_pairs[:, 0])
            Y_idx = self.id_to_idx(self.prev_rxn_pairs[:, 1])
            H_pos = positions[H_idx]
            Y_pos = positions[Y_idx]
            dists = utils.get_distances(
                utils.get_distances_xyz(H_pos, Y_pos, box_vectors)
            )
            return self.prev_rxn_pairs, dists, [None] * len(self.prev_rxn_pairs)
        return self.full_get_pairs(
            positions,
            box_vectors,
            cutoffs,
            X_idxs,
            H_idxs,
            Y_idxs,
            rxn_num,
            residues=residues,
            atoms=atoms,
            bonds=bonds,
        )

    def _get_pairs_idxs(self, pos, box_vectors, rxn_infos=None, atoms=None):
        if atoms is None:
            atoms = self.atoms
        rxn_pairs = []
        rxn_nums = []
        pair_dists = []
        hxy_angles = []
        pairs_idxs = []
        for rxn_num, rxn in enumerate(self.SI.reactions):
            if rxn_infos is None:
                rxn_info = self.get_pairs(
                    pos,
                    box_vectors,
                    rxn.cutoffs,
                    self.ST[rxn_num].X_idxs,
                    self.ST[rxn_num].H_idxs,
                    self.ST[rxn_num].Y_idxs,
                    rxn_num,
                )
            else:
                rxn_info = rxn_infos[rxn_num]
            if rxn_info is not None:
                rxn_pairs.append(rxn_info[0])
                rxn_nums.append([rxn_num] * len(rxn_info[0]))
                pair_dists.append(rxn_info[1])
                hxy_angles.append(rxn_info[2])
        if not rxn_pairs:
            rxn_pairs = np.array(rxn_pairs).reshape(-1, 2).astype(int)
            self.prev_rxn_pairs = rxn_pairs
            return [[]], rxn_pairs, {}
        pair_dists = np.concatenate(pair_dists, axis=0)
        rxn_pairs = np.concatenate(rxn_pairs, axis=0)
        self.prev_rxn_pairs = rxn_pairs
        sort = np.argsort(np.sum(rxn_pairs**2, axis=1))
        rxn_pairs = np.array(rxn_pairs[sort])
        rxn_nums = np.concatenate(rxn_nums, axis=0)[sort].astype(int)
        # hxy_angles = np.concatenate(hxy_angles, axis=0)[sort]
        pair_dists = pair_dists[sort]
        # self.dist_sort = np.argsort(pair_dists)

        # NOTE: The following is very bad and makes dangerous assumptions.
        rxn_molecules1 = np.array([atoms[pair[0]].molecule for pair in rxn_pairs])
        rxn_molecules2 = np.array([atoms[pair[1]].molecule for pair in rxn_pairs])
        rm1l = len(set(rxn_molecules1))
        rm2l = len(set(rxn_molecules2))
        rxn_molecules = rxn_molecules1
        if rm2l < rm1l:
            rxn_molecules = rxn_molecules2
        #######################

        pair_info = defaultdict(dict)
        for pair_idx, pair in enumerate(rxn_pairs):
            pair_info[pair_idx]["num"] = rxn_nums[pair_idx]
            pair_info[pair_idx]["dist"] = pair_dists[pair_idx]
            pair_info[pair_idx]["pair"] = pair
        #     self.rxn_pair_info[pair_idx + shift]["angle"] = hxy_angles[pair_idx]
        pairs_idxs = [
            list(np.arange(len(rxn_pairs))[rxn_molecules == i])
            for i in np.unique(rxn_molecules)
        ]
        if not pairs_idxs:
            pairs_idxs = [[]]
        return pairs_idxs, rxn_pairs, pair_info

    def get_systems(self, frame: Frame):
        if not self.update:
            if len(self.systems) > 1:
                return True
            return False

        pairs_idxs, rxn_pairs, pair_info = self._get_pairs_idxs(
            frame.pos, frame.box_vectors
        )
        nsites = len(pairs_idxs)
        if nsites > 1:
            raise RuntimeError(
                "No verified implementation of multi-site reactions yet."
            )
        nsite = 0
        pair_idxs = pairs_idxs[nsite]
        sites = [Site(pair=None, xhy=None, rxn_num=None, index=0, site=nsite)]
        for pair_idx in pair_idxs:
            _pair_info = pair_info[pair_idx]
            index = len(sites)
            pair = rxn_pairs[pair_idx]
            id_h, id_y = pair
            id_x = utils.get_X(id_h, self.bonds)
            sites.append(
                Site(
                    pair=pair,
                    xhy=(id_x, id_h, id_y),
                    rxn_num=_pair_info["num"],
                    index=index,
                    dist=_pair_info["dist"],
                    atoms=self.atoms,
                    bonds=self.bonds,
                    residues=self.residues,
                    qs=self.qs,
                    types=self.types,
                    shell=1,
                    parent=0,
                    site=nsite,
                )
            )

        if self.SI.shells == 1:
            return self._get_systems(sites, frame)

        # pairs_idxs_copy = deepcopy(pairs_idxs)
        # site = pairs_idxs[nsite]
        start = 0
        for shell in range(2, self.SI.shells + 1):
            stop = len(sites)
            for site in sites[start:]:
                if site.xhy is None:
                    continue
                # init_atoms = self.rxn_pair_info[pair_idx]["atoms"]
                # init_bonds = self.rxn_pair_info[pair_idx]["bonds"]
                # init_residues = self.rxn_pair_info[pair_idx]["residues"]
                # init_qs = self.rxn_pair_info[pair_idx]["qs"]
                # init_types = self.rxn_pair_info[pair_idx]["types"]
                # rxn_num = self.rxn_pair_info[pair_idx]["num"]
                # pair = self.rxn_pair_info[pair_idx]["pair"]
                # id_h, id_y = pair
                # id_x = utils.get_X(id_h, init_bonds)
                id_x, id_h, id_y = site.xhy
                hxs = site.residues[site.atoms[id_h].molecule]
                ys = site.residues[site.atoms[id_y].molecule]
                switch = False
                if id_x is None:
                    switch = True
                new_atoms = copy(site.atoms)
                new_qs = copy(site.qs)
                new_types = copy(site.types)
                for eyed in hxs + ys:
                    if eyed is None:
                        continue
                    type_changes = self.SI.reactions[site.rxn_num].type_changes1
                    if eyed in (id_x, id_h, id_y):
                        type_changes = self.SI.reactions[site.rxn_num].type_changes0
                    new_types[eyed] = new_type = type_changes[site.atoms[eyed].type]
                    new_qs[eyed] = new_q = self.SI.type_charges[new_type]
                    molecule = site.atoms[eyed].molecule
                    if eyed == id_h:
                        molecule = site.atoms[id_y].molecule
                    if eyed == id_y and switch:
                        molecule = site.atoms[id_h].molecule
                    new_atoms[eyed] = Atom(
                        idx=site.atoms[eyed].idx,
                        type=new_type,
                        charge=new_q,
                        molecule=molecule,
                        mass=site.atoms[eyed].mass,
                        image=site.atoms[eyed].image,
                    )
                new_residues = defaultdict(list)
                for eyed, atom in new_atoms.items():
                    new_residues[atom.molecule].append(eyed)
                _new_bonds = {
                    eyed: list(site.bonds.get(eyed, []))
                    for eyed in hxs + ys
                    if site.bonds.get(eyed)
                }
                if id_h in _new_bonds:
                    _new_bonds[_new_bonds[id_h][0]].remove(id_h)
                    _new_bonds[id_h][0] = id_y
                    _new_bonds.setdefault(id_y, []).append(id_h)
                new_bonds = copy(site.bonds)
                new_bonds.update({k: v for k, v in _new_bonds.items() if v})
                ST = self.generate_specific_topology(new_atoms)
                dont_idxs = self.id_to_idx(site.residues[site.atoms[id_h].molecule])
                rxn_infos = []
                for rxn_num, rxn in enumerate(self.SI.reactions):
                    new_X_idxs = self._remove_site(ST[rxn_num].X_idxs, dont_idxs)
                    new_H_idxs = self._remove_site(ST[rxn_num].H_idxs, dont_idxs)
                    # print(f"{ST[rxn_num].H_idxs=}")
                    # print(f"{dont_idxs=}")
                    # print(f"{new_H_idxs=}")
                    new_Y_idxs = self._remove_site(ST[rxn_num].Y_idxs, dont_idxs)
                    rxn_info = self.get_pairs(
                        frame.pos,
                        frame.box_vectors,
                        rxn.cutoffs,
                        new_X_idxs,
                        new_H_idxs,
                        new_Y_idxs,
                        rxn_num,
                        residues=new_residues,
                        atoms=new_atoms,
                        bonds=new_bonds,
                    )
                    rxn_infos.append(rxn_info)
                pair_idxs, rxn_pairs, pair_info = self._get_pairs_idxs(
                    frame.pos, frame.box_vectors, rxn_infos=rxn_infos
                )
                if len(pair_idxs) > 1:
                    raise RuntimeError("Undefined behaviour.")
                pair_idxs = pair_idxs[0]
                for pair_idx in pair_idxs:
                    _pair_info = pair_info[pair_idx]
                    index = len(sites)
                    pair = rxn_pairs[pair_idx]
                    id_h, id_y = pair
                    id_x = utils.get_X(id_h, new_bonds)
                    sites.append(
                        Site(
                            pair=pair,
                            xhy=(id_x, id_h, id_y),
                            rxn_num=_pair_info["num"],
                            index=index,
                            dist=_pair_info["dist"] + sites[site.index].dist,
                            atoms=new_atoms,
                            bonds=new_bonds,
                            residues=new_residues,
                            qs=new_qs,
                            types=new_types,
                            shell=shell,
                            parent=site.index,
                            site=nsite,
                        )
                    )
            start = stop

        return self._get_systems(sites, frame)

    def _get_systems(self, sites: list, frame: Frame):
        grouped_site_idxs = defaultdict(list)
        for site in sites:
            grouped_site_idxs[site.site].append(site.index)
        grouped_site_idxs = list(grouped_site_idxs.values())
        self.num_sites = len(grouped_site_idxs)
        systems_idxs = np.array(list(product(*grouped_site_idxs)))
        systems = []
        num_systems = 0
        for system_idxs in systems_idxs:
            system_sites = [sites[system_idx] for system_idx in system_idxs]
            system = self.generate_system(system_sites, num_systems, frame)
            if system is None:
                continue
            systems.append(system)
            num_systems += 1

        self.systems = systems
        self.num_systems = len(self.systems)
        if len(self.systems) > 1:
            return True
        return False

    def generate_system(
        self, sites: list[Site], system_idx: int, frame: Frame
    ) -> Union[System, None]:
        check_sites = {n: {} for n in range(self.num_sites)}
        for n, site in enumerate(sites):
            if site.xhy is None:
                check_sites[n]["a"] = {}
                check_sites[n]["b"] = {}
                continue
            id_x, id_h, id_y = site.xhy
            hxs = site.residues[site.atoms[id_h].molecule]
            ys = site.residues[site.atoms[id_y].molecule]
            ids_for_change = hxs + ys
            _new_bonds = {
                eyed: list(site.bonds.get(eyed, []))
                for eyed in ids_for_change
                if site.bonds.get(eyed)
            }
            if id_h in _new_bonds:
                _new_bonds[_new_bonds[id_h][0]].remove(id_h)
                _new_bonds[id_h][0] = id_y
                _new_bonds.setdefault(id_y, []).append(id_h)
            new_bonds = {k: v for k, v in _new_bonds.items() if v}
            new_imgs, yids = self.get_new_imgs(id_h, id_y, frame, site=site)
            new_atoms = {}
            for eyed in ids_for_change:
                if eyed is None:
                    continue
                type_changes = self.SI.reactions[site.rxn_num].type_changes1
                if eyed in (id_x, id_h, id_y):
                    type_changes = self.SI.reactions[site.rxn_num].type_changes0
                new_type = type_changes[site.atoms[eyed].type]
                new_charge = self.SI.type_charges[new_type]
                init_atom = site.atoms[eyed]
                new_img = init_atom.image
                if eyed in yids:
                    new_img = new_imgs[np.argwhere(yids == eyed)[0][0]]
                new_atoms[eyed] = Atom(
                    idx=init_atom.idx,
                    type=new_type,
                    charge=new_charge,
                    molecule=init_atom.molecule,  # not correct molecule ID but doesn't get used
                    mass=init_atom.mass,
                    image=new_img,
                )
            atom_diffs = {
                k: site.atoms[k]
                for k in site.atoms.keys() & self.atoms
                if site.atoms[k] != self.atoms[k]
            }
            atom_diffs.update(new_atoms)
            check_sites[n]["a"] = atom_diffs
            bond_diffs = {
                k: site.bonds[k]
                for k in site.bonds.keys() & self.bonds
                if site.bonds[k] != self.bonds[k]
            }
            bond_diffs.update(new_bonds)
            if self.SI.shells > 1:
                bdk = bond_diffs.keys()
                adk = atom_diffs.keys()
                nkb, nka = [], []
                for v in bond_diffs.values():
                    for j in v:
                        if j not in bdk:
                            nkb.append(j)
                        if j not in adk:
                            nka.append(j)
                for i in nkb:
                    bond_diffs[i] = site.bonds[i]
                for i in nka:
                    atom_diffs[i] = site.atoms[i]

            check_sites[n]["b"] = bond_diffs

        if self.num_sites == 1:
            return System(
                index=system_idx,
                sites=sites,
                atom_changes=check_sites[0]["a"],
                bond_changes=check_sites[0]["b"],
            )

        combined_changes = self.check_collisions(check_sites)
        if combined_changes is None:
            # Collisions occured, cannot create system
            return None
        a, b = combined_changes
        return System(index=system_idx, sites=sites, atom_changes=a, bond_changes=b)

    def check_collisions(self, check_sites: dict) -> Union[Tuple[dict, dict], None]:
        # TODO: test
        combined_atom_changes = {}
        combined_bond_changes = {}
        for site_idx in range(self.num_sites):
            atom_diffs = check_sites[site_idx]["a"]
            if atom_diffs.keys() in combined_atom_changes.keys():
                return None
            combined_atom_changes.update(atom_diffs)
            bond_diffs = check_sites[site_idx]["b"]
            combined_bond_changes.update(bond_diffs)
        return combined_atom_changes, combined_bond_changes

    @staticmethod
    def _remove_site(idxs, rxn_pair):
        return idxs[np.isin(idxs, rxn_pair.flatten(), invert=True)]

    def grab_system(self, system_idx: int):
        try:
            return self.systems[system_idx]
        except IndexError:
            return self.empty_system()

    def empty_system(self):
        return System(index=0, sites=[Site(pair=None, xhy=None, rxn_num=None, index=0)])

    def set_lmp(self, lmp_obj):
        self.lmp = lmp_obj

    def get_new_imgs(
        self, h: int, y: int, frame: Frame, site: Union[Site, None, "Topology"] = None
    ) -> Tuple[np.ndarray, list]:
        if site is None:
            site = self
        yids = site.residues[site.atoms[y].molecule]
        yidxs = [site.atoms[ID].idx for ID in yids]
        hpos = frame.pos[site.atoms[h].idx]
        himg = frame.images[site.atoms[h].idx]
        ypos = frame.pos[yidxs]
        disp = utils.get_distances_xyz(ypos, hpos, frame.box_vectors)
        uhpos = utils.unwrap_coordinates(
            hpos, frame.box_matrix, frame.inv_box_matrix, himg
        )
        uypos = uhpos + disp
        new_imgs = utils.get_periodic_images(uypos, frame.inv_box_matrix)
        if not self.SI.pbc:
            new_imgs *= 0
        return new_imgs, yids

    def set_traj_frame(self, frame):
        self.lmp.command("delete_bonds all multi remove")
        types = np.vectorize(lambda typ: self.SI.atom_types[typ.replace(" ", "")])(
            frame.types
        )
        charges = np.vectorize(lambda typ: self.SI.type_charges[typ])(types)
        set_type_charges = []
        for idx in range(frame.natoms):
            set_type_charges.append(f"set atom {self.ids[idx]} type {types[idx]}")
            set_type_charges.append(f"set atom {self.ids[idx]} charge {charges[idx]}")
            img = frame.imgs[idx]
            set_type_charges.append(
                f"set atom {self.ids[idx]} image {img[0]} {img[1]} {img[2]}"
            )
        self.lmp.commands_list(set_type_charges)
        add_bonds = np.array([])
        if frame.bonds.any():
            add_bonds = np.apply_along_axis(
                lambda row: "create_bonds single/bond {:10} {:10} {:10} special no ".format(
                    *row
                ),
                axis=1,
                arr=frame.bonds,
            )
        add_angles = np.array([])
        if frame.angles.any():
            add_angles = np.apply_along_axis(
                lambda row: "create_bonds single/angle {:10} {:10} {:10} {:10} special no ".format(
                    *row
                ),
                axis=1,
                arr=frame.angles,
            )
        add_impropers = np.array([])
        if frame.impropers.any():
            add_impropers = np.apply_along_axis(
                lambda row: "create_bonds single/improper {:10} {:10} {:10} {:10} {:10} special no ".format(
                    *row
                ),
                axis=1,
                arr=frame.impropers,
            )
        add_dihedrals = np.array([])
        if frame.dihedrals.any():
            add_dihedrals = np.apply_along_axis(
                lambda row: "create_bonds single/dihedral {:10} {:10} {:10} {:10} {:10} special no ".format(
                    *row
                ),
                axis=1,
                arr=frame.dihedrals,
            )
        full = list(
            np.concatenate(
                [add_bonds, add_angles, add_impropers, add_dihedrals]
            ).flatten()
        )
        full[-1] = full[-1].replace("no", "yes")
        [self.lmp.command(cmd) for cmd in full]
        self.lmp.command("reset_atoms mol all single yes")

    def _set_top(self, total_ids, type_str, fmt_str):
        add = self.top_ref[type_str][
            np.any(np.isin(self.top_ref[type_str][:, 1:], total_ids), axis=1)
        ]
        fmtd_add = np.array([])
        if add.any():
            fmtd_add = np.apply_along_axis(
                lambda row: fmt_str.format(*row),
                axis=1,
                arr=add,
            )
        return fmtd_add.reshape(-1, 1)

    def _reset_lmp_topology(self, total_ids):
        types = np.vectorize(lambda ID: self.atoms[ID].type)(total_ids)
        charges = np.vectorize(lambda ID: self.atoms[ID].charge)(total_ids)

        set_type_charge_img = []
        for idx, ID in enumerate(total_ids):
            set_type_charge_img.append(f"set atom {ID} type {types[idx]}")
            set_type_charge_img.append(f"set atom {ID} charge {charges[idx]}")
            images = self.atoms[ID].image
            set_type_charge_img.append(
                f"set atom {ID} image {images[0]} {images[1]} {images[2]}"
            )
        self.lmp.commands_list(set_type_charge_img)

        b = self._set_top(
            total_ids,
            type_str="bonds",
            fmt_str="create_bonds single/bond {:10} {:10} {:10} special no ",
        )
        a = self._set_top(
            total_ids,
            type_str="angles",
            fmt_str="create_bonds single/angle {:10} {:10} {:10} {:10} special no ",
        )
        i = self._set_top(
            total_ids,
            type_str="impropers",
            fmt_str="create_bonds single/improper {:10} {:10} {:10} {:10} {:10} special no ",
        )
        d = self._set_top(
            total_ids,
            type_str="dihedrals",
            fmt_str="create_bonds single/dihedral {:10} {:10} {:10} {:10} {:10} special no ",
        )
        full = list(np.concatenate([b, a, i, d]).flatten())
        if full:
            full[-1] = full[-1].replace("no", "yes")
        [self.lmp.command(cmd) for cmd in full]
        self.lmp.command("reset_atoms mol all single yes")

    def reset_lmp_topology(self):
        self.lmp.command("delete_bonds all multi remove")
        self._reset_lmp_topology(self.ids)
        # self.lmp.command("run 0 pre yes post no")

    def change_topology_to_system(self, lmp, system, frame):
        self.current_system = system
        if system.index == 0:
            return
        lammps_commands = []
        hxy_group_str = "group HXY id "
        for eyed, atom in system.atom_changes.items():
            hxy_group_str += f"{eyed} "
            set_type_cmd = f"set atom {eyed} type {atom.type}"
            set_charge_cmd = f"set atom {eyed} charge {atom.charge}"
            set_img_cmd = (
                f"set atom {eyed} image {atom.image[0]} {atom.image[1]} {atom.image[2]}"
            )
            lammps_commands.append(set_type_cmd)
            lammps_commands.append(set_charge_cmd)
            lammps_commands.append(set_img_cmd)
        # removes all bonds, angles, dihedrals and impropers involving these ids
        lammps_commands.insert(0, "delete_bonds HXY multi remove")
        lammps_commands.insert(0, hxy_group_str)
        lammps_commands += self.create_bonds(system.atom_changes, system.bond_changes)
        lammps_commands[-1] = lammps_commands[-1].replace("no", "yes")
        lammps_commands.append("group HXY delete")
        lammps_commands.append("reset_atoms mol all single yes")
        lmp.commands_list(lammps_commands)
        # if self.skips[system.index]:
        #     return
        create_bonds = []
        set_type_charge = []
        # for nsite, site in enumerate(system.sites):
        #     if site is None:
        #         continue
        #     parent_systems = []
        #     shell = site.shell
        #     parent = site.parent
        #     while shell > 0:
        #         parent_system = self.systems[parent]
        #         parent_systems.append(parent_system)
        #         if parent_system.index != 0:
        #             parent = parent_system.sites[nsite].parent
        #         shell -= 1
        #     parent_systems.reverse()
        #     change_topology_systems = parent_systems + [system]
        #     for _system in change_topology_systems[1:]:
        #         site = _system.sites[nsite]
        #         # for rxn_pair, rxn_num in zip(system.pairs, rxn_nums):
        #         # bonds = site.bonds
        #         # atoms = site.atoms
        #         # residues = site.residues
        #         rxn_pair = site.pair
        #         rxn_num = site.rxn_num
        #         id_h, id_y = rxn_pair
        #         id_x = utils.get_X(id_h, site.bonds)
        #         # create groups
        #         hxy_group_str = "group HXY id "
        #         hxs = site.residues[site.atoms[id_h].molecule]
        #         ys = site.residues[site.atoms[id_y].molecule]
        #         for eyed in hxs + ys:
        #             hxy_group_str += f"{eyed} "
        #
        #         # removes all bonds, angles, dihedrals and impropers involving these ids
        #         # lmp.commands_list([hxy_group_str, "delete_bonds HXY multi remove"])
        #         create_bonds += [hxy_group_str, "delete_bonds HXY multi remove"]
        #         ids_for_change = hxs + ys
        #         new_bonds_dict = {
        #             eyed: list(site.bonds.get(eyed, []))
        #             for eyed in ids_for_change
        #             if site.bonds.get(eyed)
        #         }
        #         if id_h in new_bonds_dict:
        #             new_bonds_dict[new_bonds_dict[id_h][0]].remove(id_h)
        #             new_bonds_dict[id_h][0] = id_y
        #             new_bonds_dict.setdefault(id_y, []).append(id_h)
        #         nbd = {k: v for k, v in new_bonds_dict.items() if v}
        #         # changes types and charges
        #         # its possible to just edit the array returned from extract_atoms
        #         # or call scatter_atoms, probably faster than looping through set
        #         new_types = {}
        #         rxn_ids = (id_h, id_y, id_x)
        #         for eyed in rxn_ids:
        #             if eyed is None:
        #                 continue
        #             new_types[eyed] = self.SI.reactions[rxn_num].type_changes0[
        #                 site.atoms[eyed].type
        #             ]
        #             set_type_charge.append(f"set atom {eyed} type {new_types[eyed]}")
        #             set_type_charge.append(
        #                 f"set atom {eyed} charge {self.SI.type_charges[new_types[eyed]]}"
        #             )
        #             ids_for_change.remove(eyed)
        #         for eyed in ids_for_change:
        #             if eyed is None:
        #                 continue
        #             new_types[eyed] = self.SI.reactions[rxn_num].type_changes1[
        #                 site.atoms[eyed].type
        #             ]
        #             set_type_charge.append(f"set atom {eyed} type {new_types[eyed]}")
        #             set_type_charge.append(
        #                 f"set atom {eyed} charge {self.SI.type_charges[new_types[eyed]]}"
        #             )
        #         new_imgs, yids = self.get_new_imgs(id_h, id_y, frame, site=site)
        #         if not self.SI.pbc:
        #             new_imgs *= 0
        #         create_bonds += [
        #             f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
        #             for ID, imgs in zip(yids, new_imgs)
        #         ]
        #         create_bonds += self.create_bonds(new_types, nbd)
        #         create_bonds += ["group HXY delete"]
        # create_bonds[-2] = create_bonds[-2].replace("no", "yes")
        # lmp.commands_list(set_type_charge + create_bonds)
        # self.lmp.command("reset_atoms mol all single yes")

    def create_bonds(self, atoms: dict, bonds: dict) -> list:
        types = {eyed: atom.type for eyed, atom in atoms.items()}
        cmd_list = []
        angles, propers, impropers = self._get_angles_dihedrals(bonds)
        # bonds
        # TODO: currently doesn't work if a bond between two IDs has more than one bond type
        if self.SI.bond_types:
            dont = set()
            for id_0, id_arr in bonds.items():
                new_type_0 = types[id_0]
                for id_1 in id_arr:
                    if f"{id_0}-{id_1}" not in dont and f"{id_1}-{id_0}" not in dont:
                        new_type_1 = types[id_1]
                        try:
                            bond_type = self.SI.bond_types[f"{new_type_0}-{new_type_1}"]
                            cmd_list.append(
                                f"create_bonds single/bond {bond_type} {id_0} {id_1} special no"
                            )
                            dont.add(f"{id_1}-{id_0}")
                        except KeyError:
                            try:
                                bond_type = self.SI.bond_types[
                                    f"{new_type_1}-{new_type_0}"
                                ]
                                cmd_list.append(
                                    f"create_bonds single/bond {bond_type} {id_1} {id_0} special no"
                                )
                                dont.add(f"{id_0}-{id_1}")
                            except KeyError:
                                pass

        # angles
        if self.SI.angle_types and angles:
            dont = set()
            for id_0, id_1, id_2 in angles:
                if f"{id_0}-{id_1}-{id_2}" in dont or f"{id_2}-{id_1}-{id_0}" in dont:
                    continue
                new_type_0 = types[id_0]
                new_type_1 = types[id_1]
                new_type_2 = types[id_2]
                try:
                    angle_type = self.SI.angle_types[
                        f"{new_type_0}-{new_type_1}-{new_type_2}"
                    ]
                    dont.add(f"{id_0}-{id_1}-{id_2}")
                except KeyError:
                    continue
                cmd_list.append(
                    f"create_bonds single/angle {angle_type} {id_0} {id_1} {id_2} special no"
                )

        # propers
        if self.SI.proper_types and propers:
            dont = set()
            for id_0, id_1, id_2, id_3 in propers:
                if (
                    f"{id_0}-{id_1}-{id_2}-{id_3}" in dont
                    or f"{id_3}-{id_2}-{id_1}-{id_0}" in dont
                ):
                    continue
                new_type_0 = types[id_0]
                new_type_1 = types[id_1]
                new_type_2 = types[id_2]
                new_type_3 = types[id_3]
                ids = [id_0, id_1, id_2, id_3]
                try:
                    proper_type = self.SI.proper_types[
                        f"{new_type_0}-{new_type_1}-{new_type_2}-{new_type_3}"
                    ]
                except:
                    proper_type = self.SI.proper_types[
                        f"{new_type_3}-{new_type_2}-{new_type_1}-{new_type_0}"
                    ]
                    ids.reverse()
                dont.add(f"{ids[0]}-{ids[1]}-{ids[2]}-{ids[3]}")
                cmd_list.append(
                    f"create_bonds single/dihedral {proper_type} {ids[0]} {ids[1]} {ids[2]} {ids[3]} special no"
                )

        # impropers
        if self.SI.improper_types and impropers:
            dont = set()
            for id_0, id_1, id_2, id_3 in impropers:
                if f"{id_0}-{id_1}-{id_2}-{id_3}" in dont:
                    continue
                new_type_0 = types[id_0]
                new_type_1 = types[id_1]
                new_type_2 = types[id_2]
                new_type_3 = types[id_3]
                try:
                    improper_type = self.SI.improper_types[
                        f"{new_type_0}-{new_type_1}-{new_type_2}-{new_type_3}"
                    ]
                except KeyError:
                    continue
                dont.add(f"{id_0}-{id_1}-{id_2}-{id_3}")
                cmd_list.append(
                    f"create_bonds single/improper {improper_type} {id_0} {id_1} {id_2} {id_3} special no"
                )
        return cmd_list

    @staticmethod
    def _get_ids_from_type(type_int, atom_info):
        ids = np.sort(
            np.array([id for id, atom in atom_info.items() if atom.type == type_int])
        )
        idxs = np.array([atom_info[id].idx for id in ids])
        return ids, idxs

    @staticmethod
    def _get_angles_dihedrals(bonds_dict):
        angles = set()
        propers = set()
        impropers = set()
        for a, bs in bonds_dict.items():
            for b1, b2 in combinations(bs, 2):
                # angles
                angles.add((b1, a, b2))
                # dihedrals
                # propers
                # check if b1 or b2 are in any other bonds
                check_a = np.isin(bonds_dict[b1], [a], invert=True)
                if check_a.any():
                    for id in np.array(bonds_dict[b1])[check_a]:
                        propers.add((id, b1, a, b2))
                check_b = np.isin(bonds_dict[b2], [a], invert=True)
                if check_b.any():
                    for id in np.array(bonds_dict[b2])[check_b]:
                        propers.add((b1, a, b2, id))
                # impropers
                check_c = np.isin(bonds_dict[a], [b1, b2], invert=True)
                if check_c.any():
                    for id in np.array(bonds_dict[a])[check_c]:
                        s = sorted((b1, b2, id))
                        impropers.add((a, *s))

        return list(angles), list(propers), list(impropers)
