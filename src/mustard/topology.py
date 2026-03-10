import numpy as np
import warnings

from typing import Union, Tuple, Dict, ClassVar, Set
from lammps import lammps
from itertools import combinations, product, permutations
from dataclasses import dataclass, field
from copy import copy
from collections import defaultdict

from . import utils


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
class Site:
    """
    Do not use dist or dist_xyz in coupling function, they will not be correct.
    """

    pair: Union[np.ndarray, None]
    xhy: Union[Tuple[Union[None, int], int, int], None]
    rxn_num: Union[int, None]
    index: int = 0
    # dist: float = 0.0
    total_dist: float = 0.0
    # dist_xyz: np.ndarray = np.array([0.0, 0.0, 0.0])
    atoms: dict = field(default_factory=dict)
    bonds: dict = field(default_factory=dict)
    residues: dict = field(default_factory=dict)
    qs: np.ndarray = field(default_factory=lambda: np.ndarray([]))
    types: np.ndarray = field(default_factory=lambda: np.ndarray([]))
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
            f"distance={self.total_dist:.2f}, "
            f"shell={self.shell}, "
            f"parent={self.parent}"
            ")"
        )


@dataclass  # partially immutable
class Snapshot:
    """
    Class for storing information accessible to coupling function calls.
    """

    PRIVATE_VARIABLES: ClassVar[tuple] = (
        "frame",
        "ids",
        "site",
        "step",
        "energies",
        "forces",
        "computes",
    )
    INIT: ClassVar[bool] = False
    frame: Frame
    ids: np.ndarray
    site: Site
    step: int
    energies: Dict[str, float]
    forces: Dict[str, np.ndarray]
    computes: Union[Dict[str, dict], None]

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


@dataclass
class PseudoSite:
    atoms: dict
    bonds: dict
    residues: dict


@dataclass
class System:
    index: int
    sites: list[Site]
    parents: Union[None, Set[int]] = None
    site_parents: Union[None, Dict[int, int]] = None
    atom_changes: dict = field(default_factory=dict)
    bond_changes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pairs = np.array(
            [site.pair if site.pair is not None else [-1, -1] for site in self.sites]
        )
        # self.distances = np.array([site.dist for site in self.sites])
        self._parents = tuple(site.parent for site in self.sites)

    def __repr__(self):
        return (
            "System("
            f"index={self.index}, "
            f"sites={self.sites}, "
            f"parents={self.parents}"
            ")"
        )

    def get_parents(self):
        if not self.site_parents:
            self.parents = None
            return
        self.parents = set(p for p in self.site_parents.keys())


@dataclass
class Pair:
    ids: np.ndarray
    distance: float
    distance_xyz: np.ndarray
    angle: Union[None, float]
    reaction_type: int

    def __post_init__(self):
        a, b = self.ids
        self.tag = 0.5 * (a + b) * (a + b + 1) + b


@dataclass
class Reaction:
    num: int
    distance_cutoff: float
    angle_cutoff: Union[None, float]


class Topology:

    @staticmethod
    def empty_system() -> System:
        return System(index=0, sites=[Site(pair=None, xhy=None, rxn_num=None, index=0)])

    EMPTY_SYSTEM = empty_system()

    def __init__(self, lmp_obj, system_info):
        self.SI = system_info
        self.lmp = lmp_obj
        self.build_topology()
        self.systems: list
        self.num_systems: int
        self.num_sites: int
        self.grouped_site_idxs: list[list] = [[]]
        self.current_system = self.empty_system()
        self.rxn_pair_info = dict()
        self.distsort: np.ndarray
        self.prev_pairs_info: list = []
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
        self.reactions = [
            Reaction(
                num=rxn_num,
                distance_cutoff=rxn.cutoffs["distance"],
                angle_cutoff=rxn.cutoffs["angle"],
            )
            for rxn_num, rxn in enumerate(self.SI.reactions)
        ]
        self._id_to_idx = np.vectorize(lambda x: self.atoms[x].idx)
        self.id_to_mass = np.vectorize(lambda x: self.atoms[x].mass)
        self.masses = self.id_to_mass(self.ids)

    def id_to_idx(self, id_arr):
        if len(id_arr) == 0:
            return np.array([])
        return self._id_to_idx(id_arr)

    def generate_generic_topology(self, lmp: lammps) -> Tuple[
        np.ndarray,
        np.ndarray,
        Dict[int, Atom],
        Dict[int, list],
        Dict[int, list],
        np.ndarray,
    ]:
        ids = np.array(utils.gather_atoms(lmp, "id", 0, 1))
        types = np.array(utils.gather_atoms(lmp, "type", 0, 1))
        qs = np.array(utils.gather_atoms(lmp, "q", 1, 1))
        mols = np.array(utils.gather_atoms(lmp, "molecule", 0, 1))
        imgs = utils.get_images(lmp)
        masses = lmp.numpy.extract_atom("mass")
        if masses is None:
            raise ValueError("Could not extract masses.")
        masses = masses[types]

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

    def _generate_specific_topology(
        self, X: int, H: int, Y: int, atoms: Dict[int, Atom]
    ) -> SpecificTopology:
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

    def generate_specific_topology(self, atoms: Dict[int, Atom]) -> list:
        return [
            self._generate_specific_topology(rxn.X, rxn.H, rxn.Y, atoms)
            for rxn in self.SI.reactions
        ]

    def init_snapshot(self) -> Snapshot:
        return Snapshot(
            frame=self.frame,
            ids=self.ids,
            site=Site(pair=None, xhy=None, rxn_num=None, index=0),
            step=0,
            energies={},
            forces={},
            computes=None,
        )

    def update_snapshot(
        self,
        frame: Frame,
        step: int,
        site: Site,
        energies: Dict[str, float],
        forces: Dict[str, np.ndarray],
        computes: Union[Dict[str, dict], None] = None,
    ) -> None:
        self.snapshot.setattr("frame", frame)
        self.snapshot.setattr("step", step)
        self.snapshot.setattr("ids", self.ids)
        self.snapshot.setattr("site", site)
        self.snapshot.setattr("energies", energies)
        self.snapshot.setattr("forces", forces)
        if computes is not None:
            self.snapshot.setattr("computes", computes)

    def full_get_pairs(
        self,
        frame: Frame,
        reaction: Reaction,
        specific_topology: SpecificTopology,
        topology: Union[
            "Topology", PseudoSite
        ],  # NOTE: 'Topology' will become typing.Self in future.
    ) -> Union[None, list[Pair]]:
        # TODO: too many args, make more readable
        if specific_topology.H_idxs.size == 0 or specific_topology.Y_idxs.size == 0:
            return None
        H_pos = frame.pos[specific_topology.H_idxs]
        Y_pos = frame.pos[specific_topology.Y_idxs]
        dists_xyz = utils.get_distances_comb_xyz(H_pos, Y_pos, frame.box_vectors)
        dists = utils.get_distances(dists_xyz)
        if self.update:
            dists_bool = np.array(dists < reaction.distance_cutoff)
        else:
            if type(dists) != np.ndarray:
                raise RuntimeError("Dists not array.")
            dists_bool = np.ones(shape=dists.shape) == 1
        if not dists_bool.any():
            return None
        hmask = dists_bool.nonzero()[0]
        ymask = dists_bool.nonzero()[1]
        pair_dists_xyz = dists_xyz[hmask, ymask]
        pair_dists = dists[hmask, ymask]  # type: ignore
        # grab indexes of pairs that meet dist cutoff
        H_ready_idx = specific_topology.H_idxs[hmask]
        Y_ready_idx = specific_topology.Y_idxs[ymask]
        rxn_pairs = np.array([self.ids[H_ready_idx], self.ids[Y_ready_idx]]).T
        X_ready_idx = [
            topology.atoms[eyed].idx
            for idx in H_ready_idx
            for eyed in topology.residues[topology.atoms[self.ids[idx]].molecule]
            if topology.atoms[eyed].idx in specific_topology.X_idxs
        ]
        hid = [utils.get_X(self.ids[idx], topology.bonds) for idx in H_ready_idx]
        mask = np.array(
            [
                (
                    topology.atoms[eyed].type == self.SI.reactions[reaction.num].X
                    if eyed is not None and self.SI.reactions[reaction.num].X is not None
                    else True
                )
                for eyed in hid
            ]
        )
        for n, (i, j) in enumerate(rxn_pairs):
            if i == j:
                mask[n] = False

        rxn_pairs = rxn_pairs[mask]
        pair_dists = pair_dists[mask]
        pair_dists_xyz = pair_dists_xyz[mask]
        if not specific_topology.X_idxs.any() or reaction.angle_cutoff is None:
            return [
                Pair(
                    ids=ids,
                    distance=pair_dists[n],
                    distance_xyz=pair_dists_xyz[n],
                    angle=None,
                    reaction_type=reaction.num,
                )
                for n, ids in enumerate(rxn_pairs)
            ]

        # NOTE: Angle has not been verified yet.
        # check angles work as well.
        H_pos = frame.pos[np.array(H_ready_idx)[mask]]
        X_pos = frame.pos[np.array(X_ready_idx)[mask]]
        Y_pos = frame.pos[np.array(Y_ready_idx)[mask]]
        # get angles..
        angles = utils.get_angles_comb(H_pos, X_pos, Y_pos, frame.box_vectors)
        angle_bool = angles < reaction.angle_cutoff
        if not angle_bool.any():
            return None
        hxy_angles = angles[angle_bool]
        rxn_pairs = rxn_pairs[angle_bool]
        return [
            Pair(
                ids=ids,
                distance=pair_dists[n],
                distance_xyz=pair_dists_xyz[n],
                angle=hxy_angles[n],
                reaction_type=reaction.num,
            )
            for n, ids in enumerate(rxn_pairs)
        ]

    def get_pairs(
        self,
        frame: Frame,
        reaction: Reaction,
        specific_topology: SpecificTopology,
        topology: Union[PseudoSite, "Topology"],
    ):
        if not self.update:
            if not self.prev_pairs_info:
                return None
            reaction_pairs = self.id_to_idx(
                np.array([pair.ids for pair in self.prev_pairs_info])
            )
            H_idx = reaction_pairs[:, 0]
            Y_idx = reaction_pairs[:, 1]
            H_pos = frame.pos[H_idx]
            Y_pos = frame.pos[Y_idx]
            dists_xyz = utils.get_distances_xyz(H_pos, Y_pos, frame.box_vectors)
            dists = utils.get_distances(dists_xyz)
            return [
                Pair(
                    ids=pair.ids,
                    distance=dists[n],  # type: ignore
                    distance_xyz=dists_xyz[n],
                    angle=None,
                    reaction_type=pair.reaction_type,
                )
                for n, pair in enumerate(self.prev_pairs_info)
            ]
        return self.full_get_pairs(frame, reaction, specific_topology, topology)

    def get_pairs_info(
        self,
        frame: Frame,
        topology: Union[PseudoSite, "Topology"],
        specific_topologies: list[SpecificTopology],
    ) -> list[Pair]:
        pairs_info = []
        for reaction in self.reactions:
            pair_info = self.get_pairs(
                frame, reaction, specific_topologies[reaction.num], topology
            )
            if pair_info is None:
                continue
            pairs_info += pair_info
        return pairs_info

    def get_pairs_by_site(
        self, pairs_info: list[Pair], topology: Union[PseudoSite, "Topology"]
    ) -> Tuple[list[list], dict[int, Pair]]:
        self.prev_pairs_info = pairs_info
        if not pairs_info:
            return [[]], {}
        sort = np.argsort([pair.tag for pair in pairs_info])
        sorted_pairs_info = [pairs_info[s] for s in sort]

        # NOTE: This is done to group pairs by the same residue \
        # to allow for multi-site solving.
        residue_info = np.array(
            [
                [
                    topology.atoms[pair.ids[0]].molecule,
                    topology.atoms[pair.ids[1]].molecule,
                ]
                for pair in sorted_pairs_info
            ]
        )
        residues = residue_info[:, 0]
        if len(np.unique(residue_info[:, 1])) < len(np.unique(residues)):
            residues = residue_info[:, 1]

        idx_to_pair = {
            pair_idx: pair for pair_idx, pair in enumerate(sorted_pairs_info)
        }
        pairs_idxs = [
            list(np.arange(len(sorted_pairs_info))[residues == i])
            for i in np.unique(residues)
        ]
        if not pairs_idxs:
            pairs_idxs = [[]]
        return pairs_idxs, idx_to_pair

    def get_systems(self, frame: Frame):
        if not self.update:
            if len(self.systems) > 1:
                return True
            return False

        pairs_info = self.get_pairs_info(frame, self, self.ST)
        pairs_idxs, idx_to_pair = self.get_pairs_by_site(pairs_info, self)
        nsites = len(pairs_idxs)
        sites = []
        shift = 0
        for nsite, pair_idxs in enumerate(pairs_idxs):
            if nsite > 0:
                shift += len(sites)
            sites.append(
                Site(
                    pair=None,
                    xhy=None,
                    rxn_num=None,
                    index=len(sites),
                    site=nsite,
                    parent=-1,
                )
            )
            for pair_idx in pair_idxs:
                pair = idx_to_pair[pair_idx]
                index = len(sites)
                id_h, id_y = pair.ids
                id_x = utils.get_X(id_h, self.bonds)
                sites.append(
                    Site(
                        pair=pair.ids,
                        xhy=(id_x, id_h, id_y),
                        rxn_num=pair.reaction_type,
                        index=index,
                        # dist=pair.distance,
                        total_dist=pair.distance,
                        # dist_xyz=pair.distance_xyz,
                        atoms=self.atoms,
                        bonds=self.bonds,
                        residues=self.residues,
                        qs=self.qs,
                        types=self.types,
                        shell=1,
                        parent=shift,
                        site=nsite,
                    )
                )

        if self.SI.shells == 1:
            return self.sites_to_systems(sites, frame)

        if nsites > 1:
            raise RuntimeError(
                "No verified implementation of multi-site with multi-shell."
            )
        start = 0
        nsite = 0
        for shell in range(2, self.SI.shells + 1):
            stop = len(sites)
            for site in sites[start:]:
                if site.xhy is None:
                    continue
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
                    new_types[site.atoms[eyed].idx] = new_type = type_changes[
                        site.atoms[eyed].type
                    ]
                    new_qs[site.atoms[eyed].idx] = new_q = self.SI.type_charges[
                        new_type
                    ]
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
                specific_topologies = [
                    SpecificTopology(
                        X_idxs=self._remove_site(ST[reaction.num].X_idxs, dont_idxs),
                        Y_idxs=self._remove_site(ST[reaction.num].Y_idxs, dont_idxs),
                        H_idxs=self._remove_site(ST[reaction.num].H_idxs, dont_idxs),
                        X_ids=np.array([]),
                        Y_ids=np.array([]),
                        H_ids=np.array([]),
                    )
                    for reaction in self.reactions
                ]
                topology = PseudoSite(
                    atoms=new_atoms, residues=new_residues, bonds=new_bonds
                )
                pairs_info = self.get_pairs_info(frame, topology, specific_topologies)
                pairs_idxs, idx_to_pair = self.get_pairs_by_site(pairs_info, topology)
                if len(pairs_idxs) > 1:
                    raise RuntimeError("Undefined behaviour.")
                pair_idxs = pairs_idxs[0]
                for pair_idx in pair_idxs:
                    pair = idx_to_pair[pair_idx]
                    index = len(sites)
                    id_h, id_y = pair.ids
                    id_x = utils.get_X(id_h, new_bonds)
                    sites.append(
                        Site(
                            pair=pair.ids,
                            xhy=(id_x, id_h, id_y),
                            rxn_num=pair.reaction_type,
                            index=index,
                            # dist=pair.distance,
                            total_dist=pair.distance + sites[site.index].total_dist,
                            # dist_xyz=pair.distance_xyz,
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

        return self.sites_to_systems(sites, frame)

    def sites_to_systems(self, sites: list, frame: Frame) -> bool:
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
        if self.num_sites == 1:
            for system in systems:
                system.parents = set(system._parents)
        else:
            system_parents = self.find_system_parents(
                [system._parents for system in systems]
            )
            for index, site_parents in system_parents.items():
                systems[index].site_parents = site_parents
                systems[index].get_parents()

        self.sites = sites
        self.systems = systems
        self.num_systems = len(self.systems)
        self.systems_idxs = systems_idxs
        self.grouped_site_idxs = grouped_site_idxs
        if len(self.systems) > 1:
            return True
        return False

    @staticmethod
    def find_system_parents(site_parents: list) -> Dict[int, list]:
        pairs = {}
        for idx1, pair1 in enumerate(site_parents):
            pairs[idx1] = {}
            for idx2, pair2 in enumerate(site_parents[:idx1]):
                diff_count = sum(1 for x, y in zip(pair1, pair2) if x != y)
                if diff_count == 1:
                    differing_index = next(
                        i for i, (a, b) in enumerate(zip(pair1, pair2)) if a != b
                    )
                    pairs[idx1][idx2] = differing_index
        return pairs

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
            if self.SI.reactions[site.rxn_num].toggle_bond:
                    # Validate that id_h and id_y are different
                    if id_h == id_y:
                        raise ValueError(f"Cannot toggle bond: id_h and id_y are the same ({id_h})")
                
                    # Ensure both id_h and id_y are in _new_bonds
                    if id_h not in _new_bonds:
                        _new_bonds[id_h] = list(site.bonds.get(id_h, []))
                    if id_y not in _new_bonds:
                        _new_bonds[id_y] = list(site.bonds.get(id_y, []))
                
                    # Check bond consistency before proceeding
                    h_has_y = id_y in _new_bonds.get(id_h, [])
                    y_has_h = id_h in _new_bonds.get(id_y, [])
                
                    if h_has_y != y_has_h:
                        raise RuntimeError(
                            f"Inconsistent bond state: id_h={id_h} "
                            f"{'has' if h_has_y else 'does not have'} bond to id_y={id_y}, "
                            f"but id_y {'has' if y_has_h else 'does not have'} bond to id_h. "
                            f"Bond graph is corrupted."
                        )
                
                    # Toggle the bond
                    if h_has_y:  # Bond exists - break it
                        try:
                            _new_bonds[id_h].remove(id_y)
                            _new_bonds[id_y].remove(id_h)
                        except ValueError as e:
                            raise RuntimeError(
                                f"Failed to remove bond between {id_h} and {id_y}: {e}"
                            )
                    else:  # Bond doesn't exist - create it
                        _new_bonds.setdefault(id_h, []).append(id_y)
                        _new_bonds.setdefault(id_y, []).append(id_h)
            elif id_h in _new_bonds:
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
            if any(k in combined_atom_changes for k in atom_diffs):
                return None
            combined_atom_changes.update(atom_diffs)
            bond_diffs = check_sites[site_idx]["b"]
            combined_bond_changes.update(bond_diffs)
        return combined_atom_changes, combined_bond_changes

    @staticmethod
    def _remove_site(idxs, rxn_pair):
        return idxs[np.isin(idxs, rxn_pair.flatten(), invert=True)]

    def _grab_system(self, system_idx: int) -> System:
        try:
            return self.systems[system_idx]
        except IndexError:
            return self.empty_system()

    def grab_system(self, system_idx: int) -> System:
        system = self._grab_system(system_idx)
        self.current_system = system
        return system

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

    def set_traj_frame(self, frame: TrajectoryFrame) -> None:
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

    def change_topology_to_system(self, lmp: lammps, system: System) -> None:
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
                            dont.add(f"{id_0}-{id_1}")
                        except KeyError:
                            continue

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
                p1 = f"{id_0}-{id_1}-{id_2}-{id_3}"
                p2 = f"{id_3}-{id_2}-{id_1}-{id_0}"
                if p1 in dont or p2 in dont:
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
                except KeyError:
                    continue
                dont.add(p1)
                dont.add(p2)
                cmd_list.append(
                    f"create_bonds single/dihedral {proper_type} {ids[0]} {ids[1]} {ids[2]} {ids[3]} special no"
                )

        # impropers
        if self.SI.improper_types and impropers:
            dont = set()
            for improper in impropers:
                for id_0, id_1, id_2, id_3 in permutations(improper):
                    nt_0 = types[id_0] 
                    nt_1 = types[id_1]
                    nt_2 = types[id_2]
                    nt_3 = types[id_3]
                    idstr = f"{id_0}-{id_1}-{id_2}-{id_3}"
                    improper_string = f"{nt_0}-{nt_1}-{nt_2}-{nt_3}"
                    if improper_string in dont:
                        continue
                    improper_type = self.SI.improper_types.get(improper_string, None)
                    if improper_type is None:
                        continue
                    dont.add(idstr)
                    cmd_list.append(
                        f"create_bonds single/improper {improper_type} {id_0} {id_1} {id_2} {id_3} special no"
                    )
        return cmd_list

    @staticmethod
    def _get_ids_from_type(atom_type: int, atoms: Dict[int, Atom]):
        ids = np.sort(
            np.array([id for id, atom in atoms.items() if atom.type == atom_type])
        )
        idxs = np.array([atoms[id].idx for id in ids])
        return ids, idxs

    @staticmethod
    def _get_angles_dihedrals(bonds: Dict[int, list]) -> Tuple[list, list, list]:
        angles = set()
        propers = set()
        impropers = set()
        for a, bs in bonds.items():
            for b1, b2 in combinations(bs, 2):
                # angles
                angles.add((b1, a, b2))
                # propers
                # check if b1 or b2 are in any other bonds
                check_a = np.isin(bonds[b1], [a], invert=True)
                if check_a.any():
                    for id in np.array(bonds[b1])[check_a]:
                        propers.add((id, b1, a, b2))
                check_b = np.isin(bonds[b2], [a], invert=True)
                if check_b.any():
                    for id in np.array(bonds[b2])[check_b]:
                        propers.add((b1, a, b2, id))
                # impropers
                check_c = np.isin(bonds[a], [b1, b2], invert=True)
                if check_c.any():
                    for id in np.array(bonds[a])[check_c]:
                        s = sorted((b1, b2, id))
                        impropers.add((a, *s))

        return list(angles), list(propers), list(impropers)
