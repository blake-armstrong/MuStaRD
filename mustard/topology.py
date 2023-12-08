from typing import NamedTuple
from itertools import combinations, product

# from .utils import gather_atoms, extract_box
from . import utils
import numpy as np
import warnings


class Topology:
    def __init__(self, lmp_obj, SystemInfo):
        self.SI = SystemInfo
        self.lmp = lmp_obj
        self.build_topology()
        self.rxn_pairs: list
        self.systems_idxs: np.ndarray
        self.pair_dists: np.ndarray
        self.hxy_angles: np.ndarray
        self.pairs_idxs: list
        self.num_systems: int
        self.current_system = np.array(tuple())

    def build_topology(self):
        (
            self.ids,
            self.types,
            self.atoms,
            self.residues,
            self.bonds,
        ) = self.generate_generic_topology(self.lmp)
        self.ST = self.generate_specific_topology()
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
            atom_info[ID] = Topology.Atom(
                idx=idx,
                type=types[idx],
                charge=qs[idx],
                molecule=mols[idx],
                mass=masses[idx],
                image=imgs[idx],
            )

        # dict converts residue ID to list of IDs
        mol_to_ids = {}
        for mol in set(mols):
            mol_to_ids[mol] = []
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

        return ids, types, atom_info, mol_to_ids, bonds_dict

    def _generate_specific_topology(self, X, H, Y):
        H_ids, H_idxs = self._get_ids_from_type(H, self.atoms)
        Y_ids, Y_idxs = self._get_ids_from_type(Y, self.atoms)
        X_ids, X_idxs = self._get_ids_from_type(X, self.atoms)
        return Topology.SpecificTopology(
            H_ids=H_ids,
            H_idxs=H_idxs,
            X_ids=X_ids,
            X_idxs=X_idxs,
            Y_ids=Y_ids,
            Y_idxs=Y_idxs,
        )

    def generate_specific_topology(self):
        return [
            self._generate_specific_topology(rxn.X, rxn.H, rxn.Y)
            for rxn in self.SI.reactions
        ]

    def _get_snapshot(self, frame, step):
        return Topology.Snapshot(
            frame=frame,
            ids=self.ids,
            types=self.types,
            atoms=self.atoms,
            residues=self.residues,
            bonds=self.bonds,
            step=step,
        )

    def rxn_pairs_to_systems(self, rxn_pairs, rxn_nums, pair_dists, hxy_angles):
        rxn_molecules = np.array([self.atoms[pair[0]].molecule for pair in rxn_pairs])
        self.rxn_nums_dict = {
            tuple(pair): rxn_nums[n] for n, pair in enumerate(rxn_pairs)
        }
        self.rxn_taper_info = {
            tuple(pair): (pair_dists[n], hxy_angles[n])
            for n, pair in enumerate(rxn_pairs)
        }
        pairs_idxs = [
            [None] + list(np.arange(len(rxn_pairs))[rxn_molecules == i])
            for i in np.unique(rxn_molecules)
        ]
        systems_idxs = np.array(list(product(*pairs_idxs)))

        return systems_idxs, pairs_idxs

    def get_pairs(self, pos, xyz_pbc):
        rxn_pairs = []
        rxn_nums = []
        pair_dists = []
        hxy_angles = []
        systems_idxs = []
        pairs_idxs = []
        for rxn_num, rxn in enumerate(self.SI.reactions):
            rxn_info = utils.get_pairs(
                pos,
                xyz_pbc,
                rxn.cutoffs,
                rxn.X,
                self.ST[rxn_num].H_idxs,
                self.ST[rxn_num].Y_idxs,
                self,
            )
            if rxn_info is not None:
                rxn_pairs.append(rxn_info[0])
                rxn_nums.append([rxn_num] * len(rxn_info[0]))
                pair_dists.append(rxn_info[1])
                hxy_angles.append(rxn_info[2])
        if rxn_pairs:
            pair_dists = np.concatenate(pair_dists, axis=0)
            sort = np.argsort(pair_dists)
            rxn_pairs = np.concatenate(rxn_pairs, axis=0)[sort]
            rxn_nums = np.concatenate(rxn_nums, axis=0)[sort]
            hxy_angles = np.concatenate(hxy_angles, axis=0)[sort]
            pair_dists = pair_dists[sort]
            systems_idxs, pairs_idxs = self.rxn_pairs_to_systems(
                rxn_pairs,
                rxn_nums,
                pair_dists,
                hxy_angles,
            )
        self.rxn_pairs = list(rxn_pairs)
        self.systems_idxs = np.array(systems_idxs)
        self.pair_dists = np.array(pair_dists)
        self.hxy_angles = np.array(hxy_angles)
        self.pairs_idxs = pairs_idxs
        print(self.pairs_idxs)
        self.num_systems = len(systems_idxs)

    def grab_system(self, system_idx: np.intp):
        if system_idx >= len(self.systems_idxs):
            return np.array(tuple())
        system_idxs = self.systems_idxs[system_idx]
        return np.array(
            tuple(
                self.rxn_pairs[pair_idxs]
                for pair_idxs in system_idxs
                if pair_idxs is not None
            )
        )

    def set_lmp(self, lmp_obj):
        self.lmp = lmp_obj

    def _get_new_imgs(self, h, y, frame):
        yids = self.residues[self.atoms[y].molecule]
        yidxs = [self.atoms[ID].idx for ID in yids]
        hpos = frame.pos[self.atoms[h].idx]
        himg = frame.images[self.atoms[h].idx]
        ypos = frame.pos[yidxs]
        disp = ypos - hpos
        disp -= frame.xyz_pbc * (disp / frame.xyz_pbc).round()
        abcabc, _ = utils.extract_box(frame.box_data)
        uhpos = himg * abcabc[:3] + hpos
        uypos = uhpos + disp
        new_imgs = np.floor(uypos / abcabc[:3]).astype(int)
        return new_imgs, yids

    # def set_bonds(self, bonds):

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
        full[-1] = full[-1].replace("no", "yes")
        [self.lmp.command(cmd) for cmd in full]
        self.lmp.command("reset_atoms mol all single yes")

    def reset_lmp_topology(self):
        # system = self.current_system
        # if not system.any():
        #    return
        # rxn_nums = [self.rxn_nums_dict[tuple(pair)] for pair in system]
        # total_ids = []
        # for rxn_pair, _ in zip(system, rxn_nums):
        #    id_h, id_y = rxn_pair
        #    # create groups
        #    hxy_group_str = "group HXY id "
        #    hxs = self.residues[self.atoms[id_h].molecule]
        #    ys = self.residues[self.atoms[id_y].molecule]
        #    for eyed in hxs + ys:
        #        hxy_group_str += f"{eyed} "
        #        total_ids.append(eyed)
        #    # removes all bonds, angles, dihedrals and impropers involving these ids
        #    self.lmp.commands_list(
        #        [hxy_group_str, "delete_bonds HXY multi remove", "group HXY delete"]
        #    )
        # total_ids = np.array(total_ids)
        # self._reset_lmp_topology(total_ids)
        # self.current_systen = np.array(tuple())
        self.lmp.command("delete_bonds all multi remove")
        self._reset_lmp_topology(self.ids)

    def change_topology_to_system(self, lmp, system, frame):
        self.current_system = system
        create_bonds = []
        set_type_charge = []
        rxn_nums = [self.rxn_nums_dict[tuple(pair)] for pair in system]
        for rxn_pair, rxn_num in zip(system, rxn_nums):
            id_h, id_y = rxn_pair
            try:
                id_x = self.bonds[id_h][
                    0
                ]  # NOTE assumes transferring atom is only bonded to one other atom
            except KeyError:
                id_x = None
            # create groups
            hxy_group_str = "group HXY id "
            hxs = self.residues[self.atoms[id_h].molecule]
            ys = self.residues[self.atoms[id_y].molecule]
            for eyed in hxs + ys:
                hxy_group_str += f"{eyed} "

            # removes all bonds, angles, dihedrals and impropers involving these ids
            lmp.commands_list([hxy_group_str, "delete_bonds HXY multi remove"])
            ids_for_change = hxs + ys
            new_bonds_dict = {
                eyed: list(self.bonds.get(eyed, []))
                for eyed in ids_for_change
                if self.bonds.get(eyed)
            }
            if id_h in new_bonds_dict:
                new_bonds_dict[new_bonds_dict[id_h][0]].remove(id_h)
                new_bonds_dict[id_h][0] = id_y
                new_bonds_dict.setdefault(id_y, []).append(id_h)
            nbd = {k: v for k, v in new_bonds_dict.items() if v}
            #        angles, propers, impropers = self._get_angles_dihedrals(nbd)

            # changes types and charges
            # its possible to just edit the array returned from extract_atoms
            # or call scatter_atoms, probably faster than looping through set
            new_types = {}
            for eyed in (id_h, id_y, id_x):
                if eyed is None:
                    continue
                new_types[eyed] = self.SI.reactions[rxn_num].type_changes0[
                    self.atoms[eyed].type
                ]
                set_type_charge.append(f"set atom {eyed} type {new_types[eyed]}")
                set_type_charge.append(
                    f"set atom {eyed} charge {self.SI.type_charges[new_types[eyed]]}"
                )
                ids_for_change.remove(eyed)

            for eyed in ids_for_change:
                new_types[eyed] = self.SI.reactions[rxn_num].type_changes1[
                    self.atoms[eyed].type
                ]
                set_type_charge.append(f"set atom {eyed} type {new_types[eyed]}")
                set_type_charge.append(
                    f"set atom {eyed} charge {self.SI.type_charges[new_types[eyed]]}"
                )
            new_imgs, yids = self._get_new_imgs(id_h, id_y, frame)
            create_bonds += [
                f"set atom {ID} image {imgs[0]} {imgs[1]} {imgs[2]}"
                for ID, imgs in zip(yids, new_imgs)
            ]
            create_bonds += self.create_bonds(new_types, nbd)
        create_bonds[-1] = create_bonds[-1].replace("no", "yes")
        lmp.commands_list(set_type_charge + create_bonds + ["group HXY delete"])
        self.lmp.command("reset_atoms mol all single yes")

    def create_bonds(self, types, bonds):
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
            for id_0, id_1, id_2 in angles:
                new_type_0 = types[id_0]
                new_type_1 = types[id_1]
                new_type_2 = types[id_2]
                try:
                    angle_type = self.SI.angle_types[
                        f"{new_type_0}-{new_type_1}-{new_type_2}"
                    ]
                except KeyError:
                    continue
                cmd_list.append(
                    f"create_bonds single/angle {angle_type} {id_0} {id_1} {id_2} special no"
                )

        # propers
        if self.SI.proper_types and propers:
            for id_0, id_1, id_2, id_3 in propers:
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
                cmd_list.append(
                    f"create_bonds single/dihedral {proper_type} {ids[0]} {ids[1]} {ids[2]} {ids[3]} special no"
                )

        # impropers
        if self.SI.improper_types and impropers:
            for id_0, id_1, id_2, id_3 in impropers:
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
                cmd_list.append(
                    f"create_bonds single/improper {improper_type} {id_0} {id_1} {id_2} {id_3} special no"
                )
        return cmd_list

    def get_new_imgs(self, h, y, frame):
        yids = self.residues[self.atoms[y].molecule]
        yidxs = [self.atoms[ID].idx for ID in yids]
        hpos = frame.pos[self.atoms[h].idx]
        himg = frame.images[self.atoms[h].idx]
        ypos = frame.pos[yidxs]
        disp = ypos - hpos
        disp -= frame.xyz_pbc * (disp / frame.xyz_pbc).round()
        abcabc, _ = utils.extract_box(frame.box_data)
        uhpos = himg * abcabc[:3] + hpos
        uypos = uhpos + disp
        new_imgs = np.floor(uypos / abcabc[:3]).astype(int)
        return new_imgs, yids

    class Frame:
        def __init__(self, lmp, natoms: int, dim: int, scale_box=False):
            self._pos: np.ndarray = np.zeros((natoms, dim))
            self._images: np.ndarray = np.zeros((natoms, dim))
            self._box_data: tuple = (None, None, None, None, None, None)
            self._xyz_pbc: np.ndarray = np.zeros(3)
            self._vel: np.ndarray = np.zeros((natoms, dim))
            self._forces: np.ndarray = np.zeros((natoms, dim))
            self.scale_box = scale_box
            # self.__call__ = self._generate_call()
            self._generate_call()
            self._get_one(lmp)
            self._get_two(lmp)

        def __call__(self, lmp, pos=None, vel=None, forces=None, imgs=None):
            return self.call(self, lmp, pos=pos, vel=vel, forces=forces, imgs=imgs)

        def _generate_call(self):
            if self.scale_box:

                def call(self, lmp, pos=None, vel=None, forces=None, imgs=None):
                    self._get_one(lmp, pos=pos, vel=vel, forces=forces, imgs=imgs)
                    self._get_two(lmp)

                self.call = call
                return

            def call(self, lmp, pos=None, vel=None, forces=None, imgs=None):
                self._get_one(lmp, pos=pos, vel=vel, forces=forces, imgs=imgs)

            self.call = call

        def _get_one(self, lmp, pos=None, vel=None, forces=None, imgs=None):
            if pos is None:
                pos = utils.get_positions(lmp)
            self._pos = pos
            if vel is None:
                vel = utils.get_velocities(lmp)
            self._vel = vel
            if imgs is None:
                imgs = utils.get_images(lmp)
            self._images = imgs
            if forces is None:
                forces = utils.get_forces(lmp)
            self._forces = forces
            return self

        def _get_two(self, lmp):
            self._box_data = utils.get_box_data(lmp)
            self._xyz_pbc = np.array(self._box_data[1]) - np.array(self._box_data[0])

        def _update_vel(self, vel):
            self._vel = vel

        @property
        def pos(self):
            return self._pos

        @pos.setter
        def pos(self, pos):
            self.err()
            self.pos = pos

        @property
        def images(self):
            return self._images

        @images.setter
        def images(self, images):
            self.err()
            self.images = images

        @property
        def box_data(self):
            return self._box_data

        @box_data.setter
        def box_data(self, box_data):
            self.err()
            self.box_data = box_data

        @property
        def xyz_pbc(self):
            return self._xyz_pbc

        @xyz_pbc.setter
        def xyz_pbc(self, xyz_pbc):
            self.err()
            self.xyz_pbx = xyz_pbc

        @property
        def vel(self):
            return self._vel

        @vel.setter
        def vel(self, vel):
            self.err()
            self.vel = vel

        @property
        def forces(self):
            return self._forces

        @forces.setter
        def forces(self, forces):
            self.err()
            self.forces = forces

        @staticmethod
        def err():
            raise RuntimeWarning("Frame attributes should not be externally modified.")

    class Snapshot(NamedTuple):
        frame: "Topology.Frame"
        ids: np.ndarray
        types: np.ndarray
        atoms: dict
        residues: dict
        bonds: dict
        step: int

    class SpecificTopology(NamedTuple):
        H_ids: np.ndarray
        H_idxs: np.ndarray
        X_ids: np.ndarray
        X_idxs: np.ndarray
        Y_ids: np.ndarray
        Y_idxs: np.ndarray

    class Atom(NamedTuple):
        idx: int
        type: int
        charge: float
        molecule: int
        mass: float
        image: np.ndarray

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
