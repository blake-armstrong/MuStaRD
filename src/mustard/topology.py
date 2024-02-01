from typing import NamedTuple, Union, Tuple
from itertools import combinations, product

import time
from . import utils
import numpy as np
import warnings
from copy import copy, deepcopy
from collections import defaultdict


class Topology:
    def __init__(self, lmp_obj, SystemInfo):
        self.SI = SystemInfo
        self.lmp = lmp_obj
        self.build_topology()
        self.systems: list
        self.num_systems: int
        self.num_sites: int
        self.current_system = self.empty_system()
        self.rxn_pair_info = dict()

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
            atom_info[ID] = Topology.Atom(
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
        return Topology.SpecificTopology(
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

    def _get_snapshot(self, frame, step):
        return Snapshot(
            frame=frame,
            ids=self.ids,
            types=self.types,
            atoms=self.atoms,
            residues=self.residues,
            bonds=self.bonds,
            qs=self.qs,
            step=step,
        )

    def get_pairs(
        self,
        positions,
        xyz_pbc,
        cutoffs,
        X_idxs,
        H_idxs,
        Y_idxs,
        residues=None,
        atoms=None,
    ):
        if residues is None:
            residues = self.residues
        if atoms is None:
            atoms = self.atoms
        dist_cut, ang_cut = (
            cutoffs["distance"],
            cutoffs["angle"],
        )
        if H_idxs.size == 0 or Y_idxs.size == 0:
            return None
        H_pos = positions[H_idxs]
        Y_pos = positions[Y_idxs]
        dists = utils.get_distances(H_pos, Y_pos, xyz_pbc)
        dists_bool = dists < dist_cut
        if not dists_bool.any():
            return None
        pair_dists = dists[dists_bool.nonzero()[0], dists_bool.nonzero()[1]]
        sort = np.argsort(pair_dists)
        # grab indexes of pairs that meet dist cutoff
        H_ready_idx = H_idxs[(dists_bool).nonzero()[0]]
        Y_ready_idx = Y_idxs[(dists_bool).nonzero()[1]]
        rxn_pairs = np.array([self.ids[H_ready_idx], self.ids[Y_ready_idx]]).T
        if ang_cut is None or not X_idxs.any():
            return rxn_pairs[sort], pair_dists[sort], [None] * len(rxn_pairs)
        X_ready_idx = [
            self.atoms[id].idx
            for idx in H_ready_idx
            for id in residues[atoms[self.ids[idx]].molecule]
            if self.atoms[id].idx in X_idxs
        ]
        # check angles work as well.
        # positions of atoms in angle
        H_pos = positions[H_ready_idx]
        X_pos = positions[X_ready_idx]
        Y_pos = positions[Y_ready_idx]
        # get angles..
        angles = utils.get_angles(H_pos, X_pos, Y_pos, xyz_pbc)
        angle_bool = angles < ang_cut
        if not angle_bool.any():
            return None
        hxy_angles = angles[angle_bool]
        rxn_pairs = rxn_pairs[angle_bool]
        rxn_pairs = rxn_pairs[np.argsort(rxn_pairs[:, 0])]
        return rxn_pairs[sort], pair_dists[sort], hxy_angles[sort]

    def _get_pairs_idxs(self, pos, xyz_pbc, rxn_infos=None):
        rxn_pairs = []
        rxn_nums = []
        pair_dists = []
        hxy_angles = []
        pairs_idxs = []
        for rxn_num, rxn in enumerate(self.SI.reactions):
            if rxn_infos is None:
                rxn_info = self.get_pairs(
                    pos,
                    xyz_pbc,
                    rxn.cutoffs,
                    self.ST[rxn_num].X_idxs,
                    self.ST[rxn_num].H_idxs,
                    self.ST[rxn_num].Y_idxs,
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
            return pairs_idxs, rxn_pairs
        pair_dists = np.concatenate(pair_dists, axis=0)
        sort = np.argsort(pair_dists)
        rxn_pairs = np.concatenate(rxn_pairs, axis=0)[sort]
        rxn_nums = np.concatenate(rxn_nums, axis=0)[sort]
        hxy_angles = np.concatenate(hxy_angles, axis=0)[sort]
        pair_dists = pair_dists[sort]
        rxn_molecules = np.array([self.atoms[pair[0]].molecule for pair in rxn_pairs])
        for n, pair in enumerate(rxn_pairs):
            self.rxn_pair_info[tuple(pair)]["num"] = rxn_nums[n]
            self.rxn_pair_info[tuple(pair)]["dist"] = pair_dists[n]
            self.rxn_pair_info[tuple(pair)]["angle"] = hxy_angles[n]
        pairs_idxs = [
            [None] + list(np.arange(len(rxn_pairs))[rxn_molecules == i])
            for i in np.unique(rxn_molecules)
        ]
        return pairs_idxs, rxn_pairs

    def get_systems(self, pos, xyz_pbc):
        self.rxn_pair_info = defaultdict(dict)
        pairs_idxs, rxn_pairs = self._get_pairs_idxs(pos, xyz_pbc)
        for pair in rxn_pairs:
            self.rxn_pair_info[tuple(pair)]["atoms"] = self.atoms
            self.rxn_pair_info[tuple(pair)]["bonds"] = self.bonds
            self.rxn_pair_info[tuple(pair)]["residues"] = self.residues
            self.rxn_pair_info[tuple(pair)]["parent"] = 0
            self.rxn_pair_info[tuple(pair)]["shell"] = 1
            self.rxn_pair_info[tuple(pair)]["tot_dists"] = [
                self.rxn_pair_info[tuple(pair)]["dist"]
            ]
        if self.SI.shells > 1:
            if len(pairs_idxs) > 1:
                raise RuntimeError(
                    "still need to implement multi-site multi-shell reactions"
                )
            pairs_idxs_copy = deepcopy(pairs_idxs)
            shells = utils.nested_defaultdict()
            for nsite, site in enumerate(pairs_idxs):
                site = pairs_idxs[nsite]
                for shell in range(2, self.SI.shells + 1):
                    dont = np.array(pairs_idxs_copy).flatten()
                    dont = self.id_to_idx(
                        rxn_pairs[dont[dont != None].astype(int)].flatten()
                    )
                    end_of_site = len(pairs_idxs_copy[nsite])
                    for state in site:
                        if state is None:
                            continue
                        shells[nsite][1][state]["atoms"] = self.atoms
                        shells[nsite][1][state]["bonds"] = self.bonds
                        shells[nsite][1][state]["residues"] = self.residues
                        atoms = shells[nsite][shell - 1][state]["atoms"]
                        bonds = shells[nsite][shell - 1][state]["bonds"]
                        residues = shells[nsite][shell - 1][state]["residues"]
                        pair = rxn_pairs[state]
                        id_h, id_y = pair
                        try:
                            id_x = bonds[id_h][
                                0
                            ]  # NOTE assumes transferring atom is only bonded to one other atom
                        except KeyError:
                            id_x = None
                        # update topology to reflect new reaction
                        rxn_num = self.rxn_pair_info[tuple(pair)]["num"]
                        rxn = self.SI.reactions[rxn_num]
                        hxs = residues[atoms[id_h].molecule]
                        ys = residues[atoms[id_y].molecule]
                        reactive_ids1 = [
                            eyed for eyed in hxs + ys if eyed not in (id_x, id_h, id_y)
                        ]
                        new_types = {
                            eyed: rxn.type_changes1[atoms[eyed].type]
                            for eyed in reactive_ids1
                        }
                        for eyed in (id_x, id_h, id_y):
                            if eyed is None:
                                continue
                            new_types[eyed] = rxn.type_changes0[atoms[eyed].type]

                        new_atoms = copy(atoms)
                        for eyed in hxs + ys:
                            typ = new_types[eyed]
                            molecule = self.atoms[eyed].molecule
                            if eyed == id_h:
                                molecule = self.atoms[id_y].molecule
                            new_atoms[eyed] = Topology.Atom(
                                idx=self.atoms[eyed].idx,
                                type=typ,
                                charge=self.SI.type_charges[typ],
                                molecule=molecule,
                                mass=self.atoms[eyed].mass,
                                image=self.atoms[eyed].image,
                            )

                        new_residues = defaultdict(list)
                        for ID, atom in new_atoms.items():
                            new_residues[atom.molecule].append(ID)

                        ST = self.generate_specific_topology(new_atoms)
                        rxn_infos = []
                        for rxn_num, rxn in enumerate(self.SI.reactions):
                            rxn_info = self.get_pairs(
                                pos,
                                xyz_pbc,
                                rxn.cutoffs,
                                self._remove_site(ST[rxn_num].X_idxs, dont),
                                self._remove_site(ST[rxn_num].H_idxs, dont),
                                self._remove_site(ST[rxn_num].Y_idxs, dont),
                                residues=new_residues,
                                atoms=new_atoms,
                            )
                            rxn_infos.append(rxn_info)
                        new_pairs_idxs, new_rxn_pairs = self._get_pairs_idxs(
                            pos, xyz_pbc, rxn_infos=rxn_infos
                        )
                        shift = len(pairs_idxs_copy[nsite]) - 1
                        new_pairs_idxs = [
                            element + shift
                            for row in new_pairs_idxs
                            for element in row
                            if element is not None
                        ]
                        pairs_idxs_copy[nsite].extend(new_pairs_idxs)
                        rxn_pairs = np.concatenate([rxn_pairs, new_rxn_pairs], axis=0)
                        _new_bonds_dict = {
                            eyed: list(bonds.get(eyed, []))
                            for eyed in hxs + ys
                            if bonds.get(eyed)
                        }
                        if id_h in _new_bonds_dict:
                            _new_bonds_dict[_new_bonds_dict[id_h][0]].remove(id_h)
                            _new_bonds_dict[id_h][0] = id_y
                            _new_bonds_dict.setdefault(id_y, []).append(id_h)
                        new_bonds_dict = copy(bonds)
                        new_bonds_dict.update(
                            {k: v for k, v in _new_bonds_dict.items() if v}
                        )
                        shells[nsite][shell][state]["atoms"] = new_atoms
                        shells[nsite][shell][state]["residues"] = new_residues
                        shells[nsite][shell][state]["bonds"] = new_bonds_dict
                        for idx in new_pairs_idxs:
                            shells[nsite][shell][idx]["atoms"] = shells[nsite][shell][
                                state
                            ]["atoms"]
                            shells[nsite][shell][idx]["residues"] = shells[nsite][
                                shell
                            ][state]["residues"]
                            shells[nsite][shell][idx]["bonds"] = shells[nsite][shell][
                                state
                            ]["bonds"]
                            pair = rxn_pairs[idx]
                            self.rxn_pair_info[tuple(pair)]["atoms"] = new_atoms
                            self.rxn_pair_info[tuple(pair)]["bonds"] = new_bonds_dict
                            self.rxn_pair_info[tuple(pair)]["residues"] = new_residues
                            self.rxn_pair_info[tuple(pair)]["parent"] = state + 1
                            self.rxn_pair_info[tuple(pair)]["shell"] = shell
                            self.rxn_pair_info[tuple(pair)][
                                "tot_dists"
                            ] = self.rxn_pair_info[tuple(rxn_pairs[state])][
                                "tot_dists"
                            ] + [
                                self.rxn_pair_info[tuple(pair)]["dist"]
                            ]

                    site = pairs_idxs_copy[nsite][end_of_site:]
            pairs_idxs = pairs_idxs_copy

        systems_idxs = np.array(list(product(*pairs_idxs)))

        systems = []
        for n, system_idxs in enumerate(systems_idxs):
            pairs = np.array(
                [
                    rxn_pairs[system_idx] if system_idx is not None else None
                    for system_idx in system_idxs
                ]
            )
            sites = []
            for m, pair in enumerate(pairs):
                if pair is None:
                    sites.append(None)
                    continue
                dists = self.rxn_pair_info[tuple(pair)]["tot_dists"]
                atoms = self.rxn_pair_info[tuple(pair)]["atoms"]
                bonds = self.rxn_pair_info[tuple(pair)]["bonds"]
                residues = self.rxn_pair_info[tuple(pair)]["residues"]
                parent = self.rxn_pair_info[tuple(pair)]["parent"]
                shell = self.rxn_pair_info[tuple(pair)]["shell"]
                rxn_num = self.rxn_pair_info[tuple(pair)]["num"]
                site = Site(
                    index=m,
                    rxn_num=rxn_num,
                    pair=pair,
                    dists=dists,
                    atoms=atoms,
                    bonds=bonds,
                    residues=residues,
                    parent=parent,
                    shell=shell,
                )
                sites.append(site)
            system = System(
                index=n,
                sites=tuple(sites),
            )
            systems.append(system)

        self.num_systems = len(systems_idxs)
        self.num_sites = len(pairs_idxs)
        self.systems = systems
        if len(self.systems) > 1:
            return True
        return False

    @staticmethod
    def _remove_site(idxs, rxn_pair):
        return idxs[np.isin(idxs, rxn_pair.flatten(), invert=True)]

    def grab_system(self, system_idx: int):
        try:
            return self.systems[system_idx]
        except IndexError:
            return self.empty_system()

    def empty_system(self):
        return System(sites=(Site(pair=np.array([]), rxn_num=None),))

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
        if system.index == 0:
            return
        create_bonds = []
        set_type_charge = []
        for nsite, site in enumerate(system.sites):
            parent_systems = []
            shell = site.shells
            parent = site.parents
            while shell > 0:
                parent_system = self.systems[parent]
                parent_systems.append(parent_system)
                if parent_system.index != 0:
                    parent = parent_system.sites[nsite].parents
                shell -= 1
            parent_systems.reverse()
            change_topology_systems = parent_systems + [system]
            for _system in change_topology_systems[1:]:
                site = _system.sites[nsite]

                # for rxn_pair, rxn_num in zip(system.pairs, rxn_nums):
                # bonds = site.bonds
                # atoms = site.atoms
                # residues = site.residues
                rxn_pair = site.pair
                rxn_num = site.rxn_num
                id_h, id_y = rxn_pair
                try:
                    id_x = site.bonds[id_h][
                        # id_x = self.bonds[id_h][
                        0
                    ]  # NOTE assumes transferring atom is only bonded to one other atom
                except KeyError:
                    id_x = None
                # create groups
                hxy_group_str = "group HXY id "
                # hxs = self.residues[self.atoms[id_h].molecule]
                # ys = self.residues[self.atoms[id_y].molecule]
                hxs = site.residues[site.atoms[id_h].molecule]
                ys = site.residues[site.atoms[id_y].molecule]
                for eyed in hxs + ys:
                    hxy_group_str += f"{eyed} "

                # removes all bonds, angles, dihedrals and impropers involving these ids
                # lmp.commands_list([hxy_group_str, "delete_bonds HXY multi remove"])
                create_bonds += [hxy_group_str, "delete_bonds HXY multi remove"]
                ids_for_change = hxs + ys
                new_bonds_dict = {
                    eyed: list(site.bonds.get(eyed, []))
                    for eyed in ids_for_change
                    if site.bonds.get(eyed)
                }
                if id_h in new_bonds_dict:
                    new_bonds_dict[new_bonds_dict[id_h][0]].remove(id_h)
                    new_bonds_dict[id_h][0] = id_y
                    new_bonds_dict.setdefault(id_y, []).append(id_h)
                nbd = {k: v for k, v in new_bonds_dict.items() if v}

                # changes types and charges
                # its possible to just edit the array returned from extract_atoms
                # or call scatter_atoms, probably faster than looping through set
                new_types = {}
                for eyed in (id_h, id_y, id_x):
                    if eyed is None:
                        continue
                    new_types[eyed] = self.SI.reactions[rxn_num].type_changes0[
                        site.atoms[eyed].type
                    ]
                    set_type_charge.append(f"set atom {eyed} type {new_types[eyed]}")
                    set_type_charge.append(
                        f"set atom {eyed} charge {self.SI.type_charges[new_types[eyed]]}"
                    )
                    ids_for_change.remove(eyed)

                for eyed in ids_for_change:
                    new_types[eyed] = self.SI.reactions[rxn_num].type_changes1[
                        site.atoms[eyed].type
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
                create_bonds += ["group HXY delete"]

        create_bonds[-2] = create_bonds[-2].replace("no", "yes")
        lmp.commands_list(set_type_charge + create_bonds)
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

    def get_new_imgs(self, h, y, frame, site):
        yids = site.residues[site.atoms[y].molecule]
        yidxs = [site.atoms[ID].idx for ID in yids]
        hpos = frame.pos[site.atoms[h].idx]
        himg = frame.images[site.atoms[h].idx]
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


class Snapshot:
    def __init__(
        self,
        frame: Topology.Frame,
        ids: np.ndarray,
        types: np.ndarray,
        atoms: dict,
        residues: dict,
        bonds: dict,
        qs: np.ndarray,
        step: int,
    ):
        self.frame = frame
        self.ids = ids
        self.types = types
        self.atoms = atoms
        self.residues = residues
        self.bonds = bonds
        self.qs = qs
        self.step = step


class Site:
    def __init__(
        self,
        pair: np.ndarray,
        rxn_num: Union[int, None],
        index: int = 0,
        dists: list = [],
        atoms: dict = {},
        bonds: dict = {},
        residues: dict = {},
        shell: int = 1,
        parent: int = 0,
    ):
        self.index = index
        self.rxn_num = rxn_num
        self.pair = pair
        self.dists = dists
        self.tot_dist = np.round(np.sum(self.dists), 2)
        self.atoms = atoms
        self.bonds = bonds
        self.residues = residues
        self.shells = shell
        self.parents = parent

    def __repr__(self):
        return (
            f"Index: {self.index}, "
            f"Pair: {self.pair}, "
            f"Reaction no. {self.rxn_num}, "
            f"Total distance: {self.tot_dist}, "
            f"Shell: {self.shells}, "
            f"Parent: {self.parents}. "
        )


class System:
    def __init__(
        self,
        sites: Tuple[Union[Site, None], ...],
        index=0,
    ):
        self.index = index
        self.sites = sites
        self.pairs = np.array([site.pair for site in self.sites if site is not None])

    def __repr__(self):
        return f"System: {self.index}, " f"Sites: {self.sites}."
