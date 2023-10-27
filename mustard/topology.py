from typing import NamedTuple
from itertools import combinations, product
from .utils import gather_atoms, extract_box
import numpy as np
import warnings


class Topology:
    def __init__(self, lmp_obj, SystemInfo):
        self.SI = SystemInfo
        self.lmp = lmp_obj
        self.build_topology()

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
        self.id_to_idx = np.vectorize(lambda x: self.atoms[x].idx)

    def generate_generic_topology(self, lmp):
        ids = np.array(gather_atoms(lmp, "id", 0, 1))
        types = np.array(gather_atoms(lmp, "type", 0, 1))
        qs = np.array(gather_atoms(lmp, "q", 1, 1))
        mols = np.array(gather_atoms(lmp, "molecule", 0, 1))
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
            )

        # dict converts residue ID to list of IDs
        mol_to_ids = {}
        for mol in set(mols):
            mol_to_ids[mol] = []
        for ID, atom in atom_info.items():
            mol_to_ids[atom.molecule].append(ID)

        with warnings.catch_warnings(record=True):
            bonds = lmp.numpy.gather_bonds().astype(int)[:, 1:]
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

    def set_lmp(self, lmp_obj):
        self.lmp = lmp_obj

    # def update_lmp_topology(self):
    #     self.lmp_obj.
    #     self.lmp_obj.command("read_data /tmp/mustard.data")
    #     # cmds_list = []
    #     # cmds_list.append("delete_bonds all multi remove")
    #     # for ID, atom in self.atoms.items():
    #     #     cmds_list.append(f"set atom {ID} type {atom.type}")
    #     #     cmds_list.append(f"set atom {ID} charge {atom.charge}")
    #     # types = {id: atom.type for id, atom in self.atoms.items()}
    #     # create_bonds = self.create_bonds(types, self.bonds)
    #     # cmds_list += create_bonds
    #     # cmds_list[-1] = cmds_list[-1].replace("no", "yes")
    #     # cmds_list += ["reset_atoms mol all single yes"] + ["run 0 post no"]
    #     # self.lmp.commands_list(cmds_list)

    def _get_new_imgs(self, h, y, frame):
        yids = self.residues[self.atoms[y].molecule]
        yidxs = [self.atoms[ID].idx for ID in yids]
        hpos = frame.pos[self.atoms[h].idx]
        himg = frame.images[self.atoms[h].idx]
        ypos = frame.pos[yidxs]
        disp = ypos - hpos
        disp -= frame.xyz_pbc * (disp / frame.xyz_pbc).round()
        abcabc, _ = extract_box(frame.box_data)
        uhpos = himg * abcabc[:3] + hpos
        uypos = uhpos + disp
        new_imgs = np.floor(uypos / abcabc[:3]).astype(int)
        return new_imgs, yids

    def change_topology_to_system(self, system, frame):
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
            self.lmp.commands_list([hxy_group_str, "delete_bonds HXY multi remove"])
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
        self.lmp.commands_list(set_type_charge + create_bonds + ["group HXY delete"])

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

    class Frame(NamedTuple):
        pos: np.ndarray
        images: np.ndarray
        box_data: tuple
        xyz_pbc: np.ndarray
        vel: np.ndarray

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
