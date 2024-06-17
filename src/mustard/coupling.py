import numpy as np

from typing import Dict, Tuple

from . import utils
from .topology import Snapshot


class BaseCoupling:

    def __init__(self, doi: str = str(None)):
        self.coupling_function = self.regular_coupling_function
        self.cutoff = 0.0
        self.taper = 0.0
        self.doi = doi

    def __str__(self):
        return f"{self.__class__.__name__}. DOI = {self.doi}"

    def __repr__(self):
        return self.__str__()

    def regular_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        cpl = 0.0
        cpl_forces = np.zeros(shape=snapshot.forces["new"].shape)
        return cpl, cpl_forces

    def tapered_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        cpl_val, cpl_forces = self.regular_coupling_function(snapshot)
        h_id, y_id = snapshot.site.pair  # type: ignore
        h_idx = snapshot.site.atoms[h_id].idx
        y_idx = snapshot.site.atoms[y_id].idx
        h_pos = snapshot.frame.pos[h_idx]
        y_pos = snapshot.frame.pos[y_idx]
        dist_xyz = utils.get_distance_xyz(h_pos, y_pos, snapshot.frame.box_vectors)
        dist = utils.get_distances(dist_xyz)
        mdf_taper = utils.MDF(dist, self.taper, self.cutoff)
        cpl_val_tpr = float(cpl_val * mdf_taper)
        taper_derivative = 0
        if snapshot.site.dist < self.cutoff and snapshot.site.dist > self.taper:
            dx, dy, dz = dist_xyz
            taper_derivative = utils.dMDF(dx, dy, dz, self.taper, self.cutoff)
        cpl_forces *= -1
        cpl_forces *= mdf_taper
        cpl_forces[h_idx] += taper_derivative * cpl_val
        cpl_forces[y_idx] -= taper_derivative * cpl_val
        cpl_forces *= -1
        return cpl_val_tpr, cpl_forces

    def add_taper(self, cutoff: float, taper: float) -> None:
        if taper > cutoff:
            raise ValueError("Taper is larger than cutoff.")
        self.cutoff = cutoff
        self.taper = taper
        self.coupling_function = self.tapered_coupling_function

    def __call__(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        return self.coupling_function(snapshot)


class Raiteri2011(BaseCoupling):
    DOI = "10.1088/0953-8984/23/33/334213"

    def __init__(self, lmb: float, zeta: float):
        super().__init__(__class__.DOI)
        self.lmb = lmb
        self.zeta = zeta

    def get_coupling_value(self, Q: float) -> float:
        return self.lmb * np.exp(-self.zeta * Q**2)

    def regular_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        x_id, h_id, y_id = snapshot.site.xhy  # type: ignore
        h_idx = snapshot.site.atoms[h_id].idx
        x_idx = snapshot.site.atoms[x_id].idx
        y_idx = snapshot.site.atoms[y_id].idx
        h_pos = snapshot.frame.pos[h_idx]
        x_pos = snapshot.frame.pos[x_idx]
        y_pos = snapshot.frame.pos[y_idx]
        dHX = utils.get_distance_xyz(h_pos, x_pos, snapshot.frame.box_vectors)
        rHX = utils.get_distances(dHX)
        dHY = utils.get_distance_xyz(h_pos, y_pos, snapshot.frame.box_vectors)
        rHY = utils.get_distances(dHY)
        _Q = rHY - rHX
        cpl = self.get_coupling_value(abs(float(_Q)))
        cpl_forces = np.zeros(shape=snapshot.forces["new"].shape)
        prefactor = -2 * self.zeta * cpl * _Q
        derivHY = prefactor * (dHY.flatten() / rHY)
        derivHX = prefactor * -(dHX.flatten() / rHX)
        fHY = -derivHY
        fHX = -derivHX
        cpl_forces[h_idx] += fHY
        cpl_forces[y_idx] -= fHY
        cpl_forces[h_idx] += fHX
        cpl_forces[x_idx] -= fHX
        return cpl, cpl_forces


class Raiteri2011_wrong(BaseCoupling):
    DOI = "10.1088/0953-8984/23/33/334213"

    def __init__(self, lmb: float, zeta: float):
        super().__init__(__class__.DOI)
        self.lmb = lmb
        self.zeta = zeta

    def get_coupling_value(self, Q: float) -> float:
        return self.lmb * np.exp(-self.zeta * Q**2)

    def regular_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        x_id, h_id, y_id = snapshot.site.xhy  # type: ignore
        h_idx = snapshot.site.atoms[h_id].idx
        x_idx = snapshot.site.atoms[x_id].idx
        y_idx = snapshot.site.atoms[y_id].idx
        h_pos = snapshot.frame.pos[h_idx]
        x_pos = snapshot.frame.pos[x_idx]
        y_pos = snapshot.frame.pos[y_idx]
        dHX = utils.get_distance_xyz(h_pos, x_pos, snapshot.frame.box_vectors)
        rHX = utils.get_distances(dHX)
        dHY = utils.get_distance_xyz(h_pos, y_pos, snapshot.frame.box_vectors)
        rHY = utils.get_distances(dHY)
        _Q = rHY - rHX
        cpl = self.get_coupling_value(abs(float(_Q)))
        cpl_forces = np.zeros(shape=snapshot.forces["new"].shape)
        prefactor = -2 * self.zeta * cpl * _Q
        derivHY = prefactor * (dHY.flatten() / rHY)
        derivHX = prefactor * -(dHX.flatten() / rHX)
        fHY = -derivHY
        fHX = -derivHX
        cpl_forces[h_idx] += fHY
        cpl_forces[y_idx] -= fHY
        cpl_forces[h_idx] += fHX
        cpl_forces[x_idx] -= fHX
        cpl_forces *= 0
        return cpl, cpl_forces


class Vuilleumier1998(BaseCoupling):
    DOI = "https://doi.org/10.1016/S0009-2614(97)01365-1"

    def __init__(self, v12: float, alpha: float, gamma: float):
        super().__init__(__class__.DOI)
        self.v12 = float(v12)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def get_coupling_value(self, Q: float, q: float) -> float:
        return self.v12 * np.exp(-self.alpha * Q - self.gamma * q**2)

    def regular_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        x_id, h_id, y_id = snapshot.site.xhy  # type: ignore
        h_idx = snapshot.site.atoms[h_id].idx
        x_idx = snapshot.site.atoms[x_id].idx
        y_idx = snapshot.site.atoms[y_id].idx
        h_pos = snapshot.frame.pos[h_idx]
        x_pos = snapshot.frame.pos[x_idx]
        y_pos = snapshot.frame.pos[y_idx]
        dQpos = utils.get_distance_xyz(x_pos, y_pos, snapshot.frame.box_vectors)
        Q = float(utils.get_distances(dQpos))
        centrexy_pos = 0.5 * (x_pos + y_pos)
        dqpos = utils.get_distance_xyz(h_pos, centrexy_pos, snapshot.frame.box_vectors)
        q = float(utils.get_distances(dqpos))
        cpl = self.get_coupling_value(Q, q)
        cpl_forces = np.zeros(shape=snapshot.forces["new"].shape)
        derivOO = -self.alpha * cpl * dQpos.flatten() / Q
        derivHOO = -self.gamma * 2 * cpl * dqpos.flatten()
        fOO = -derivOO
        fHOO = derivHOO
        cpl_forces[x_idx] += fOO
        cpl_forces[y_idx] -= fOO
        cpl_forces[h_idx] -= fHOO
        cpl_forces[x_idx] += 0.5 * fHOO
        cpl_forces[y_idx] += 0.5 * fHOO
        return cpl, cpl_forces


class Wu2008(BaseCoupling):
    DOI = "https://doi.org/10.1021/jp076658h"

    def __init__(
        self,
        vconst: float,
        gamma: float,
        p: float,
        k: float,
        doo: float,
        beta: float,
        alpha: float,
        roo0: float,
        pp: float,
        roo02: float,
        type_to_exchange_q: Dict[int, float],
        h_exchange_q: float,
        conversion_factor: float = 14.399645,
    ):
        """
        conversion_factor: qq/r -> energy
        type_to_exchange_q: dict which converts atom type (int) to exchange charge
        h_exchang_q: exchange charge for exchanged proton
        """

        super().__init__(__class__.DOI)
        self.vconst = float(vconst)
        self.gamma = float(gamma)
        self.p = float(p)
        self.k = float(k)
        self.doo = float(doo)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.roo0 = float(roo0)
        self.pp = float(pp)
        self.roo02 = float(roo02)
        self.type_to_exchange_q = np.vectorize(dict(type_to_exchange_q).__getitem__)
        self.h_exchange_q = float(h_exchange_q)
        self.conversion_factor = float(conversion_factor)

    def get_coupling_value(self, Vex: float, A: float) -> float:
        return (self.vconst + Vex) * A

    @staticmethod
    def sech2(x: float) -> float:
        return 1 - (np.tanh(x)) ** 2

    def regular_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        x_id, h_id, y_id = snapshot.site.xhy  # type: ignore
        h_idx = snapshot.site.atoms[h_id].idx
        x_idx = snapshot.site.atoms[x_id].idx
        y_idx = snapshot.site.atoms[y_id].idx
        h_pos = snapshot.frame.pos[h_idx]
        x_pos = snapshot.frame.pos[x_idx]
        y_pos = snapshot.frame.pos[y_idx]
        dRoopos = utils.get_distance_xyz(x_pos, y_pos, snapshot.frame.box_vectors)
        Roo = utils.get_distances(dRoopos)
        centrexy_pos = 0.5 * (x_pos + y_pos)
        dqpos = utils.get_distance_xyz(centrexy_pos, h_pos, snapshot.frame.box_vectors)
        q = utils.get_distances(dqpos)
        group1_ids = np.concatenate(
            [
                snapshot.site.residues[snapshot.site.atoms[h_id].molecule],
                snapshot.site.residues[snapshot.site.atoms[y_id].molecule],
            ]
        )
        group1_idxs = np.array([snapshot.site.atoms[eyed].idx for eyed in group1_ids])
        exch_qs = self.type_to_exchange_q(snapshot.site.types[group1_idxs])
        exch_qs[np.where(group1_idxs == h_idx)[0]] = self.h_exchange_q
        group2_idxs = np.delete(np.arange(len(snapshot.frame.pos)), group1_idxs)
        group1_pos = snapshot.frame.pos[group1_idxs]
        group1_qs = exch_qs
        group2_pos = snapshot.frame.pos[group2_idxs]
        group2_qs = snapshot.site.qs[group2_idxs]
        dxs = utils.get_distances_comb_xyz(
            group1_pos, group2_pos, snapshot.frame.box_vectors
        )
        dists = utils.get_distances(dxs)
        dxr = dxs / dists[:, :, None]  # type: ignore
        q_prod = group1_qs[:, None] * group2_qs[None, :]
        q_prod_r = q_prod / dists
        V_ex = np.sum(q_prod_r) * self.conversion_factor
        ds = dxr * -(q_prod_r / dists)[:, :, None]
        term1 = np.exp(-self.gamma * q**2)
        term2 = self.p * np.exp(-self.k * (Roo - self.doo) ** 2)
        term3 = self.beta * (Roo - self.roo0)
        term4 = self.pp * np.exp(-self.alpha * (Roo - self.roo02))
        A_val = float(term1 * (1 + term2) * (0.5 * (1 - np.tanh(term3)) + term4))
        cpl_val = self.get_coupling_value(V_ex, A_val)
        dA_dq = -2 * self.gamma * A_val
        dA_dq_H = -1 * dA_dq * dqpos
        dA_dq_O = 0.5 * dA_dq * dqpos
        dA_dr_term1 = -2 * self.k * (Roo - self.doo) * term2
        dA_dr_term2 = (
            -0.5 * self.beta * self.sech2(float(self.beta * (Roo - self.roo0)))
            - self.alpha * term4
        )
        dA_dr = term1 * (
            (dA_dr_term1 * (0.5 * (1 - np.tanh(term3)) + term4))
            + (dA_dr_term2 * (1 + term2))
        )
        dA_dx = dA_dr * dRoopos / Roo
        group1_deriv = np.sum(ds, axis=1) * self.conversion_factor
        group2_deriv = -1 * np.sum(ds, axis=0) * self.conversion_factor

        # V_ex derivs
        V_ex_derivs = np.zeros(shape=snapshot.frame.pos.shape)
        V_ex_derivs[group1_idxs] += group1_deriv
        V_ex_derivs[group2_idxs] += group2_deriv

        # A derivs
        A_derivs = np.zeros(shape=snapshot.frame.pos.shape)
        A_derivs[h_idx] += dA_dq_H
        A_derivs[y_idx] += dA_dq_O
        A_derivs[x_idx] += dA_dq_O
        A_derivs[y_idx] -= dA_dx
        A_derivs[x_idx] += dA_dx

        full_derivs = (self.vconst + V_ex) * A_derivs + V_ex_derivs * A_val
        cpl_forces = full_derivs * -1

        return cpl_val, cpl_forces


class Grimme2015(BaseCoupling):
    DOI = "https://doi.org/10.1039/C5CP02580J"

    def __init__(self, a: float, b: float):
        super().__init__(__class__.DOI)
        self.a = float(a)
        self.b = float(b)

    def get_coupling_value(self, dE: float) -> float:
        return self.a * np.exp(-self.b * dE**2)

    def regular_coupling_function(self, snapshot: Snapshot) -> Tuple[float, np.ndarray]:
        dE = snapshot.energies["new"] - snapshot.energies["initial"]
        cpl = self.get_coupling_value(dE)
        nF = snapshot.forces["new"]
        iF = snapshot.forces["initial"]
        cpl_forces = np.zeros(nF.shape)
        dF = iF - nF
        dCddE = -2 * self.b * dE * cpl * dF
        cpl_forces -= dCddE

        return cpl, cpl_forces
