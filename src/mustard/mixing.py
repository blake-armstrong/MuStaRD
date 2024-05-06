import numpy as np

_TOLERANCE_1 = 1e-10
_TOLERANCE_2 = 1e-12


def get_FD_occupancies(
    eig_vals: np.ndarray,
    RT: float,
    tol1: float = _TOLERANCE_1,
    tol2: float = _TOLERANCE_2,
) -> np.ndarray:
    """
    Finds the Fermi energy and returns the occupation weights of states.
    Based on the routine fermid from SIESTA originally written by J.M.Soler.
    """

    nitermax = 150
    nb = len(eig_vals)
    mu = np.mean(np.sort(eig_vals)[0:2])
    for _ in range(nitermax):
        xs = (eig_vals - mu) / RT
        stepf = np.zeros(nb)
        dstepf = np.zeros(nb)
        stepf[xs < -100] = 1.0
        mask = (xs < 100) & (xs > -100)
        stepf[mask] = 1 / (1 + np.exp(xs[mask]))
        dstepf[mask] = np.exp(xs[mask]) / (RT * (np.exp(xs[mask]) + 1) ** 2)
        occupancies = stepf
        sumq = np.sum(occupancies)
        dsumq = np.sum(dstepf)

        if abs(sumq - 1) < tol1:
            return occupancies

        if abs(dsumq) > tol2:
            mu += (1 - sumq) / dsumq

    raise RuntimeError("Calculation of Fermi level energy has failed to converge.")
