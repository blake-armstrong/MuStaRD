import numpy as np
from scipy.optimize import minimize, least_squares
from mpi4py import MPI

RANK = MPI.COMM_WORLD.Get_rank()


def _calc_FD_occupancies(mu, energy, temp, tol, RT):
    # calculate Fermi-Dirac statistic for a particular energy state \
    # and chemical potential at the requested mixing temperature
    occupancies = np.zeros(len(energy))
    if temp > tol:
        e = np.clip((energy - mu) / RT, -700, 700)
        e1 = np.clip(np.exp(e), 1e-8, 1e8)
        e2 = np.clip(np.exp(-e), 1e-8, 1e8)
        elm = energy < mu
        occupancies += e2 / (1 + e2) * ~elm * 1 + 1 / (1 + e1) * elm * 1
        return occupancies
    else:
        occupancies[energy <= mu] = 1
        return occupancies


# def penalty_term(*args):
#     return (2 * np.sum(_calc_FD_occupancies(*args)) - 1.0) ** 2


def objective(*args):
    occ = _calc_FD_occupancies(*args)
    return np.sum(occ / np.sum(occ))

    # print("out", out)
    # if out < 1.0 - args[3]:
    #     print(out)
    #     return 1e10


def get_FD_occupancies(eig_vals, temp, tols, RT):
    # this is the fermi level at 0 K
    tol1, tol2 = tols
    mu = np.mean(np.sort(eig_vals)[0:2])
    occupancies = _calc_FD_occupancies(mu, eig_vals, temp, tol1, RT)

    if temp > 0.0:
        occ_sum = np.sum(occupancies)
        # iteratively converge the chemical potential, mu, so that fermi occupancies integrate to 1
        # print("occ_sum", occ_sum)
        # print("eig_vals", eig_vals)
        if not 1.0 + tol2 >= occ_sum >= 1.0 - tol2:
            print("mu", mu)
            print("occ", occupancies)
            print("occ_sum", occ_sum)
            result = minimize(
                objective,
                x0=mu,
                args=(eig_vals, temp, tol1, RT),
                # bounds=[(mu - 2 * RT, mu + 2 * RT)],
                tol=tol2,
                # tol=tol2,
                # method="Nelder-Mead",
            )
            print(result)
            print("aaa")
            # result = least_squares(
            #    objective,
            #    x0=mu,
            #    args=(eig_vals, temp, tol1, RT),
            #    # bounds=[(mu - 10 * RT, mu + 10 * RT)],
            #    ftol=1e-8,
            #    # method="CG",
            # )
            o_mu = result.x[0]
            # occupancies = _calc_FD_occupancies(o_mu, eig_vals, temp, tol1, RT)
            # if RANK == 1:
            print("o_mu", o_mu)
            print("occ", _calc_FD_occupancies(o_mu, *(eig_vals, temp, tol1, RT)))
            print(
                "occ_sum",
                np.sum(_calc_FD_occupancies(o_mu, *(eig_vals, temp, tol1, RT))),
            )
            exit()
            #
            # i = 0
            # while not 1.0 + tol2 > occ_sum > 1.0 - tol2:
            #    if RANK == 1:
            #        print("diff", abs(occ_sum - 1))
            #        print("pdiff", tol2 / abs(occ_sum - 1))
            #        print("occ_sum", occ_sum)
            #        print("occ", occupancies)
            #        print("mu", mu)
            #        print("d", d)

            # mu += d
            # occupancies = _calc_FD_occupancies(eig_vals, mu, temp, tol1, RT)
            # occ_sum = np.sum(occupancies)
            # d = 0.1 * abs(d) * -(np.sign(occ_sum - 1))
            # i += 1
            # if i == 10:
    else:
        min_states = np.where(eig_vals == np.amin(eig_vals))
        occupancies = ((eig_vals <= mu) * 1) / (len(min_states) + 1)
    return occupancies
