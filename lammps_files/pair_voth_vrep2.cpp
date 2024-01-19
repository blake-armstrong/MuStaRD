/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_voth_vrep2.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairVoth2::PairVoth2(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairVoth2::~PairVoth2()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(cut_inner);
    memory->destroy(cut_inner_sq);
    memory->destroy(C);
    memory->destroy(c);
    memory->destroy(d0);
    memory->destroy(switch_1);
    memory->destroy(switch_2);
    memory->destroy(switch_3);
    memory->destroy(switch_4);
  }
}

/* ---------------------------------------------------------------------- */

void PairVoth2::compute(int eflag, int vflag)
{
  int i, j, h, hh, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, force_oh, pe, tt, dt;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  if (natoms != atom->natoms) error->all(FLERR, "pair voth/vrep2 requires a fixed number of atoms");

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;

      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;    //OO distance

      if (rsq > cutsq[itype][jtype]) {
        if (eflag) { evdwl = 0.0; }
        continue;
      }
      r = sqrt(rsq);
      pe = C[itype][jtype] * exp(-c[itype][jtype] * (r - d0[itype][jtype]));
      force_oh = -c[itype][jtype] * pe;
      if (rsq > cut_inner_sq[itype][jtype]) {
        // switching function
        tt = 1 +
            (cut_inner_sq[itype][jtype] * cut_inner[itype][jtype] - switch_4[itype][jtype] * rsq -
             switch_3[itype][jtype] * rsq + 2 * r * rsq -
             switch_3[itype][jtype] * cut_inner_sq[itype][jtype] + r * switch_2[itype][jtype]) *
                switch_1[itype][jtype];
        // switching function derivate w.r.t. r
        dt = (6 * rsq - 2 * switch_3[itype][jtype] * r - 2 * switch_4[itype][jtype] * r +
              switch_2[itype][jtype]) *
            switch_1[itype][jtype];

        force_oh = force_oh * tt + pe * dt;
      } else {
        tt = 1;
      }
      if (eflag) { evdwl = pe * tt; }
      fpair = -1 * force_oh / r;

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      f[j][0] -= delx * fpair;
      f[j][1] -= dely * fpair;
      f[j][2] -= delz * fpair;

      if (evflag) { ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz); }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairVoth2::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");

  memory->create(cut, n, n, "pair:cut");
  memory->create(cut_inner, n, n, "pair:cut_inner");
  memory->create(cut_inner_sq, n, n, "pair:cut_inner_sq");
  memory->create(C, n, n, "pair:C");
  memory->create(c, n, n, "pair:c");
  memory->create(d0, n, n, "pair:d0");
  memory->create(switch_1, n, n, "pair:switch_1");
  memory->create(switch_2, n, n, "pair:switch_2");
  memory->create(switch_3, n, n, "pair:switch_3");
  memory->create(switch_4, n, n, "pair:switch_4");

  natoms = atom->natoms;
  if (!natoms) error->all(FLERR, "No atoms found");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairVoth2::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Illegal pair_style command");

  cut_inner_global = utils::numeric(FLERR, arg[0], false, lmp);
  cut_global = utils::numeric(FLERR, arg[1], false, lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) {
          cut[i][j] = cut_global;
          cut_inner[i][j] = cut_inner_global;
        }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairVoth2::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 7) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double C_one = utils::numeric(FLERR, arg[2], false, lmp);
  double c_one = utils::numeric(FLERR, arg[3], false, lmp);
  double d0_one = utils::numeric(FLERR, arg[4], false, lmp);

  double cut_one = cut_global;
  double cut_inner_one = cut_inner_global;
  if (narg == 7) {
    cut_one = utils::numeric(FLERR, arg[5], false, lmp);
    cut_inner_one = utils::numeric(FLERR, arg[6], false, lmp);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      C[i][j] = C_one;
      c[i][j] = c_one;
      d0[i][j] = d0_one;
      cut[i][j] = cut_one;
      cut_inner[i][j] = cut_inner_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

// void PairVoth2::init_style()
// {
//   // request regular or rRESPA neighbor list
//
//   if (atom->tag_enable == 0) error->all(FLERR, "Pair style E3B requires atom IDs");
//   if (force->newton_pair == 0) error->all(FLERR, "Pair style E3B requires newton pair on");
// }

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairVoth2::init_one(int i, int j)
{

  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  cut_inner_sq[i][j] = cut_inner[i][j] * cut_inner[i][j];
  cut[j][i] = cut[i][j];
  cut_inner[j][i] = cut_inner[i][j];
  cut_inner_sq[j][i] = cut_inner_sq[i][j];

  C[j][i] = C[i][j];
  c[j][i] = c[i][j];
  d0[j][i] = d0[i][j];

  switch_1[i][j] = 1 /
      ((cut[i][j] - cut_inner[i][j]) * (cut[i][j] - cut_inner[i][j]) *
       (cut[i][j] - cut_inner[i][j]));
  switch_2[i][j] = 6 * cut[i][j] * cut_inner[i][j];
  switch_3[i][j] = 3 * cut[i][j];
  switch_4[i][j] = 3 * cut_inner[i][j];
  switch_1[j][i] = switch_1[i][j];
  switch_2[j][i] = switch_2[i][j];
  switch_3[j][i] = switch_3[i][j];
  switch_4[j][i] = switch_4[i][j];

  return cut[i][j];
}
