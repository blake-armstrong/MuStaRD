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
   Contributing author: Carsten Svaneborg (SDU)
------------------------------------------------------------------------- */

#include "pair_const.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairConst::PairConst(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairConst::~PairConst()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

/* ---------------------------------------------------------------------- */

void PairConst::compute(int eflag, int vflag)
{
  int i, count;
  double evdwl;
  evdwl = 0.0;
  ev_init(eflag, vflag);
  int newton_pair = 1;
  if (eflag) {
    int *type = atom->type;

    count = 0;
    for (i = 0; i < natoms; i++) {
      if (type[i] == atomtype) { count++; }
    }
    evdwl = count * atomtype_energy_shift;
  }
  if (evflag) ev_tally(0, 0, 0, newton_pair, evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairConst::allocate()
{
  natoms = atom->natoms;
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairConst::settings(int narg, char **arg)
{
  if (narg != 0) utils::missing_cmd_args(FLERR, "pair_style const", error);
  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i + 1; j <= atom->ntypes; j++) cut[i][j] = 1;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairConst::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  atomtype = utils::numeric(FLERR, arg[2], false, lmp);
  atomtype_energy_shift = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = 1;
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairConst::init_style()
{
  neighbor->add_request(this, NeighConst::REQ_DEFAULT);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairConst::init_one(int i, int j)
{
  if (setflag[i][j] == 0) { cut[i][j] = mix_distance(cut[i][i], cut[j][j]); }
  return cut[i][j];
}
