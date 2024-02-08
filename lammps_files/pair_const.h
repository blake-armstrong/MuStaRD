/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Pair zero is a dummy pair interaction useful for requiring a
   force cutoff distance in the absence of pair-interactions or
   with hybrid/overlay if a larger force cutoff distance is required.

   This can be used in conjunction with bond/create to create bonds
   that are longer than the cutoff of a given force field, or to
   calculate radial distribution functions for models without
   pair interactions.

------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(const,PairConst);
// clang-format on
#else

#ifndef LMP_PAIR_CONST_H
#define LMP_PAIR_CONST_H

#include "pair.h"

namespace LAMMPS_NS {

class PairConst : public Pair {
 public:
  PairConst(class LAMMPS *);
  ~PairConst() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  double **cut;
  int atomtype;
  double atomtype_energy_shift;
  int natoms;

  virtual void allocate();
};
}    // namespace LAMMPS_NS
#endif
#endif
