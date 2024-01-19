/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(voth/vrep2,PairVoth2);
// clang-format on
#else

#ifndef LMP_PAIR_VOTH2_H
#define LMP_PAIR_VOTH2_H

#include "pair.h"

namespace LAMMPS_NS {

class PairVoth2 : public Pair {
 public:
  PairVoth2(class LAMMPS *);
  ~PairVoth2() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  // void init_style() override;
  double init_one(int, int) override;

 protected:
  double cut_global, cut_inner_global;
  double **cut, **cut_inner, **cut_inner_sq;
  double **C, **c, **d0, **switch_1, **switch_2, **switch_3, **switch_4;
  int natoms;    //to make sure number of atoms is constant

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
