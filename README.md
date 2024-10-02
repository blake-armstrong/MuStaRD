
## Installation

```
git clone https://github.com/blake-armstrong/MuStaRD.git
git clone https://github.com/lammps/lammps.git
```
python3 >= python3.10
```
python3 -m venv mustard-venv
. mustard-venv/bin/activate
pip install MuStaRD/
```

To use the extra pair styles required for adding constant energy offsets and using the Wu2008 hydronium ion model you need to copy the .cpp and .h files from the lammps_files directory into the lammps/src directory.

```
cd lammps/
mkdir build
cd build
cmake -C ../../MuStaRD/lammps_files/mustard.cmake -DBUILD_SHARED_LIBS=yes ../cmake
cmake --build . -- -j 4
make install-python
```

## Specifying reaction parameters
The reaction parameters are passed into the main Mustard object upon its creation. The reaction parameters need to take the form of a dictionary. The following keys are available:

`"atom_types"` -- dict
The atom types key should point to a dictionary containing your atom types as strings as the keys and the values should be the corresponding number used in LAMMPS. E.g., if you had atom types "A" and "B" and in LAMMPS you set A as 1 and B as 2 the dictionary should be `{"A": 1, "B": 2}`. This is done so that you can use string representations when specifying the reaction later on. If you'd like, you can forgo the string reprsentation and pass a dictionary like: `{1:1, 2:2}`.

`"bond_types"` -- dict\\
`"angle_types"` -- dict
`"proper_types"` -- dict
`"improper_types"` -- dict
The bond, angle, proper and improper types keys serve the same purpose as the atom_types key. They should point to dictionaries that tell Mustard how to convert the any intramolecular potentials found to the corresponding potential index in lammps. E.g., for bonds it should be `{"A-B": 1, "C-D": 2}`, for angles `{"A-B-C": 1, "D-E-F": 2}` where B and E are the central atoms of the angle potential, for impropers `{"A-B-C-D": 1}` where A is the central atom in the imroper torsion, and for propers `{"A-B-C-D": 1}` where A, B, C and D are in order of the proper torsion.

`"type_charges"` -- dict
The type charges key should be a dictionary that converts the type to its partial charge: `{"A": 0.500, "B": -0.500}`.

`"reactions"` -- list
The reactions key is where the reactions are specified. It should point to a list which contains a dictionary per reaction. For one reaction it will look something like this:

```
[
    {
        "reaction": ("A", "B", "C"),
        "type_changes0": {"A": "C", "C": "A", "B": "B"},
        "type_changes1": {"I": "J", "J": "I"},
        "cutoffs": {
            "distance": 2.0,
            "angle": None,
        },
        "coupling_function": coupling_func,
    }
]
```

This will work for a system where one molecule is I bonded to A which is bonded to B, and the second molecule is C bonded to J. The `"reaction"` key is always a three element tuple/list where element 0 is bonded to element 1 which reacts with element 2. In this case, A is bonded to B which will react with C. If there are any C within 2.0 (`"dist_cutoff"`) of B, mustard will break the bond between A-B and form a bond between B-C. At the same time, it will change the types of A, B and C according to the `"type_change0"` dictionary. Every other atom in either molecule will have its type changed according to `"type_changes1"`. I.e., I will become J and J will become I. This example represents a perfectly symmetrical reaction where I-A-B + C-J -> J-C + B-A-I. The `"coupling_function"` key is a function that can be called that will take in one argument of a Snapshot object (see mustard.topology) and access the atom positions, system energy and any other relevant property one might need to calculate a coupling value and an array of per-atom forces. For convenience, in mustard.coupling there is a BaseCoupling class that can be used as a starting point for creating any arbitrary coupling function. In that file there are a number of coupling functions from literature that have been created from the BaseCoupling class, which can be used as a reference for creating your own.

`"fermi_mixing"` -- Bool
`"temperature"` -- float
`"fermi_tolerance_1"` -- float,
`"fermi_tolerance_2"` -- float,
The above keys concern whether or not to Fermi smearing to partially occupy the various eigensolutions. `"fermi_mixing"` is a Boolean (on/off). The `"temperature"` key is a float to set the temperature for the Fermi level. `"fermi_tolerance_1"` and `"fermi_tolerance_2"` set convergence tolerances as floats. To see exactly what they do, look in mustard.mixing.

`"lammps_unit_system"` -- string
Sets the unit system that for Mustard to know how to calculate certain values. Should be a string. Needs to be the same as your LAMMPS setting.

`"constant_volume"` -- bool
Not properly implemented yet. Need a way to get virial from coupling function... don't want to overcomplicate things yet.

`"neighbour_list_update"` -- int
How often to rebuild the neighbour list in number of steps. Needs to be manually specified due to the nature of using the API. Will depend on what kind of system you have. Probably never should be more than 10.

`"topology_update"` -- int
How often to check for a reaction and update the surrounding topology. Should always be 1 for accuracy. Sometimes when it's not 1 the simulation crashes. Need to figure out why...

`"shells"` -- int
Number of shells of reactivity. 

`"pbc"` -- bool
Whether or not your simulation has periodic boundary conditions. Boolean.

`"eig_solver"` -- string
The software used to generate the eigenvalues and eigenvectors from the Hamiltonian. Either "numpy" or "sympy". "numpy" is faster and should always be used. "sympy" is just there for validation.
