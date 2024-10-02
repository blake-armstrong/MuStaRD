
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
