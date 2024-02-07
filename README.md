```
git clone https://github.com/blake-armstrong/MuStaRD.git
git clone https://github.com/blake-armstrong/lammps-mustard.git
```
python3 >= python3.10
```
python3 -m venv mustard-venv
. mustard-venv/bin/activate
pip install MuStaRD/
```
```
cd lammps-mustard/
mkdir build
cd build
cmake -C ../cmake/presets/mustard.cmake -DBUILD_SHARED_LIBS=yes ../cmake
cmake --build . -- -j 4
```
