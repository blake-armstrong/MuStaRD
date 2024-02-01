# constants
# codata 2018
AVOGADRO_CONSTANT_NA = 6.02214076e23  # / mole
BOLTZMANN_CONSTANT_kB = 1.380649e-23  # joule / kelvin
MOLAR_GAS_CONSTANT_R = (
    AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB
)  # joule / mole / kelvin
eV = 1.602176634e-19  # joule

# convsersions
j2cal = 1 / 4.184


UNITS = {
    "real": {
        "boltz": MOLAR_GAS_CONSTANT_R * j2cal,  # kcal/mol/K
        "pr2vir": 68568.7,
    },
    "metal": {
        "boltz": MOLAR_GAS_CONSTANT_R / AVOGADRO_CONSTANT_NA / eV,  # eV/K
        "pr2vir": 1602176.6,
    },
    "si": {"boltz": BOLTZMANN_CONSTANT_kB, "pr2vir": 1.0},  # J/K
    "cgs": {
        "boltz": BOLTZMANN_CONSTANT_kB * 10000000,  # ergs/K
        "pr2vir": 1.0,
    },
    "electron": {
        "boltz": BOLTZMANN_CONSTANT_kB * 2.2937123e17,  # Hartrees/K
        "pr2vir": 2.9421196e13,
    },
}
