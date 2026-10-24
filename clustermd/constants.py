"""Define shared scientific and presentation constants."""

import re

R_KCAL = 0.0019872041                # Ideal gas constant in kcal/(mol*K)
MAX_FREE_ENERGY_KCAL = 9.0           # Cap for free-energy surface plots (kcal/mol)
FALLBACK_ATOM_FLAGS = ["-atomid", "CA", "-atomid", "C", "-atomid", "N", "-atomid", "O"]
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

class Colors:
    LIGHT_BLUE = '\033[96m'   # Processing / Info steps
    BLUE = '\033[94m'         # Success / Milestones
    YELLOW = '\033[93m'       # Warnings
    RED = '\033[91m'          # Errors
    BOLD = '\033[1m'
    RESET = '\033[0m'

