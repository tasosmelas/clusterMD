"""Collect Python and native-software environment metadata."""

import numpy as np
import re
import scipy
import shutil
import subprocess
import sys

def get_software_env():
    """Detects software and library versions for the provenance report."""
    try:
        import importlib.metadata
        fh_ver = importlib.metadata.version('fast_hdbscan')
    except Exception:
        try:
            import fast_hdbscan
            fh_ver = fast_hdbscan.__version__ if hasattr(fast_hdbscan, '__version__') else "not detected"
        except ImportError:
            fh_ver = "not found"
            
    carma_path = shutil.which("carma")
    carma_ver = "not detected"
    if carma_path:
        try:
            res = subprocess.run(["carma"], capture_output=True, text=True)
            for line in (res.stderr + "\n" + res.stdout).splitlines():
                if "version" in line.lower() or " v." in line.lower() or "carma v" in line.lower():
                    match = re.search(r'v\.?\s*(\d+\.\d+)', line.lower())
                    if match:
                        carma_ver = match.group(1)
                    else:
                        parts = line.strip().split()
                        carma_ver = parts[1] if len(parts) > 1 else line.strip()
                    break
        except Exception:
            pass
            
    return {
        "clusterMD_ver": "1.2.0",
        "python_ver": sys.version.split()[0],
        "numpy_ver": np.__version__,
        "scipy_ver": scipy.__version__,
        "fh_ver": fh_ver,
        "carma_path": carma_path if carma_path else "NOT FOUND",
        "carma_ver": carma_ver
    }


def get_gromacs_env():
    """Detect the GROMACS executable and version independently of carma provenance."""
    gromacs_path = shutil.which("gmx")
    gromacs_ver = "not detected"

    if gromacs_path:
        try:
            res = subprocess.run(
                ["gmx", "--version"],
                capture_output=True,
                text=True,
            )
            version_text = res.stdout + "\n" + res.stderr
            for line in version_text.splitlines():
                if "gromacs version" in line.lower():
                    _, _, value = line.partition(":")
                    gromacs_ver = value.strip() if value.strip() else line.strip()
                    break
        except Exception:
            pass

    return {
        "gromacs_path": gromacs_path if gromacs_path else "NOT FOUND",
        "gromacs_ver": gromacs_ver,
    }
