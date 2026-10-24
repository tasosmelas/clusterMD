"""Shared, plain-text presentation helpers for clusterMD."""

from clustermd.utils.environment import get_software_env

CONSOLE_DESCRIPTION = "HDBSCAN Analysis of Molecular-Dynamics PCA Coordinates"
LABEL_WIDTH = 26


def _headline(title):
    return f"{title}\n{'-' * len(title)}"


def console_header(version=None, title=None):
    version = version or get_software_env()["clusterMD_ver"]
    return _headline(f"clusterMD {version} | {title or CONSOLE_DESCRIPTION}")


def report_header(version=None):
    version = version or get_software_env()["clusterMD_ver"]
    return _headline(f"clusterMD {version} | ANALYSIS REPORT")


def section(title):
    return f"\n{title.upper()}\n"


def field(label, value):
    return f"  {label:<{LABEL_WIDTH}} {value}\n"


def table(headers, rows, right_aligned=None):
    """Render narrow, deterministic columns without terminal-dependent spacing."""
    right_aligned = set(right_aligned or ())
    data = [tuple(str(value) for value in headers)] + [tuple(str(value) for value in row) for row in rows]
    if not data:
        return ""
    widths = [max(len(row[index]) for row in data) for index in range(len(headers))]
    def render(row):
        return "  " + "  ".join(
            f"{value:>{widths[i]}}" if i in right_aligned else f"{value:<{widths[i]}}"
            for i, value in enumerate(row)
        ).rstrip()
    header_line = render(data[0])
    return "\n".join([header_line, "  " + "-" * (len(header_line) - 2)] + [render(row) for row in data[1:]]) + "\n"
