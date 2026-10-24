"""Shared backend-aware structural completion assessment for TXT and JSON reports."""

import numpy as np


def structural_status(spec, info, unique_clusters, *, skip_structural=False,
                      fatal_struct_error=False):
    """Return a human-readable extraction status from positive backend evidence."""
    if skip_structural:
        return "Not performed (invalid frame identifiers)"
    if not spec.trajectory_file or not spec.topology_file:
        return "Skipped (trajectory/topology unavailable)"
    if fatal_struct_error:
        return "Failed or partially completed; inspect execution log"
    if info.get("compatible") is False:
        return "Failed (structural inputs incompatible)"
    if info.get("bounds_valid") is False:
        return "Failed (trajectory frame bounds)"

    expected = {int(cid) for cid in unique_clusters}
    if not expected:
        return "Not performed (no assigned clusters)"
    backend = spec.structural.backend.strip().lower()
    if backend == "carma":
        success = info.get("success_count")
        failed = info.get("failed_count")
        if not isinstance(success, int) or not isinstance(failed, int):
            return "Not confirmed (extraction counts unavailable)"
        if failed > 0:
            return (f"Partial ({success} successful, {failed} failed)" if success > 0
                    else f"Failed (0 successful, {failed} failed)")
        if success == len(expected):
            return "Completed"
        if 0 < success < len(expected):
            return f"Partial ({success} of {len(expected)} clusters confirmed)"
        return "Not confirmed (no successful cluster extraction recorded)"
    if backend == "gromacs":
        completed = {int(cid) for cid in info if isinstance(cid, (int, np.integer))}
        completed &= expected
        if completed == expected:
            return "Completed"
        if completed:
            return f"Partial ({len(completed)} of {len(expected)} clusters confirmed)"
        return "Not confirmed (no completed cluster records)"
    return "Not confirmed (unknown structural backend)"



def status_code(status):
    """Normalize a human-readable structural status for machine-readable records."""
    prefix = status.split(" (", 1)[0].lower().replace(" ", "_")
    return {"completed": "completed", "partial": "partial", "failed": "failed",
            "skipped": "skipped", "not_confirmed": "not_confirmed",
            "not_performed": "not_performed"}.get(prefix, "not_confirmed")
