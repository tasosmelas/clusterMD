"""Format human-readable values used in analysis reports."""



def format_duration(seconds):
    """Converts raw seconds into a human-readable time format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    elif m > 0: return f"{m}m {s}s"
    else: return f"{seconds:.2f}s"


def format_record(raw_line, frame_val, cid_val):
    """Format projection coordinates and frame/cluster labels for readable logs."""
    coords_str = " ".join([f"{c:9.5f}" for c in raw_line])
    return f"Frame {frame_val:7d} | Cluster {cid_val:4d} | Coordinates: {coords_str}"

