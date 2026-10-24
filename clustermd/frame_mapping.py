'Validate physical-time mappings between projections and trajectories.\n\nTime values are in ps and GROMACS source-frame ordinals are zero-based.'
import numpy as np
from clustermd.models import PipelineExecutionError

def validate_times(times):
    """Require a finite, nonempty, strictly increasing time axis in ps."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise PipelineExecutionError('Trajectory times must be a finite, nonempty vector.')
    if np.any(np.diff(times) <= 0):
        raise PipelineExecutionError('Trajectory times must be strictly increasing; repeated/reset times are ambiguous.')
    return times


def map_times_to_frames(projection_times, trajectory_times, atol=1e-4):
    """Require exactly one trajectory timestamp within an absolute ps tolerance."""
    trajectory_times = validate_times(trajectory_times)
    projection_times = validate_times(projection_times)
    left = np.searchsorted(trajectory_times, projection_times - atol, side='left')
    right = np.searchsorted(trajectory_times, projection_times + atol, side='right')
    if np.any(right - left != 1):
        raise PipelineExecutionError('Projection times do not map uniquely to trajectory frames. '
                                     'Check the trajectory, XVG time units (ps), and output precision.')
    if np.any(np.diff(left) <= 0):
        raise PipelineExecutionError('Projection samples map to repeated or reordered trajectory frames.')
    return left.astype(np.int64)


def check_output_times(actual, expected, context, atol=1e-4):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=atol):
        raise PipelineExecutionError(f'{context}: output timestamps do not match the assigned trajectory frames.')


def write_frame_selector(path, trajectory_times, frames):
    """Write selected zero-based trajectory ordinals for native extraction."""
    times = validate_times(trajectory_times)
    frames = np.asarray(frames)
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu':
        raise PipelineExecutionError('Frame selector requires a nonempty integer vector.')
    if np.any(frames < 0) or np.any(frames >= len(times)) or len(np.unique(frames)) != len(frames):
        raise PipelineExecutionError('Frame selector contains invalid or duplicate trajectory ordinals.')
    keep = np.zeros(len(times), dtype=int)
    keep[frames] = 1
    np.savetxt(path, np.column_stack((times, keep)), fmt=['%.17g', '%d'])
