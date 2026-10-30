"""Validate measured outcomes before they enter the search or selection."""
import math
from numbers import Real, Integral


def valid_sr(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def validate_evaluation(result, *, require_counts):
    if not isinstance(result, dict):
        raise ValueError('Evaluation must return a result dictionary')
    if result.get('sr') is None:
        return result
    if not valid_sr(result['sr']):
        raise ValueError('Evaluation SR must be finite and in [0,1]')
    if result.get('status') in {'failed', 'dry_run', 'incomplete'}:
        raise ValueError('Failed or incomplete evaluations cannot carry a valid SR')
    if require_counts:
        successes, episodes = result.get('successes'), result.get('episodes')
        if (result.get('status') != 'completed' or
            not isinstance(episodes, Integral) or isinstance(episodes, bool) or episodes < 1 or
            not isinstance(successes, Integral) or isinstance(successes, bool) or not 0 <= successes <= episodes):
            raise ValueError('Real evaluations require completed status and valid success/episode counts')
        if not math.isclose(result['sr'], successes / episodes, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Evaluation SR disagrees with success/episode counts')
    return result
