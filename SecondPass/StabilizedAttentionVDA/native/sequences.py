"""Supervised observation protocol (does not alter the RL environment API)."""

SUPERVISED_SEQUENCE_VERSION = "blank_first_v1"


def supervised_frame_indices(T, frame_window=1, frame_stride=1):
    """Physical window endpoints used by the recurrent model (including last)."""
    ends = list(range(frame_window - 1, T, frame_stride))
    if ends[-1] != T - 1:
        ends.append(T - 1)
    return ends


def collect_supervised_sequence(env):
    """Return physical frames t=0..T-1, including reset, never terminal-return t=T.

    Use the observations themselves, not re-rendering: this preserves the seeded
    orientation/pixel noise draws and frame-repeat cache. Supervised collectors
    intentionally do not take the final RL action or update terminal curriculum.
    """
    first = env.reset()
    return [first] + [env.step(0)[0] for _ in range(int(env.T) - 1)]
