"""
Shared base for the RViT+ FiLM task battery.

BaseChangeDetectionEnv generalises the published cued orientation-change-detection
task to an arbitrary grid_rows × grid_cols stimulus array (one Gabor per cell, indexed
ROW-MAJOR so token i ⇔ stimulus i in the conv front-end), with a cue that marks one
cell and shows its reliability (ring completeness = proportion = validity) and,
optionally, its value (colour). Subclasses override only the reward / target / response
rules to realise the specific battery tasks.

Trial timeline (T steps): t=0,2 blank · t=1 cue · t≥3 Gabors (orientation noise every
frame) · at change_time∈[min,max] one stimulus steps by Δ (50% of trials are change
trials). Actions: 0=wait, 1=declare change.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

COLOR_RGB = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0),
             "blue": (0.0, 0.0, 1.0), "white": (1.0, 1.0, 1.0)}


def cell_bounds(extent: int, n: int):
    edges = np.linspace(0, extent, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


class BaseChangeDetectionEnv(gym.Env):
    def __init__(
        self,
        grid_rows: int = 2,
        grid_cols: int = 2,
        theta: float = 65.0,                  # paper: Δ ~ U(−k,k), k starts at 65
        noise_multiplier: float = 5.0,
        reward_scale: float = 1.0,            # multiply the correct-response reward magnitude (see _reward_value)
        cue_positions=None,                   # restrict the cue to these stim indices (paper: S1/S4)
        # Paper timeline: 7 steps (t=0 black, t=1 cue, t=2 black, t=3-6 stimuli) with the
        # change FIXED at t=5. (Set min<max and a larger T for the longer random-onset
        # variant.)
        min_change_time: int = 5,
        max_change_time: int = 5,
        T: int = 7,
        frame_repeat: int = 1,                # hold each LOGICAL frame for this many physical steps
        proportions=(0.25, 0.5, 0.75, 1.0),
        value_cues: bool = False,
        color_values=None,
        image_size: int = 50,
        # Curriculum on the orientation-change magnitude (paper: start large/easy and
        # SHRINK Δ as performance improves, to make detection progressively harder).
        # `self.theta` is the current max |Δ| — the harness logs it each iteration.
        curriculum: bool = True,
        curr_window: int = 1000,              # paper: average over 1000 trials
        curr_threshold: float = 0.85,         # paper: success >= 85%
        curr_step: float = 3.0,               # paper: drop max |Δ| by 3 degrees per cleared window
        curr_decay: float = 0.9,              # (legacy multiplicative decay; used only if curr_step<=0)
        theta_floor: float = 8.0,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(image_size, image_size, 3),
                                            dtype=np.float32)
        self.grid_rows, self.grid_cols = int(grid_rows), int(grid_cols)
        self.n_stim = self.grid_rows * self.grid_cols
        self.cue_positions = list(cue_positions) if cue_positions else list(range(self.n_stim))
        self.theta = float(theta)
        self.noise_multiplier = float(noise_multiplier)
        self.reward_scale = float(reward_scale)
        self.min_change_time, self.max_change_time = int(min_change_time), int(max_change_time)
        if self.min_change_time < 3:
            raise ValueError("min_change_time must be >= 3 (Gabors start at t=3)")
        # FRAME-REPEAT: T (input) is the LOGICAL frame count; physical T = T × frame_repeat.
        # change_time / cue / Gabor timing are all in LOGICAL frames; pixels are frozen within a frame.
        self.frame_repeat = int(frame_repeat)
        self.n_logical = int(T)
        self.T = int(T) * self.frame_repeat
        self._frame_cache = None
        self.S = int(image_size)
        self.proportions = list(proportions)
        self.value_cues = bool(value_cues)
        self.color_values = dict(color_values) if color_values else {"red": 5, "green": 3, "blue": 1}
        self.curriculum = bool(curriculum)
        self.theta_start = float(theta)
        self.theta_floor = float(theta_floor)
        self.curr_window = int(curr_window)
        self.curr_threshold = float(curr_threshold)
        self.curr_step = float(curr_step)
        self.curr_decay = float(curr_decay)
        self._recent_correct: list = []
        self._rb = cell_bounds(self.S, self.grid_rows)
        self._cb = cell_bounds(self.S, self.grid_cols)
        self.cells = [(r0, r1, c0, c1) for (r0, r1) in self._rb for (c0, c1) in self._cb]
        self.t = 0
        self.reset()

    # ── trial setup ─────────────────────────────────────────────────────────
    def reset(self):
        self.t = 0
        self._frame_cache = None
        self.change_time = np.random.randint(self.min_change_time, self.max_change_time + 1)
        self.orientations = [np.random.uniform(0, 360) for _ in range(self.n_stim)]
        self.change_true = int(np.random.rand() >= 0.5)
        self.orientation_change = np.random.uniform(-self.theta, self.theta)
        self.cue_index = int(np.random.choice(self.cue_positions))
        self.proportion = float(np.random.choice(self.proportions))
        if self.value_cues:
            self.cue_color = str(np.random.choice(list(self.color_values.keys())))
        else:
            self.cue_color = "white"
        self.change_index = self._draw_change_index()
        self.valid = int(self.change_true == 1 and self.change_index == self.cue_index)
        return self._next_observation()

    def _draw_change_index(self) -> int:
        """Where the change occurs on a change trial (legacy base-task semantics)."""
        if self.change_true == 0:
            return -1
        if np.random.rand() < self.proportion:
            return self.cue_index
        return int(np.random.randint(self.n_stim))

    def _draw_validity_target(self, candidates=None) -> int:
        """Draw a target with displayed validity equal to realized cue validity.

        The invalid branch samples only non-cued candidates. VDA tasks opt into this
        helper; the base task keeps its legacy sampling behavior.
        """
        if self.change_true == 0:
            return -1
        choices = list(range(self.n_stim)) if candidates is None else [int(i) for i in candidates]
        if self.cue_index not in choices:
            raise ValueError("cue_index must be among target candidates")
        uncued = [i for i in choices if i != self.cue_index]
        if not uncued or np.random.rand() < self.proportion:
            return int(self.cue_index)
        return int(np.random.choice(uncued))

    # ── rendering ─────────────────────────────────────────────────────────────
    def _next_observation(self):
        L = self.t // self.frame_repeat                          # LOGICAL frame
        # frozen pixels: re-render only on the FIRST physical step of a logical frame
        if self.frame_repeat > 1 and self.t % self.frame_repeat != 0 and self._frame_cache is not None:
            return self._frame_cache
        obs = np.zeros((self.S, self.S, 3), dtype=np.float32)
        if L in (0, 2):
            out = obs
        elif L == 1:
            out = self._render_cue(obs)
        else:
            out = self._render_stimuli(obs, L)
        self._frame_cache = out
        return out

    def _current_feature(self, i: int, L: int) -> float:
        """Per-cell feature (orientation), with the change applied from change_time on. One place for
        the change logic regardless of what the stimulus renderer draws."""
        f = self.orientations[i]
        if L >= self.change_time and self.change_true == 1 and self.change_index == i:
            f += self.orientation_change
        return f

    def _render_stimuli(self, obs, L):
        """Draw the per-cell stimulus for a stimulus frame (default: a Gabor per cell + orientation
        noise). Subclasses override ONLY this to change which/what stimuli are drawn (e.g. show a
        distractor only on some trials) without touching the timeline/cue/change/reward logic."""
        for i, (r0, r1, c0, c1) in enumerate(self.cells):
            ori = self._current_feature(i, L) + self.noise_multiplier * np.random.normal()
            g = self._gabor(ori, r1 - r0, c1 - c0)
            obs[r0:r1, c0:c1, :] = np.stack([g, g, g], axis=-1)
        return obs

    def _render_cue(self, obs):
        r0, r1, c0, c1 = self.cells[self.cue_index]
        h, w = r1 - r0, c1 - c0
        n = min(h, w)
        yy, xx = np.ogrid[0:h, 0:w]                      # match the actual (h, w) cell
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rr = (xx - cx) ** 2 + (yy - cy) ** 2
        disc_r = 0.32 * n
        ring_in, ring_out = 0.40 * n, 0.48 * n
        cue = np.zeros((h, w), dtype=np.float32)
        cue[rr <= disc_r ** 2] = 1.0
        ring = (rr <= ring_out ** 2) & (rr >= ring_in ** 2)
        ang = np.arctan2(yy - cy, xx - cx) + np.pi
        ring &= ~(ang < 2 * np.pi * (1 - self.proportion))
        cue[ring] = 1.0
        rgb = COLOR_RGB.get(self.cue_color, (1.0, 1.0, 1.0))
        for k in range(3):
            obs[r0:r1, c0:c1, k] = cue * rgb[k]
        return obs

    def _gabor(self, orientation, h, w):
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        d = np.sqrt(x * x + y * y)
        sigma, th, Lambda, psi, gamma = 0.5, np.deg2rad(orientation), 0.3, 0, 1
        xt = x * np.cos(th) + y * np.sin(th)
        yt = -x * np.sin(th) + y * np.cos(th)
        g = np.exp(-0.5 * (xt ** 2 + yt ** 2 / gamma ** 2) / sigma ** 2) * np.cos(2 * np.pi * xt / Lambda + psi)
        noise = np.random.uniform(-0.11, 0.11, size=g.shape)
        g[d > 0.5] = 0.0; noise[d > 0.5] = 0.0
        return (g + noise).astype(np.float32)

    # ── reward (overridable hooks) ────────────────────────────────────────────
    def _reward_value(self) -> float:
        """Reward for a correct response on this trial (× reward_scale)."""
        base = float(self.color_values.get(self.cue_color, 1)) if self.value_cues else 1.0
        return self.reward_scale * base

    def _is_reportable_change(self) -> bool:
        """Whether the change on this trial should be reported (True for plain
        detection; subclasses restrict it, e.g. Krauzlis = cued only)."""
        return self.change_true == 1

    def step(self, action):
        t_before = int(self.t)
        L_before = t_before // self.frame_repeat                 # LOGICAL frame the action was taken on
        self.t += 1
        reward, done = 0.0, False
        obs = self._next_observation()
        if action == 1:
            done = True
            if L_before >= self.change_time and self._is_reportable_change():
                reward = self._reward_value()
            else:
                reward = 0.0                                # premature / false alarm
        if self.t >= self.T and not done:
            done = True
            if not self._is_reportable_change():            # correct rejection: no REPORTABLE change to declare —
                reward = self._reward_value()               # true no-change, OR a distractor to ignore (Krauzlis).
                # NB: for every task that does NOT restrict reportability this is exactly `change_true == 0`
                # (unchanged). It only differs for attend-here/ignore-here (Krauzlis), where correctly WAITING
                # through a distractor change is now a rewarded correct-rejection instead of scoring 0 (a miss).
        if done and self.curriculum:
            self._update_curriculum(reward > 0)
        return obs, reward, done, {"theta": self.theta, "correct": float(reward > 0)}

    def _update_curriculum(self, correct: bool) -> None:
        """Paper curriculum: once the agent averages >= curr_threshold (85%) over a non-overlapping
        window of curr_window (1000) trials, drop the max orientation change |Δ| by curr_step (3 deg),
        making detection progressively harder. Δ ~ U(-theta, theta); theta starts at 65 and floors at
        theta_floor. (Set curr_step<=0 to fall back to the legacy multiplicative decay.)"""
        self._recent_correct.append(bool(correct))
        if len(self._recent_correct) >= self.curr_window:
            acc = sum(self._recent_correct) / len(self._recent_correct)
            if acc >= self.curr_threshold:
                if self.curr_step > 0:
                    self.theta = max(self.theta_floor, self.theta - self.curr_step)
                else:
                    self.theta = max(self.theta_floor, self.theta * self.curr_decay)
            self._recent_correct = []

    def training_state_dict(self) -> dict:
        """Return the curriculum state needed for an exact training resume."""
        return {
            "schema_version": 1,
            "theta": float(self.theta),
            "recent_correct": [bool(value) for value in self._recent_correct],
            "curriculum": bool(self.curriculum),
            "theta_start": float(self.theta_start),
            "theta_floor": float(self.theta_floor),
            "curr_window": int(self.curr_window),
            "curr_threshold": float(self.curr_threshold),
            "curr_step": float(self.curr_step),
            "curr_decay": float(self.curr_decay),
            "environment_config": {
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "grid_rows": self.grid_rows, "grid_cols": self.grid_cols,
                "cue_positions": list(self.cue_positions),
                "noise_multiplier": self.noise_multiplier, "reward_scale": self.reward_scale,
                "min_change_time": self.min_change_time, "max_change_time": self.max_change_time,
                "frame_repeat": self.frame_repeat, "T": self.T, "image_size": self.S,
                "proportions": list(self.proportions), "value_cues": self.value_cues,
                "color_values": dict(self.color_values),
            },
        }

    def load_training_state_dict(self, state: dict) -> None:
        """Restore curriculum progress while rejecting an incompatible environment."""
        if state.get("schema_version") != 1:
            raise ValueError("unsupported environment checkpoint schema")
        if state.get("environment_config") != self.training_state_dict()["environment_config"]:
            raise ValueError("checkpoint environment configuration mismatch")
        expected = {
            "curriculum": bool(self.curriculum),
            "theta_start": float(self.theta_start),
            "theta_floor": float(self.theta_floor),
            "curr_window": int(self.curr_window),
            "curr_threshold": float(self.curr_threshold),
            "curr_step": float(self.curr_step),
            "curr_decay": float(self.curr_decay),
        }
        mismatches = [key for key, value in expected.items() if state.get(key) != value]
        if mismatches:
            raise ValueError(f"checkpoint curriculum configuration mismatch: {mismatches}")
        recent = state.get("recent_correct", [])
        if len(recent) >= self.curr_window:
            raise ValueError("checkpoint recent_correct must be shorter than curr_window")
        theta = float(state["theta"])
        if not math.isfinite(theta):
            raise ValueError("checkpoint theta must be finite")
        if not self.theta_floor <= theta <= self.theta_start:
            raise ValueError("checkpoint theta is outside configured curriculum bounds")
        self.theta = theta
        self._recent_correct = [bool(value) for value in recent]

    def render(self):
        return None

    def close(self):
        pass
