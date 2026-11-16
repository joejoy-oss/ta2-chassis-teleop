"""Tests for Runtime's loop timing against a fake clock: steps are due at fixed times, so what a sleep oversleeps does
not add up, and a step that overruns its period restarts the schedule without a burst of catch-up steps."""

import pytest

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import runtime as _runtime

T0 = 1000.0


class _Clock:
    """Stands in for the time module inside runtime.py: sleep(s) moves the clock by s plus `oversleep`, as a real sleep
    returns late."""

    def __init__(self, oversleep: float = 0.0):
        self.now = T0
        self.oversleep = oversleep
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds + self.oversleep


class _Environment(_environment.Environment):
    """Notes the clock when each step starts (get_observation); step k (counted over the whole run) takes
    durations.get(k, default) seconds."""

    def __init__(self, clock: _Clock, durations: dict[int, float], default: float):
        self._clock = clock
        self._durations = durations
        self._default = default
        self.starts: list[float] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> dict:
        self.starts.append(self._clock.now)
        self._clock.now += self._durations.get(len(self.starts) - 1, self._default)
        return {}

    def apply_action(self, action: dict) -> None:
        pass


class _Agent(_agent.Agent):
    def get_action(self, observation: dict) -> dict:
        return {}

    def reset(self) -> None:
        pass


def _run(monkeypatch, *, max_hz, steps, episodes=1, oversleep=0.0, durations=None, default=0.005):
    clock = _Clock(oversleep)
    monkeypatch.setattr(_runtime, "time", clock)
    environment = _Environment(clock, durations or {}, default)
    runtime = _runtime.Runtime(environment, _Agent(), [], max_hz=max_hz, num_episodes=episodes, max_episode_steps=steps)
    runtime.run()
    return clock, environment


def test_steps_are_due_at_fixed_times_so_oversleeping_does_not_add_up(monkeypatch):
    period = 1 / 50
    clock, environment = _run(monkeypatch, max_hz=50, steps=600, oversleep=0.001)
    starts = environment.starts
    assert len(starts) == 601
    assert starts[0] == T0
    # Every step after the first starts one oversleep after its due time, however many came before it.
    for k in range(1, len(starts)):
        assert starts[k] - (T0 + k * period) == pytest.approx(0.001, abs=1e-9)
    # Sleeping a full period after each step would have drifted by 600 oversleeps.
    assert starts[-1] - starts[0] == pytest.approx(600 * period + 0.001, abs=1e-9)
    assert len(clock.sleeps) == 601


def test_an_overrun_restarts_the_schedule_without_a_burst(monkeypatch):
    period = 1 / 50
    clock, environment = _run(monkeypatch, max_hz=50, steps=10, durations={3: 2.5 * period})
    gaps = [b - a for a, b in zip(environment.starts, environment.starts[1:])]
    # The step after the long one starts as soon as it ends, the ones after that a period apart again: no steps run
    # back to back to make up for the lost time.
    assert gaps == pytest.approx([period] * 3 + [2.5 * period] + [period] * 6, abs=1e-9)
    assert len(clock.sleeps) == 10  # no sleep after the overrun


def test_each_episode_runs_max_episode_steps_plus_one_steps_and_hz_0_never_sleeps(monkeypatch):
    clock, environment = _run(monkeypatch, max_hz=0, steps=5, episodes=2)
    assert len(environment.starts) == 2 * 6
    assert environment.resets == 3  # one per episode and the final one
    assert clock.sleeps == []
