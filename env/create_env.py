import time
import re
from copy import deepcopy

import numpy as np
from pogema import AnimationConfig, AnimationMonitor, pogema_v0, GridConfig
import gymnasium

from follower.training_config import Environment
from env.custom_maps import MAPS_REGISTRY
from follower.preprocessing import wrap_preprocessors, PreprocessorConfig


class ProvideGlobalObstacles(gymnasium.Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()


def create_env_base(config: Environment):
    env = pogema_v0(grid_config=config.grid_config)
    env = ProvideGlobalObstacles(env)
    if config.use_maps:
        env = MultiMapWrapper(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(directory='renders', egocentric_idx=None))

    env = MetricsWrapper(env)

    return env


class MetricsWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None
        self._env_step_time = None
        self._num_agents: int = 0
        self._step: int = 0
        self._total_targets_reached: int = 0
        self._agent_ever_reached: set = set()
        self._collisions: int = 0
        self._makespan: int = 0
        self._all_done: bool = False
        self._total_step_time: float = 0.0
        self._step_count: int = 0
        self._prev_rewards = None

    def step(self, actions):
        env_step_start = time.monotonic()
        t_start = time.perf_counter()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        step_time = time.perf_counter() - t_start
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start

        self._step += 1
        self._total_step_time += step_time
        self._step_count += 1

        for i, reward in enumerate(rewards):
            if reward > 0:
                self._total_targets_reached += 1
                self._agent_ever_reached.add(i)

        positions = self._get_positions()
        self._collisions += self._count_collisions(positions)

        if not self._all_done:
            if all(terminated):
                self._makespan = self._step
                self._all_done = True
            elif all(truncated):
                self._makespan = self._step
                self._all_done = True

        if all(terminated) or all(truncated):
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)

        throughput = self._total_targets_reached / max(self._step, 1)
        success_rate = len(self._agent_ever_reached) / max(self._num_agents, 1)
        avg_planner_time = (self._total_step_time / self._step_count if self._step_count > 0 else 0.0)

        metrics = {
            "throughput": round(throughput, 4),
            "success_rate": round(success_rate, 4),
            "num_collisions": self._collisions,
            "makespan": self._makespan,
            "planner_time": round(avg_planner_time, 6),
        }

        if infos:
            existing = infos[0].get('metrics', {})
            existing.update(metrics)
            infos[0]['metrics'] = existing

        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._start_time = time.monotonic()
        self._env_step_time = 0.0

        grid_config = self.unwrapped.grid_config
        self._num_agents = grid_config.num_agents
        self._step = 0
        self._total_targets_reached = 0
        self._agent_ever_reached = set()
        self._collisions = 0
        self._makespan = 0
        self._all_done = False
        self._total_step_time = 0.0
        self._step_count = 0
        self._prev_rewards = [0.0] * self._num_agents
        return obs, info

    def _get_positions(self):
        try:
            raw_positions = self.unwrapped.grid.get_agents_xy()
            return [tuple(pos) for pos in raw_positions]
        except Exception:
            return []

    def _count_collisions(self, positions):
        count = 0
        n = len(positions)

        for i in range(n):
            for j in range(i + 1, n):
                if positions[i] == positions[j]:
                    count += 1

        return count


class MultiMapWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            for map_name in sorted(MAPS_REGISTRY):
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def reset(self, seed=None, **kwargs):
        self._rnd = np.random.default_rng(seed)
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs))
            cfg = deepcopy(self._configs[map_idx])
            self.env.unwrapped.grid_config = cfg
            self.env.unwrapped.grid_config.seed = seed
        return self.env.reset(seed=seed, **kwargs)


def main():
    env = create_env_base(config=Environment())
    env = wrap_preprocessors(env, config=PreprocessorConfig())
    env.reset()
    env.render()


if __name__ == '__main__':
    main()
