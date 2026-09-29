"""
Stage 2 MAPPO Training
======================

Three heterogeneous learning agents:

    agent_0 -> aggressive
    agent_1 -> neutral
    agent_2 -> conservative

Each agent has:
    local observation: 47 dimensions
    action space: 4 discrete actions

Centralized critic:
    joint observation: 3 * 47 = 141 dimensions

Actors are decentralized:
    each actor sees only its own 47-dimensional observation.

Stage-1 selected PPO checkpoints are used to warm-start the three
Stage-2 decentralized actors through a local 47-D -> Stage-1 40-D
observation adapter. The centralized Stage-2 critic still uses the full
141-D joint Stage-2 observation. No communication or ToM is added.

Run from project root:

    python src/stage2/training/train_mappo.py

Short smoke training:

    python src/stage2/training/train_mappo.py --total-updates 5

Normal training:

    python src/stage2/training/train_mappo.py --total-updates 1000

CPU:

    python src/stage2/training/train_mappo.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()

# .../emergent_coordination/src/stage2/training/train_mappo.py
PROJECT_ROOT = THIS_FILE.parents[3]
STAGE2_DIR = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))


# ============================================================
# STAGE 2 ENVIRONMENT
# ============================================================

from intersection_env import (
    SELECTED_MODELS,
    Stage2Config,
    Stage2IntersectionEnv,
)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class MAPPOConfig:
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed: int = 42

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    total_updates: int = 1000
    rollout_steps: int = 2048

    # --------------------------------------------------------
    # PPO / GAE
    # --------------------------------------------------------

    gamma: float = 0.995
    gae_lambda: float = 0.95

    clip_coef: float = 0.20
    value_clip_coef: float | None = None

    entropy_coef: float = 0.02
    value_coef: float = 0.50

    max_grad_norm: float = 0.50

    update_epochs: int = 6
    minibatch_size: int = 256

    # --------------------------------------------------------
    # Learning rates
    # --------------------------------------------------------

    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    hidden_dim: int = 128

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    normalize_observations: bool = True
    normalize_advantages: bool = True

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    background_traffic: bool = True

    # Stage-1 behavioral warm start.
    stage1_warm_start: bool = True
    stage1_checkpoint_dir: str = "models/stage1_selected"

    # Curriculum: first learn the three-agent intersection without
    # scripted traffic, then continue from the same policies with
    # the full Stage-2 traffic distribution.
    warmup_updates: int = 100

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_interval: int = 25
    eval_episodes: int = 20
    final_eval_episodes: int = 20

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_interval: int = 50

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir: str = "models/stage2_mappo"


# ============================================================
# RANDOM SEEDS
# ============================================================


def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# RUNNING MEAN / STANDARD DEVIATION
# ============================================================


class RunningMeanStd:

    def __init__(self, dimension: int):

        self.mean = np.zeros(
            dimension,
            dtype=np.float64,
        )

        self.var = np.ones(
            dimension,
            dtype=np.float64,
        )

        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if x.ndim == 1:
            x = x[None, :]

        batch_mean = np.mean(
            x,
            axis=0,
        )

        batch_var = np.var(
            x,
            axis=0,
        )

        batch_count = x.shape[0]

        delta = (
            batch_mean
            - self.mean
        )

        total_count = (
            self.count
            + batch_count
        )

        new_mean = (
            self.mean
            + delta
            * batch_count
            / total_count
        )

        mean_a = (
            self.var
            * self.count
        )

        mean_b = (
            batch_var
            * batch_count
        )

        correction = (
            delta
            * delta
            * self.count
            * batch_count
            / total_count
        )

        new_var = (
            mean_a
            + mean_b
            + correction
        ) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)

        self.count = total_count

    def normalize(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        normalized = (
            x - self.mean
        ) / np.sqrt(
            self.var + 1e-8
        )

        return normalized.astype(
            np.float32
        )

    def state_dict(self) -> dict:

        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:

        self.mean = np.asarray(
            state["mean"],
            dtype=np.float64,
        )

        self.var = np.asarray(
            state["var"],
            dtype=np.float64,
        )

        self.count = float(
            state["count"]
        )


# ============================================================
# ACTOR
# ============================================================


class Actor(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                observation_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(
            observations
        )

    def distribution(
        self,
        observations: torch.Tensor,
    ) -> Categorical:

        logits = self.forward(
            observations
        )

        return Categorical(
            logits=logits
        )


# ============================================================
# CENTRALIZED CRITIC
# ============================================================


class CentralizedCritic(nn.Module):

    def __init__(
        self,
        joint_observation_dim: int,
        hidden_dim: int,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                joint_observation_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                3,
            ),
        )

    def forward(
        self,
        joint_observation: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(
            joint_observation
        )


# ============================================================
# ROLLOUT BUFFER
# ============================================================


class RolloutBuffer:

    def __init__(
        self,
        rollout_steps: int,
        num_agents: int,
        observation_dim: int,
        actor_observation_dim: int | None = None,
    ):

        self.rollout_steps = rollout_steps

        self.num_agents = num_agents

        self.observation_dim = (
            observation_dim
        )
        self.actor_observation_dim = (
            actor_observation_dim
            if actor_observation_dim is not None
            else observation_dim
        )

        self.local_observations = np.zeros(
            (
                rollout_steps,
                num_agents,
                self.actor_observation_dim,
            ),
            dtype=np.float32,
        )

        self.joint_observations = np.zeros(
            (
                rollout_steps,
                num_agents
                * observation_dim,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (
                rollout_steps,
                num_agents,
            ),
            dtype=np.int64,
        )

        self.log_probabilities = np.zeros(
            (
                rollout_steps,
                num_agents,
            ),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (
                rollout_steps,
                num_agents,
            ),
            dtype=np.float32,
        )

        # Centralized value per agent: V_i(global state).
        self.values = np.zeros(
            (rollout_steps, num_agents),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            rollout_steps,
            dtype=np.float32,
        )
        self.timeouts = np.zeros(
            rollout_steps,
            dtype=np.float32,
        )
        self.timeout_values = np.zeros(
            (rollout_steps, num_agents),
            dtype=np.float32,
        )

        # Storage is time-major: [t, agent, ...].
        self.position = 0

    def add(
        self,
        local_observations,
        joint_observation,
        actions,
        log_probabilities,
        rewards,
        value,
        done,
        timeout=False,
        timeout_value=None,
    ) -> None:

        index = self.position

        self.local_observations[
            index
        ] = local_observations

        self.joint_observations[
            index
        ] = joint_observation

        self.actions[
            index
        ] = actions

        self.log_probabilities[
            index
        ] = log_probabilities

        self.rewards[
            index
        ] = rewards

        self.values[
            index
        ] = value

        self.dones[
            index
        ] = float(done)
        self.timeouts[index] = float(timeout)
        if timeout_value is not None:
            self.timeout_values[index] = timeout_value

        self.position += 1

    def compute_gae(
        self,
        last_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ):

        # Per-agent GAE with a centralized multi-head critic.
        advantages = np.zeros(
            (self.rollout_steps, self.values.shape[1]),
            dtype=np.float32,
        )

        last_gae = np.zeros(
            self.values.shape[1],
            dtype=np.float32,
        )

        for step in reversed(range(self.rollout_steps)):
            next_nonterminal = 1.0 - self.dones[step]
            bootstrap_nonterminal = (
                1.0
                if self.dones[step] < 0.5 or self.timeouts[step] > 0.5
                else 0.0
            )
            if self.dones[step] > 0.5:
                next_value = (
                    self.timeout_values[step]
                    if self.timeouts[step] > 0.5
                    else np.zeros(self.values.shape[1], dtype=np.float32)
                )
            elif step == self.rollout_steps - 1:
                next_value = np.asarray(last_value, dtype=np.float32)
            else:
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * bootstrap_nonterminal
                - self.values[step]
            )

            last_gae = (
                delta
                + gamma
                * gae_lambda
                * next_nonterminal
                * last_gae
            )

            advantages[step] = last_gae

        returns = advantages + self.values
        return advantages, returns


# ============================================================
# MAPPO TRAINER
# ============================================================


class MAPPOTrainer:

    def __init__(
        self,
        config: MAPPOConfig,
    ):

        self.config = config

        set_seed(
            config.seed
        )

        if (
            config.device
            == "cuda"
            and not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        self.device = torch.device(
            config.device
        )

        # ----------------------------------------------------
        # Agents
        # ----------------------------------------------------

        self.agent_ids = [
            "agent_0",
            "agent_1",
            "agent_2",
        ]

        self.num_agents = 3

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        self.env = Stage2IntersectionEnv(
            seed=config.seed,
            config=Stage2Config(),
            background_traffic=(
                config.background_traffic
            ),
        )

        self.observation_dim = (
            self.env.observation_dim
        )

        self.action_dim = (
            self.env.action_dim
        )

        self.actor_observation_dim = 40

        self.joint_observation_dim = (
            self.num_agents
            * self.observation_dim
        )

        # ----------------------------------------------------
        # Actors
        # ----------------------------------------------------

        self.actors = nn.ModuleDict({

            agent_id: Actor(
                observation_dim=(
                    self.actor_observation_dim
                ),
                action_dim=(
                    self.action_dim
                ),
                hidden_dim=(
                    config.hidden_dim
                ),
            )

            for agent_id in self.agent_ids
        })

        self.actors.to(
            self.device
        )

        self.stage1_observation_stats = {}
        self.stage1_checkpoint_paths = {}
        if config.stage1_warm_start:
            self._load_stage1_warm_start()

        # ----------------------------------------------------
        # Centralized critic
        # ----------------------------------------------------

        self.critic = CentralizedCritic(
            joint_observation_dim=(
                self.joint_observation_dim
            ),
            hidden_dim=(
                config.hidden_dim
            ),
        ).to(
            self.device
        )

        # ----------------------------------------------------
        # Optimizers
        # ----------------------------------------------------

        self.actor_optimizers = {

            agent_id: torch.optim.Adam(
                self.actors[
                    agent_id
                ].parameters(),

                lr=(
                    config.actor_learning_rate
                ),

                eps=1e-5,
            )

            for agent_id
            in self.agent_ids
        }

        self.critic_optimizer = (
            torch.optim.Adam(
                self.critic.parameters(),

                lr=(
                    config.critic_learning_rate
                ),

                eps=1e-5,
            )
        )

        # ----------------------------------------------------
        # Observation normalization
        # ----------------------------------------------------

        self.observation_rms = {

            agent_id: RunningMeanStd(
                self.observation_dim
            )

            for agent_id
            in self.agent_ids
        }

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output_dir = (
            PROJECT_ROOT
            / config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.phase_name = ""
        self.phase_best_paths = {}

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        self.total_environment_steps = 0
        self._rollout_observations = None
        self._episode_returns = np.zeros(
            self.num_agents,
            dtype=np.float64,
        )
        self._episode_length = 0

        self.best_evaluation_score = (
            -float("inf")
        )

        self.env = self._build_environment(
            background_traffic=(
                config.background_traffic
            )
        )

        # ----------------------------------------------------
        # Save configuration immediately
        # ----------------------------------------------------

        self.save_training_config()

    def _resolve_stage1_checkpoint(self, filename: str) -> Path:
        candidates = [
            PROJECT_ROOT / self.config.stage1_checkpoint_dir / filename,
            PROJECT_ROOT / filename,
            PROJECT_ROOT / "models" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {filename}\nSearched:\n{searched}"
        )

    def _load_stage1_warm_start(self) -> None:
        behavior_by_agent = {
            "agent_0": "aggressive",
            "agent_1": "neutral",
            "agent_2": "conservative",
        }
        for agent_id, behavior in behavior_by_agent.items():
            filename = SELECTED_MODELS[agent_id]["checkpoint"]
            checkpoint_path = self._resolve_stage1_checkpoint(filename)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if int(checkpoint.get("obs_dim", -1)) != 40:
                raise ValueError(f"{filename}: expected obs_dim=40")
            if int(checkpoint.get("action_dim", -1)) != self.action_dim:
                raise ValueError(f"{filename}: action_dim mismatch")
            if checkpoint.get("behavior") != behavior:
                raise ValueError(f"{filename}: behavior identity mismatch")
            state = checkpoint["model_state_dict"]
            actor_state = {
                "network.0.weight": state["shared.0.weight"],
                "network.0.bias": state["shared.0.bias"],
                "network.2.weight": state["shared.2.weight"],
                "network.2.bias": state["shared.2.bias"],
                "network.4.weight": state["actor.weight"],
                "network.4.bias": state["actor.bias"],
            }
            self.actors[agent_id].load_state_dict(actor_state, strict=True)
            self.stage1_observation_stats[agent_id] = {
                "mean": np.asarray(checkpoint["obs_mean"], dtype=np.float64),
                "var": np.asarray(checkpoint["obs_var"], dtype=np.float64),
                "count": float(checkpoint["obs_count"]),
            }
            self.stage1_checkpoint_paths[agent_id] = str(checkpoint_path)

    @staticmethod
    def _cardinal_direction_value(sin_heading: float, cos_heading: float) -> float:
        angle = math.atan2(float(sin_heading), float(cos_heading))
        directions = [
            (0.0, -math.pi / 2.0),
            (0.33, math.pi / 2.0),
            (0.66, math.pi),
            (1.0, 0.0),
        ]
        return float(min(
            directions,
            key=lambda item: abs(math.atan2(
                math.sin(angle - item[1]),
                math.cos(angle - item[1]),
            )),
        )[0])

    @staticmethod
    def _stage1_normalize(values: np.ndarray, stats: dict) -> np.ndarray:
        return ((values - stats["mean"]) / np.sqrt(stats["var"] + 1e-8)).astype(np.float32)

    def _project_stage2_to_stage1(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (47,):
            raise ValueError(f"Expected Stage-2 observation shape (47,), got {obs.shape}")

        x, y, speed, acceleration = obs[0:4]
        sin_heading, cos_heading = obs[4:6]
        maneuver = obs[8:11]
        result = [float(x), float(y), float(speed), float(acceleration)]
        # Stage-2 deliberately keeps local observations compact; lane is not
        # exposed, so do not leak environment state into the actor.
        result.append(0.0)
        result.append(self._cardinal_direction_value(sin_heading, cos_heading))
        result.append(0.5 if maneuver[1] > 0.5 else (1.0 if maneuver[2] > 0.5 else 0.0))

        others = []
        for start in (11, 19):
            dx, dy, rel_speed, distance = obs[start:start + 4]
            ego_ttc = obs[start + 4]
            others.append((float(distance), float(dx), float(dy), float(rel_speed), 0.0, float(ego_ttc)))
        for start in (27, 32, 37, 42):
            dx, dy, rel_speed, distance, ttc = obs[start:start + 5]
            others.append((float(distance), float(dx), float(dy), float(rel_speed), 0.0, float(ttc)))

        min_distance = min((item[0] for item in others), default=1.0)
        min_ttc = min((item[5] for item in others), default=1.0)
        result.extend([
            float(np.clip(min_distance, 0.0, 1.0)),
            float(np.clip(min_ttc, 0.0, 1.0)),
            float(np.clip(math.sqrt(float(x) ** 2 + float(y) ** 2) / 120.0, 0.0, 1.0)),
        ])

        others.sort(key=lambda item: item[0])
        for distance, dx, dy, rel_speed, lane, tti in others[:5]:
            result.extend([
                dx,
                dy,
                rel_speed,
                float(np.clip(distance, 0.0, 1.0)),
                lane,
                float(np.clip(tti, 0.0, 1.0)),
            ])
        while len(result) < 40:
            result.append(0.0)
        return np.asarray(result[:40], dtype=np.float32)

    def prepare_actor_observations(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        processed = []
        for agent_id in self.agent_ids:
            projected = self._project_stage2_to_stage1(observations[agent_id])
            processed.append(self._stage1_normalize(projected, self.stage1_observation_stats[agent_id]))
        return np.stack(processed, axis=0).astype(np.float32)

    def _build_environment(
        self,
        background_traffic: bool,
    ) -> Stage2IntersectionEnv:

        return Stage2IntersectionEnv(
            seed=self.config.seed,
            config=Stage2Config(),
            background_traffic=background_traffic,
        )

    def set_training_environment(self) -> None:

        self.phase_name = ""
        self.config.background_traffic = True
        self.env = self._build_environment(background_traffic=True)
        self.total_environment_steps = 0
        self._rollout_observations = None
        self._episode_returns = np.zeros(
            self.num_agents,
            dtype=np.float64,
        )
        self._episode_length = 0
        self.best_evaluation_score = -float("inf")
        self.save_training_config()

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def save_training_config(
        self,
    ) -> None:

        configuration = {

            "mappo": asdict(
                self.config
            ),

            "environment": asdict(
                self.env.config
            ),

            "background_traffic": bool(
                self.env.background_traffic_enabled
            ),

            "phase_name": self.phase_name,

            "agents": SELECTED_MODELS,

            "stage1_warm_start": bool(self.config.stage1_warm_start),
            "stage1_checkpoint_paths": self.stage1_checkpoint_paths,
            "actor_observation_dim": self.actor_observation_dim,

            "observation_dim": (
                self.observation_dim
            ),

            "joint_observation_dim": (
                self.joint_observation_dim
            ),

            "action_dim": (
                self.action_dim
            ),
        }

        path = (
            self.output_dir
            / "training_config.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                configuration,
                file,
                indent=2,
            )

    # ========================================================
    # OBSERVATION NORMALIZATION
    # ========================================================

    def prepare_observations(
        self,
        observations: Dict[str, np.ndarray],
        update_statistics: bool,
    ) -> np.ndarray:

        processed = []
        for agent_id in self.agent_ids:
            observation = np.asarray(observations[agent_id], dtype=np.float32)
            if update_statistics:
                self.observation_rms[agent_id].update(observation)
            if self.config.normalize_observations:
                observation = self.observation_rms[agent_id].normalize(observation)
            processed.append(observation)

        return np.stack(processed, axis=0).astype(np.float32)

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    @torch.no_grad()
    def select_actions(
        self,
        observations,
        deterministic=False,
        update_statistics=False,
    ):
        local_observations = self.prepare_observations(
            observations,
            update_statistics,
        )
        actor_observations = self.prepare_actor_observations(observations)
        actor_tensor = torch.as_tensor(
            actor_observations,
            dtype=torch.float32,
            device=self.device,
        )

        actions = []
        log_probabilities = []
        for index, agent_id in enumerate(self.agent_ids):
            agent_observation = actor_tensor[index:index + 1]
            distribution = self.actors[agent_id].distribution(agent_observation)
            action = (
                torch.argmax(distribution.logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probability = distribution.log_prob(action)
            actions.append(int(action.item()))
            log_probabilities.append(float(log_probability.item()))

        joint_observation = local_observations.reshape(-1).astype(np.float32)
        joint_tensor = torch.as_tensor(
            joint_observation[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        critic_value = (
            self.critic(joint_tensor)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        action_dict = {
            agent_id: actions[index]
            for index, agent_id in enumerate(self.agent_ids)
        }
        return (
            action_dict,
            np.asarray(actions, dtype=np.int64),
            np.asarray(log_probabilities, dtype=np.float32),
            critic_value,
            actor_observations,
            joint_observation,
        )

    # ========================================================
    # ROLLOUT COLLECTION
    # ========================================================

    def collect_rollout(
        self,
        update_index: int,
    ):

        buffer = RolloutBuffer(

            rollout_steps=(
                self.config.rollout_steps
            ),

            num_agents=(
                self.num_agents
            ),

            observation_dim=(
                self.observation_dim
            ),
            actor_observation_dim=(
                self.actor_observation_dim
            ),
        )

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        if self._rollout_observations is None:
            self._rollout_observations = self.env.reset(
                seed=self.config.seed + update_index
            )
            self._episode_returns = np.zeros(
                self.num_agents,
                dtype=np.float64,
            )
            self._episode_length = 0

        observations = self._rollout_observations
        episode_returns = self._episode_returns
        episode_length = self._episode_length

        completed_episodes = []

        # ----------------------------------------------------
        # Collect transitions
        # ----------------------------------------------------

        for _ in range(
            self.config.rollout_steps
        ):

            (
                action_dict,
                action_array,
                log_probabilities,
                critic_value,
                actor_observations,
                joint_observation,
            ) = self.select_actions(

                observations,

                # PPO/MAPPO training must sample stochastically.
                deterministic=False,

                update_statistics=True,
            )
            (
                next_observations,
                rewards,
                terminated,
                truncated,
                infos,
            ) = self.env.step(
                action_dict
            )

            reward_array = np.asarray(

                [
                    rewards[agent_id]

                    for agent_id
                    in self.agent_ids
                ],

                dtype=np.float32,
            )

            done = (
                terminated
                or truncated
            )

            timeout = truncated and not terminated
            timeout_value = None
            if timeout:
                timeout_local_observations = self.prepare_observations(
                    next_observations,
                    update_statistics=False,
                )
                timeout_joint_observation = torch.as_tensor(
                    timeout_local_observations.reshape(-1)[None, :],
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    timeout_value = (
                        self.critic(timeout_joint_observation)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

            buffer.add(

                actor_observations,

                joint_observation,

                action_array,

                log_probabilities,

                reward_array,

                critic_value,

                done,

                timeout,

                timeout_value,
            )

            episode_returns += (
                reward_array
            )

            episode_length += 1

            self.total_environment_steps += 1

            observations = (
                next_observations
            )

            # ------------------------------------------------
            # Episode finished
            # ------------------------------------------------

            if done:

                global_info = infos.get(
                    "global",
                    {},
                )

                completed_episodes.append({

                    "returns": (
                        episode_returns.copy()
                    ),

                    "length": (
                        episode_length
                    ),

                    "agent_collisions": (
                        global_info.get(
                            "agent_collisions",
                            0,
                        )
                    ),

                    "background_collisions": (
                        global_info.get(
                            "traffic_collisions",
                            0,
                        )
                    ),

                    "near_collisions": (
                        global_info.get(
                            "near_collisions",
                            0,
                        )
                    ),

                    "minimum_distance": (
                        global_info.get(
                            "minimum_distance",
                            float("inf"),
                        )
                    ),

                    "minimum_ttc": (
                        global_info.get(
                            "minimum_ttc",
                            float("inf"),
                        )
                    ),

                    "all_completed": (
                        global_info.get(
                            "all_completed",
                            False,
                        )
                    ),
                })

                observations = (
                    self.env.reset(
                        seed=(
                            self.config.seed
                            + self.total_environment_steps
                        )
                    )
                )

                episode_returns = (
                    np.zeros(
                        self.num_agents,
                        dtype=np.float64,
                    )
                )

                episode_length = 0

        self._rollout_observations = observations
        self._episode_returns = episode_returns
        self._episode_length = episode_length

        # ----------------------------------------------------
        # Bootstrap value
        # ----------------------------------------------------

        with torch.no_grad():

            final_local_observations = (
                self.prepare_observations(
                    observations,
                    update_statistics=False,
                )
            )

            final_joint_observation = (
                final_local_observations
                .reshape(-1)
                .astype(np.float32)
            )

            final_tensor = torch.as_tensor(

                final_joint_observation[
                    None,
                    :
                ],

                dtype=torch.float32,

                device=self.device,
            )

            last_value = (
                self.critic(final_tensor)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        advantages, returns = (
            buffer.compute_gae(

                last_value,

                self.config.gamma,

                self.config.gae_lambda,
            )
        )

        assert advantages.shape == (
            self.config.rollout_steps,
            self.num_agents,
        )

        assert returns.shape == (
            self.config.rollout_steps,
            self.num_agents,
        )

        return (
            buffer,
            advantages,
            returns,
            completed_episodes,
            episode_length,
        )

    # ========================================================
    # PPO / MAPPO UPDATE
    # ========================================================

    def update(
        self,
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
    ):

        rollout_steps = buffer.rollout_steps

        # ----------------------------------------------------
        # Flatten local observations
        # ----------------------------------------------------

        # Buffer storage is time-major: [time, agent, ...]. Each decentralized actor needs its
        # own contiguous [time, ...] training stream.
        local_observations = torch.as_tensor(
            buffer.local_observations
            .transpose(1, 0, 2)
            .reshape(
                -1,
                self.actor_observation_dim,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Joint observations
        # ----------------------------------------------------

        joint_observations = torch.as_tensor(
            buffer.joint_observations,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        actions = torch.as_tensor(
            buffer.actions
            .transpose(1, 0)
            .reshape(-1),
            dtype=torch.long,
            device=self.device,
        )

        # ----------------------------------------------------
        # Old log probabilities
        # ----------------------------------------------------

        old_log_probabilities = torch.as_tensor(
            buffer.log_probabilities
            .transpose(1, 0)
            .reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Per-agent advantages / returns
        # ----------------------------------------------------

        advantages_tensor = torch.as_tensor(
            advantages.transpose(1, 0).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Normalize advantages
        # ----------------------------------------------------

        if (
            self.config
            .normalize_advantages
        ):
            advantages_by_agent = advantages_tensor.reshape(
                self.num_agents,
                rollout_steps,
            )
            advantages_by_agent = (
                advantages_by_agent
                - advantages_by_agent.mean(dim=1, keepdim=True)
            ) / (
                advantages_by_agent.std(dim=1, keepdim=True, unbiased=False)
                + 1e-8
            )
            advantages_tensor = advantages_by_agent.reshape(-1)
        metrics = {

            "actor_loss": 0.0,

            "critic_loss": 0.0,

            "entropy": 0.0,

            "approx_kl": 0.0,

            "clip_fraction": 0.0,
        }

        actor_metric_count = 0

        critic_metric_count = 0

        # ----------------------------------------------------
        # Old critic values
        # ----------------------------------------------------

        with torch.no_grad():

            old_values = (
                self.critic(
                    joint_observations
                )
            )

        # ====================================================
        # PPO EPOCHS
        # ====================================================

        for _ in range(
            self.config.update_epochs
        ):

            permutation = (
                torch.randperm(
                    rollout_steps,
                    device=self.device,
                )
            )

            # ------------------------------------------------
            # Minibatches
            # ------------------------------------------------

            for start in range(

                0,

                rollout_steps,

                self.config.minibatch_size,
            ):

                indices = permutation[

                    start:
                    start
                    + self.config.minibatch_size
                ]

                # =================================================
                # CENTRALIZED CRITIC UPDATE
                # =================================================

                minibatch_joint = (
                    joint_observations[
                        indices
                    ]
                )

                # Per-agent return for each centralized value head.
                minibatch_returns = (
                    returns_tensor[
                        indices
                    ]
                )

                old_value = (
                    old_values[
                        indices
                    ]
                )

                new_value = (
                    self.critic(
                        minibatch_joint
                    )
                )

                value_loss_unclipped = (
                    new_value
                    - minibatch_returns
                ).pow(2)

                # Match the proven Stage-1 PPO objective by default: plain
                # squared value error. A value clip can still be enabled
                # explicitly for ablations.
                if self.config.value_clip_coef is None:
                    value_loss = 0.5 * value_loss_unclipped.mean()
                else:
                    clipped_value = (
                        old_value
                        + torch.clamp(
                            new_value - old_value,
                            -self.config.value_clip_coef,
                            self.config.value_clip_coef,
                        )
                    )
                    value_loss_clipped = (
                        clipped_value
                        - minibatch_returns
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    )

                critic_loss = (

                    self.config.value_coef
                    * value_loss
                )

                self.critic_optimizer.zero_grad(
                    set_to_none=True
                )

                critic_loss.backward()

                nn.utils.clip_grad_norm_(

                    self.critic.parameters(),

                    self.config.max_grad_norm,
                )

                self.critic_optimizer.step()

                metrics[
                    "critic_loss"
                ] += float(
                    critic_loss.item()
                )

                critic_metric_count += 1

                # =================================================
                # DECENTRALIZED ACTOR UPDATES
                # =================================================

                for agent_index, agent_id in enumerate(
                    self.agent_ids
                ):

                    agent_offset = (
                        agent_index
                        * rollout_steps
                    )

                    minibatch_indices = (
                        agent_offset
                        + indices
                    )

                    agent_observations = (
                        local_observations[
                            minibatch_indices
                        ]
                    )

                    agent_actions = (
                        actions[
                            minibatch_indices
                        ]
                    )

                    agent_old_log_probs = (
                        old_log_probabilities[
                            minibatch_indices
                        ]
                    )

                    agent_advantages = (
                        advantages_tensor[
                            minibatch_indices
                        ]
                    )

                    distribution = (
                        self.actors[
                            agent_id
                        ].distribution(
                            agent_observations
                        )
                    )

                    new_log_probs = (
                        distribution.log_prob(
                            agent_actions
                        )
                    )

                    entropy = (
                        distribution
                        .entropy()
                        .mean()
                    )

                    probability_ratio = (
                        torch.exp(
                            new_log_probs
                            - agent_old_log_probs
                        )
                    )

                    unclipped_objective = (

                        probability_ratio
                        * agent_advantages
                    )

                    clipped_ratio = (
                        torch.clamp(

                            probability_ratio,

                            1.0
                            - self.config.clip_coef,

                            1.0
                            + self.config.clip_coef,
                        )
                    )

                    clipped_objective = (

                        clipped_ratio
                        * agent_advantages
                    )

                    policy_loss = -torch.min(

                        unclipped_objective,

                        clipped_objective,

                    ).mean()

                    actor_loss = (

                        policy_loss

                        - self.config.entropy_coef
                        * entropy
                    )

                    optimizer = (
                        self.actor_optimizers[
                            agent_id
                        ]
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    actor_loss.backward()

                    nn.utils.clip_grad_norm_(

                        self.actors[
                            agent_id
                        ].parameters(),

                        self.config.max_grad_norm,
                    )

                    optimizer.step()

                    with torch.no_grad():

                        approx_kl = (

                            agent_old_log_probs
                            - new_log_probs
                        ).mean()

                        clip_fraction = (

                            torch.abs(
                                probability_ratio
                                - 1.0
                            )

                            > self.config.clip_coef

                        ).float().mean()

                    metrics[
                        "actor_loss"
                    ] += float(
                        actor_loss.item()
                    )

                    metrics[
                        "entropy"
                    ] += float(
                        entropy.item()
                    )

                    metrics[
                        "approx_kl"
                    ] += float(
                        approx_kl.item()
                    )

                    metrics[
                        "clip_fraction"
                    ] += float(
                        clip_fraction.item()
                    )

                    actor_metric_count += 1

        # ----------------------------------------------------
        # Average metrics
        # ----------------------------------------------------

        if actor_metric_count > 0:

            metrics[
                "actor_loss"
            ] /= actor_metric_count

            metrics[
                "entropy"
            ] /= actor_metric_count

            metrics[
                "approx_kl"
            ] /= actor_metric_count

            metrics[
                "clip_fraction"
            ] /= actor_metric_count

        if critic_metric_count > 0:

            metrics[
                "critic_loss"
            ] /= critic_metric_count

        return metrics

    # ========================================================
    # EVALUATION
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        episodes: int,
        background_traffic: bool | None = None,
        seed_offset: int = 100000,
    ):

        eval_background_traffic = (
            self.config.background_traffic
            if background_traffic is None
            else bool(background_traffic)
        )

        eval_env = Stage2IntersectionEnv(
            seed=self.config.seed,
            config=Stage2Config(),
            background_traffic=eval_background_traffic,
        )

        episode_rewards = []

        episode_lengths = []

        collisions = []

        near_collisions = []

        minimum_distances = []

        minimum_ttcs = []

        completions = []
        agent_completion_counts = np.zeros(
            self.num_agents,
            dtype=np.float64,
        )
        agent_collision_counts = np.zeros(
            self.num_agents,
            dtype=np.float64,
        )
        agent_agent_collision_counts = np.zeros(
            self.num_agents,
            dtype=np.float64,
        )
        agent_traffic_contact_counts = np.zeros(
            self.num_agents,
            dtype=np.float64,
        )
        agent_progress_fractions = {
            agent_id: []
            for agent_id in self.agent_ids
        }
        agent_collision_episodes = []
        traffic_collision_episodes = []
        collision_terminations = []
        time_limit_episodes = []

        for episode in range(
            episodes
        ):

            observations = (
                eval_env.reset(
                    seed=(
                        seed_offset
                        + self.config.seed
                        + episode
                    )
                )
            )

            episode_reward = np.zeros(
                self.num_agents,
                dtype=np.float64,
            )

            steps_taken = 0

            infos = {}

            for step in range(
                eval_env.config.max_steps
            ):

                (
                    action_dict,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.select_actions(

                    observations,

                    deterministic=True,

                    update_statistics=False,
                )

                (
                    observations,
                    rewards,
                    terminated,
                    truncated,
                    infos,
                ) = eval_env.step(
                    action_dict
                )

                for index, agent_id in enumerate(
                    self.agent_ids
                ):

                    episode_reward[
                        index
                    ] += rewards[
                        agent_id
                    ]

                steps_taken = (
                    step + 1
                )

                if (
                    terminated
                    or truncated
                ):

                    break

            global_info = infos.get(
                "global",
                {},
            )

            episode_rewards.append(
                episode_reward
            )

            episode_lengths.append(
                steps_taken
            )

            collisions.append(
                global_info.get(
                    "agent_involved_collisions",
                    global_info.get("total_collisions", 0),
                )
            )

            near_collisions.append(
                global_info.get(
                    "near_collisions",
                    0,
                )
            )

            minimum_distances.append(
                global_info.get(
                    "minimum_distance",
                    float("inf"),
                )
            )

            minimum_ttcs.append(
                global_info.get(
                    "minimum_ttc",
                    float("inf"),
                )
            )

            completions.append(
                float(
                    global_info.get(
                        "all_completed",
                        False,
                    )
                )
            )

            for index, agent_id in enumerate(self.agent_ids):
                vehicle = eval_env.vehicles[agent_id]
                agent_completion_counts[index] += float(vehicle.completed)
                agent_collision_counts[index] += float(vehicle.collided)
                agent_agent_collision_counts[index] += float(getattr(vehicle, "agent_collided", False))
                agent_traffic_contact_counts[index] += float(getattr(vehicle, "traffic_collided", False))
                agent_progress_fractions[agent_id].append(
                    vehicle.progress / max(vehicle.route_distance, 1.0)
                )

            agent_collision_episodes.append(
                float(global_info.get("agent_collisions", 0) > 0)
            )
            traffic_collision_episodes.append(
                float(global_info.get("traffic_collisions", 0) > 0)
            )
            collision_terminations.append(
                float(
                    global_info.get("agent_involved_collisions", 0) > 0
                    and not global_info.get("all_completed", False)
                )
            )
            time_limit_episodes.append(
                float(truncated and not terminated)
            )

        reward_matrix = np.asarray(
            episode_rewards
        )

        finite_ttc = [

            value

            for value
            in minimum_ttcs

            if np.isfinite(value)
        ]

        result = {

            "mean_reward": float(
                reward_matrix.mean()
            ),

            "agent_mean_reward": [

                float(value)

                for value
                in reward_matrix.mean(
                    axis=0
                )
            ],

            "mean_episode_length": float(
                np.mean(
                    episode_lengths
                )
            ),

            "collision_rate": float(
                np.mean(
                    np.asarray(
                        collisions
                    ) > 0
                )
            ),

            "mean_collisions": float(
                np.mean(
                    collisions
                )
            ),

            "mean_near_collisions": float(
                np.mean(
                    near_collisions
                )
            ),

            "mean_minimum_distance": float(
                np.mean(
                    minimum_distances
                )
            ),

            "mean_minimum_ttc": (

                float(
                    np.mean(
                        finite_ttc
                    )
                )

                if finite_ttc

                else float("inf")
            ),

            "completion_rate": float(
                np.mean(
                    completions
                )
            ),
            "agent_completion_rate": {
                agent_id: float(agent_completion_counts[index] / episodes)
                for index, agent_id in enumerate(self.agent_ids)
            },
            "agent_collision_rate": {
                agent_id: float(agent_collision_counts[index] / episodes)
                for index, agent_id in enumerate(self.agent_ids)
            },
            "agent_agent_collision_rate": {
                agent_id: float(agent_agent_collision_counts[index] / episodes)
                for index, agent_id in enumerate(self.agent_ids)
            },
            "agent_background_contact_rate": {
                agent_id: float(agent_traffic_contact_counts[index] / episodes)
                for index, agent_id in enumerate(self.agent_ids)
            },
            "agent_mean_route_progress": {
                agent_id: float(np.mean(agent_progress_fractions[agent_id]))
                for agent_id in self.agent_ids
            },
            "agent_collision_episode_rate": float(
                np.mean(agent_collision_episodes)
            ),
            "agent_agent_collision_episode_rate": float(
                np.mean(agent_collision_episodes)
            ),
            "agent_traffic_collision_episode_rate": float(
                np.mean(traffic_collision_episodes)
            ),
            "collision_termination_rate": float(
                np.mean(collision_terminations)
            ),
            "collision_free_rate": float(
                1.0 - np.mean(np.asarray(collisions) > 0)
            ),
            "time_limit_rate": float(
                np.mean(time_limit_episodes)
            ),
        }

        return result

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        update: int,
        evaluation=None,
        best: bool = False,
        phase_name: str | None = None,
    ) -> None:

        checkpoint = {

            "update": update,

            "total_environment_steps": (
                self.total_environment_steps
            ),

            "seed": self.config.seed,

            "background_traffic": bool(
                self.env.background_traffic_enabled
            ),

            "phase_name": (
                phase_name
                if phase_name is not None
                else self.phase_name
            ),

            "best_evaluation_score": (
                self.best_evaluation_score
            ),

            # ---------------------------------------------
            # Actors
            # ---------------------------------------------

            "actors": {

                agent_id:
                self.actors[
                    agent_id
                ].state_dict()

                for agent_id
                in self.agent_ids
            },

            # ---------------------------------------------
            # Critic
            # ---------------------------------------------

            "critic": (
                self.critic.state_dict()
            ),

            # ---------------------------------------------
            # Optimizers
            # ---------------------------------------------

            "actor_optimizers": {

                agent_id:
                self.actor_optimizers[
                    agent_id
                ].state_dict()

                for agent_id
                in self.agent_ids
            },

            "critic_optimizer": (
                self.critic_optimizer.state_dict()
            ),

            # ---------------------------------------------
            # Observation normalization
            # ---------------------------------------------

            "observation_rms": {

                agent_id:
                self.observation_rms[
                    agent_id
                ].state_dict()

                for agent_id
                in self.agent_ids
            },

            # ---------------------------------------------
            # Config
            # ---------------------------------------------

            "mappo_config": asdict(
                self.config
            ),

            "environment_config": asdict(
                self.env.config
            ),

            # ---------------------------------------------
            # Stage-1 selected identities
            # ---------------------------------------------

            "selected_stage1_models": (
                SELECTED_MODELS
            ),

            "stage1_warm_start": bool(self.config.stage1_warm_start),
            "stage1_checkpoint_paths": self.stage1_checkpoint_paths,
            "actor_observation_dim": self.actor_observation_dim,

            # ---------------------------------------------
            # Dimensions
            # ---------------------------------------------

            "observation_dim": (
                self.observation_dim
            ),

            "joint_observation_dim": (
                self.joint_observation_dim
            ),

            "action_dim": (
                self.action_dim
            ),

            # ---------------------------------------------
            # Evaluation
            # ---------------------------------------------

            "evaluation": evaluation,
        }

        phase_prefix = (
            f"{phase_name or self.phase_name}_"
            if phase_name is not None
            or self.phase_name not in {"phase_0", ""}
            else ""
        )

        if best:

            filename = (
                f"{phase_prefix}best.pt"
            )

        else:

            filename = (
                f"{phase_prefix}checkpoint_{update:06d}.pt"
            )

        torch.save(

            checkpoint,

            self.output_dir
            / filename,
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        for agent_id in self.agent_ids:
            self.actors[agent_id].load_state_dict(
                checkpoint["actors"][agent_id]
            )

            self.actor_optimizers[agent_id].load_state_dict(
                checkpoint["actor_optimizers"][agent_id]
            )

            self.observation_rms[agent_id].load_state_dict(
                checkpoint["observation_rms"][agent_id]
            )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        self.total_environment_steps = int(
            checkpoint.get(
                "total_environment_steps",
                0,
            )
        )

        self.best_evaluation_score = float(
            checkpoint.get(
            "best_evaluation_score",
            checkpoint.get("best_evaluation_reward", -float("inf")),
            )
        )

        if "background_traffic" in checkpoint:
            self.config.background_traffic = bool(
                checkpoint["background_traffic"]
            )

        self.actors.to(self.device)
        self.critic.to(self.device)

        return checkpoint

    def train_run(
        self,
        total_updates: int,
        background_traffic: bool,
        phase_name: str,
        keep_best: bool = True,
    ) -> dict:

        self.config.background_traffic = background_traffic
        self.env = self._build_environment(background_traffic=background_traffic)
        self._rollout_observations = None
        self._episode_returns = np.zeros(self.num_agents, dtype=np.float64)
        self._episode_length = 0
        self.phase_name = phase_name
        self.best_evaluation_score = -float("inf")
        if keep_best:
            self.phase_best_paths.pop(phase_name, None)
        self.save_training_config()

        training_history = []

        print()
        print("=" * 72)
        print(f"{phase_name.upper()} MAPPO TRAINING")
        print("=" * 72)
        print(f"background traffic: {'enabled' if background_traffic else 'disabled'}")
        print(f"updates: {total_updates}")
        print()

        for update in range(
            1,
            total_updates + 1,
        ):

            (
                buffer,
                advantages,
                returns,
                episodes,
                current_episode_length,
            ) = self.collect_rollout(
                update
            )

            metrics = self.update(
                buffer,
                advantages,
                returns,
            )

            rollout_reward = float(
                np.mean(buffer.rewards)
            )

            if episodes:
                rollout_length = float(
                    np.mean(
                        [
                            episode["length"]
                            for episode in episodes
                        ]
                    )
                )
                rollout_collision_rate = float(
                    np.mean(
                        [
                            episode["agent_collisions"] > 0
                            or episode["background_collisions"] > 0
                            for episode in episodes
                        ]
                    )
                )
                rollout_agent_agent_collision_rate = float(
                    np.mean(
                        [
                            episode["agent_collisions"] > 0
                            for episode in episodes
                        ]
                    )
                )
                rollout_agent_traffic_collision_rate = float(
                    np.mean(
                        [
                            episode["background_collisions"] > 0
                            for episode in episodes
                        ]
                    )
                )
                rollout_completion_rate = float(
                    np.mean(
                        [
                            episode["all_completed"]
                            for episode in episodes
                        ]
                    )
                )
            else:
                rollout_length = float(
                    current_episode_length
                )
                rollout_collision_rate = 0.0
                rollout_agent_agent_collision_rate = 0.0
                rollout_agent_traffic_collision_rate = 0.0
                rollout_completion_rate = 0.0

            completed_episode_count = sum(
                bool(episode["all_completed"])
                for episode in episodes
            )

            record = {
                "phase": phase_name,
                "update": update,
                "environment_steps": self.total_environment_steps,
                "rollout_reward": rollout_reward,
                "rollout_episode_length": rollout_length,
                "rollout_collision_rate": rollout_collision_rate,
                "rollout_agent_agent_collision_rate": rollout_agent_agent_collision_rate,
                "rollout_agent_traffic_collision_rate": rollout_agent_traffic_collision_rate,
                "rollout_completion_rate": rollout_completion_rate,
                "completed_episodes": completed_episode_count,
                "episodes_observed": len(episodes),
                "actor_loss": metrics["actor_loss"],
                "critic_loss": metrics["critic_loss"],
                "entropy": metrics["entropy"],
                "approx_kl": metrics["approx_kl"],
                "clip_fraction": metrics["clip_fraction"],
            }

            print(
                f"[{phase_name}] update {update:3d} "
                f"reward={rollout_reward:8.3f} "
                f"completion={rollout_completion_rate:5.3f} "
                f"episodes={len(episodes):3d} "
                f"collision={rollout_collision_rate:5.3f} "
                f"agent_agent={rollout_agent_agent_collision_rate:5.3f} "
                f"agent_traffic={rollout_agent_traffic_collision_rate:5.3f} "
                f"length={rollout_length:7.1f} "
                f"actor={metrics['actor_loss']:8.4f} "
                f"critic={metrics['critic_loss']:8.4f} "
                f"entropy={metrics['entropy']:6.4f}"
            )

            if (
                self.config.eval_interval > 0
                and update % self.config.eval_interval == 0
            ):
                evaluation = self.evaluate(
                    self.config.eval_episodes,
                    background_traffic=background_traffic,
                )
                record["evaluation"] = evaluation
                print(
                    f"  eval reward={evaluation['mean_reward']:.3f} "
                    f"completion={evaluation['completion_rate']:.3f} "
                    f"collision={evaluation['collision_rate']:.3f} "
                    f"near={evaluation['mean_near_collisions']:.1f} "
                    f"min_dist={evaluation['mean_minimum_distance']:.3f} "
                    f"min_ttc={evaluation['mean_minimum_ttc']:.3f}"
                )
                print(
                    f"  agent completion={evaluation['agent_completion_rate']} "
                    f"route progress={evaluation['agent_mean_route_progress']} "
                    f"agent-agent={evaluation['agent_agent_collision_episode_rate']:.3f} "
                    f"agent-traffic={evaluation['agent_traffic_collision_episode_rate']:.3f} "
                    f"timeouts={evaluation['time_limit_rate']:.3f}"
                )
                selection_score = (
                    1000.0 * evaluation["completion_rate"]
                    - 100.0 * evaluation["collision_rate"]
                    + 0.01 * evaluation["mean_reward"]
                )
                if keep_best and selection_score > self.best_evaluation_score:
                    self.best_evaluation_score = selection_score
                    self.save_checkpoint(
                        update,
                        evaluation,
                        best=True,
                        phase_name=phase_name,
                    )
                    self.phase_best_paths[phase_name] = (
                        self.output_dir
                        / f"{phase_name}_best.pt"
                    )
                    print(f"  best checkpoint: {phase_name}_best.pt")

            if (
                self.config.save_interval > 0
                and update % self.config.save_interval == 0
            ):
                self.save_checkpoint(
                    update,
                    phase_name=phase_name,
                )

            training_history.append(record)

            history_path = (
                self.output_dir
                / f"{phase_name}_training_history.json"
            )
            with open(history_path, "w", encoding="utf-8") as file:
                json.dump(training_history, file, indent=2)

        final_evaluation = self.evaluate(
            self.config.final_eval_episodes,
            background_traffic=background_traffic,
            seed_offset=1000000,
        )

        with open(
            self.output_dir / f"{phase_name}_final_evaluation.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(final_evaluation, file, indent=2)

        self.save_checkpoint(
            total_updates,
            evaluation=final_evaluation,
            phase_name=phase_name,
        )

        if keep_best and phase_name not in self.phase_best_paths:
            self.phase_best_paths[phase_name] = (
                self.output_dir
                / f"{phase_name}_checkpoint_{total_updates:06d}.pt"
            )

        print(f"[{phase_name}] final evaluation: reward={final_evaluation['mean_reward']:.3f} completion={final_evaluation['completion_rate']:.3f} collision={final_evaluation['collision_rate']:.3f}")
        if keep_best and phase_name in self.phase_best_paths:
            print(f"[{phase_name}] best checkpoint: {self.phase_best_paths[phase_name].name}")
        return final_evaluation

    def train(self) -> None:

        print()
        print("=" * 72)
        print("STAGE 2 MAPPO TRAINING")
        print("=" * 72)
        print(f"Device: {self.device}")
        print(f"Agents: {self.agent_ids}")
        print(f"Local observation dimension: {self.observation_dim}")
        print(f"Centralized critic dimension: {self.joint_observation_dim} -> {self.num_agents} value heads")
        print(f"Action dimension: {self.action_dim}")
        print(f"Rollout steps/update: {self.config.rollout_steps}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 72)

        warmup_updates = min(
            max(int(self.config.warmup_updates), 0),
            max(self.config.total_updates - 1, 0),
        )
        traffic_updates = self.config.total_updates - warmup_updates

        if warmup_updates > 0:
            self.train_run(
                warmup_updates,
                background_traffic=False,
                phase_name="warmup_no_traffic",
                keep_best=False,
            )

        self.train_run(
            traffic_updates,
            background_traffic=True,
            phase_name="traffic",
            keep_best=True,
        )

        final_best = self.phase_best_paths["traffic"]
        self.load_checkpoint(final_best)
        final_evaluation = self.evaluate(
            self.config.final_eval_episodes,
            background_traffic=True,
            seed_offset=2000000,
        )

        with open(
            self.output_dir / "final_evaluation.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(final_evaluation, file, indent=2)

        final_path = self.output_dir / "final.pt"
        if final_best.exists():
            final_checkpoint = torch.load(
                final_best,
                map_location=self.device,
                weights_only=False,
            )
            final_checkpoint["evaluation"] = final_evaluation
            torch.save(final_checkpoint, final_path)

        print()
        print("FINAL EVALUATION")
        print(f"  reward: {final_evaluation['mean_reward']:.3f}")
        print(f"  completion: {final_evaluation['completion_rate']:.3f}")
        print(f"  collision rate: {final_evaluation['collision_rate']:.3f}")
        print(f"  final checkpoint: {final_path.name}")
        print("=" * 72)
        print("MAPPO TRAINING COMPLETE")
        print("=" * 72)


# ============================================================
# ARGUMENT PARSER
# ============================================================


def parse_arguments():

    parser = argparse.ArgumentParser(

        description=(
            "Train heterogeneous "
            "Stage-2 MAPPO."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--total-updates",
        type=int,
        default=1000,
        help="Total MAPPO updates across the warmup and traffic phases.",
    )

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=100,
        help="Initial no-background-traffic curriculum updates.",
    )

    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--update-epochs",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=[
            "cpu",
            "cuda",
        ],
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "models/stage2_mappo"
        ),
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
    )

    arguments = (
        parser.parse_args()
    )
    if arguments.total_updates <= 0:
        parser.error("--total-updates must be positive.")

    return MAPPOConfig(

        seed=arguments.seed,

        total_updates=arguments.total_updates,

        warmup_updates=arguments.warmup_updates,

        rollout_steps=(
            arguments.rollout_steps
        ),

        actor_learning_rate=(
            arguments.actor_learning_rate
        ),

        critic_learning_rate=(
            arguments.critic_learning_rate
        ),

        update_epochs=(
            arguments.update_epochs
        ),

        minibatch_size=(
            arguments.minibatch_size
        ),

        device=arguments.device,

        background_traffic=True,

        output_dir=arguments.output_dir,

        eval_interval=(
            arguments.eval_interval
        ),

        eval_episodes=(
            arguments.eval_episodes
        ),

        final_eval_episodes=(
            arguments.final_eval_episodes
        ),

        save_interval=(
            arguments.save_interval
        ),

    )


# ============================================================
# MAIN
# ============================================================


def main():

    config = parse_arguments()

    trainer = MAPPOTrainer(
        config
    )

    trainer.train()


if __name__ == "__main__":
    main()