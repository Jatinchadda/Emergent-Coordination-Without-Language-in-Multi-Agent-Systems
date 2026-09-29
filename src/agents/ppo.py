from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cpu")


# =========================================================
# ACTOR / CRITIC NETWORK
# =========================================================

class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.shared = nn.Sequential(
            nn.Linear(
                self.obs_dim,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                128,
            ),
            nn.Tanh(),
        )

        self.actor = nn.Linear(
            128,
            self.action_dim,
        )

        self.critic = nn.Linear(
            128,
            1,
        )

        # Attached after training.
        self.obs_mean = None
        self.obs_var = None

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, obs):

        x = self.shared(obs)

        logits = self.actor(x)

        value = self.critic(x).squeeze(-1)

        return logits, value

    # =====================================================
    # ACTION
    # =====================================================

    def get_action(
        self,
        obs,
        deterministic=False,
    ):

        logits, value = self.forward(obs)

        dist = Categorical(
            logits=logits
        )

        if deterministic:
            action = torch.argmax(
                logits,
                dim=-1,
            )
        else:
            action = dist.sample()

        log_prob = dist.log_prob(
            action
        )

        entropy = dist.entropy()

        return (
            action,
            log_prob,
            entropy,
            value,
        )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(
        self,
        obs,
        actions,
    ):

        logits, values = self.forward(
            obs
        )

        dist = Categorical(
            logits=logits
        )

        log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy()

        return (
            log_probs,
            entropy,
            values,
        )


# =========================================================
# RUNNING OBSERVATION NORMALIZER
# =========================================================

class RunningMeanStd:

    def __init__(
        self,
        shape,
    ):

        self.mean = np.zeros(
            shape,
            dtype=np.float64,
        )

        self.var = np.ones(
            shape,
            dtype=np.float64,
        )

        self.count = 1e-4

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self, x):

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if x.ndim == 1:
            x = x.reshape(
                1,
                -1,
            )

        batch_mean = np.mean(
            x,
            axis=0,
        )

        batch_var = np.var(
            x,
            axis=0,
        )

        batch_count = x.shape[0]

        self._update_from_moments(
            batch_mean,
            batch_var,
            batch_count,
        )

    # =====================================================
    # MOMENTS
    # =====================================================

    def _update_from_moments(
        self,
        batch_mean,
        batch_var,
        batch_count,
    ):

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

        m_a = (
            self.var
            * self.count
        )

        m_b = (
            batch_var
            * batch_count
        )

        correction = (
            delta ** 2
            * self.count
            * batch_count
            / total_count
        )

        new_var = (
            m_a
            + m_b
            + correction
        ) / total_count

        self.mean = new_mean

        self.var = np.maximum(
            new_var,
            1e-8,
        )

        self.count = total_count


# =========================================================
# DISCOUNTED RETURNS / GAE
# =========================================================

def compute_gae(
    rewards,
    values,
    dones,
    last_value,
    gamma,
    gae_lambda,
):

    advantages = np.zeros(
        len(rewards),
        dtype=np.float32,
    )

    gae = 0.0

    for t in reversed(
        range(len(rewards))
    ):

        if t == len(rewards) - 1:

            next_value = (
                last_value
            )

        else:

            next_value = (
                values[t + 1]
            )

        non_terminal = (
            1.0
            - float(dones[t])
        )

        delta = (
            rewards[t]
            + gamma
            * next_value
            * non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * gae
        )

        advantages[t] = gae

    returns = (
        advantages
        + np.asarray(
            values,
            dtype=np.float32,
        )
    )

    return (
        advantages,
        returns,
    )


# =========================================================
# PPO TRAINING
# =========================================================

def train_ppo(
    env,
    total_steps=100_000,
    seed=42,
    save_path="models/model.pt",

    # -----------------------------------------------------
    # New stable PPO configuration
    # -----------------------------------------------------

    learning_rate=1e-4,

    gamma=0.995,

    gae_lambda=0.95,

    clip_eps=0.20,

    entropy_coef=0.02,

    value_coef=0.5,

    max_grad_norm=0.5,

    rollout_size=2048,

    batch_size=256,

    update_epochs=6,
):

    # =====================================================
    # SEED EVERYTHING
    # =====================================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        env.rng.seed(seed)
    except Exception:
        pass

    try:
        env.traffic_generator.rng.seed(seed)
    except Exception:
        pass

    # =====================================================
    # MODEL
    # =====================================================

    model = ActorCritic(
        env.observation_dim,
        env.action_dim,
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # =====================================================
    # OBSERVATION NORMALIZATION
    # =====================================================

    obs_stats = RunningMeanStd(
        env.observation_dim
    )

    # =====================================================
    # TRAINING STATE
    # =====================================================

    obs, _ = env.reset(
        seed=seed
    )

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    steps_done = 0

    episode_reward = 0.0
    episode_length = 0
    episode_collision = False

    episode_rewards = []
    episode_lengths = []
    episode_collisions = []
    episode_completions = []

    # =====================================================
    # ROLLOUT STORAGE
    # =====================================================

    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    dones = []

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    while steps_done < total_steps:

        observations.clear()
        actions.clear()
        rewards.clear()
        values.clear()
        log_probs.clear()
        dones.clear()

        # -------------------------------------------------
        # Collect rollout
        # -------------------------------------------------

        rollout_count = min(
            rollout_size,
            total_steps - steps_done,
        )

        for _ in range(
            rollout_count
        ):

            # ---------------------------------------------
            # Normalize current observation
            # ---------------------------------------------

            obs_stats.update(
                obs
            )

            normalized_obs = (
                (
                    obs
                    - obs_stats.mean
                )
                /
                np.sqrt(
                    obs_stats.var
                    + 1e-8
                )
            ).astype(
                np.float32
            )

            obs_tensor = torch.tensor(
                normalized_obs,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            # ---------------------------------------------
            # Sample action
            # ---------------------------------------------

            with torch.no_grad():

                (
                    action_tensor,
                    log_prob_tensor,
                    _entropy,
                    value_tensor,
                ) = model.get_action(
                    obs_tensor,
                    deterministic=False,
                )

            action = int(
                action_tensor.item()
            )

            old_log_prob = float(
                log_prob_tensor.item()
            )

            value = float(
                value_tensor.item()
            )

            # ---------------------------------------------
            # Environment step
            # ---------------------------------------------

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            done = (
                terminated
                or truncated
            )

            # ---------------------------------------------
            # Store transition
            # ---------------------------------------------

            observations.append(
                normalized_obs.copy()
            )

            actions.append(
                action
            )

            rewards.append(
                float(reward)
            )

            values.append(
                value
            )

            log_probs.append(
                old_log_prob
            )

            dones.append(
                done
            )

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            episode_reward += float(
                reward
            )

            episode_length += 1

            if info.get(
                "collision",
                False,
            ):
                episode_collision = True

            # ---------------------------------------------
            # Global step
            # ---------------------------------------------

            steps_done += 1

            # ---------------------------------------------
            # Episode termination
            # ---------------------------------------------

            if done:

                episode_rewards.append(
                    episode_reward
                )

                episode_lengths.append(
                    episode_length
                )

                episode_collisions.append(
                    float(
                        episode_collision
                    )
                )

                episode_completions.append(
                    float(
                        info.get(
                            "completed",
                            False,
                        )
                    )
                )

                episode_reward = 0.0
                episode_length = 0
                episode_collision = False

                if steps_done < total_steps:

                    obs, _ = env.reset()

                    obs = np.asarray(
                        obs,
                        dtype=np.float32,
                    )

                else:

                    # We are done collecting.
                    obs = np.asarray(
                        next_obs,
                        dtype=np.float32,
                    )

            else:

                obs = np.asarray(
                    next_obs,
                    dtype=np.float32,
                )

            if steps_done >= total_steps:
                break

        # =================================================
        # BOOTSTRAP VALUE
        # =================================================

        if dones and dones[-1]:

            last_value = 0.0

        else:

            obs_stats.update(
                obs
            )

            normalized_last_obs = (
                (
                    obs
                    - obs_stats.mean
                )
                /
                np.sqrt(
                    obs_stats.var
                    + 1e-8
                )
            ).astype(
                np.float32
            )

            last_obs_tensor = torch.tensor(
                normalized_last_obs,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            with torch.no_grad():

                _logits, last_value_tensor = (
                    model.forward(
                        last_obs_tensor
                    )
                )

            last_value = float(
                last_value_tensor.item()
            )

        # =================================================
        # GAE
        # =================================================

        (
            advantages,
            returns,
        ) = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # =================================================
        # TENSORS
        # =================================================

        obs_tensor = torch.tensor(
            np.asarray(
                observations,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        action_tensor = torch.tensor(
            np.asarray(
                actions,
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=DEVICE,
        )

        old_log_prob_tensor = torch.tensor(
            np.asarray(
                log_probs,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        advantage_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=DEVICE,
        )

        return_tensor = torch.tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        # =================================================
        # ADVANTAGE NORMALIZATION
        # =================================================

        if len(
            advantage_tensor
        ) > 1:

            advantage_tensor = (
                advantage_tensor
                - advantage_tensor.mean()
            ) / (
                advantage_tensor.std()
                + 1e-8
            )

        # =================================================
        # PPO UPDATE
        # =================================================

        n = len(
            obs_tensor
        )

        for _ in range(
            update_epochs
        ):

            indices = np.random.permutation(
                n
            )

            for start in range(
                0,
                n,
                batch_size,
            ):

                batch_idx = indices[
                    start:
                    start + batch_size
                ]

                (
                    new_log_prob,
                    entropy,
                    new_values,
                ) = model.evaluate(
                    obs_tensor[
                        batch_idx
                    ],
                    action_tensor[
                        batch_idx
                    ],
                )

                ratio = torch.exp(
                    new_log_prob
                    - old_log_prob_tensor[
                        batch_idx
                    ]
                )

                surrogate_1 = (
                    ratio
                    * advantage_tensor[
                        batch_idx
                    ]
                )

                surrogate_2 = (
                    torch.clamp(
                        ratio,
                        1.0 - clip_eps,
                        1.0 + clip_eps,
                    )
                    * advantage_tensor[
                        batch_idx
                    ]
                )

                actor_loss = (
                    -torch.min(
                        surrogate_1,
                        surrogate_2,
                    ).mean()
                )

                critic_loss = (
                    value_coef
                    * (
                        return_tensor[
                            batch_idx
                        ]
                        - new_values
                    )
                    .pow(2)
                    .mean()
                )

                entropy_bonus = (
                    entropy.mean()
                )

                loss = (
                    actor_loss
                    + critic_loss
                    - entropy_coef
                    * entropy_bonus
                )

                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

                optimizer.step()

        # =================================================
        # TRAINING LOG
        # =================================================

        if (
            steps_done % 5000 < rollout_count
            or steps_done >= total_steps
        ):

            recent_rewards = (
                episode_rewards[-10:]
            )

            recent_collisions = (
                episode_collisions[-10:]
            )

            recent_completions = (
                episode_completions[-10:]
            )

            avg_reward = (
                float(
                    np.mean(
                        recent_rewards
                    )
                )
                if recent_rewards
                else 0.0
            )

            collision_rate = (
                float(
                    np.mean(
                        recent_collisions
                    )
                )
                if recent_collisions
                else 0.0
            )

            completion_rate = (
                float(
                    np.mean(
                        recent_completions
                    )
                )
                if recent_completions
                else 0.0
            )

            print(
                f"steps={steps_done:6d} "
                f"avg_reward={avg_reward:9.3f} "
                f"collision_rate="
                f"{collision_rate:.3f} "
                f"completion_rate="
                f"{completion_rate:.3f}"
            )

    # =====================================================
    # ATTACH NORMALIZATION
    # =====================================================

    model.obs_mean = (
        obs_stats.mean.astype(
            np.float32
        )
    )

    model.obs_var = (
        obs_stats.var.astype(
            np.float32
        )
    )

    # =====================================================
    # SAVE
    # =====================================================

    save_path = str(
        save_path
    )

    directory = os.path.dirname(
        save_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "obs_dim":
            env.observation_dim,

        "action_dim":
            env.action_dim,

        "behavior":
            env.behavior,

        "seed":
            seed,

        "total_steps":
            total_steps,

        "gamma":
            gamma,

        "gae_lambda":
            gae_lambda,

        "clip_eps":
            clip_eps,

        "learning_rate":
            learning_rate,

        "entropy_coef":
            entropy_coef,

        "value_coef":
            value_coef,

        "max_grad_norm":
            max_grad_norm,

        "rollout_size":
            rollout_size,

        "batch_size":
            batch_size,

        "update_epochs":
            update_epochs,

        "obs_mean":
            model.obs_mean,

        "obs_var":
            model.obs_var,

        "obs_count":
            obs_stats.count,

        "episode_rewards":
            episode_rewards,

        "episode_lengths":
            episode_lengths,

        "episode_collisions":
            episode_collisions,

        "episode_completions":
            episode_completions,
    }

    torch.save(
        checkpoint,
        save_path,
    )

    return model, {
        "episode_rewards":
            episode_rewards,

        "episode_lengths":
            episode_lengths,

        "episode_collisions":
            episode_collisions,

        "episode_completions":
            episode_completions,
    }


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = ActorCritic(
        checkpoint["obs_dim"],
        checkpoint["action_dim"],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.obs_mean = np.asarray(
        checkpoint["obs_mean"],
        dtype=np.float32,
    )

    model.obs_var = np.asarray(
        checkpoint["obs_var"],
        dtype=np.float32,
    )

    model.eval()

    return model


# =========================================================
# NORMALIZE FOR MODEL
# =========================================================

def normalize_for_model(
    model,
    obs,
):

    if (
        model.obs_mean is None
        or model.obs_var is None
    ):
        raise RuntimeError(
            "Model does not contain "
            "observation normalization "
            "statistics."
        )

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    normalized = (
        obs
        - model.obs_mean
    ) / np.sqrt(
        model.obs_var
        + 1e-8
    )

    return normalized.astype(
        np.float32
    )