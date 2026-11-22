import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations, action_mask, logit_bias):
        logits = self.actor(observations).squeeze(-1) + logit_bias
        logits = logits.masked_fill(~action_mask, -1e9)
        weights = action_mask.unsqueeze(-1).to(observations.dtype)
        pooled = (observations * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        values = self.critic(pooled).squeeze(-1)
        return logits, values


class SharedPPOLearner:
    def __init__(self, input_size, model_path, training=False, agents_per_team=2,
                 collect_only=False, rollout_path=None, seed=17):
        if agents_per_team < 1:
            raise ValueError('agents_per_team must be positive')
        self.model_path = Path(model_path)
        self.training = training
        self.agents_per_team = agents_per_team
        self.collect_only = collect_only
        self.rollout_path = Path(rollout_path) if rollout_path is not None else None
        if self.collect_only and self.rollout_path is None:
            raise ValueError('rollout_path is required in collect-only mode')
        self.completed_episodes = []
        use_cuda = training and not collect_only and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
        self.model = ActorCritic(input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        if self.model_path.exists():
            with self.model_path.open('rb') as checkpoint_file:
                checkpoint_prefix = checkpoint_file.read(1)
            if checkpoint_prefix == b'{':
                checkpoint_data = json.loads(self.model_path.read_text(encoding='utf-8'))
                if (
                    isinstance(checkpoint_data, dict)
                    and {'input_size', 'hidden_size', 'hidden_weights'} <= checkpoint_data.keys()
                ):
                    raise ValueError(
                        f'{self.model_path} contains legacy Q-learning JSON weights, '
                        'which are not compatible with PPO. Start a PPO run with '
                        '--fresh to initialize and save a PPO checkpoint.'
                    )
                raise ValueError(
                    f'{self.model_path} contains JSON, not a PyTorch PPO checkpoint. '
                    'Start a PPO run with --fresh to initialize a PPO checkpoint.'
                )
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint)

    def act(self, action_features, action_biases, deterministic=False):
        action_count = len(action_features)
        if action_count == 0:
            raise ValueError('PPO requires at least one legal action')
        if action_count != len(action_biases):
            raise ValueError('Each legal action must have exactly one policy bias')
        feature_count = self.model.actor[0].in_features
        observations = torch.zeros(
            (1, 5, feature_count), dtype=torch.float32, device=self.device
        )
        action_mask = torch.zeros((1, 5), dtype=torch.bool, device=self.device)
        logit_bias = torch.zeros((1, 5), dtype=torch.float32, device=self.device)
        action_indices = []
        for (action, features), bias in zip(action_features, action_biases):
            action_index = _action_index(action)
            observations[0, action_index] = torch.tensor(
                features, dtype=torch.float32, device=self.device
            )
            action_mask[0, action_index] = True
            logit_bias[0, action_index] = bias
            action_indices.append(action_index)

        with torch.no_grad():
            logits, value = self.model(observations, action_mask, logit_bias)
            distribution = Categorical(logits=logits)
            selected = logits.argmax(dim=1) if deterministic else distribution.sample()
            action_index = int(selected.item())
            log_probability = float(distribution.log_prob(selected).item())
        action = action_features[action_indices.index(action_index)][0]
        transition = {
            'observations': observations[0].tolist(),
            'action_mask': action_mask[0].tolist(),
            'logit_bias': logit_bias[0].tolist(),
            'action': action_index,
            'log_probability': log_probability,
            'value': float(value.item()),
        }
        return action, transition

    def finish_episode(self, trajectory):
        if not self.training:
            return
        if trajectory:
            self.completed_episodes.append(trajectory)
        if len(self.completed_episodes) >= self.agents_per_team:
            episodes = self.completed_episodes
            self.completed_episodes = []
            if self.collect_only:
                self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
                self.rollout_path.write_text(
                    json.dumps(episodes, separators=(',', ':'))
                )
            else:
                self.update_episodes(episodes)

    def update_episodes(self, episodes):
        if not self.training:
            raise RuntimeError('Cannot update a PPO learner created for inference')
        self._update(episodes)
        self.save()

    def _update(self, episodes):
        observations = []
        action_masks = []
        logit_biases = []
        actions = []
        old_log_probabilities = []
        returns = []
        advantages = []

        for episode in episodes:
            episode_advantages = [0.0] * len(episode)
            next_value = 0.0
            gae = 0.0
            for index in range(len(episode) - 1, -1, -1):
                transition = episode[index]
                delta = transition['reward'] + 0.99 * next_value - transition['value']
                gae = delta + 0.99 * 0.95 * gae
                episode_advantages[index] = gae
                next_value = transition['value']

            for transition, advantage in zip(episode, episode_advantages):
                observations.append(transition['observations'])
                action_masks.append(transition['action_mask'])
                logit_biases.append(transition['logit_bias'])
                actions.append(transition['action'])
                old_log_probabilities.append(transition['log_probability'])
                advantages.append(advantage)
                returns.append(advantage + transition['value'])

        if not observations:
            return
        observations = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )
        action_masks = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
        logit_biases = torch.tensor(logit_biases, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probabilities = torch.tensor(
            old_log_probabilities, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-8)

        batch_size = len(observations)
        for _ in range(4):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, 64):
                indices = permutation[start:start + 64]
                logits, values = self.model(
                    observations[indices], action_masks[indices], logit_biases[indices]
                )
                distribution = Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(actions[indices])
                ratios = (new_log_probabilities - old_log_probabilities[indices]).exp()
                unclipped = ratios * advantages[indices]
                clipped = ratios.clamp(0.8, 1.2) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns[indices])
                entropy_loss = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.model_path.with_suffix(self.model_path.suffix + '.tmp')
        torch.save(self.model.state_dict(), temporary_path)
        temporary_path.replace(self.model_path)


def _action_index(action):
    from contest.game import Directions

    action_indices = {
        Directions.NORTH: 0,
        Directions.SOUTH: 1,
        Directions.EAST: 2,
        Directions.WEST: 3,
        Directions.STOP: 4,
    }
    try:
        return action_indices[action]
    except KeyError as error:
        raise ValueError(f'Unsupported Pacman action: {action}') from error
