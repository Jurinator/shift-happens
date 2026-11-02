import json
import random
from pathlib import Path


class LinearQLearner:
    def __init__(
        self,
        feature_names,
        actions,
        model_path,
        training=False,
        learning_rate=0.01,
        discount_factor=0.94,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        seed=31,
    ):
        self.feature_names = tuple(feature_names)
        self.actions = tuple(actions)
        self.model_path = Path(model_path)
        self.training = training
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = random.Random(seed)
        self.weights = {
            action: {feature: 0.0 for feature in self.feature_names}
            for action in self.actions
        }

        if self.model_path.exists():
            self._load()

    def get_q_value(self, features, action):
        return sum(
            self.weights[action][feature] * value
            for feature, value in features.items()
        )

    def choose_action(self, action_features):
        if not action_features:
            raise ValueError('Cannot choose an action without legal actions')
        if self.training and self.rng.random() < self.epsilon:
            return self.rng.choice(tuple(action_features))

        values = {
            action: self.get_q_value(features, action)
            for action, features in action_features.items()
        }
        best_value = max(values.values())
        best_actions = [
            action for action, value in values.items() if value == best_value
        ]
        return self.rng.choice(best_actions)

    def update_weights(
        self,
        state_features,
        action,
        reward,
        next_action_features,
        done,
    ):
        current_q = self.get_q_value(state_features, action)
        max_future_q = 0.0
        if not done and next_action_features:
            max_future_q = max(
                self.get_q_value(features, next_action)
                for next_action, features in next_action_features.items()
            )

        target = reward + self.discount_factor * max_future_q
        td_error = target - current_q
        for feature, value in state_features.items():
            self.weights[action][feature] += (
                self.learning_rate * td_error * value
            )

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.model_path.with_suffix(
            self.model_path.suffix + '.tmp'
        )
        temporary_path.write_text(
            json.dumps(
                {
                    'feature_names': self.feature_names,
                    'actions': self.actions,
                    'weights': self.weights,
                    'epsilon': self.epsilon,
                },
                separators=(',', ':'),
            )
        )
        temporary_path.replace(self.model_path)

    def _load(self):
        model = json.loads(self.model_path.read_text())
        if tuple(model.get('feature_names', ())) != self.feature_names:
            raise ValueError(
                f'Saved linear model has incompatible features: {self.model_path}'
            )
        if tuple(model.get('actions', ())) != self.actions:
            raise ValueError(
                f'Saved linear model has incompatible actions: {self.model_path}'
            )

        weights = model.get('weights')
        if not isinstance(weights, dict) or set(weights) != set(self.actions):
            raise ValueError(
                f'Saved linear model has invalid action weights: {self.model_path}'
            )
        for action in self.actions:
            action_weights = weights[action]
            if (
                not isinstance(action_weights, dict)
                or set(action_weights) != set(self.feature_names)
            ):
                raise ValueError(
                    f'Saved linear model has invalid feature weights for '
                    f'{action}: {self.model_path}'
                )
            self.weights[action] = {
                feature: float(action_weights[feature])
                for feature in self.feature_names
            }
        self.epsilon = float(model['epsilon'])
