import tempfile
import unittest
from pathlib import Path

from q_network import LinearQLearner
from train_q_agent import _select_winner


class LinearQLearnerTests(unittest.TestCase):
    def make_learner(self, model_path, **kwargs):
        return LinearQLearner(
            ('bias',),
            ('left', 'right'),
            model_path,
            **kwargs,
        )

    def test_update_uses_maximum_next_action_value(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = self.make_learner(Path(directory) / 'weights.json')
            learner.learning_rate = 0.1
            learner.weights['right']['bias'] = 2.0

            learner.update_weights(
                {'bias': 1.0},
                'left',
                reward=1.0,
                next_action_features={
                    'left': {'bias': 1.0},
                    'right': {'bias': 1.0},
                },
                done=False,
            )

            self.assertAlmostEqual(learner.weights['left']['bias'], 0.288)

    def test_terminal_update_does_not_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = self.make_learner(Path(directory) / 'weights.json')
            learner.learning_rate = 0.1
            learner.weights['right']['bias'] = 100.0

            learner.update_weights(
                {'bias': 1.0},
                'left',
                reward=3.0,
                next_action_features={'right': {'bias': 1.0}},
                done=True,
            )

            self.assertAlmostEqual(learner.weights['left']['bias'], 0.3)

    def test_weights_and_epsilon_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'weights.json'
            learner = self.make_learner(
                model_path,
                training=True,
                epsilon=0.5,
                epsilon_decay=0.5,
                min_epsilon=0.1,
            )
            learner.weights['left']['bias'] = 1.25
            learner.decay_epsilon()
            learner.save()

            loaded = self.make_learner(model_path)

            self.assertEqual(loaded.weights['left']['bias'], 1.25)
            self.assertEqual(loaded.epsilon, 0.25)

    def test_epsilon_decay_respects_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = self.make_learner(
                Path(directory) / 'weights.json',
                epsilon=0.2,
                epsilon_decay=0.5,
                min_epsilon=0.1,
            )

            learner.decay_epsilon()
            learner.decay_epsilon()

            self.assertEqual(learner.epsilon, 0.1)

    def test_best_candidate_only_replaces_better_champion(self):
        results = [
            {'generation': 0, 'average_score': -20.0},
            {'generation': 1, 'average_score': -18.0},
        ]

        self.assertEqual(_select_winner(results, -19.0), results[1])
        self.assertIsNone(_select_winner(results, -18.0))
        self.assertIsNone(_select_winner(results, -17.0))


if __name__ == '__main__':
    unittest.main()
