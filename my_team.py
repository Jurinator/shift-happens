# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from pathlib import Path

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from ppo_network import SharedPPOLearner


FEATURE_COUNT = 30
ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
MODEL_PATH = Path(__file__).with_name('ppo_weights.pt')


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='PPOCaptureAgent', second='PPOCaptureAgent', num_training=0,
                model_path=MODEL_PATH, rollout_path=None, ppo_seed=17):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    agents = []
    requested_agents = ((first_index, first), (second_index, second))
    agent_classes = [(index, name, eval(name)) for index, name in requested_agents]
    ppo_count = sum(agent_class is PPOCaptureAgent for _, _, agent_class in agent_classes)
    learner = None
    if ppo_count:
        learner = SharedPPOLearner(
            FEATURE_COUNT, model_path, training=int(num_training) > 0,
            agents_per_team=ppo_count,
            collect_only=rollout_path is not None,
            rollout_path=rollout_path,
            seed=int(ppo_seed),
        )

    for index, _, agent_class in agent_classes:
        if agent_class is PPOCaptureAgent:
            agents.append(agent_class(index, learner))
        else:
            agents.append(agent_class(index))
    return agents


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


class PPOFeatureCaptureAgent(CaptureAgent):
    def __init__(self, index, learner):
        super().__init__(index, time_for_computing=0.1)
        self.learner = learner
        self.start = None
        self.last_transition = None
        self.dist_scale = 40.0
        self.width = 1
        self.height = 1

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        self.width = walls.width
        self.height = walls.height
        self.dist_scale = float(max(self.width, self.height) * 2)
        CaptureAgent.register_initial_state(self, game_state)

    def _remember_reward_baseline(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        self.last_score = self.get_score(game_state)
        self.last_food_count = len(self.get_food(game_state).as_list())
        self.last_carrying = agent_state.num_carrying
        self.last_returned = agent_state.num_returned

    def _reward_since_last_action(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        score_change = self.get_score(game_state) - self.last_score
        food_eaten = max(0, self.last_food_count - len(self.get_food(game_state).as_list()))
        returned = max(0, agent_state.num_returned - self.last_returned)
        lost_carrying = max(0, self.last_carrying - agent_state.num_carrying - returned)
        position = game_state.get_agent_position(self.index)
        food_distance = self._nearest_distance(position, self.get_food(game_state).as_list())
        home_distance = self.get_maze_distance(position, self.start)
        previous_food_distance = self.last_transition[7] * self.dist_scale
        previous_home_distance = self.last_transition[8] * self.dist_scale
        food_progress = previous_food_distance - food_distance
        home_progress = previous_home_distance - home_distance
        carrying_fraction = self.last_transition[1]
        return (
            score_change
            + 0.15 * food_eaten
            + 0.03 * food_progress
            + 0.08 * carrying_fraction * home_progress
            - 0.5 * lost_carrying
        )

    def _action_features(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        position = successor.get_agent_position(self.index)
        if position != nearest_point(position):
            successor = successor.generate_successor(self.index, action)
            position = successor.get_agent_position(self.index)

        agent_state = successor.get_agent_state(self.index)
        enemy_food = self.get_food(successor).as_list()
        own_food = self.get_food_you_are_defending(successor).as_list()
        capsules = self.get_capsules(successor)
        opponents = [successor.get_agent_state(index) for index in self.get_opponents(successor)]
        visible_opponents = [enemy for enemy in opponents if enemy.get_position() is not None]
        invaders = [enemy for enemy in visible_opponents if enemy.is_pacman]
        vulnerable_ghosts = [
            enemy for enemy in visible_opponents
            if not enemy.is_pacman and enemy.scared_timer > 0
        ]
        teammate_positions = [
            successor.get_agent_position(index)
            for index in self.get_team(successor)
            if index != self.index and successor.get_agent_position(index) is not None
        ]

        current_position = game_state.get_agent_position(self.index)
        current_food_distance = self._nearest_distance(
            current_position, self.get_food(game_state).as_list()
        )
        current_home_distance = self.get_maze_distance(current_position, self.start)
        current_danger_distance = self._nearest_distance(
            current_position,
            [enemy.get_position() for enemy in visible_opponents
             if not enemy.is_pacman and enemy.scared_timer == 0],
        )
        danger_distance = self._nearest_distance(
            position,
            [enemy.get_position() for enemy in visible_opponents
             if not enemy.is_pacman and enemy.scared_timer == 0],
        )
        food_distance = self._nearest_distance(position, enemy_food)
        food_eaten = max(0, len(self.get_food(game_state).as_list()) - len(enemy_food))
        home_distance = self.get_maze_distance(position, self.start)
        capsule_distance = self._nearest_distance(position, capsules)
        opponent_distance = self._nearest_distance(
            position, [enemy.get_position() for enemy in visible_opponents]
        )
        invader_distance = self._nearest_distance(
            position, [enemy.get_position() for enemy in invaders]
        )
        vulnerable_distance = self._nearest_distance(
            position, [enemy.get_position() for enemy in vulnerable_ghosts]
        )
        teammate_distance = self._nearest_distance(position, teammate_positions)
        x, y = position
        oriented_x = x / self.width if self.red else (self.width - x) / self.width
        current_agent_state = game_state.get_agent_state(self.index)
        reverse_action = Directions.REVERSE[current_agent_state.configuration.direction]

        features = [
            float(agent_state.is_pacman),
            min(1.0, agent_state.num_carrying / 10.0),
            min(1.0, agent_state.num_returned / 30.0),
            max(-1.0, min(1.0, self.get_score(successor) / 15.0)),
            max(0.0, min(1.0, successor.data.timeleft / 1200.0)),
            len(enemy_food) / 30.0,
            len(own_food) / 30.0,
            food_distance / self.dist_scale,
            home_distance / self.dist_scale,
            capsule_distance / self.dist_scale,
            opponent_distance / self.dist_scale,
            invader_distance / self.dist_scale,
            vulnerable_distance / self.dist_scale,
            agent_state.scared_timer / 40.0,
            len(invaders) / 2.0,
            len(visible_opponents) / 2.0,
            oriented_x,
            y / self.height,
            teammate_distance / self.dist_scale,
            float(self.index == self.get_team(successor)[-1]),
        ]
        features.extend(float(action == candidate) for candidate in ACTIONS)
        features.extend([
            float(agent_state.is_pacman),
            (current_food_distance - food_distance + 2.0 * food_eaten) / self.dist_scale,
            (current_home_distance - home_distance) / self.dist_scale,
            (current_danger_distance - danger_distance) / self.dist_scale,
            float(action == reverse_action),
        ])
        return features

    def _policy_prior(self, features):
        carrying = features[1]
        food_progress = features[26] * self.dist_scale
        home_progress = features[27] * self.dist_scale
        if features[25] > 0.5:
            if carrying >= 0.3:
                score = (0.35 + 0.3 * carrying) * home_progress
            else:
                score = 0.35 * food_progress
            if features[10] * self.dist_scale < 6.0:
                score -= 0.5 * features[28] * self.dist_scale
            return score
        if features[14] > 0.0:
            return -0.35 * features[11] * self.dist_scale
        if features[19] > 0.5:
            return -0.15 * abs(features[16] - 0.5) * self.dist_scale
        return 0.25 * food_progress

    def _nearest_distance(self, position, targets):
        if position is None or not targets:
            return self.dist_scale
        return min(self.get_maze_distance(position, target) for target in targets)


class PPOCaptureAgent(PPOFeatureCaptureAgent):
    def __init__(self, index, learner):
        super().__init__(index, learner)
        self.pending_transition = None
        self.trajectory = []

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        moving_actions = [action for action in actions if action != Directions.STOP]
        if moving_actions:
            actions = moving_actions

        action_features = [
            (action, self._action_features(game_state, action))
            for action in actions
        ]
        if self.pending_transition is not None and self.learner.training:
            self.pending_transition['reward'] = self._reward_since_last_action(game_state)
            self.trajectory.append(self.pending_transition)
            self.pending_transition = None

        action, transition = self.learner.act(
            action_features,
            [self._policy_prior(features) for _, features in action_features],
            deterministic=not self.learner.training,
        )
        if self.learner.training:
            self.pending_transition = transition
            self.last_transition = next(
                features for candidate, features in action_features if candidate == action
            )
            self._remember_reward_baseline(game_state)
        return action

    def final(self, game_state):
        if not self.learner.training:
            return
        if self.pending_transition is not None:
            reward = self._reward_since_last_action(game_state)
            final_score = self.get_score(game_state)
            reward += 1.0 if final_score > 0 else -1.0 if final_score < 0 else 0.0
            self.pending_transition['reward'] = reward
            self.trajectory.append(self.pending_transition)
        self.pending_transition = None
        self.last_transition = None
        self.learner.finish_episode(self.trajectory)
        self.trajectory = []
