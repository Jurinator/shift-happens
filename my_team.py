# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import heapq
import random
from pathlib import Path

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from q_network import LinearQLearner


ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
FEATURE_NAMES = (
    'pacman',
    'carrying',
    'returned',
    'score',
    'time_remaining',
    'enemy_food_remaining',
    'defended_food_remaining',
    'food_distance',
    'home_distance',
    'capsule_distance',
    'opponent_distance',
    'invader_distance',
    'vulnerable_ghost_distance',
    'scared_timer',
    'invader_count',
    'visible_opponent_count',
    'oriented_x',
    'y_position',
    'teammate_distance',
    'is_second_teammate',
    'successor_is_pacman',
    'food_progress',
    'home_progress',
    'danger_progress',
    'reverse_action',
    'bias',
)
MODEL_PATH = Path(__file__).with_name('linear_q_weights.json')


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='StrategicCaptureAgent', second='StrategicCaptureAgent', num_training=0,
                model_path=MODEL_PATH):
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
    selected_agents = [
        (index, eval(name))
        for index, name in ((first_index, first), (second_index, second))
    ]
    q_learning_indices = [
        index for index, agent_class in selected_agents
        if agent_class is QLearningCaptureAgent
    ]
    learner = None
    if q_learning_indices:
        learner = LinearQLearner(
            FEATURE_NAMES,
            ACTIONS,
            model_path,
            training=int(num_training) > 0,
        )

    agents = []
    for index, agent_class in selected_agents:
        if agent_class is QLearningCaptureAgent:
            agents.append(
                agent_class(
                    index,
                    learner,
                    decay_exploration=index == min(q_learning_indices),
                )
            )
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


class StrategicCaptureAgent(CaptureAgent):
    def __init__(self, index):
        super().__init__(index, time_for_computing=0.1)
        self.home_entries = []
        self.walls = None
        self.midline = 0

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        walls = game_state.get_walls()
        self.walls = walls
        self.midline = walls.width // 2
        home_x = self.midline - 1 if self.red else self.midline
        self.home_entries = [
            (home_x, y) for y in range(1, walls.height - 1)
            if not walls[home_x][y]
        ]
        if not self.home_entries:
            raise ValueError('The layout has no traversable home-side boundary cells')

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            raise RuntimeError(f'No legal actions available for agent {self.index}')

        current_position = nearest_point(
            game_state.get_agent_state(self.index).get_position()
        )
        agent_state = game_state.get_agent_state(self.index)
        opponents = [
            game_state.get_agent_state(index)
            for index in self.get_opponents(game_state)
        ]
        visible_opponents = [
            opponent for opponent in opponents
            if opponent.get_position() is not None
        ]
        defenders = [
            opponent for opponent in visible_opponents
            if not opponent.is_pacman and opponent.scared_timer <= 0
        ]
        invaders = [
            opponent for opponent in visible_opponents
            if opponent.is_pacman
        ]
        capsules = self.get_capsules(game_state)
        food = self.get_food(game_state).as_list()
        home_distance = self._distance_to_home(current_position)
        threat_distance = self._nearest_distance(
            current_position,
            [opponent.get_position() for opponent in defenders],
        )
        remaining_turns = game_state.data.timeleft / max(
            1, game_state.get_num_agents()
        )
        carrying = agent_state.num_carrying

        goal = None
        if invaders and not agent_state.is_pacman and agent_state.scared_timer <= 0:
            invader = min(
                invaders,
                key=lambda enemy: self.get_maze_distance(
                    current_position, nearest_point(enemy.get_position())
                ),
            )
            if self._is_closest_teammate_to(
                game_state, current_position, nearest_point(invader.get_position())
            ):
                goal = nearest_point(invader.get_position())

        should_return = carrying > 0 and (
            carrying >= 4
            or remaining_turns <= home_distance + 3
            or (carrying >= 1 and threat_distance <= 2)
        )
        if goal is None and should_return:
            goal = self._nearest_home_entry(current_position)

        if goal is None and defenders and capsules:
            capsule = min(
                capsules,
                key=lambda target: self.get_maze_distance(current_position, target),
            )
            capsule_distance = self.get_maze_distance(current_position, capsule)
            food_distance = self._nearest_distance(current_position, food)
            if capsule_distance <= 5 and capsule_distance <= food_distance + 1:
                goal = capsule

        if goal is None and food:
            goal = self._select_food_target(game_state, current_position, food)
        if goal is None and carrying:
            goal = self._nearest_home_entry(current_position)
        if goal is None:
            goal = self._nearest_home_entry(current_position)

        route_costs = self._risk_weighted_distances(
            goal, defenders, carrying
        )
        action_scores = []
        current_direction = agent_state.configuration.direction
        reverse_action = Directions.REVERSE[current_direction]
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            successor_position = successor.get_agent_state(self.index).get_position()
            if successor_position != nearest_point(successor_position):
                successor = successor.generate_successor(self.index, action)
                successor_position = successor.get_agent_state(self.index).get_position()
            successor_position = nearest_point(successor_position)

            score = route_costs.get(
                successor_position,
                self.get_maze_distance(successor_position, goal),
            )
            if action == reverse_action and len(actions) > 1:
                score += 0.05
            action_scores.append((action, score))

        best_score = min(score for _, score in action_scores)
        best_actions = [
            action for action, score in action_scores
            if score <= best_score + 1e-9
        ]
        return random.choice(best_actions)

    def _distance_to_home(self, position):
        return min(self.get_maze_distance(position, entry) for entry in self.home_entries)

    def _nearest_home_entry(self, position):
        return min(
            self.home_entries,
            key=lambda entry: self.get_maze_distance(position, entry),
        )

    def _risk_weighted_distances(self, goal, defenders, carrying):
        distances = {goal: self._cell_risk_cost(goal, defenders, carrying)}
        frontier = [(distances[goal], goal)]
        walls = self.walls
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

        while frontier:
            cost, position = heapq.heappop(frontier)
            if cost > distances[position]:
                continue
            x, y = position
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor
                if (
                    nx < 0 or nx >= walls.width
                    or ny < 0 or ny >= walls.height
                    or walls[nx][ny]
                ):
                    continue
                candidate = (
                    cost + 1.0
                    + self._cell_risk_cost(neighbor, defenders, carrying)
                )
                if candidate < distances.get(neighbor, float('inf')):
                    distances[neighbor] = candidate
                    heapq.heappush(frontier, (candidate, neighbor))
        return distances

    def _cell_risk_cost(self, position, defenders, carrying):
        if not self._is_enemy_territory(position) or not defenders:
            return 0.0
        distance = self._nearest_distance(
            position, [defender.get_position() for defender in defenders]
        )
        return self._danger_penalty(distance) * (
            1.0 + min(carrying, 5) * 0.15
        )

    def _is_enemy_territory(self, position):
        x, _ = position
        return x >= self.midline if self.red else x < self.midline

    def _nearest_distance(self, position, targets):
        if position is None or not targets:
            return float('inf')
        return min(
            self.get_maze_distance(position, nearest_point(target))
            for target in targets
        )

    def _select_food_target(self, game_state, position, food):
        teammate_positions = [
            game_state.get_agent_position(index)
            for index in self.get_team(game_state)
            if index != self.index
            and game_state.get_agent_position(index) is not None
        ]
        defenders = [
            game_state.get_agent_state(index)
            for index in self.get_opponents(game_state)
            if game_state.get_agent_position(index) is not None
            and not game_state.get_agent_state(index).is_pacman
            and game_state.get_agent_state(index).scared_timer <= 0
        ]

        def target_cost(target):
            distance = self.get_maze_distance(position, target)
            nearby_defender = self._nearest_distance(
                target, [enemy.get_position() for enemy in defenders]
            )
            if nearby_defender <= 2:
                distance += 8
            elif nearby_defender <= 4:
                distance += 3
            if teammate_positions:
                teammate_distance = self._nearest_distance(target, teammate_positions)
                if teammate_distance + 1 < distance:
                    distance += 2
            return distance

        return min(food, key=target_cost)

    def _is_closest_teammate_to(self, game_state, position, target):
        team_positions = [
            (index, game_state.get_agent_position(index))
            for index in self.get_team(game_state)
            if game_state.get_agent_position(index) is not None
        ]
        if len(team_positions) <= 1:
            return True
        own_distance = self.get_maze_distance(position, target)
        teammate_distances = [
            self.get_maze_distance(teammate_position, target)
            for index, teammate_position in team_positions
            if index != self.index
        ]
        return not teammate_distances or own_distance <= min(teammate_distances)

    @staticmethod
    def _danger_penalty(distance):
        if distance <= 1:
            return 100
        if distance <= 2:
            return 18
        if distance <= 3:
            return 8
        if distance <= 4:
            return 3
        return 0


class QLearningCaptureAgent(CaptureAgent):
    def __init__(self, index, learner, decay_exploration=False):
        super().__init__(index, time_for_computing=0.1)
        self.learner = learner
        self.decay_exploration = decay_exploration
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

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            raise RuntimeError(f'No legal actions available for agent {self.index}')
        action_features = {
            action: self._action_features(game_state, action)
            for action in actions
        }
        if self.last_transition is not None and self.learner.training:
            previous_action, previous_features = self.last_transition
            self.learner.update_weights(
                previous_features,
                previous_action,
                self._reward_since_last_action(game_state),
                action_features,
                done=False,
            )

        action = self.learner.choose_action(action_features)

        chosen_features = action_features[action]
        if self.learner.training:
            self.last_transition = (action, chosen_features)
            self._remember_reward_baseline(game_state)
        return action

    def final(self, game_state):
        if self.learner.training and self.last_transition is not None:
            reward = self._reward_since_last_action(game_state)
            final_score = self.get_score(game_state)
            reward += 1.0 if final_score > 0 else -1.0 if final_score < 0 else 0.0
            action, features = self.last_transition
            self.learner.update_weights(
                features,
                action,
                reward,
                {},
                done=True,
            )
            self.last_transition = None
            if self.decay_exploration:
                self.learner.decay_epsilon()
            self.learner.save()

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
        previous_features = self.last_transition[1]
        previous_food_distance = previous_features['food_distance'] * self.dist_scale
        previous_home_distance = previous_features['home_distance'] * self.dist_scale
        food_progress = previous_food_distance - food_distance
        home_progress = previous_home_distance - home_distance
        carrying_fraction = previous_features['carrying']
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

        values = [
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
        values.extend([
            float(agent_state.is_pacman),
            (current_food_distance - food_distance + 2.0 * food_eaten) / self.dist_scale,
            (current_home_distance - home_distance) / self.dist_scale,
            (current_danger_distance - danger_distance) / self.dist_scale,
            float(action == reverse_action),
            1.0,
        ])
        return dict(zip(FEATURE_NAMES, values))

    def _nearest_distance(self, position, targets):
        if position is None or not targets:
            return self.dist_scale
        return min(self.get_maze_distance(position, target) for target in targets)
