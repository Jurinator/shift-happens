# Pacman Agent

A template for coding a pacman agent.

## Default agent
The default team uses a goal-based planner. It selects between collecting food,
returning carried food, taking a nearby capsule when defenders are close, and
intercepting visible invaders. It uses a risk-weighted shortest-path search
(Dijkstra) to route around nearby unscared defenders, increases route risk when
carrying food, and coordinates food targets with its teammate.

The linear Q-learning agent remains available for experiments. It learns a
separate feature-weight vector for each legal action, using an epsilon-greedy
policy and temporal-difference updates. `train_q_agent.py` trains and evaluates
it; it is not required to run the default planner.

## Training the linear Q-learning agent
Install the contest dependencies as described below, then run from the
repository root:

```sh
python train_q_agent.py --generations 1 --workers 1 --episodes 2 --eval-games 4 --time 120
```

This brief run checks the training/evaluation setup. For longer games and more
training, increase `--time`, `--generations`, and `--episodes`, for example:

```sh
python train_q_agent.py --generations 12 --workers 4 --episodes 100 --eval-games 12
```

Each generation trains separate candidate weights, tests each candidate on
held-out games as both red and blue. When continuing from existing weights,
the trainer also evaluates the starting champion on those same games and only
replaces it if a candidate scores strictly higher. A worse run still trains
and saves its candidate checkpoints, but keeps the stronger champion.
Candidates start from the current weights; pass `--fresh` to train from zero
weights and deliberately skip the champion comparison. Existing neural-network
weights in `q_network_weights.json` are incompatible with this linear model
and are left untouched.

To play the trained agent against the baseline from the contest directory:

```sh
cd pacman-contest/src/contest
python capture.py -r baseline_team -b ../../../my_team.py \
  --blueOpts first=QLearningCaptureAgent,second=QLearningCaptureAgent,model_path=../../../linear_q_weights.json
```

The default team remains the strategic planner. Training and evaluating the
linear agent does not change it.

## Setting up the environment
1. Copy the code from this framework to create your Pacman Agent.
2. Clone your own pacman-agent repository (e.g., `git clone git@github.com:Username/pacman-agent.git`)
3. Go into pacman-agent folder, `cd pacman-agent/`
4. Run `git submodule update --init --remote` to pull the last pacman-contest
5. Create a virtual environment (e.g., `python3.8 -m venv venv`)
6. Activate the virtual environment with `source venv/bin/activate`
7. Go to the pacman-contest folder and install the requirements:
    - `cd pacman-contest/`
    - `pip install -r requirements.txt`
    - `pip install -e .`

## Coding a new agent
In the root folder do the following:
1. Create in `my_team.py` a class with the name of your agent that inherits from `CaptureAgent`, e.g. `class ReflexCaptureAgent(CaptureAgent):`
2. In the new class, override the `def choose_action(self, game_state):` function to return the best next action (check the given source code example).
3. (Optional) Add any other functions to the class for reasoning / learning and improving your agents decision which could also use other code sources in the same folder.

To debug the agent you can run `capture.py` between the `baseline_team` and your current agent:
1. `cd pacman-contest/src/contest/`
2. `python capture.py -r baseline_team -b ../../../my_team.py`
