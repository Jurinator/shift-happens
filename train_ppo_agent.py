import argparse
import concurrent.futures
import json
import os
import re
import shutil
import time
from pathlib import Path

from ppo_network import SharedPPOLearner
from training_utils import _resolve_layout, _resolve_opponent, _run_match


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate the shared PPO Pacman agent.'
    )
    parser.add_argument('--episodes', type=int, default=100,
                        help='Training games, split across red and blue')
    parser.add_argument('--eval-games', type=int, default=8,
                        help='Held-out evaluation games (minimum 4)')
    parser.add_argument('--workers', type=int, default=min(os.cpu_count() or 1, 4),
                        help='Parallel game simulations (default: up to 4)')
    parser.add_argument('--layout', default='RANDOM')
    parser.add_argument(
        '--opponent', default=None,
        help='Opponent team Python file (default: contest baseline_team.py)',
    )
    parser.add_argument('--time', type=int, default=1200, help='Game plies per episode')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model-path', type=Path, default=None,
                        help='Checkpoint path (default: ppo_weights.pt in the repository)')
    parser.add_argument('--fresh', action='store_true',
                        help='Train from a new policy instead of the current checkpoint')
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_games < 4 or args.time < 1 or args.workers < 1:
        parser.error('--episodes, --time, and --workers must be positive; --eval-games must be at least 4')

    root = Path(__file__).resolve().parent
    capture = root / 'pacman-contest' / 'src' / 'contest' / 'capture.py'
    team = root / 'my_team.py'
    default_opponent = root / 'pacman-contest' / 'src' / 'contest' / 'baseline_team.py'
    champion = args.model_path or root / 'ppo_weights.pt'
    if not champion.is_absolute():
        champion = root / champion
    if not capture.exists():
        parser.error('pacman-contest submodule is missing; initialize it with git submodule update --init')
    environment = os.environ.copy()
    source_path = str(root / 'pacman-contest' / 'src')
    environment['PYTHONPATH'] = os.pathsep.join(
        path for path in (source_path, environment.get('PYTHONPATH', '')) if path
    )
    layout = _resolve_layout(root, args.layout, parser)
    opponent = _resolve_opponent(root, args.opponent, default_opponent, parser)
    run_directory = root / 'training_runs' / ('ppo-' + time.strftime('%Y%m%d-%H%M%S'))
    run_directory.mkdir(parents=True, exist_ok=True)
    candidate = run_directory / 'candidate.pt'
    if champion.exists() and not args.fresh:
        shutil.copy2(champion, candidate)

    learner = SharedPPOLearner(
        30, candidate, training=True, agents_per_team=2, seed=args.seed
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch_start in range(0, args.episodes, args.workers):
            batch_size = min(args.workers, args.episodes - batch_start)
            rollout_directory = run_directory / f'rollouts_{batch_start:05d}'
            rollout_directory.mkdir(parents=True, exist_ok=True)
            futures = []
            for offset in range(batch_size):
                game_index = batch_start + offset
                side = 'red' if game_index % 2 == 0 else 'blue'
                rollout_path = rollout_directory / f'game_{game_index:05d}.json'
                futures.append(pool.submit(
                    _collect_rollout,
                    str(capture), str(team), str(opponent), str(layout),
                    environment, str(run_directory), str(candidate), side,
                    args.time, args.seed + game_index, str(rollout_path),
                    game_index,
                ))
            for future in concurrent.futures.as_completed(futures):
                future.result()

            episodes = []
            for rollout_path in sorted(rollout_directory.glob('game_*.json')):
                episodes.extend(json.loads(rollout_path.read_text()))
            if len(episodes) != batch_size * 2:
                raise RuntimeError(
                    f'Expected {batch_size * 2} agent rollouts, received {len(episodes)}'
                )
            learner.update_episodes(episodes)
            print(
                f'Trained on {min(batch_start + batch_size, args.episodes)}/'
                f'{args.episodes} games ({batch_size} parallel workers)',
                flush=True,
            )

    candidate_wins, candidate_games = _evaluate_parallel(
        str(candidate), str(capture), str(team), str(opponent), str(layout),
        environment, run_directory, args.eval_games, args.workers, args.time,
        args.seed + 1_000_000, 'ppo_eval_candidate',
    )
    candidate_win_rate = candidate_wins / candidate_games
    print(
        f'Candidate win rate: {candidate_win_rate:.1%} '
        f'({candidate_wins}/{candidate_games})'
    )
    if champion.exists() and not args.fresh:
        incumbent_wins, incumbent_games = _evaluate_parallel(
            str(champion), str(capture), str(team), str(opponent), str(layout),
            environment, run_directory, args.eval_games, args.workers, args.time,
            args.seed + 1_000_000, 'ppo_eval_incumbent',
        )
        incumbent_win_rate = incumbent_wins / incumbent_games
        print(
            f'Incumbent win rate: {incumbent_win_rate:.1%} '
            f'({incumbent_wins}/{incumbent_games})'
        )
        if candidate_wins * incumbent_games <= incumbent_wins * candidate_games:
            print(f'Kept incumbent; candidate checkpoint: {candidate}')
            return

    shutil.copy2(candidate, champion)
    print(f'Promoted PPO checkpoint: {champion}')


def _collect_rollout(capture, team, opponent, layout, environment, workdir,
                     model_path, side, game_time, seed, rollout_path, game_index):
    team_options = [
        'first=PPOCaptureAgent',
        'second=PPOCaptureAgent',
        f'rollout_path={rollout_path}',
        f'ppo_seed={seed}',
    ]
    _run_match(
        capture, team, opponent, layout, environment, workdir, model_path,
        side, 1, game_time, str(seed), training=True,
        run_name=f'ppo_train_{game_index:05d}', agent_options=team_options,
    )
    if not Path(rollout_path).is_file():
        raise RuntimeError(f'PPO rollout worker did not write {rollout_path}')


def _evaluate_parallel(model_path, capture, team, opponent, layout, environment,
                       workdir, games, workers, game_time, seed, run_name):
    side_counts = {
        'red': (games + 1) // 2,
        'blue': games // 2,
    }
    jobs = []
    job_index = 0
    for side, side_count in side_counts.items():
        while side_count:
            batch_size = 2 if side_count == 4 else min(3, side_count)
            side_count -= batch_size
            jobs.append((job_index, side, batch_size))
            job_index += 1

    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for index, side, batch_size in jobs:
            future = pool.submit(
                _run_match,
                capture, team, opponent, layout, environment, str(workdir),
                model_path, side, batch_size, game_time, str(seed + index),
                False, f'{run_name}_{index:05d}',
                ['first=PPOCaptureAgent', 'second=PPOCaptureAgent'],
            )
            futures[future] = (side, batch_size, index)

    total_wins = 0
    for future, (side, batch_size, index) in futures.items():
        output = future.result()
        match = re.search(r'Red Win Rate:\s*(\d+)/(\d+)', output)
        if match is None:
            raise RuntimeError(
                f'Could not parse evaluation win rate for {run_name}/{index}: '
                f'{output[-1000:]}'
            )
        red_wins, games_played = map(int, match.groups())
        if games_played != batch_size:
            raise RuntimeError(
                f'Expected {batch_size} evaluation games for {run_name}/{index}, '
                f'engine reported {games_played}'
            )
        if side == 'red':
            total_wins += red_wins
        else:
            blue_match = re.search(r'Blue Win Rate:\s*(\d+)/(\d+)', output)
            if blue_match is None:
                raise RuntimeError(
                    f'Could not parse blue evaluation wins for {run_name}/{index}: '
                    f'{output[-1000:]}'
                )
            blue_wins, blue_games = map(int, blue_match.groups())
            if blue_games != batch_size:
                raise RuntimeError(
                    f'Expected {batch_size} evaluation games for {run_name}/{index}, '
                    f'engine reported {blue_games}'
                )
            total_wins += blue_wins
    return total_wins, games


if __name__ == '__main__':
    main()
