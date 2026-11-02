import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Train linear Q-learning candidates in parallel, then promote the best.'
    )
    parser.add_argument('--generations', type=int, default=12)
    parser.add_argument('--workers', type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument('--episodes', type=int, default=20, help='Training games per candidate')
    parser.add_argument('--eval-games', type=int, default=8,
                        help='Held-out games per candidate, split across colors (minimum 4)')
    parser.add_argument('--layout', default='RANDOM')
    parser.add_argument('--time', type=int, default=1200, help='Game plies per episode')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--fresh', action='store_true', help='Start all candidates from random weights')
    args = parser.parse_args()
    if min(args.generations, args.workers, args.episodes, args.eval_games, args.time) < 1:
        parser.error('--generations, --workers, --episodes, --eval-games, and --time must be positive')
    if args.eval_games < 4:
        parser.error('--eval-games must be at least 4 so each color gets a score summary')

    root = Path(__file__).resolve().parent
    capture = root / 'pacman-contest' / 'src' / 'contest' / 'capture.py'
    team = root / 'my_team.py'
    baseline = root / 'pacman-contest' / 'src' / 'contest' / 'baseline_team.py'
    champion = root / 'linear_q_weights.json'
    if not capture.exists():
        parser.error('pacman-contest submodule is missing; initialize it with git submodule update --init')
    environment = os.environ.copy()
    source_path = str(root / 'pacman-contest' / 'src')
    environment['PYTHONPATH'] = os.pathsep.join(
        path for path in (source_path, environment.get('PYTHONPATH', '')) if path
    )
    layout = _resolve_layout(root, args.layout, parser)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_directory = root / 'training_runs' / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    initial_model = champion if champion.exists() and not args.fresh else None

    champion_score = None
    if initial_model is not None:
        print(
            f"Evaluating starting champion on {args.eval_games} held-out games...",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix='pacman-q-champion-eval-') as workdir:
            champion_result = _evaluate_model(
                'starting champion',
                str(initial_model),
                str(capture),
                str(team),
                str(baseline),
                str(layout),
                environment,
                workdir,
                args.eval_games,
                args.time,
                args.seed,
            )
        champion_score = champion_result['average_score']
        print(
            f"Starting champion: held-out average {champion_score:+.2f} "
            f"({champion_result['games']} games)",
            flush=True,
        )

    jobs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for generation in range(args.generations):
            model_path = run_directory / f'candidate_{generation:02d}.json'
            if initial_model is not None:
                shutil.copy2(initial_model, model_path)
            jobs.append(pool.submit(
                _run_candidate,
                generation,
                str(model_path),
                str(capture),
                str(team),
                str(baseline),
                str(layout),
                environment,
                args.episodes,
                args.eval_games,
                args.time,
                args.seed + generation * 1009,
                args.seed,
            ))

        results = []
        for job in concurrent.futures.as_completed(jobs):
            try:
                result = job.result()
                results.append(result)
                print(
                    f"Candidate {result['generation']:02d}: "
                    f"held-out average {result['average_score']:+.2f} "
                    f"({result['games']} games)",
                    flush=True,
                )
            except Exception as error:
                print(f'Candidate failed: {error}', file=sys.stderr, flush=True)

    if not results:
        raise RuntimeError('Every candidate failed; the existing champion was left unchanged')
    winner = _select_winner(results, champion_score)
    if winner is None:
        print(
            f"Keeping existing champion ({champion_score:+.2f}); "
            f"best candidate scored {max(result['average_score'] for result in results):+.2f}. "
            f"Trained candidates are saved in {run_directory}",
            flush=True,
        )
    else:
        shutil.copy2(winner['model_path'], champion)
        print(
            f"Promoted candidate {winner['generation']:02d}: "
            f"held-out average {winner['average_score']:+.2f} "
            f"({winner['games']} games). Weights: {champion}",
            flush=True,
        )
    print(f'Candidate checkpoints: {run_directory}')


def _run_candidate(generation, model_path, capture, team, baseline, layout,
                   environment, episodes, eval_games, game_time, seed, eval_seed):
    model_path = Path(model_path)
    red_train = episodes // 2
    blue_train = episodes - red_train
    with tempfile.TemporaryDirectory(prefix=f'pacman-q-{generation}-') as workdir:
        for side, count in (('blue', blue_train), ('red', red_train)):
            if count:
                _run_match(
                    capture, team, baseline, layout, environment, workdir,
                    model_path, side, count, game_time, f'{seed}-{side}',
                    training=True, run_name=f'train_{generation}_{side}',
                )

        evaluation = _evaluate_model(
            f'candidate_{generation}',
            str(model_path),
            capture,
            team,
            baseline,
            layout,
            environment,
            workdir,
            eval_games,
            game_time,
            eval_seed,
        )

    return {
        'generation': generation,
        'model_path': str(model_path),
        **evaluation,
    }


def _evaluate_model(label, model_path, capture, team, baseline, layout,
                     environment, workdir, eval_games, game_time, eval_seed):
    blue_eval = (eval_games + 1) // 2
    red_eval = eval_games // 2
    scores = []
    for side, count in (('blue', blue_eval), ('red', red_eval)):
        if count:
            output = _run_match(
                capture, team, baseline, layout, environment, workdir,
                model_path, side, count, game_time, f'{eval_seed}-eval-{side}',
                training=False, run_name=f'eval_{label}_{side}',
            )
            red_score = _parse_average_score(output, label)
            scores.extend([red_score if side == 'red' else -red_score] * count)

    return {
        'average_score': sum(scores) / len(scores),
        'games': len(scores),
    }


def _parse_average_score(output, label):
    match = re.search(r'Average Score:\s*(-?\d+(?:\.\d+)?)', output)
    if match is None:
        raise RuntimeError(
            f'Could not parse evaluation score for {label}: {output[-1000:]}'
        )
    return float(match.group(1))


def _select_winner(results, champion_score):
    winner = max(results, key=lambda result: result['average_score'])
    if champion_score is not None and winner['average_score'] <= champion_score:
        return None
    return winner


def _run_match(capture, team, baseline, layout, environment, workdir,
               model_path, side, games, game_time, seed, training, run_name):
    options = [
        'first=QLearningCaptureAgent',
        'second=QLearningCaptureAgent',
        f'model_path={model_path}',
    ]
    if training:
        options.append('num_training=1')
    if side == 'blue':
        red_team, blue_team = baseline, team
        team_option = '--blueOpts'
    else:
        red_team, blue_team = team, baseline
        team_option = '--redOpts'
    command = [
        sys.executable, capture,
        '--quiet', '--catch-exceptions',
        '--num_games', str(games), team_option, ','.join(options),
        '--layout', layout, '--time', str(game_time), '--delay-step', '0',
        '--setRandomSeed', str(seed),
        '--contest-name', run_name, '--match-identifier', str(os.getpid()),
        '--red', red_team, '--blue', blue_team,
    ]
    completed = subprocess.run(
        command, cwd=workdir, env=environment, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f'{run_name} exited with {completed.returncode}:\n'
            f'{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}'
        )
    return completed.stdout


def _resolve_layout(root, requested, parser):
    if requested == 'RANDOM' or requested.startswith('RANDOM'):
        return requested
    path = Path(requested)
    if not path.is_absolute():
        layout_directory = root / 'pacman-contest' / 'src' / 'contest' / 'layouts'
        candidates = [root / path, layout_directory / path]
        if path.suffix != '.lay':
            candidates.extend([root / path.with_suffix('.lay'), layout_directory / path.with_suffix('.lay')])
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.exists():
        parser.error(f'layout not found: {requested}')
    return path.resolve()


if __name__ == '__main__':
    main()