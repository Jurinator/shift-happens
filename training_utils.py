import os
import subprocess
import sys
from pathlib import Path


def _run_match(capture, team, opponent, layout, environment, workdir,
              model_path, side, games, game_time, seed, training, run_name,
              agent_options=None):
    options = [f'model_path={model_path}']
    if agent_options is None:
        options.extend([
            'first=PPOCaptureAgent',
            'second=PPOCaptureAgent',
        ])
    else:
        options.extend(agent_options)
    if training:
        options.append('num_training=1')
    if side == 'blue':
        red_team, blue_team = opponent, team
        team_option = '--blueOpts'
    else:
        red_team, blue_team = team, opponent
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


def _resolve_opponent(root, requested, default_opponent, parser):
    if requested is None:
        return str(default_opponent.resolve())

    path = Path(requested)
    if not path.is_absolute():
        candidates = [
            root / path,
            root / 'pacman-contest' / 'src' / 'contest' / path,
        ]
        if path.suffix != '.py':
            candidates.extend([
                root / path.with_suffix('.py'),
                root / 'pacman-contest' / 'src' / 'contest' / path.with_suffix('.py'),
            ])
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        parser.error(f'opponent team file not found: {requested}')
    return str(path.resolve())


def _resolve_layout(root, requested, parser):
    if requested == 'RANDOM' or requested.startswith('RANDOM'):
        return requested
    path = Path(requested)
    if not path.is_absolute():
        layout_directory = root / 'pacman-contest' / 'src' / 'contest' / 'layouts'
        candidates = [root / path, layout_directory / path]
        if path.suffix != '.lay':
            candidates.extend([
                root / path.with_suffix('.lay'),
                layout_directory / path.with_suffix('.lay'),
            ])
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    if not path.exists():
        parser.error(f'layout not found: {requested}')
    return path.resolve()
