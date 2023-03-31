import sys
import argparse
from pathlib import Path
import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str,
        help='Path to config file (.yaml). See cli/configs/ for examples.'
    )
    parser.add_argument('-e', '--estimate', action='store_true',
        help='Estimate the total runtime of the script'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh.read())

    # Check if config file is correct
    if config['cli'] != 'generate_answers':
        print(f'Config file is not for generate_answers.py: "{args.config}"')
        sys.exit()

    local_dir = config['results'].get('local_dir', None)
    filename = config['results']['filename']

    # Check if results file already exists
    if local_dir and (Path(local_dir) / filename).exists():
        print(f'Results file already exists locally: "{local_dir}/{filename}"')
        sys.exit()

    raise NotImplementedError  # TODO: still incomplete
