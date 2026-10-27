"""Prepare RESISC45 first, then forward training arguments unchanged to the runner."""
import argparse
import subprocess
import sys
from pathlib import Path
from prepare_resisc45 import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__,epilog='All other arguments are passed to run_resisc45.py; see that script --help.')
    p.add_argument('--data-root',default='./data/NWPU-RESISC45')
    p.add_argument('--download-only',action='store_true',help='Download/check data without starting training; no GPU needed')
    p.add_argument('--archive',help='Offline: use the checksum-matching NWPU-RESISC45.zip')
    a, training_args = p.parse_known_args()
    try:
        root = prepare(a.data_root,archive=a.archive)
    except (OSError,ValueError,RuntimeError) as error:
        p.exit(1,'Dataset preparation failed: '+str(error)+'\n')
    if a.download_only:
        return
    script = Path(__file__).resolve().with_name('run_resisc45.py')
    print('Data ready; starting the original training/unlearning runner.',flush=True)
    result = subprocess.run([sys.executable,'-u',str(script),'--data-root',str(root),*training_args])
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
