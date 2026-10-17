"""CPU end-to-end smoke run. Synthetic data and tiny model; no paper result claim."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='work/smoke')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(a.out).resolve()
    data, ckpt = out/'data', out/'checkpoints'
    common = ['--size', '32', '--threads', '2', '--device', 'cpu']
    train = ['--data', str(data), '--out', str(ckpt), '--base-channels', '8',
             '--time-dim', '32', '--batch-size', '2', '--epochs', '1', '--updates', '2']
    def run(command, *args):
        subprocess.run([sys.executable, '-m', 'lithogrpo', command, *common, *args], cwd=root, check=True)
    run('toy-data', '--out', str(data), '--count', '8')
    run('train', *train)
    run('train', *train, '--stage', 'sft', '--toy', '--lr', '0.00002',
        '--checkpoint', str(ckpt/'pretrain_last.pt'))
    metric = ['--toy', '--shot-size', '32', '--batch-size', '2', '--data', str(data)]
    run('baseline', *metric, '--checkpoint', str(ckpt/'sft_last.pt'), '--out', str(out/'baseline.json'))
    run('train', *train, *metric, '--stage', 'rlft', '--lr', '0.000002', '--group-size', '3',
        '--sde-steps', '3', '--checkpoint', str(ckpt/'sft_last.pt'), '--baseline', str(out/'baseline.json'))
    run('infer', '--checkpoint', str(ckpt/'rlft_last.pt'), '--target', str(data/'target/00000.png'),
        '--out', str(out/'inference'))
    run('evaluate', *metric, '--split', 'test', '--checkpoint', str(ckpt/'rlft_last.pt'),
        '--out', str(out/'evaluation'))
    print(f'Smoke pipeline completed: {out}')


if __name__ == '__main__':
    main()
