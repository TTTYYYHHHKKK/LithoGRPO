"""Run with python -m lithogrpo --help. Paths are relative to the current directory."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .data import LayoutPairs, synthetic_data, image_tensor
from .flow import (matching_loss, ode_sample, sde_sample, advantages, grpo_loss,
                   trajectory_log_ratio, Trajectory, transition)
from .metrics import Metrics, KEYS, normalized_reward
from .model import FlowUNet
from .physics import Lithography, litho_losses


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def make_sim(a):
    return Lithography(kernel_dir=a.kernel_dir, toy=a.toy, kernel_num=a.kernel_num,
                       threshold=a.threshold, steepness=a.steepness,
                       dose_nom=a.dose_nom, dose_max=a.dose_max, dose_min=a.dose_min).to(a.device)


def make_metrics(a, sim):
    return Metrics(sim, a.shot_backend, a.shot_size, a.field_nm, a.epe_backend, a.lithobench_root)


def load_model(a):
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True) if a.checkpoint else None
    spec = checkpoint['model_config'] if checkpoint else dict(base_channels=a.base_channels, time_dim=a.time_dim)
    model = FlowUNet(**spec).to(a.device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    return model, checkpoint, spec


def data_loader(a, paired=True, shuffle=False):
    ds = LayoutPairs(a.data, a.size, a.split, paired=paired)
    return DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers)


def snapshot(model, optimizer, a, spec, epoch, update):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                model_config=spec, stage=a.stage, epoch=epoch, update=update,
                settings=vars(a), torch_rng=torch.get_rng_state())


def train(a):
    model, previous, spec = load_model(a)
    model.train()
    if a.stage != 'pretrain' and previous is None:
        raise ValueError('SFT/RLFT require a trained --checkpoint')
    loader = data_loader(a, paired=a.stage != 'rlft', shuffle=True)
    sim = make_sim(a) if a.stage != 'pretrain' else None
    metric = make_metrics(a, sim) if a.stage == 'rlft' else None
    baseline = None
    if a.stage == 'rlft':
        if not a.baseline:
            raise ValueError('RLFT requires --baseline generated from the end-of-SFT model')
        saved = json.loads(Path(a.baseline).read_text(encoding='utf8'))
        baseline = saved['metrics']
        if saved.get('toy') != a.toy:
            raise ValueError('Baseline optics mode does not match training')
        for key in ('shot_backend', 'shot_size', 'size', 'field_nm', 'epe_backend', 'data', 'split',
                    'kernel_num', 'threshold', 'steepness', 'dose_nom', 'dose_max', 'dose_min'):
            if saved['settings'].get(key) != getattr(a, key):
                raise ValueError(f'Baseline setting {key} does not match training')
        if not a.toy and Path(saved['settings']['kernel_dir']).resolve() != Path(a.kernel_dir).resolve():
            raise ValueError('Baseline optical kernels must match training')
        if not a.resume and Path(saved['checkpoint']).resolve() != Path(a.checkpoint).resolve():
            raise ValueError('RLFT must start from the SFT checkpoint used to compute its baseline')
        if a.group_size < 2:
            raise ValueError('GRPO needs at least two candidates per layout')
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    start_epoch, updates = 0, 0
    if a.resume:
        if not previous or previous['stage'] != a.stage:
            raise ValueError('--resume needs a checkpoint from the same stage')
        optimizer.load_state_dict(previous['optimizer'])
        start_epoch, updates = previous['epoch']+1, previous['update']
        torch.set_rng_state(previous['torch_rng'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/f'{a.stage}_settings.json', vars(a))
    max_updates = a.updates if a.updates else (1000 if a.stage == 'rlft' else a.epochs*len(loader))
    epoch_limit = max(a.epochs, math.ceil(max_updates/len(loader))+start_epoch) if a.stage == 'rlft' else a.epochs
    for epoch in range(start_epoch, epoch_limit):
        for target, mask, _ in loader:
            if updates >= max_updates:
                break
            target, mask = target.to(a.device), mask.to(a.device)
            if a.stage == 'pretrain':
                lr = a.lr * (.1 if epoch >= max(1, a.epochs//2) else 1)
            elif a.stage == 'rlft':
                warm = max(1, round(max_updates*.05))
                phase = max(0., (updates-warm)/max(1, max_updates-warm))
                lr = a.lr*((updates+1)/warm if updates < warm else .05+.95*(1+math.cos(math.pi*phase))/2)
            else:
                lr = a.lr
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            values = {}
            if a.stage != 'rlft':
                total = 0.
                for start in range(0, len(target), a.microbatch):
                    t, m = target[start:start+a.microbatch], mask[start:start+a.microbatch]
                    flow, predicted = matching_loss(model, t, m)
                    if a.stage == 'sft':
                        l2, pvb = litho_losses(sim, predicted.clamp(0, 1), t)
                        loss = (1-a.lambda_l2-a.lambda_pvb)*flow + a.lambda_l2*l2 + a.lambda_pvb*pvb
                    else:
                        loss = flow
                    (loss.sum()/len(target)).backward()
                    total += float(loss.detach().sum())/len(target)
                values['loss'] = total
            else:
                repeated = target.repeat_interleave(a.group_size, dim=0)
                # Store rollouts on CPU to bound accelerator memory.
                rollouts, all_metrics = [], {k: [] for k in KEYS}
                for start in range(0, len(repeated), a.microbatch):
                    t = repeated[start:start+a.microbatch]
                    generated, trajectory = sde_sample(model, t, a.sde_steps, a.noise_level,
                                                       a.start_time, a.noise_type, a.cutoff)
                    measured = metric(generated, t)
                    for k in KEYS:
                        all_metrics[k].append(measured[k])
                    trajectory.states = [x.cpu() for x in trajectory.states]
                    trajectory.means = [x.cpu() for x in trajectory.means]
                    rollouts.append((start, trajectory))
                measured = {k: torch.cat(v) for k, v in all_metrics.items()}
                rewards = normalized_reward(measured, baseline, a.baseline_floor)
                adv = advantages(rewards, a.group_size)
                loss_total = 0.
                for start, trajectory in rollouts:
                    trajectory.states = [x.to(a.device) for x in trajectory.states]
                    trajectory.means = [x.to(a.device) for x in trajectory.means]
                    t = repeated[start:start+len(trajectory.states[0])]
                    # First obtain the trajectory ratio, then its scalar derivative.
                    # A second pass backprops each transition separately, keeping
                    # activation memory independent of the number of SDE steps.
                    with torch.no_grad():
                        log_ratio = trajectory_log_ratio(model, t, trajectory, a.noise_level)
                    log_ratio.requires_grad_(True)
                    loss = grpo_loss(log_ratio, adv[start:start+len(t)], a.clip)*len(t)/len(repeated)
                    coefficient, = torch.autograd.grad(loss, log_ratio)
                    for x, nxt, old, ti, var in zip(trajectory.states[:-1], trajectory.states[1:],
                                                  trajectory.means, trajectory.times, trajectory.variances):
                        mean, _ = transition(model, x, t, ti, trajectory.dt, a.noise_level)
                        residual, delta = nxt.double()-old.double(), mean.double()-old.double()
                        part = ((2*residual*delta-delta.square())/(2*var)).flatten(1).sum(1)
                        (part*coefficient).sum().backward()
                    loss_total += float(loss.detach())
                values = dict(loss=loss_total, reward=float(rewards.mean()),
                              advantage_std=float(adv.std(unbiased=False)),
                              **{k: float(v.mean()) for k, v in measured.items()})
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            record = dict(epoch=epoch+1, update=updates, lr=lr, grad_norm=float(grad), **values)
            with (out/f'{a.stage}.jsonl').open('a', encoding='utf8') as handle:
                handle.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            if updates % a.save_every == 0:
                torch.save(snapshot(model, optimizer, a, spec, epoch, updates), out/f'{a.stage}_{updates:06d}.pt')
        torch.save(snapshot(model, optimizer, a, spec, epoch, updates), out/f'{a.stage}_last.pt')
        if updates >= max_updates:
            break


@torch.no_grad()
def baseline(a):
    model, previous, _ = load_model(a)
    if previous is None or previous.get('stage') != 'sft':
        raise ValueError('Reward baseline must use an end-of-SFT checkpoint')
    model.eval()
    sim = make_sim(a)
    metric = make_metrics(a, sim)
    values = {k: [] for k in KEYS}
    for target, _, _ in data_loader(a, paired=False):
        target = target.to(a.device)
        for k, v in metric(ode_sample(model, target, a.steps), target).items():
            values[k].extend(v.cpu().tolist())
    result = dict(metrics={k: float(np.mean(v)) for k, v in values.items()},
                  checkpoint=str(Path(a.checkpoint).resolve()), toy=a.toy,
                  shot_backend=a.shot_backend, shot_size=a.shot_size,
                  epe_backend=a.epe_backend, settings=vars(a))
    save_json(a.out, result)
    print(json.dumps(result, indent=2))


@torch.no_grad()
def inference(a):
    if not a.checkpoint:
        raise ValueError('Inference requires --checkpoint')
    model, _, _ = load_model(a)
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.command == 'infer':
        target = image_tensor(a.target, a.size)[None].to(a.device)
        result = ode_sample(model, target, a.steps)[0, 0].clamp(0, 1).cpu().numpy()
        Image.fromarray((result*255).astype(np.uint8)).save(out/'mask_soft.png')
        Image.fromarray((result >= .5).astype(np.uint8)*255).save(out/'mask.png')
        return
    sim, rows = make_sim(a), []
    metric = make_metrics(a, sim)
    for target, _, names in data_loader(a, paired=False):
        target = target.to(a.device)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        mask = ode_sample(model, target, a.steps)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = (time.perf_counter()-begin)/len(target)
        measured = metric(mask, target)
        resist, _, _ = sim((mask >= .5).float(), hard=True)
        for i, name in enumerate(names):
            rows.append(dict(name=name, seconds=seconds, **{k: float(v[i]) for k, v in measured.items()}))
            image = torch.cat((target[i, 0], (mask[i, 0] >= .5).float(), resist[i, 0]), dim=1)
            Image.fromarray((image.cpu().numpy()*255).astype(np.uint8)).save(out/f'{name}_target_mask_resist.png')
    report = dict(mean={k: float(np.mean([r[k] for r in rows])) for k in (*KEYS, 'seconds')},
                  samples=rows, settings=vars(a),
                  note='Synthetic optics; not a paper benchmark' if a.toy else
                       ('Local EPE is approximate; see REPRODUCTION.md' if a.epe_backend == 'local'
                        else 'Native LithoBench EPE; verify optical resolution and protocol before comparison'))
    save_json(out/'metrics.json', report)
    print(json.dumps(report['mean'], indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('toy-data', 'train', 'baseline', 'infer', 'evaluate'))
    p.add_argument('--config', help='JSON defaults; explicit CLI flags override them')
    p.add_argument('--stage', choices=('pretrain', 'sft', 'rlft'), default='pretrain')
    p.add_argument('--data')
    p.add_argument('--out', default='results/reconstruction')
    p.add_argument('--target')
    p.add_argument('--checkpoint')
    p.add_argument('--resume', action='store_true', help='Continue at next epoch boundary; see documentation')
    p.add_argument('--baseline')
    p.add_argument('--split', choices=('train', 'test', 'all'), default='train')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--base-channels', type=int, default=64)
    p.add_argument('--time-dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--updates', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=12, help='Number of layouts, RLFT expands this by G')
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--lambda-l2', type=float, default=.002)
    p.add_argument('--lambda-pvb', type=float, default=.248)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--group-size', type=int, default=6)
    p.add_argument('--sde-steps', type=int, default=10)
    p.add_argument('--start-time', type=float, default=.1)
    p.add_argument('--noise-level', type=float, default=.1)
    p.add_argument('--noise-type', choices=('colored', 'white'), default='colored')
    p.add_argument('--cutoff', type=float, default=.1)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--baseline-floor', type=float, default=1.)
    p.add_argument('--toy', action='store_true')
    p.add_argument('--kernel-dir')
    p.add_argument('--kernel-num', type=int, default=24)
    p.add_argument('--threshold', type=float, default=.225)
    p.add_argument('--steepness', type=float, default=50.)
    p.add_argument('--dose-nom', type=float, default=1.)
    p.add_argument('--dose-max', type=float, default=1.02)
    p.add_argument('--dose-min', type=float, default=.98)
    p.add_argument('--field-nm', type=float, default=2048.)
    p.add_argument('--shot-backend', choices=('fast', 'reference'), default='fast')
    p.add_argument('--shot-size', type=int, default=512)
    p.add_argument('--epe-backend', choices=('local', 'lithobench'), default='local')
    p.add_argument('--lithobench-root')
    return p


def main():
    p = parser()
    first, _ = p.parse_known_args()
    if first.config:
        config = json.loads(Path(first.config).read_text(encoding='utf8'))
        unknown = set(config)-{x.dest for x in p._actions}
        if unknown:
            p.error(f'Unknown config keys: {unknown}')
        p.set_defaults(**config)
    a = p.parse_args()
    if a.size < 16 or a.size % 16:
        p.error('--size must be a positive multiple of 16')
    if min(a.batch_size, a.microbatch, a.epochs, a.save_every, a.threads) < 1:
        p.error('batch sizes, epochs, save interval and threads must be positive')
    if a.lambda_l2 < 0 or a.lambda_pvb < 0 or a.lambda_l2+a.lambda_pvb > 1:
        p.error('SFT weights must be nonnegative and sum to at most one')
    torch.set_num_threads(a.threads)
    seed_all(a.seed)
    if a.command == 'toy-data':
        synthetic_data(a.out, a.count, a.size, a.seed)
    elif a.command == 'train':
        train(a)
    elif a.command == 'baseline':
        baseline(a)
    else:
        inference(a)
