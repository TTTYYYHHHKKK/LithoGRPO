"""Physical metrics and frozen end-of-SFT reward normalization (Eq.13)."""
import numpy as np
import torch
from scipy.ndimage import binary_erosion

KEYS = ('L2', 'PVB', 'EPE', 'Shot')


def _runs(a):
    padded = np.pad(a.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def epe_normal_checks(printed, target, field_nm=2048., tolerance_nm=15., spacing_nm=40.):
    """Manhattan normal-probe EPE approximation with physical pixel scaling.

    Both inward missing material and outward extra material count as violations.
    Native LithoBench protocol should be used for publishable benchmark comparison.
    """
    h, w = target.shape
    if h != w:
        raise ValueError('EPE currently expects a square field')
    pixel_nm = field_nm/h
    distance = max(1, round(tolerance_nm/pixel_nm))
    spacing = max(1, round(spacing_nm/pixel_nm))
    boundary = target & ~binary_erosion(target, structure=np.ones((3, 3)), border_value=0)
    p = np.pad(target, 1)
    vertical = boundary & ~(p[1:-1, :-2] & p[1:-1, 2:])
    horizontal = boundary & ~(p[:-2, 1:-1] & p[2:, 1:-1])
    count = 0
    def at(arr, y, x):
        return bool(arr[y, x]) if 0 <= y < h and 0 <= x < w else False
    for transpose, edges in ((False, vertical.T), (True, horizontal)):
        for fixed, row in enumerate(edges):
            for start, end in _runs(row):
                last = end-1
                center = (start+last)//2
                sites = [center] if last-start <= 2*spacing else sorted(set(
                    list(range(start+spacing, center+1, spacing))+
                    list(range(last-spacing, center, -spacing))))
                for varying in sites:
                    y, x = (fixed, varying) if transpose else (varying, fixed)
                    dy, dx = (1, 0) if transpose else (0, 1)
                    plus, minus = at(target, y+dy, x+dx), at(target, y-dy, x-dx)
                    if plus == minus:
                        continue  # corner / ambiguous normal
                    sign = 1 if plus else -1
                    count += not at(printed, y+sign*distance*dy, x+sign*distance*dx)
                    count += at(printed, y-sign*distance*dy, x-sign*distance*dx)
    return float(count)


class Metrics:
    def __init__(self, simulator, shot_backend='fast', shot_size=512, field_nm=2048.,
                 epe_backend='local', lithobench_root=None):
        self.simulator, self.shot_size, self.field_nm = simulator, shot_size, field_nm
        if shot_backend == 'fast':
            from shotcount import FastShotCounter
            self.counter = FastShotCounter(device='cpu')
        elif shot_backend == 'reference':
            from shotcount import ShotCounter
            self.counter = ShotCounter(device='cpu')
        else:
            raise ValueError('shot backend must be fast or reference')
        self.native_epe = None
        if epe_backend == 'lithobench':
            # Use upstream geometry code at 1 nm/pixel; no new simulation is needed.
            import os, sys, importlib
            from pathlib import Path
            if lithobench_root is None:
                raise ValueError('--lithobench-root required for native EPE')
            root = Path(lithobench_root).resolve()
            old = Path.cwd()
            sys.path.insert(0, str(root))
            try:
                os.chdir(root)
                self.native_epe = importlib.import_module('pyilt.evaluation')
            finally:
                os.chdir(old)

    @torch.no_grad()
    def __call__(self, mask, target):
        mask = (mask >= .5).float()
        nominal, high, low = self.simulator(mask, hard=True)
        result = {'L2': (nominal-target).square().flatten(1).sum(1),
                  'PVB': (high != low).float().flatten(1).sum(1)}
        epe, shots = [], []
        for m, z, t in zip(mask[:, 0], nominal[:, 0], target[:, 0]):
            if self.native_epe:
                import torch.nn.functional as F
                size = round(self.field_nm)
                zfull = F.interpolate(z[None, None], (size, size), mode='nearest')[0, 0]
                tfull = F.interpolate(t[None, None], (size, size), mode='nearest')[0, 0]
                if not tfull.any():
                    epe.append(0.)
                else:
                    poses = self.native_epe.boundaries(tfull)
                    inner, outer, _ = self.native_epe.epecheck(zfull, tfull, *poses)
                    epe.append(float(inner+outer))
            else:
                epe.append(epe_normal_checks(z.cpu().numpy().astype(bool),
                                            t.cpu().numpy() >= .5, self.field_nm))
            shots.append(self.counter.run(m, shape=(self.shot_size, self.shot_size)))
        result['EPE'] = torch.tensor(epe, device=mask.device)
        result['Shot'] = torch.tensor(shots, device=mask.device, dtype=torch.float32)
        return result


def normalized_reward(metrics, baseline, floor=1.):
    # EPE can saturate at zero after SFT. A configurable floor avoids division by 0.
    return -sum(metrics[k]/max(float(baseline[k]), floor) for k in KEYS)
