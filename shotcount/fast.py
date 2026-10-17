"""Fast shot counting via maximal-rectangle cover.

`FastShotCounter` estimates the number of e-beam shots (rectangles) needed to
write a binary mask. It first enumerates all maximal all-ones rectangles, prunes
those that are fully contained in another, and then solves a minimum
set-cover ILP so that every foreground pixel is covered by at least one chosen
rectangle. The number of selected rectangles is the shot count.

This is the >130x-faster manufacturability estimator described in the LithoGRPO
paper. It preserves the ranking of the slower reference counter
(`shotcount.reference.ShotCounter`) while being cheap enough to use inside the
RL reward loop.
"""

from collections import defaultdict
from typing import List, Sequence

import numpy as np
import torch
try:
    from numba import njit
except ImportError:
    # Same pruning algorithm without JIT; useful for lightweight CPU smoke runs.
    def njit(function):
        return function
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def _scipy_cover(rectangles, mask, scan_line):
    """Same binary set-cover ILP via SciPy/HiGHS if PuLP is unavailable."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    rows, cols = [], []
    constraint = 0
    for y in range(len(mask)):
        active = [i for i, (r, c, h, w) in enumerate(rectangles) if r <= y < r+h]
        if scan_line:
            cuts = sorted({p for i in active for p in (rectangles[i][1],
                                                        rectangles[i][1]+rectangles[i][3])})
            points = [x for x in cuts[:-1] if mask[y, x]]
        else:
            points = np.flatnonzero(mask[y])
        for x in points:
            covering = [i for i in active if rectangles[i][1] <= x < rectangles[i][1]+rectangles[i][3]]
            if not covering:
                raise RuntimeError('Foreground pixel has no covering rectangle')
            rows.extend([constraint]*len(covering))
            cols.extend(covering)
            constraint += 1
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(constraint, len(rectangles))).tocsc()
    result = milp(np.ones(len(rectangles)), integrality=np.ones(len(rectangles)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, 1, np.inf),
                  options={'time_limit': 300, 'mip_rel_gap': 0.})
    if result.status != 0:
        raise RuntimeError(f'Shot ILP did not prove optimality: {result.message}')
    return round(result.fun)

REALTYPE = torch.float32


@njit
def check_containment_optimized(y1_s, x1_s, y2_s, x2_s, areas, n):
    """Greedy containment pruning, in descending area order (numba-accelerated)."""
    keep_flag = np.ones(n, dtype=np.bool_)
    keep_list = np.empty(n, dtype=np.int32)
    keep_count = 0

    for i in range(n):
        cur_y1, cur_x1 = y1_s[i], x1_s[i]
        cur_y2, cur_x2 = y2_s[i], x2_s[i]

        contained = False
        for j in range(keep_count):
            k = keep_list[j]
            if (y1_s[k] <= cur_y1 and cur_y2 <= y2_s[k] and
                    x1_s[k] <= cur_x1 and cur_x2 <= x2_s[k]):
                contained = True
                break

        if not contained:
            keep_list[keep_count] = i
            keep_count += 1
        else:
            keep_flag[i] = False

    return keep_flag


class FastShotCounter:
    """Fast shot counter based on maximal-rectangle cover + ILP set cover.

    Args:
        scan_line: use scan-line segment constraints to keep the ILP compact.
                   Default True.
    """

    def __init__(self, scan_line=True, device=None):
        self.scan_line = scan_line
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def rle_rows(self, img):
        """Run-length encode the foreground runs of each row."""
        runs = []
        H, W = img.shape
        for r in range(H):
            c = 0
            while c < W:
                while c < W and img[r, c] == 0:
                    c += 1
                s = c
                while c < W and img[r, c] == 1:
                    c += 1
                if s < c:
                    runs.append((r, s, c - 1))
        return runs

    def prune_rectangles_numba(self, rects: Sequence[Sequence[int]]) -> List[List[int]]:
        """Remove rectangles contained in another, greedily by area (numba)."""
        n = len(rects)
        if n <= 1:
            return [list(r) for r in rects]

        arr = np.asarray(rects, dtype=np.int32)
        y1, x1, h, w = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        y2 = y1 + h
        x2 = x1 + w

        areas = h * w
        order = np.argsort(-areas)

        y1_s = y1[order].astype(np.int32)
        x1_s = x1[order].astype(np.int32)
        y2_s = y2[order].astype(np.int32)
        x2_s = x2[order].astype(np.int32)

        keep_flag = check_containment_optimized(y1_s, x1_s, y2_s, x2_s, areas, n)
        kept_indices = order[keep_flag]
        return arr[kept_indices].tolist()

    def maximal_rectangles(self, img):
        """Enumerate maximal all-ones rectangles via the histogram/stack method."""
        H, W = img.shape
        h = np.zeros(W, int)
        rects = set()
        for r in range(H):
            h = np.where(img[r] == 1, h + 1, 0)
            stack = []
            for c in range(W + 1):
                cur = h[c] if c < W else 0
                last = c
                while stack and cur < stack[-1][1]:
                    pos, height = stack.pop()
                    # pos is the inherited LEFT boundary, not a histogram index.
                    # Deriving it from the previous stack entry includes zeros in
                    # shapes such as heights=[2, 1, 2].
                    rects.add((r - height + 1, pos, height, c - pos))
                    last = pos
                if cur > 0 and (not stack or cur > stack[-1][1]):
                    stack.append((last, cur))
        return list(rects)

    def run(self, mask, shape=(512, 512)):
        """Return the number of shots needed to write `mask`.

        Args:
            mask:  2D array / tensor of {0, 1}. Resized (nearest) to `shape`.
            shape: working resolution for the cover (default 512x512).
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if mask.ndim != 2 or not torch.isfinite(mask).all():
            raise ValueError('mask must be a finite two-dimensional array')
        mask = (mask >= 0.5).to(dtype=REALTYPE)
        mask = torch.nn.functional.interpolate(
            mask[None, None, :, :], size=shape, mode="nearest")[0, 0]

        if mask.device.type != 'cpu':
            mask = mask.detach().cpu().numpy().astype(np.uint8)
        else:
            mask = mask.detach().numpy().astype(np.uint8)

        R = self.maximal_rectangles(mask)
        R = self.prune_rectangles_numba(R)
        if not mask.any():
            return 0
        if not R:
            raise RuntimeError('Foreground has no covering rectangles')
        if not HAS_PULP:
            return _scipy_cover(R, mask, self.scan_line)

        H, W = shape

        prob = LpProblem('rect_cover', LpMinimize)
        x = LpVariable.dicts('x', range(len(R)), 0, 1, cat='Binary')

        row_rects = defaultdict(list)
        for i, (y, x0, h, w) in enumerate(R):
            for r in range(y, y + h):
                row_rects[r].append(i)

        if not self.scan_line:
            for r, s, e in self.rle_rows(mask):
                rects_in_row = row_rects[r]
                for c in range(s, e + 1):
                    prob += lpSum(x[i] for i in rects_in_row
                                  if R[i][1] <= c < R[i][1] + R[i][3]) >= 1
        else:
            row_runs = defaultdict(list)
            for r, s, e in self.rle_rows(mask):
                row_runs[r].append((s, e))

            for r in range(H):
                rects_in_row = row_rects[r]
                if not rects_in_row:
                    continue

                cuts = {R[i][1] for i in rects_in_row} | {R[i][1] + R[i][3] for i in rects_in_row}
                cuts |= {col for (s, e) in row_runs[r] for col in (s, e + 1)}
                segments = sorted(cuts)

                for a, b in zip(segments, segments[1:]):
                    c0 = a
                    if mask[r, c0] == 0:
                        continue
                    prob += lpSum(x[i] for i in rects_in_row
                                  if R[i][1] <= c0 < R[i][1] + R[i][3]) >= 1

        prob += lpSum(x.values())
        prob.solve(PULP_CBC_CMD(timeLimit=300, msg=False))
        if LpStatus[prob.status] != 'Optimal' or prob.sol_status != 1:
            raise RuntimeError(f'Shot ILP did not prove optimality: {LpStatus[prob.status]}')
        answer = [R[i] for i in range(len(R)) if x[i].value() == 1]
        return len(answer)
