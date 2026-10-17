import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.proj = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        f = torch.exp(-math.log(10000) * torch.arange(self.width // 2, device=t.device)
                      / max(self.width // 2 - 1, 1))
        phase = t[:, None] * f[None] * 1000
        return self.proj(torch.cat((phase.sin(), phase.cos()), -1))


class Residual(nn.Module):
    def __init__(self, cin, cout, time_dim):
        super().__init__()
        self.time = nn.Linear(time_dim, cin)
        self.main = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                  nn.GroupNorm(8, cout), nn.SiLU(),
                                  nn.Conv2d(cout, cout, 3, padding=1),
                                  nn.GroupNorm(8, cout), nn.SiLU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        return self.skip(x) + self.main(x + self.time(t)[:, :, None, None])


class FlowUNet(nn.Module):
    """Appendix D interpretation: pre-pool skips, two residual blocks per level.

    Each decoder stage upsamples once and uses matching spatial resolution skips;
    see REPRODUCTION.md for interpretation and parameter-count differences.
    """
    def __init__(self, base_channels=64, time_dim=256):
        super().__init__()
        if base_channels % 8 or time_dim % 2:
            raise ValueError('base_channels must divide into 8 groups; time_dim must be even')
        widths = [base_channels * 2 ** i for i in range(5)]
        self.time = TimeEmbedding(time_dim)
        self.input = nn.Conv2d(2, widths[0], 3, padding=1)
        self.down = nn.ModuleList([nn.ModuleList([Residual(a, b, time_dim),
                                                  Residual(b, b, time_dim)])
                                   for a, b in zip(widths[:-1], widths[1:])])
        self.middle = Residual(widths[-1], widths[-1], time_dim)
        self.up = nn.ModuleList([nn.ModuleList([Residual(2*b, a, time_dim),
                                                Residual(a, a, time_dim)])
                                 for a, b in reversed(list(zip(widths[:-1], widths[1:])))])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x, t, target):
        if x.shape != target.shape or x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError('x/target must match BCHW shapes with H,W divisible by 16')
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0])
        emb = self.time(t)
        h = self.input(torch.cat((x, target), 1))
        skips = []
        for first, second in self.down:
            h = second(first(h, emb), emb)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.middle(h, emb)
        for (first, second), skip in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = second(first(torch.cat((h, skip), 1), emb), emb)
        return self.output(h)
