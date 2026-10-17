"""Rectified flow and paper Eq.15 SDE; the t=0 boundary is regularized explicitly."""
from dataclasses import dataclass
import math
import torch


def matching_loss(model, target, mask):
    noise = torch.randn_like(mask)
    t = torch.rand(len(mask), device=mask.device)
    t4 = t[:, None, None, None]
    xt = (1-t4)*noise + t4*mask
    velocity = model(xt, t, target)
    loss = (velocity-(mask-noise)).square().flatten(1).mean(1)
    return loss, xt+(1-t4)*velocity


@torch.no_grad()
def ode_sample(model, target, steps=1):
    if steps < 1:
        raise ValueError('steps must be positive')
    x = torch.randn_like(target)
    for i in range(steps):
        x = x + model(x, i/steps, target)/steps
    return x


def colored_noise(x, cutoff=.1):
    if cutoff <= 0:
        raise ValueError('noise cutoff must be positive')
    noise = torch.randn_like(x)
    h, w = x.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=x.device)[None, :]
    filt = torch.exp(-(fy.square()+fx.square())/(2*cutoff**2))
    # Normalize expected variance, preserving a Gaussian (no per-image rescaling).
    fullfx = torch.fft.fftfreq(w, device=x.device)[None, :]
    power = torch.exp(-(fy.square()+fullfx.square())/cutoff**2).mean()
    return torch.fft.irfft2(torch.fft.rfft2(noise)*filt, s=(h, w))/power.sqrt()


def transition(model, x, target, t, dt, noise_level):
    v = model(x, t, target)
    variance_rate = noise_level**2*(1-t)/t
    mean = x + (v + variance_rate/(2*t)*(x+(1-t)*v))*dt
    return mean, variance_rate*dt


@dataclass
class Trajectory:
    states: list
    times: list
    dt: float
    means: list
    variances: list


@torch.no_grad()
def sde_sample(model, target, steps=10, noise_level=.1, start_time=.1,
               noise_type='colored', cutoff=.1):
    if not (0 < start_time < 1) or noise_level <= 0 or steps < 1:
        raise ValueError('SDE needs 0 < start_time < 1, positive steps and noise_level')
    x = torch.randn_like(target)
    # Deterministic warm start avoids the singular coefficients at t=0.
    x = x + start_time*model(x, 0., target)
    states, times, means, variances = [x], [], [], []
    dt = (1-start_time)/steps
    for i in range(steps):
        t = start_time + i*dt
        mean, var = transition(model, x, target, t, dt, noise_level)
        noise = colored_noise(x, cutoff) if noise_type == 'colored' else torch.randn_like(x)
        x = mean + math.sqrt(var)*noise
        if not torch.isfinite(x).all():
            raise FloatingPointError('SDE diverged; increase start_time/steps or reduce noise_level')
        times.append(t)
        means.append(mean)
        variances.append(var)
        states.append(x)
    return x, Trajectory(states, times, dt, means, variances)


def trajectory_log_ratio(model, target, trajectory, noise_level):
    """Eq.18-19: SUM over pixels and timesteps, independent-Gaussian approximation.

    Difference of squared residuals avoids subtracting huge Gaussian constants.
    Warm start is held fixed, so this is a conditional trajectory objective.
    """
    ratio = torch.zeros(len(target), device=target.device, dtype=torch.float64)
    for x, nxt, old, t, var in zip(trajectory.states[:-1], trajectory.states[1:],
                                  trajectory.means, trajectory.times, trajectory.variances):
        new, _ = transition(model, x, target, t, trajectory.dt, noise_level)
        residual = nxt.double()-old.double()
        delta = new.double()-old.double()
        ratio = ratio + ((2*residual*delta-delta.square())/(2*var)).flatten(1).sum(1)
    return ratio


def advantages(rewards, group_size):
    r = rewards.reshape(-1, group_size)
    return ((r-r.mean(1, keepdim=True))/(r.std(1, keepdim=True, unbiased=False)+1e-8)).flatten()


def grpo_loss(log_ratio, advantage, clip=.2):
    ratio = log_ratio.clamp(-60, 60).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean()
