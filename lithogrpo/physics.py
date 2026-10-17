"""Differentiable sum-of-coherent-systems optics, equations (1)-(5)."""
from pathlib import Path
import torch
from torch import nn


class Lithography(nn.Module):
    def __init__(self, kernel_dir=None, kernel_num=24, threshold=0.225, steepness=50.,
                 dose_nom=1., dose_max=1.02, dose_min=.98, toy=False):
        super().__init__()
        self.toy = toy
        self.threshold, self.steepness = threshold, steepness
        self.doses = (dose_nom, dose_max, dose_min)
        if toy:
            # Explicitly synthetic optics, never a substitute for benchmark kernels.
            coords = torch.arange(-4, 5).float()
            y, x = torch.meshgrid(coords, coords, indexing='ij')
            kernels = []
            for sigma in (1., 1.5):
                k = torch.exp(-(x*x+y*y)/(2*sigma*sigma))
                kernels.append((k/k.sum())[None, None])
            self.register_buffer('toy_focus', kernels[0])
            self.register_buffer('toy_defocus', kernels[1])
        else:
            if kernel_dir is None:
                raise ValueError('Real optics requires --kernel-dir (LithoBench kernel/). Use --toy only for smoke runs.')
            for name in ('focus', 'defocus'):
                root = Path(kernel_dir)
                k = torch.load(root/'kernels'/f'{name}.pt', map_location='cpu', weights_only=True)
                w = torch.load(root/'scales'/f'{name}.pt', map_location='cpu', weights_only=True)
                k = k.permute(2, 0, 1)[:kernel_num].contiguous().to(torch.complex64)
                w = w.flatten()[:kernel_num].float()
                if len(k) != len(w) or not len(k) or (w < 0).any():
                    raise ValueError('Invalid SOCS kernel weights')
                self.register_buffer(name, k)
                self.register_buffer(name+'_weights', w)

    def aerial(self, mask, defocus=False, dose=1.):
        if self.toy:
            kernel = self.toy_defocus if defocus else self.toy_focus
            return torch.nn.functional.conv2d(mask*dose, kernel, padding=4).square()
        name = 'defocus' if defocus else 'focus'
        k, w = getattr(self, name), getattr(self, name+'_weights')
        h, width = mask.shape[-2:]
        kh, kw = k.shape[-2:]
        if kh > h or kw > width:
            raise ValueError('Image smaller than optical kernel')
        # Input kernels store DC at index floor(size/2), matching pylitho.exact.
        transfer = torch.zeros(len(k), h, width, dtype=k.dtype, device=k.device)
        r, c = h//2-kh//2, width//2-kw//2
        transfer[:, r:r+kh, c:c+kw] = k
        transfer = torch.fft.ifftshift(transfer, dim=(-2, -1))
        spectrum = torch.fft.fft2(mask*dose, norm='forward')
        fields = torch.fft.ifft2(spectrum*transfer[None], norm='forward')
        return (fields.abs().square()*w[None, :, None, None]).sum(1, keepdim=True)

    def forward(self, mask, hard=False):
        corners = (self.aerial(mask, dose=self.doses[0]),
                   self.aerial(mask, dose=self.doses[1]),
                   self.aerial(mask, defocus=True, dose=self.doses[2]))
        if hard:
            return tuple((a >= self.threshold).float() for a in corners)
        return tuple(torch.sigmoid(self.steepness*(a-self.threshold)) for a in corners)


def litho_losses(simulator, mask, target):
    nominal, high, low = simulator(mask)
    dims = (1, 2, 3)
    return (nominal-target).square().sum(dims), (high-low).square().sum(dims)
