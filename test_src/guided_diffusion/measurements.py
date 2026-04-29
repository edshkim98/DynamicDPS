'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure 

MSSIM = MultiScaleStructuralSimilarityIndexMeasure()

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size=11, intensity=2.0, gamma=1.0, down_scale=1.0, device='cuda'):
        self.device = device
        self.kernel_size = int(kernel_size)
        self.gamma = float(gamma)
        self.intensity = float(intensity)
        self.down_scale = down_scale
        self._kernel = None  # cached fixed kernel after bake
        self._bake_kernel()
        print("Gaussian blur kernel size: ", self.kernel_size)

    def _build_kernel(self, sigma, dtype, device):
        """Build a 2D Gaussian kernel. Differentiable if sigma is a tensor."""
        coords = torch.arange(self.kernel_size, device=device, dtype=dtype) - self.kernel_size // 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = torch.outer(kernel, kernel)
        kernel = kernel / kernel.sum()
        return kernel

    def _bake_kernel(self):
        """Freeze the kernel using current self.intensity (no grad)."""
        with torch.no_grad():
            sigma = torch.tensor(self.intensity, device=self.device, dtype=torch.float32)
            k = self._build_kernel(sigma, dtype=torch.float32, device=self.device)
            self._kernel = k.detach()

    def _apply_blur(self, image, kernel):
        C = image.shape[1]
        k = kernel.to(dtype=image.dtype, device=image.device)
        k = k.view(1, 1, self.kernel_size, self.kernel_size).repeat(C, 1, 1, 1)
        padding = self.kernel_size // 2
        return F.conv2d(image, k, padding=padding, groups=C)

    def forward(self, image, eps=1e-7, gamma=None, intensity=None, **kwargs):
        """
        Inference: uses fixed cached kernel and self.gamma.
        Fitting: pass `gamma` and `intensity` as tensors to keep graph alive.
        """
        g = gamma if gamma is not None else self.gamma
        image = image.clamp(min=eps) ** g

        h, w = int(256 // self.down_scale), int(256 // self.down_scale)
        image = Resize((h, w))(image)
        image = Resize((256, 256))(image)

        if intensity is not None:
            # differentiable path: rebuild kernel from tensor sigma
            kernel = self._build_kernel(intensity, dtype=image.dtype, device=image.device)
        else:
            kernel = self._kernel

        return self._apply_blur(image, kernel)

    def fit_A(self, x_hf: torch.Tensor, y_lf: torch.Tensor):
        gamma = nn.Parameter(torch.tensor(1.0, device=x_hf.device, dtype=x_hf.dtype))
        sigma = nn.Parameter(torch.tensor(1.0, device=x_hf.device, dtype=x_hf.dtype))
        optimizer = torch.optim.Adam([gamma, sigma], lr=5e-3, weight_decay=1e-3)

        best_loss, best_gamma, best_sigma = float('inf'), None, None
        loss_history = []
        early_stop, cnt = 100, 0

        for step in range(1000):
            y_hat = self.forward(image=x_hf, gamma=gamma, intensity=sigma)
            loss = F.mse_loss(y_hat, y_lf)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gamma.data.clamp_(0.3, 3.0)
            sigma.data.clamp_(0.1, 5.0)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_gamma = gamma.item()
                best_sigma = sigma.item()
                cnt = 0
            else:
                cnt += 1
                if cnt >= early_stop:
                    break

        # freeze optimized values and bake the kernel
        self.gamma = best_gamma
        self.intensity = best_sigma
        self._bake_kernel()

        print(f"Best gamma: {best_gamma}, Best sigma: {best_sigma}, Best loss: {best_loss}")
        return best_gamma, best_sigma, best_loss

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)
