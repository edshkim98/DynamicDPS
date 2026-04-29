from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import torch.nn as nn
from timm import create_model
from torch.nn.functional import interpolate
import pandas as pd
import warnings
from torch.optim import LBFGS
import os

warnings.filterwarnings("ignore")

__CONDITIONING_METHOD__ = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchSSIMLoss(torch.nn.Module):
    def __init__(self, patch_size=32, eps=1e-6):
        """
        Initializes the Patch-based Mutual Information Loss module.
        Args:
            patch_size (int): Size of each patch (square patches are assumed).
            num_bins (int): Number of bins for the histogram.
            eps (float): Small value to avoid division by zero.
        """
        super(PatchSSIMLoss, self).__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))

    def compute_patch_ssim(self, patch_x, patch_y):
        """
        Computes ssim for a single patch.
        Args:
            patch_x (Tensor): Patch from image X (batch_size, 1, H, W).
            patch_y (Tensor): Patch from image Y (batch_size, 1, H, W).
        Returns:
            mi (Tensor): Mutual information for the patch (scalar).
        """
        ssim_val = self.SSIM(patch_x, patch_y)
        return ssim_val

    def forward(self, x, y):
        """
        Computes the patch-based mutual information loss between two images.
        Args:
            x (Tensor): Image 1 (batch_size, 1, H, W), normalized to [0, 1].
            y (Tensor): Image 2 (batch_size, 1, H, W), normalized to [0, 1].
        Returns:
            loss (Tensor): Patch-based mutual information loss (scalar).
        """
        batch_size, _, height, width = x.size()
        ssim_loss = []
        num_patches = 0

        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch_x = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patch_y = y[:, :, i:i+self.patch_size, j:j+self.patch_size]

                if patch_x.size(2) == self.patch_size and patch_x.size(3) == self.patch_size:
                    ssim_loss.append(1.0 - self.compute_patch_ssim(patch_x, patch_y))
                    num_patches += 1

        ssim_loss = torch.stack(ssim_loss)
        ssim_loss = torch.linalg.norm(ssim_loss)
        return ssim_loss

class CannyEdgeLoss(torch.nn.Module):
    def __init__(self, low_threshold=0.1, high_threshold=0.3):
        """
        Initializes the EdgeLoss module with thresholds suitable for normalized images.
        Args:
            low_threshold (float): Lower threshold for Canny edge detection (normalized scale 0–1).
            high_threshold (float): Higher threshold for Canny edge detection (normalized scale 0–1).
        """
        super(CannyEdgeLoss, self).__init__()
        self.low_threshold = int(low_threshold * 255)  # Scale for OpenCV (expects 0-255)
        self.high_threshold = int(high_threshold * 255)  # Scale for OpenCV (expects 0-255)

        # Example 3x3 Sobel kernels:
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).reshape(1,1,3,3).to(device)
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).reshape(1,1,3,3).to(device)

    def sobel_edge_magnitude(self,image):
        """Compute a differentiable approximation of edge magnitude via Sobel."""
        #print data type
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        # Edge magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-7)
        return edges

    def forward(self, image_A, image_B):
        """
        Compute edge loss between two normalized images.
        Input:
            image_A: PyTorch tensor (batch_size, C, H, W), normalized to [0, 1].
            image_B: PyTorch tensor (batch_size, C, H, W), normalized to [0, 1].
        Output:
            loss: Scalar edge loss.
        """
        
        # Check if the input images are normalized to [0, 1]
        if image_A.max() > 1:
            image_A = image_A / 2.0
        if image_B.max() > 1:
            image_B = image_B / 2.0

        edges_A = self.sobel_edge_magnitude(image_A.to(torch.float32))
        edges_B = self.sobel_edge_magnitude(image_B.to(torch.float32))
        difference = edges_A - edges_B
        loss = torch.linalg.norm(difference)  # sum of all differences
        return loss

class TotalVariationLoss(nn.Module):
    def __init__(self):
        """
        Initialize the Total Variation Loss module.
        """
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        """
        Compute the Total Variation (TV) Loss for an image.
        
        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W),
                                  where B is batch size, C is the number of channels,
                                  H is the height, and W is the width.
        
        Returns:
            tv_loss (torch.Tensor): Scalar tensor representing the total variation loss.
        """
        # Compute horizontal and vertical differences
        diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # Horizontal differences
        diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # Vertical differences
        
        # Sum the absolute differences
        tv_loss = diff_h.sum() + diff_w.sum()
        
        return tv_loss
    
class PerceptualLoss(nn.Module):
    def __init__(self, model_name="resnet18", layers=("layer2", "layer3"), device='cuda'):
        """
        Perceptual Loss using intermediate features of a Timm pre-trained model.
        Args:
            model_name: Name of the model to use from Timm (e.g., "resnet18").
            layers: Tuple of layer names from which to extract intermediate features.
        """
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.device = device

        # Load a pre-trained model from Timm
        self.feature_extractor = create_model(model_name, pretrained=True, features_only=True, out_indices=tuple(range(len(layers))))
        self.layer_names = [f"layer{i}" for i in range(len(self.feature_extractor.feature_info))]

        # Freeze the feature extractor's parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.to(self.device)

    def forward(self, input_image, target_image):
        """
        Calculate perceptual loss between input and target images.
        Args:
            input_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
            target_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
        Returns:
            loss: Scalar perceptual loss.
        """
        # Convert grayscale (1-channel) images to 3-channel by repeating
        input_image = input_image.repeat(1, 3, 1, 1)  # Shape: (B, 3, H, W)
        target_image = target_image.repeat(1, 3, 1, 1)  # Shape: (B, 3, H, W)
        
        # Normalize images using ImageNet statistics
        #input_image = (input_image - 0.485) / 0.229
        #target_image = (target_image - 0.485) / 0.229

        # Ensure input and target are resized to match the model's expected input size
        input_image = interpolate(input_image, size=(256, 256), mode="bilinear", align_corners=False)
        target_image = interpolate(target_image, size=(256, 256), mode="bilinear", align_corners=False)

        # Extract intermediate features
        input_features = self.feature_extractor(input_image)
        target_features = self.feature_extractor(target_image)

        # Calculate perceptual loss using L2 norm of feature differences
        loss = 0.0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += torch.linalg.norm(input_feat - target_feat)

        return loss


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.ssim = PatchSSIMLoss(patch_size=32)
        self.tv = TotalVariationLoss()
        self.edge_ls = CannyEdgeLoss(low_threshold=0.05, high_threshold=0.1)

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, t, use_lbfgs, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            x_0_hat = x_0_hat.clamp(min=0.)  ############# Ensure x_0_hat is clamped safely
            
            if use_lbfgs:
                x_0_hat = x_0_hat.detach().clone()
                x_0_hat.requires_grad_(True)
                def closure():
                    optimizer.zero_grad()
                    pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                    difference = measurement - pred_measurement
                    norm = torch.linalg.norm(difference)
                    edge_ls = self.edge_ls(measurement, pred_measurement)
                    ssim = self.ssim(pred_measurement.type(torch.DoubleTensor), measurement.type(torch.DoubleTensor))
                    norm_total = norm + 0.5*edge_ls + 0.5*ssim
                    norm_total.backward()
                    return norm_total
                optimizer = LBFGS([x_0_hat], lr=0.5, max_iter=10, history_size=10, line_search_fn='strong_wolfe')
                optimizer.step(closure)
                
                # 2) Now do a new forward pass for the final gradient
                with torch.enable_grad():
                    pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                    difference = measurement - pred_measurement
                    norm = torch.linalg.norm(difference)
                    edge_ls = self.edge_ls(measurement, pred_measurement)
                    ssim = self.ssim(
                        pred_measurement.type(torch.DoubleTensor),
                        measurement.type(torch.DoubleTensor)
                    )
                    new_loss = norm + 0.5*edge_ls + 0.5*ssim
                norm_grad = torch.autograd.grad(outputs=new_loss, inputs=x_prev)[0]
            else:
                pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                #measurement = self.noiser(measurement)
                
                difference = measurement - pred_measurement
                norm = torch.linalg.norm(difference)
                edge_ls = self.edge_ls(measurement, pred_measurement)
                #tv = self.tv(pred_measurement)
                ssim = self.ssim(pred_measurement.type(torch.DoubleTensor), measurement.type(torch.DoubleTensor))
                # pred_measurement[pred_measurement < 0.] = 0.
                # measurement[measurement < 0.] = 0.
                #percept = self.perceptual_loss(pred_measurement.type(torch.float32), measurement.type(torch.float32))
                norm_total = norm + 0.5*edge_ls + 0.1*ssim #+ 0.005*tv
                norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.scale_original = self.scale
        self.c1 = 1e-3      # Sufficient decrease parameter
        self.c2 = 0.8       # Curvature parameter
        self.max_line_search = 10  # Max iterations for line search
        self.alpha = self.scale_original
        self.best_ls = 1000
        self.loss_df = pd.DataFrame(columns=['Time', 'Loss'])
        self.use_lbfgs = kwargs.get('use_lbfgs', False)
        print(f"Using L-BFGS: {self.use_lbfgs}")
        
        self.csv_file = "./line_search_stepsize.csv"

        if os.path.exists(self.csv_file):
            self.stepsize_df = pd.read_csv(self.csv_file)
        else:
            self.stepsize_df = pd.DataFrame(columns=['Epoch', 'Step Size'])
        
    def line_search(self, x_prev, x_t, x_0_hat, measurement, norm_grad, norm, t, **kwargs):
        self.alpha = self.scale_original          # FIX 3: reset
        alpha_min = 0.1                          # FIX 4: sensible floor
        alpha_max = 1.5 * self.scale_original

        norm_orig = norm
        # φ'(0) = ∇f(x_t)·p where p = -norm_grad
        grad_orig = -norm_grad.reshape(-1).dot(norm_grad.reshape(-1))   # negative scalar

        for i in range(self.max_line_search):
            x_t_new = x_t - self.alpha * norm_grad

            # FIX 5: use t, not t-1; handle t==0
            x_0_hat_new = kwargs['func'](
                kwargs['model'], x_t_new, t,
                kwargs['clip_denoised'], kwargs['denoised_fn'],
                kwargs['cond_fn'], kwargs['model_kwargs']
            )['pred_xstart']

            norm_grad_new, norm_new = self.grad_and_value(
                x_prev=x_t_new, x_0_hat=x_0_hat_new,
                measurement=measurement, t=t, use_lbfgs=False, **kwargs
            )

            # Armijo: φ(α) ≤ φ(0) + c1·α·φ'(0)
            if norm_new > norm_orig + self.c1 * self.alpha * grad_orig:
                self.alpha *= 0.75
                if self.alpha < alpha_min:
                    self.alpha = alpha_min
                    break
                continue

            # Strong Wolfe curvature: |φ'(α)| ≤ c2·|φ'(0)|
            # φ'(α) = ∇f(x_new)·p = -norm_grad_new·norm_grad
            phi_prime_alpha = -norm_grad_new.reshape(-1).dot(norm_grad.reshape(-1))
            if torch.abs(phi_prime_alpha) > self.c2 * torch.abs(grad_orig):   # FIX 2: flipped
                self.alpha *= 1.25
                if self.alpha > alpha_max:
                    self.alpha = alpha_max
                    break
                continue

            return self.alpha

        return self.alpha

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, t=t, use_lbfgs=self.use_lbfgs, **kwargs)
        self.loss_df = pd.concat([self.loss_df, pd.DataFrame([{'Time': t.cpu().numpy(), 'Loss': norm.detach().cpu().numpy()}])], ignore_index=True)
        if self.best_ls > norm:
            self.best_ls = norm
            self.best_x = x_0_hat

        self.alpha = self.scale_original
        # Perform line search to find step size satisfying Wolfe conditions
        if (kwargs['line_search']) and (t > 5) and (t % 5 == 0):
            print(f"Time step: {t}")
            self.scale = self.line_search(x_prev.detach(), x_t.detach().requires_grad_(True), x_0_hat, measurement, norm_grad.detach(), norm, t, **kwargs)
            self.stepsize_df = pd.concat([self.stepsize_df, pd.DataFrame([{'Epoch': t.cpu().numpy(), 'Step Size': self.scale}])], ignore_index=True)
            self.stepsize_df.to_csv(self.csv_file)
        else:
            if t <= 5:
                self.scale = 0.1
            if kwargs['line_search']:
                self.stepsize_df = pd.concat([self.stepsize_df, pd.DataFrame([{'Epoch': t.cpu().numpy(), 'Step Size': self.scale}])], ignore_index=True)
                self.stepsize_df.to_csv(self.csv_file)
                print(f"Step size for final iteration {t}: {self.scale}")

        x_t -= norm_grad * self.scale

        if t > 0:
            return x_t, norm
        print("Returning best loss: ", self.best_ls)
        print("Re-initializing best loss")
        self.best_ls = 1000
        self.loss_df.to_csv('measurement_loss_timestep.csv')
        return self.best_x, self.best_ls
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
