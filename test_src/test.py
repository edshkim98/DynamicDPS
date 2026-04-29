"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import glob
import os

import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torch.nn.functional as F
import nibabel as nib

from guided_diffusion import logger
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Set random seeds for reproducible results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preproc_data(ref_img, meas_img, meas_resampled_img, warmstrat_img, seg_map):
    """
    ref_img: (H, W, D) - GT image
    meas_img: (H, W, D) - Measurement image (e.g. low-field)
    meas_resampled_img: (H, W, D) - Measurement image resampled to GT space
    warmstrat_img: (H, W, D) - Warm start image (e.g. synthetic image)
    seg_map: (H, W, D) - Segmentation map (if available)
    """

    slices = ref_img.shape[-1]

    # Normalize images to [0, 1]
    ref_img = ref_img / 4096.0
    meas_img = meas_img / 4096.0
    meas_resampled_img = meas_resampled_img / 4096.0
    warmstrat_img = warmstrat_img / 4096.0

    # Scale to [0, 2] because of pretrained diffusion model's expected input range
    ref_img = ref_img * 2.0
    meas_img = meas_img * 2.0
    meas_resampled_img = meas_resampled_img * 2.0
    warmstrat_img = warmstrat_img * 2.0

    # Convert to torch tensors and add batch/channel dimensions
    ref_img = torch.tensor(ref_img).unsqueeze(0).unsqueeze(0).to(torch.float32)
    meas_img = torch.tensor(meas_img).unsqueeze(0).unsqueeze(0).to(torch.float32)
    meas_resampled_img = torch.tensor(meas_resampled_img).unsqueeze(0).unsqueeze(0).to(torch.float32)
    warmstrat_img = torch.tensor(warmstrat_img).unsqueeze(0).unsqueeze(0).to(torch.float32)
    seg_map = torch.tensor(seg_map).unsqueeze(0).unsqueeze(0).to(torch.float32)

    assert ref_img.shape == warmstrat_img.shape == seg_map.shape, f"All images must have the same shape before preprocessing but got {ref_img.shape}, {warmstrat_img.shape}, {seg_map.shape}"

    #  Crop to 256x256 if needed (assuming images are already resampled to the same space) original 386x386x29 -> 256x256x29
    diff_x = ref_img.shape[2] - 256
    diff_y = ref_img.shape[3] - 256
    crop_x = diff_x // 2
    crop_y = diff_y // 2
    if ref_img.shape[2] != 256 and ref_img.shape[3] != 256:
        b, c, h, w, d = ref_img.shape
        ref_img = ref_img.permute(0, 1, 4, 2, 3).reshape(b * c * d, 1, h, w)  # (29, 1, 256, 256)
        ref_img = F.interpolate(ref_img, size=(256, 256), mode='nearest').reshape(b, c, d, 256, 256).permute(0, 1, 3, 4, 2)  # (1, 1, 256, 256, 29)
        
    if meas_img.shape[2] != 256 and meas_img.shape[3] != 256:
        b, c, h, w, d = meas_img.shape
        meas_img = meas_img.permute(0, 1, 4, 2, 3).reshape(b * c * d, 1, h, w)  # (29, 1, 256, 256)
        meas_img = F.interpolate(meas_img, size=(256, 256), mode='nearest').reshape(b, c, d, 256, 256).permute(0, 1, 3, 4, 2)  # (1, 1, 256, 256, 29)

    if meas_resampled_img.shape[2] != 256 and meas_resampled_img.shape[3] != 256:
        b, c, h, w, d = meas_resampled_img.shape
        meas_resampled_img = meas_resampled_img.permute(0, 1, 4, 2, 3).reshape(b * c * d, 1, h, w)  # (29, 1, 256, 256)
        meas_resampled_img = F.interpolate(meas_resampled_img, size=(256, 256), mode='nearest').reshape(b, c, d, 256, 256).permute(0, 1, 3, 4, 2)  # (1, 1, 256, 256, 29)

    if warmstrat_img.shape[2] != 256 and warmstrat_img.shape[3] != 256:
        b, c, h, w, d = warmstrat_img.shape
        warmstrat_img = warmstrat_img.permute(0, 1, 4, 2, 3).reshape(b * c * d, 1, h, w)  # (29, 1, 256, 256)
        warmstrat_img = F.interpolate(warmstrat_img, size=(256, 256), mode='nearest').reshape(b, c, d, 256, 256).permute(0, 1, 3, 4, 2)  # (1, 1, 256, 256, 29)

    if seg_map.shape[2] != 256 and seg_map.shape[3] != 256:
        b, c, h, w, d = seg_map.shape
        seg_map = seg_map.permute(0, 1, 4, 2, 3).reshape(b * c * d, 1, h, w)  # (29, 1, 256, 256)
        seg_map = F.interpolate(seg_map, size=(256, 256), mode='nearest').reshape(b, c, d, 256, 256).permute(0, 1, 3, 4, 2)  # (1, 1, 256, 256, 29)

    assert ref_img.shape == (1, 1, 256, 256, slices), f"Expected shape (1, 1, 256, 256, {slices}) but got {ref_img.shape}"
    assert meas_img.shape == (1, 1, 256, 256, slices), f"Expected shape (1, 1, 256, 256, {slices}) but got {meas_img.shape}"
    assert meas_resampled_img.shape == (1, 1, 256, 256, slices), f"Expected shape (1, 1, 256, 256, {slices}) but got {meas_resampled_img.shape}"
    assert warmstrat_img.shape == (1, 1, 256, 256, slices), f"Expected shape (1, 1, 256, 256, {slices}) but got {warmstrat_img.shape}"
    assert seg_map.shape == (1, 1, 256, 256, slices), f"Expected shape (1, 1, 256, 256, {slices}) but got {seg_map.shape}"

    # Slice to get the middle slice (assuming the 3rd dimension is the slice dimension)
    mid_slice = ref_img.shape[-1] // 2
    ref_img = ref_img[:, :, :, :, mid_slice]
    meas_img = meas_img[:, :, :, :, mid_slice]
    meas_resampled_img = meas_resampled_img[:, :, :, :, mid_slice]
    warmstrat_img = warmstrat_img[:, :, :, :, mid_slice]
    seg_map = seg_map[:, :, :, :, mid_slice]

    # Skull-strip using seg_map (if available) - set non-brain regions to 0
    seg_map_binary = (seg_map > 0).float()  # Assuming seg_map has positive values for brain regions
    warmstrat_img = warmstrat_img * seg_map_binary

    return ref_img, meas_img, meas_resampled_img, warmstrat_img, seg_map, mid_slice

def main():
    set_seed(42)

    # Load configuration
    config_path = './configs.yaml'
    with open(config_path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        configs=configs,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print('Using device:', device)

    # Configuration paths and file handling
    save_path = './DynamicDPS_test/'
    
    #Folder structure: ...{data_dir}/{file_id}/all.ni.gz
    files = os.listdir(configs['data_dir']) 
    print(len(files))
 
    # Prepare Operator and noise
    measure_config = configs['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
 
    # Working directory
    save_dir = '/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/results/'
    out_path = os.path.join(save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
           
    # Prepare conditioning method
    cond_config = configs['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {configs['conditioning']['method']}")

    logger.log("sampling...")
    all_images = []
    ys = []
    refs = []
    
    files_new = []
    for f in files:
        try:
            int(f)
            files_new.append(f)
        except:
            print(f"Skipping non-numeric file: {f}")
            continue
    files = files_new
    for i, fname in tqdm.tqdm(enumerate(files)):

        print(f"{i}/{len(files)}: Processing {fname}")

        ref_img = nib.load(os.path.join(configs['data_dir'], fname, configs['gt_name'])).get_fdata()
        meas_img = nib.load(os.path.join(configs['data_dir'], fname, configs['meas_name'])).get_fdata()
        meas_resampled_img = nib.load(os.path.join(configs['data_dir'], fname, configs['meas_resampled_name'])).get_fdata()
        warmstrat_img = nib.load(os.path.join(configs['data_dir'], fname, configs['warmstart_name'])).get_fdata()
        seg_map = nib.load(os.path.join(configs['data_dir'], fname, configs['segmap_name'])).get_fdata()

        # Preprocess data (normalize, crop, slice, skull-strip)
        ref_img, meas_img, meas_resampled_img, warmstrat_img, seg_map, mid_slice = preproc_data(ref_img, meas_img, meas_resampled_img, warmstrat_img, seg_map)
        ref_img, meas_img, meas_resampled_img, warmstrat_img, seg_map = ref_img.to(device), meas_img.to(device), meas_resampled_img.to(device), warmstrat_img.to(device), seg_map.to(device)

        data_dict = {}
        data_dict['file_id'] = [fname] * args.batch_size
        data_dict['slice_idx'] = [mid_slice] * args.batch_size


        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
                    
        # Forward measurement model (Ax + n)
        #y = operator.forward(ref_img)
        #y_n = noiser(y)
        y = meas_img
        y_n = y.clone()

        # Estimate parameters of the measurement model
        g, s, l = operator.fit_A(x_hf=ref_img.float(), y_lf=meas_resampled_img.float())
        logger.info(f"Estimated parameters - Gamma: {g:.4f}, Sigma: {s:.4f}, Loss: {l:.4f}")

        # Inject noise if skip_timestep is enabled, no memory bank at the moment
        # so we just start from t = closest_time and inject noise to the measurement image
        if configs['skip_timestep']:
            closest_time = 299
            skip_x0 = warmstrat_img.clone() 
            skip_timestep = 999 - closest_time
        else:
            skip_x0 = None
            skip_timestep = configs['skip_timestep']
            
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            measurement=y_n.to(torch.float32),
            measurement_cond_fn=measurement_cond_fn,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timestep,
            skip_x0=skip_x0,
            line_search=configs['line_search']
        )
        
        sample = sample.contiguous()
        all_images.append(sample.cpu().numpy()) 
        refs.append(ref_img.cpu().numpy())
        ys.append(y.cpu().numpy())
        print("One image done!")

        if data_dict is not None:
            # Save the images
            for j in range(args.batch_size):
                if not os.path.exists(f'{save_path}/{data_dict["file_id"][j]}'):
                    os.makedirs(f'{save_path}/{data_dict["file_id"][j]}')
                np.save(f'{save_path}/{data_dict["file_id"][j]}/pred_{data_dict["slice_idx"][j]}_axial.npy', 
                       sample[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/gt_{data_dict["slice_idx"][j]}_axial.npy', 
                       ref_img[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/lr_{data_dict["slice_idx"][j]}_axial.npy', 
                       y[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/warmstart_{data_dict["slice_idx"][j]}_axial.npy', 
                       skip_x0[j].cpu().numpy() if skip_x0 is not None else y[j].cpu().numpy())
    
    print("Saving the results in Numpy")
    logger.log("sampling complete")


def create_argparser():
    """Create argument parser with default values."""
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/logs_large_zero2two_HCPMoreSlice2025/model360000.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
