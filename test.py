"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import glob
import os
import pandas as pd

import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from guided_diffusion import logger
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.image_datasets import IQTDataset
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


def main():
    set_seed(42)

    # Load configuration
    config_path = '/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/DynamicDPS/configs.yaml'
    with open(config_path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        configs=configs,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    assert args.model_path != "ADD_YOUR_MODEL_PATH_HERE.pth", "Please specify the model path."
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print('Using device:', device)

    # Configuration paths and file handling
    save_path = './DynamicDPS_results/'
    lst_files = [
        '116120', '116221', '116423', '116524', '116726', '117021', '117122', 
        '117324', '117728', '117930', '118023', '118124', '118225', '118528', 
        '118730', '118831', '118932', '119025', '119126', '119732'
    ]
    
    data_dir = 'Path/To/Your/Data/Directory'  # Update this path to your data directory
    files = glob.glob(data_dir + '/T1w/T1w_acpc_dc_restore_brain.nii.gz')
    print(f"Total files: {len(files)}")
    
    # Filter files based on lst_files
    files_new = []
    for f in files:
        if f.split('/')[-3] in lst_files:
            files_new.append(f)
    files = files_new

    dataset = IQTDataset(files, configs=configs, return_id=configs['data']['return_id'])
    print(f"Files: {len(files)} Dataset size: {len(dataset)}")
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    try:
        ref_img, data_dict = next(iter(data))
        print(f"Batch: ref_img shape: {ref_img.shape}, data_dict: {data_dict}")
    except Exception as e:
        print(f"Error in batch: {e}")       
 
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
    
    for i, (ref_img, data_dict) in tqdm.tqdm(enumerate(data)):
        print(f"{i}/{len(data)}")
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
            
        ref_img = ref_img.to(device)
        
        # Load U-Net output
        print(data_dict['file_id'][0], data_dict['slice_idx'].numpy()[0])
        fname_curr, slice_curr = int(data_dict['file_id'][0]), str(data_dict['slice_idx'].numpy()[0])
        print(fname_curr, slice_curr)
        
        # Load the conditional model's output
        data = np.load(f'/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/cond_results/unet/ood_contrast/{fname_curr}/pred_{slice_curr}_axial.npy')[0]
        mean = 271.64814106698583
        std = 377.117173547721
        
        # Clip values and normalize
        data = np.clip(data, 0., 2.0)
        print("DATA shape")
        print(data.min(), data.max())

        # Load the reference memory bank
        df = pd.read_csv('./reference_memorybank.csv')
        loss_dict = {}
        group_size = 50
        for i in range(0, 1000, 1):
            curr_time = '[' + str(i) + ']'
            df_curr = df[df['Time'] == curr_time]
            loss = df_curr['Loss'].values
            loss_dict[i] = np.mean(loss)

        # 1. Sort your keys so that you can iterate in ascending or descending order
        sorted_keys = sorted(loss_dict.keys())
        sorted_values = [loss_dict[k] for k in sorted_keys]

        # 2. Chunk the keys and values into groups of size `group_size`
        chunks = [sorted_keys[i:i + group_size] for i in range(0, len(sorted_keys), group_size)]
        values_chunks = [sorted_values[i:i + group_size] for i in range(0, len(sorted_values), group_size)]

        # 3. Compute the average of each chunk
        mean_keys = [chunk[-1] for chunk in chunks]
        mean_values = [np.mean(values_chunk) for values_chunk in values_chunks]
        mean_values = torch.tensor(mean_values)

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Downsample the conditional data
        data = torch.tensor(data).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        data_low = operator.forward(data)

        # Calculate the data likelihood
        test_likelihood = torch.linalg.norm(y - data_low)**configs['tau']
        print(f"Test likelihood: {test_likelihood.item()}")

        # Find the time index corresponding to the closest mean value
        closest_time_idx = np.argmin(np.abs(mean_values - test_likelihood.item()))
        if (mean_values[closest_time_idx] < test_likelihood.item()) and (closest_time_idx > 1):
            closest_time_idx -= 1
        print(f"Closest time index: {closest_time_idx}, Mean value: {mean_values[closest_time_idx]}")
        # Get the corresponding time value
        closest_time = mean_keys[closest_time_idx]
        print(f"Closest time: {closest_time}")
        
        # Inject noise if skip_timestep is enabled 
        if configs['skip_timestep']:
            skip_x0 = data #torch.tensor(data).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
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
    
    print("Saving the results in Numpy")
    # Concatenate all the images into a single numpy array
    arr = np.array(all_images)
    arr_ys = np.array(ys)
    arr_refs = np.array(refs)
    
    mini, maxi = 0.0, 2.0 
    arr = np.clip(arr, mini, maxi)
    
    # Save the samples in numpy files
    np.savez("samples_pred", arr)
    np.savez("samples_ys", arr_ys)
    np.savez("samples_refs", arr_refs)

    logger.log("sampling complete")


def create_argparser():
    """Create argument parser with default values."""
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="ADD_YOUR_MODEL_PATH_HERE.pth",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
