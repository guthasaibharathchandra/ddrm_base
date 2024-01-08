import torch
import torchvision
import argparse
import random
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
from datasets import get_dataset, data_transform, inverse_data_transform
from torchvision.transforms import ToTensor, Compose, PILToTensor, Normalize, Resize
from lib_svd import Deblurring, SuperResolution
from lib import make_schedule, DDRM
from ckpt_util import download
from collections import namedtuple
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.kid import KernelInceptionDistance

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_model_and_initialize_ddrm(config_dict, args_dict):
    
    # This loads pretrained DDPM model and initializes DDRM class object for reverse process 
    config = dict2namespace(config_dict)
    args = dict2namespace(args_dict)

    if not os.path.exists(args.exp):
            os.mkdir(args.exp)

    model = create_model(**config_dict['model'])
    if config_dict['model']['use_fp16']:
            
            model.convert_to_fp16()
            
            if config_dict['model']['class_cond']:
                ckpt = os.path.join(args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (config_dict['model']["image_size"], config_dict['model']["image_size"]))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (config_dict['model']["image_size"], config_dict['model']["image_size"]), ckpt)
            else:
                ckpt = os.path.join(args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            model.load_state_dict(torch.load(ckpt, map_location=torch.device("cuda")))
            model.to(torch.device("cuda"))
            model.eval()
            model = torch.nn.DataParallel(model)

    # diffusion schedule
    schedule = make_schedule(scheme=config_dict['diffusion']['beta_schedule'], rvar='beta', T=config_dict['diffusion']['num_diffusion_timesteps'], 
                            start_beta=config_dict['diffusion']['beta_start'], end_beta=config_dict['diffusion']['beta_end'])
                            
    # DDRM wrapper
    ddrm_model = DDRM(schedule=schedule,model=model,weightedloss=False,cuda=True)

    return config, args, ddrm_model, schedule


def load_dataset_and_task(config, args):

    dataset, test_dataset = get_dataset(args, config)
    device = torch.device("cuda")
    args.subset_start = 0
    args.subset_end = len(test_dataset)
    print(f'Dataset has size {len(test_dataset)}')    

    def seed_worker(worker_id):
        worker_seed = args.seed % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.sampling.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
    #sigma_y = args.sigma_y

    if args.task == 'deblur':
        H = Deblurring(torch.Tensor([1/9] * 9).to(device), config.data.channels, config.data.image_size, device)        
    elif args.task == 'sr4x':
        H = SuperResolution(config.data.channels, config.data.image_size, ratio=4, device=device)
    else:
        raise NotImplementedError

    return dataset, val_loader, H


def run_evaluation(config, args, val_loader, ddrm_model, H, T_ddrm_steps=20):

        device = torch.device("cuda")
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_metric_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        kid = KernelInceptionDistance(subsets=100, subset_size=100, normalize=True, compute_on_cpu=False).cuda()
        
        for x_orig, classes in pbar:
            
            x_orig = x_orig.to(device)
            x_orig = data_transform(config, x_orig)
            y = H.H(x_orig)
            y = y + args.sigma_y * torch.randn_like(y)
            
            ##Begin DDRM
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=device,
            )
            
            with torch.no_grad():
                x = ddrm_model.reverse_diffusion_ddrm(x, y, args.sigma_y, args.eta, args.eta_b, config.data.channels, config.data.image_size, H=H, T_ddrm=T_ddrm_steps, cuda=True, onlymean=False)
        
            x = inverse_data_transform(config, x)
            
            for j in range(x.size(0)):
                    
                    orig = inverse_data_transform(config, x_orig[j])
                    rxorig = orig.reshape((1,*orig.shape))
                    rx = x[j].to(device).reshape((1,config.data.channels, config.data.image_size, config.data.image_size))
                    
                    ssim = structural_similarity_index_measure(rx, rxorig)
                    metric_psnr = peak_signal_noise_ratio(rx, rxorig)
                    mse = torch.mean((rx - rxorig) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    kid.update(rx, real=False)
                    kid.update(rxorig, real=True)

                    avg_psnr += float(psnr)
                    avg_metric_psnr += float(metric_psnr)
                    avg_ssim += float(ssim)

            idx_so_far += y.shape[0]
            pbar.set_description(f"PSNR = {round(avg_psnr / (idx_so_far - idx_init),2)}, METRIC_PSNR = {round(avg_metric_psnr / (idx_so_far - idx_init),2)}, SSIM = {round(avg_ssim / (idx_so_far - idx_init),2)}")

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        avg_ssim = avg_ssim / (idx_so_far - idx_init)
        avg_metric_psnr = avg_metric_psnr / (idx_so_far - idx_init)
        kid_mean, kid_std  = kid.compute()
        kid_mean, kid_std = round(float(kid_mean*1e3),2), round(float(kid_std*1e3),2)
        print(f"Total Average PSNR = {round(avg_psnr,2)}")
        print(f"Total Average Metric PSNR = {round(avg_metric_psnr,2)}")
        print(f"Total Average SSIM = {round(avg_ssim,2)}")
        print(f"KID (experimental) = {kid_mean}+-{kid_std}")
        print("Number of samples: %d" % (idx_so_far - idx_init))


