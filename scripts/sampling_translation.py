"""
Generate a large batch of image samples from a model and save them as a large
torch tensor array. This can be used to produce samples.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import matplotlib.pyplot as plt
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

import visdom
viz = visdom.Visdom()

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img



def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    ds= load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        test_flag=True,
        class_cond=args.class_cond,
    )
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True)
    
    data = iter(datal)
    
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
        
    model.eval()

    while len(all_images) * args.batch_size < args.num_samples:
        b, _class, path = next(data)  #should return an image, class and path of the image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
  
        slice_ID=path[0].split('\\')[-1]
        
        
        viz.image(visualize(img[0,0,...]), opts=dict(caption="img input0"))
        viz.image(visualize(img[0, 1, ...]), opts=dict(caption="img input1"))
    

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        model_kwargs = {}
        if args.class_cond:
            model_kwargs["y"]=_class["y"].to(dist_util.dev())
            
        
        for i in range(args.num_ensemble):  #this is for the generation of an translated images 5 times and choosing best out of it..
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
    
            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
            s=sample.clone().detach().requires_grad_(True)
            th.save(s, './results/'+str(slice_ID)+'_'+str(i)+'_output') 


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
