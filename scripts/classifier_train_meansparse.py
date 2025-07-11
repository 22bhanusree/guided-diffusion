"""
Training Meansparse based classifier 
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from guided_diffusion.image_datasets import load_data
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from meansparse_unet import MeanSparse
from meansparse_script_util import (
    meansparse_add_dict_to_argparser,
    meansparse_args_to_dict,
    meansparse_classifier_and_diffusion_defaults,
    meansparse_create_classifier_and_diffusion,
    meansparse_create_classifier,
    meansparse_classifier_defaults,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()



    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    
    def initialize_missing_keys(state_dict, model):
        for name, module in model.named_modules():
            if isinstance(module, MeanSparse):
                if f"{name}.running_mean" not in state_dict:
                    state_dict[f"{name}.running_mean"] = th.zeros_like(module.running_mean)
                if f"{name}.running_var" not in state_dict:
                    state_dict[f"{name}.running_var"] = th.zeros_like(module.running_var)
                if f"{name}.threshold" not in state_dict:
                    state_dict[f"{name}.threshold"] = th.zeros_like(module.threshold)
                if f"{name}.flag_update_statistics" not in state_dict:
                    state_dict[f"{name}.flag_update_statistics"] = th.tensor(1, dtype=module.flag_update_statistics.dtype, device=module.flag_update_statistics.device)
                if f"{name}.batch_num" not in state_dict:
                    state_dict[f"{name}.batch_num"] = th.tensor(16, dtype=module.batch_num.dtype, device=module.batch_num.device)
        return state_dict
     
    def create_and_initialize_meansparse_classifier(args):
        """
        Create and initialize a meansparse classifier.
        """
        meansparse_classifier = meansparse_create_classifier(**args_to_dict(args, meansparse_classifier_defaults().keys()))

        state_dict = dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        state_dict = initialize_missing_keys(state_dict, meansparse_classifier)
        meansparse_classifier.load_state_dict(state_dict)
        
        meansparse_classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            meansparse_classifier.convert_to_fp16()

        meansparse_classifier.eval()
        return meansparse_classifier

    def wrap_with_ddp(model):
        """
        Wrap the model with DistributedDataParallel.
        """
        model = DDP(
            model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )
        return model

    # Create 5 meansparse classifiers
    meansparse_classifiers = [create_and_initialize_meansparse_classifier(args) for _ in range(5)]

    # Wrap each classifier with DDP
    meansparse_classifiers_ddp = [wrap_with_ddp(meansparse_classifier) for meansparse_classifier in meansparse_classifiers]

    # Print to verify
    # for i, temp in enumerate(meansparse_classifiers_ddp):
    #     print(f"Meansparse Classifier {i+1}: {temp}")

    logger.log("loading classifier with meansparse...")

    classifier = DDP(
        classifier,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    
    logger.log("loading data ...")
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
        is_val=False,
    )
    
    timestamps = [0, 200, 400, 600, 800]
    running_means = {t: [] for t in timestamps}
    running_vars = {t: [] for t in timestamps}

    count = 0
    with th.no_grad():
        while (count<80072): #(TODO: count < no.of images in data / batch_size)
            batch, extra = next(data)
            count +=1
            labels = extra["y"].to(dist_util.dev())
            batch = batch.to(dist_util.dev())

            for idx, meanspare_classifier in enumerate(meansparse_classifiers_ddp): 
            
                timestamp = timestamps[idx]

                if timestamp == 0:
                    t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
                else:
                    t = th.full((batch.shape[0],), timestamp, dtype=th.long, device=dist_util.dev())
                    batch = diffusion.q_sample(batch, t)
                
                for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
                ):
                        
                    logits = meanspare_classifier(sub_batch, timesteps=sub_t)
            

    # Set flag_update_statistics to False for all classifiers
    for meansparse_classifier in meansparse_classifiers_ddp:
        for name, module in meansparse_classifier.named_modules():
            if isinstance(module, MeanSparse):
                module.flag_update_statistics.fill_(0)

    def save_classifiers(classifiers, filename='meansparse_classifiers.pth'):
        th.save(classifiers, filename)
        print(f'Saved meansparse classifiers to {filename}')

    save_classifiers(meansparse_classifiers_ddp, 'meansparse_classifiers.pth')


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        microbatch=-1,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()


