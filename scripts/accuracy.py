"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
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

    classifier = DDP(
        classifier,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    val_data = load_data(
        data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        is_val=True,
    )
    logger.log("loading data ...")

    # Load the saved classifiers
    def load_classifiers(filename='meansparse_classifiers.pth'):
        classifiers = th.load(filename)
        print(f'Loaded meansparse classifiers from {filename}')
        return classifiers
    
    logger.log("loading classifier with meansparse...")
    loaded_meansparse_classifiers_ddp = load_classifiers('meansparse_classifiers_temp_1.pth')
        
    
    # for meansparse_classifier in loaded_meansparse_classifiers_ddp:
    #     for name, module in meansparse_classifier.named_modules():
    #         if isinstance(module, MeanSparse):
    #             # module.flag_update_statistics.fill_(0)
    #             module.threshold.fill_(1)
    
    # for idx, meansparse_classifier in enumerate(loaded_meansparse_classifiers_ddp):
    #     print(f"Threshold for classifier {idx+1}:")
    #     for name, param in meansparse_classifier.state_dict().items():
    #         if "threshold" in name:
    #             print(name, param)

    

    # Evaluate each meansparse_classifier
    for idx, meansparse_classifier in enumerate(loaded_meansparse_classifiers_ddp):
        
        # Disable gradient computation for validation
        with th.no_grad():
            # For accumulating predictions and labels
            correct_meansparse = 0
            total_meansparse = 0
            correct_classifier = 0
            total_classifier = 0
            count =0 

            while (count < 3125) :
                val_batch, val_extra = next(val_data)
                val_labels = val_extra["y"].to(dist_util.dev())  # Move labels to the same device as the model
                val_batch = val_batch.to(dist_util.dev())  

                # val_t = th.zeros(val_batch.shape[0], dtype=th.long, device=dist_util.dev()) # to evaluate accuracy on no noise input

                val_t = th.full((val_batch.shape[0],), 200 , dtype=th.long, device=dist_util.dev()) # to evaluate accuracy on noisy input t=200 
                val_batch = diffusion.q_sample(val_batch, val_t) # to evaluate accuracy on noisy input t=200 

                # Iterate over microbatches
                for sub_batch, sub_labels, sub_t in split_microbatches(args.microbatch, val_batch, val_labels, val_t):
                    
                    # Forward pass through the current meansparse_classifier
                    logits_meansparse = meansparse_classifier(sub_batch, timesteps=sub_t)
                    _, meansparse_predictions = th.max(logits_meansparse, dim=1)
                    correct_meansparse += (meansparse_predictions == sub_labels).sum().item()
                    total_meansparse += sub_labels.size(0)

                    # Classifier predictions
                    logits_classifier = classifier(sub_batch, timesteps=sub_t)
                    _, predictions = th.max(logits_classifier, dim=1)
                    correct_classifier += (predictions == sub_labels).sum().item()
                    total_classifier += sub_labels.size(0)

                count +=1
            

            # accuracy_meansparse = correct_meansparse / total_meansparse
            accuracy_classifier = correct_classifier / total_classifier

            # print(f'Meansparse model Accuracy : {accuracy_meansparse * 100:.2f}%')
            print(f'Classifier Accuracy : {accuracy_classifier * 100:.2f}%')



    
    
    

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)



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






