"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

import random
import numpy as np
import torch as th

import wandb

def main():
    args = create_argparser().parse_args()
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        th.cuda.manual_seed_all(args.seed)
        # For full determinism, but may impact performance
        # th.backends.cudnn.deterministic = True
        # th.backends.cudnn.benchmark = False

    dist_util.setup_dist()

    wandb_config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "microbatch": args.microbatch,
        "image_size": args.image_size,
        "diffusion_steps": args.diffusion_steps,
        "noise_schedule": args.noise_schedule,
        "use_fp16": args.use_fp16,
        "ema_rate": args.ema_rate,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "learn_sigma": args.learn_sigma,
        "num_res_blocks": args.num_res_blocks,
        "attention_resolutions": args.attention_resolutions,
        "dropout": args.dropout,
        "use_scale_shift_norm": args.use_scale_shift_norm,
        "resblock_updown": args.resblock_updown,
        "timestep_respacing": args.timestep_respacing,
        "weight_decay": args.weight_decay,
        "lr_anneal_steps": args.lr_anneal_steps,
    }

    wandb.init(
      project="DDA_Cardiac",
      wandb_entity=None, #  username
      name=f"DDAC-{args.diffusion_steps}-{args.image_size}",
      config=wandb_config
    )

    logger.configure(
        format_strs=["stdout", "log", "csv", "wandb"], # Add "wandb" here
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        mask_dir=args.mask_dir,
        num_mask_classes=args.num_mask_classes,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        log_samples_interval=args.log_samples_interval, # wandb
        seed=args.seed,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        log_samples_interval=50, # wandb
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=42,# for reproducibility
        log_dir="",
        in_channels=2,
        out_channels=2,
        mask_dir=None,
        num_mask_classes=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
