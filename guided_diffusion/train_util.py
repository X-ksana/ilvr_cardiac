import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler



import random
import numpy as np
import torch as th
import wandb

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        log_samples_interval=1000, # Control sample logging frequency for wandb
        seed=42,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.log_samples_interval = log_samples_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.seed = seed
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.best_loss = float('inf')


        self.sync_cuda = th.cuda.is_available()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self._load_and_sync_parameters()
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # 1. load ckpt into single state_dict variable
                state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
                # 2. Restore the model and optimizer state
               # self.mp_trainer.load_state_dict(state_dict)
                self.model.load_state_dict(state_dict)


                # 3. Restore the rng
                try:
                    th.set_rng_state(state_dict["torch_rng_state"])
                    th.cuda.set_rng_state(state_dict["cuda_rng_state"])
                    np.random.set_state(state_dict["numpy_rng_state"])
                    random.setstate(state_dict["random_rng_state"])
                except KeyError:
                    logger.log("Could not find RNG states in checkpoint. Starting with new random state.")
                

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            
            # Periodically log a batch of geenrated samples
            if self.step % self.log_samples_interval == 0:
                self.log_samples()
            
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.last_loss < self.best_loss:
                self.best_loss = self.last_loss
                self.save_best_checkpoint()
            
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    # ---Best checkpoint and overwrite the previous one ---
    def save_best_checkpoint(self):
        """
        Saves the model, EMA, and optimizer states for the best performing model
        and removes the previous best checkpoint.
        """
        # Helper
        def _save_and_cleanup(suffix, data_to_save):
            # Define the new best checkpoint filename
            new_filename = f"best_{suffix}_{(self.step+self.resume_step):06d}.pt"
            
            # Find and delete the old best checkpoint for this suffix (e.g., 'best_model_*.pt')
            old_checkpoints = bf.glob(bf.join(get_blob_logdir(), f"best_{suffix}_*.pt"))
            for old_ckpt in old_checkpoints:
                if os.path.basename(old_ckpt) != new_filename:
                    logger.log(f"Removing old best checkpoint: {old_ckpt}")
                    bf.remove(old_ckpt)

            # Save the new best checkpoint
            with bf.BlobFile(bf.join(get_blob_logdir(), new_filename), "wb") as f:
                th.save(data_to_save, f)

        logger.log(f"Saving new best checkpoint at step {self.step}...")

        # Save the main model parameters
        model_state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        _save_and_cleanup("model", model_state_dict)

        # Save the EMA parameters
        for rate, params in zip(self.ema_rate, self.ema_params):
            ema_state_dict = self.mp_trainer.master_params_to_state_dict(params)
            _save_and_cleanup(f"ema_{rate}", ema_state_dict)
        
        # Save the optimizer state
        _save_and_cleanup("opt", self.opt.state_dict())
    
    # Add to TrainLoop class

    def log_samples(self):
        """
        Generate a batch of samples from the model and logs them to wandb
        """
        self.model.eval()
        logger.log("Logging a batch of samples")

        device = next(self.model.parameters()).device

        # Generate small batch of samples
        num_samples_to_log = 6
        shape = (num_samples_to_log, self.model.in_channels, self.model.image_size, self.model.image_size) #  flex in channels

        # In case conditional
        model_kwargs = {}
        if self.model.num_classes is not None:
            classes = th.randint(low=0, high=self.model.num_classes, size=(num_samples_to_log,), device=device)
            model_kwargs["y"] = classes

        samples = self.diffusion.p_sample_loop_validate(
            self.model,
            shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        # Convert samples to the format wandb expects for images
        # Rescale from [-1,1] to [0,255] and convert to numpy
        # Bear in mind 2 channels, one is binary cardiac mri, one is mask in 4 classes
        # 1. De-normalize the entire batch from [-1, 1] to [0, 1] range
        samples = (samples + 1) / 2.0
    # 2. Split the 4-channel output into image and mask
    #    Channel 0 is the image. Channels 1, 2, 3 are the mask classes.
        image_channels = samples[:, :1, :, :]  # Shape: (N, 1, H, W)
        mask_channels = samples[:, 1:, :, :]   # Shape: (N, 3, H, W)
        categorical_masks = th.argmax(mask_channels, dim=1).cpu().numpy() + 1
        image_channels = image_channels.clamp(0, 1)
        images_for_log = (image_channels * 255).to(th.uint8)
        images_for_log = images_for_log.repeat(1, 3, 1, 1) # (N, 1, H, W) -> (N, 3, H, W)
        images_for_log = images_for_log.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        images_for_log = images_for_log.contiguous().cpu().numpy()

        """
        samples = ((samples+1)*127.5).clamp(0,255).to(th.uint8)
        samples = samples.permute(0,2,3,1) # NCHW to NHWC
        samples = samples.contiguous().cpu().numpy
        """
        # 5. Log to Weights & Biases using their segmentation mask format
        log_list = []
        for i in range(num_samples_to_log):
            log_list.append(
                wandb.Image(
                    images_for_log[i],
                    masks={
                    "predictions": {
                        "mask_data": categorical_masks[i],
                        "class_labels": {
                            1: "LV",
                            2: "Myo",
                            3: "RV"
                        }
                    }
                }
            )
        )
        wandb.log({"validation_samples": log_list, "step": self.step})
        # Log images to wandb
       # wandb.log({
       #     "samples":[wandb.Image(sample) for sample in samples]
       # })

        self.model.train()


    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        #store the last loss for  best checkpoint
        self.last_loss = self.current_loss # set this in forward_backward

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.current_loss = loss.item()  # Add this line
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # --- TO SAVE RNG STATES ---
            state_dict["torch_rng_state"] = th.get_rng_state()
            state_dict["cuda_rng_state"] = th.cuda.get_rng_state()
            state_dict["numpy_rng_state"] = np.random.get_state()
            state_dict["random_rng_state"] = random.getstate()

            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
