# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys

_VAR_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "VAR")
sys.path.insert(0, _VAR_DIR)
from VAR.models import build_vae_var
from VAR.utils.data import normalize_01_into_pm1
sys.path.remove(_VAR_DIR)
for _m in list(sys.modules):
    if _m in ("dist", "models", "utils") or _m.startswith(("models.", "utils.")):
        del sys.modules[_m]
del _VAR_DIR

import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
import argparse
from utils import SomethingSomethingV2Dataset
from models.hierarchical_model import HPC_model, MEC_model, JointHPCMEC, Inverse_World_model

os.environ["WANDB_START_METHOD"] = "thread"
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train hierarchical model")
    
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3],
                        help="Training phase (1: HPC, 2: HPC+MEC, 3: Inverse model)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size for training and evaluation")
    parser.add_argument("--frames_per_clip", type=int, default=8,
                        help="Number of frames per video clip")
    parser.add_argument("--sliding_window", type=int, default=2,
                        help="Sliding window size for video clips")
    parser.add_argument("--clip_grad_norm", type=float, default=0.1,
                        help="Gradient clipping norm")
    parser.add_argument("--inverse_lr", type=float, default=1e-4,
                        help="Learning rate for inverse model")
    parser.add_argument("--world_lr", type=float, default=1e-4,
                        help="Learning rate for world model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--work_dir", type=str, 
                        default='./', help="Directory to save checkpoints")
    parser.add_argument("--wandb_enabled", type=bool, default=True,
                        help="Enable wandb logging")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="Path to model checkpoint for phase 2 and 3 training")
    
    return parser.parse_args()


class Trainer:
    def __init__(self, args=None):
        if args is None:
            args = parse_args()
            
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=60),
        )
        dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
        )
        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        
        # Use args to initialize parameters
        self.phase = args.phase
        self.num_epochs = args.num_epochs
        self.epoch = 0
        self.batch_size = args.batch_size
        self.frames_per_clip = args.frames_per_clip
        self.sliding_window = args.sliding_window
        self.clip_grad_norm = args.clip_grad_norm
        self.lr_dict = {
            "inverse_lr": args.inverse_lr,
            "world_lr": args.world_lr,
        }
        self._set_seed_everywhere(args.seed)

        self.work_dir = args.work_dir
        self.model_ckpt = args.model_ckpt

        # all processes use the work_dir from the main process
        if torch.distributed.is_initialized():
            objs = [str(self.work_dir)]
            torch.distributed.broadcast_object_list(objs, 0)
            self.work_dir = Path(objs[0])
        self.accelerator.wait_for_everyone()
        logger.info("Saving to {}".format(self.work_dir))

        self.exp_name = f'Coupled_model-phase{self.phase}'
        self.wandb_enabled = args.wandb_enabled

        self._setup_loaders(batch_size=self.batch_size, frames_per_clip=self.frames_per_clip, sliding_window=self.sliding_window)
        self._init_tracker(exp_name=self.exp_name, wandb_enabled=self.wandb_enabled)
        self._init_model(self.phase, model_ckpt=self.model_ckpt)


    def _init_tracker(self, exp_name=None, wandb_enabled=True):
        self.accelerator.init_trackers(
            project_name='Abstract latent action',
            init_kwargs={
                "wandb": {
                    "reinit": False,
                    "settings": {"start_method": "thread"},
                    "name": exp_name,
                    "mode": "online" if wandb_enabled else "disabled",
                    "save_code": True,
                },
            },
        )
        if self.accelerator.is_main_process:
            self.wandb_run = self.accelerator.get_tracker("wandb", unwrap=True)
            logger.info("wandb run url: %s", self.wandb_run.get_url())

    def _set_seed_everywhere(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _init_model(self, phase, model_ckpt=None):
        MODEL_DEPTH = 16   
        assert MODEL_DEPTH in {16, 20, 24, 30}
        # download checkpoint
        hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
        vae_ckpt = osp.join("VAR", "checkpoints", 'vae_ch160v4096z32.pth')
        if not osp.exists(vae_ckpt):
            os.makedirs(osp.dirname(vae_ckpt), exist_ok=True)
            os.system(f'wget -O {vae_ckpt} {hf_home}/{osp.basename(vae_ckpt)}')

        # build vae, var
        self.patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.device = self.accelerator.device
        if 'vae' not in globals():
            # build_vae_var returns (VQVAE, VAR); only the VQVAE tokenizer is needed
            self.VAE, _var = build_vae_var(
                V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
                device=self.device, patch_nums=self.patch_nums,
                num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
            )

        # load checkpoints
        self.VAE.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        self.VAE.eval()
        for p in self.VAE.parameters(): p.requires_grad_(False)
        print(f'prepare finished.')

        # Phase 1 Training
        if phase == 1:
            self.HPC_model = HPC_model().to(self.device)
            self.HPC_model.apply(self.HPC_model._init_weights)

            self.MEC_model = MEC_model().to(self.device)
            self.MEC_model.apply(self.MEC_model._init_weights)
            
            self.model = JointHPCMEC(self.HPC_model, self.MEC_model)

            self.wm_optimizer, self.wm_scheduler = \
                self.model.configure_optimizers(weight_decay=1e-4, lr=self.lr_dict, betas=(0.9, 0.999), T_max=200)

            self.model, self.wm_optimizer, self.wm_scheduler = \
                self.accelerator.prepare(self.model, self.wm_optimizer, self.wm_scheduler)

        # Phase 2 Training
        elif phase == 2:
            self.HPC_model = HPC_model().to(self.device)
            self.MEC_model = MEC_model().to(self.device)
            self.JointHPCMEC_model = JointHPCMEC(self.HPC_model, self.MEC_model)
            self.model = Inverse_World_model(self.JointHPCMEC_model).to(self.device)

            self.model.apply(self.model._init_weights)
            self.JointHPCMEC_model.load_state_dict(torch.load(model_ckpt, map_location='cpu'), strict=True)
            self.model.period = 1

            self.wm_optimizer, self.wm_scheduler = \
                self.model.configure_optimizers(weight_decay=1e-4, lr=self.lr_dict, betas=(0.9, 0.999), T_max=200)
            
            self.model, self.wm_optimizer, self.wm_scheduler = \
                    self.accelerator.prepare(self.model, self.wm_optimizer, self.wm_scheduler)
        
        # Phase 3 Fine-tuning
        elif phase == 3:
            self.HPC_model = HPC_model().to(self.device)
            self.MEC_model = MEC_model().to(self.device)
            self.JointHPCMEC_model = JointHPCMEC(self.HPC_model, self.MEC_model)
            self.model = Inverse_World_model(self.JointHPCMEC_model).to(self.device)
            self.model.load_state_dict(torch.load(model_ckpt, map_location='cpu'), strict=True)
            self.model.period = None

            self.wm_optimizer, self.wm_scheduler = \
                self.model.configure_optimizers(weight_decay=1e-4, lr=self.lr_dict, betas=(0.9, 0.999), T_max=200, finetune_world_model=True)
            
            self.model, self.wm_optimizer, self.wm_scheduler = \
                    self.accelerator.prepare(self.model, self.wm_optimizer, self.wm_scheduler)

        else:
            raise ValueError("Invalid phase. Choose from 1, 2, or 3.")

    def _setup_loaders(self, batch_size=64, frames_per_clip=16, num_workers=4, sliding_window=4):
        root_dir = './dataset/20bn-something-something-v2/rawframes'
        train_file = './dataset/labels/train.json'
        val_file = './dataset/labels/validation.json'

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ])

        train_dataset = SomethingSomethingV2Dataset(root_dir=root_dir, annotations_file=train_file, transform=transform, frames_per_clip=frames_per_clip, sliding_window=sliding_window)
        val_dataset = SomethingSomethingV2Dataset(root_dir=root_dir, annotations_file=val_file, transform=transform, frames_per_clip=frames_per_clip, sliding_window=sliding_window)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.val_loader = self.accelerator.prepare(self.val_loader)

    def train(self, epoch=None):
        self.model.train()  # Set model to training mode

        epoch_loss = 0
        max_losses = 13 # Based on phase 3
        epoch_separate_loss = np.zeros(max_losses) 

        for batch_data in self.train_loader:
            # data to device
            X = batch_data.to(self.device)
            X = X.flatten(0, 1) # Flatten the batch and time dimensions
            # forward
            with self.accelerator.autocast():
                batch_size, seq_len, *_ = X.size()
                X_enc = self.VAE.quant_conv(self.VAE.encoder(X.flatten(0, 1)))
                X_enc = self.VAE.quantize.f_to_idxBl_or_fhat(X_enc, to_fhat=True, v_patch_nums=self.patch_nums)
                X_enc = X_enc[-1]
                X_enc = rearrange(X_enc, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len)
                res_dict = self.model(X_enc) # Use encoded X

            self.wm_optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = self.accelerator.unwrap_model(self.model).total_loss(res_dict)
            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            )
            self.wm_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()
            epoch_loss += loss.item()

            # Gather and average separate losses
            gathered_losses = {}
            for key, value in loss_dict.items():
                 gathered_losses[key] = self.accelerator.gather_for_metrics(value).mean().item()

            # Map to the fixed-size array based on phase
            if self.phase == 1:
                keys = ['sensory_inf', 'place_inf', 'sensory_recon', 'covariance', 'variance', 'smoothing']
            elif self.phase == 2:
                keys = ['sensory_inf', 'sensory_gen', 'sensory_recon', 'place_inf', 'place_gen', 'grid', 'action', 'cycle', 'covariance', 'variance']
            elif self.phase == 3:
                keys = ['sensory_inf', 'sensory_gen', 'sensory_recon', 'place_inf', 'place_gen', 'grid', 'action', 'cycle', 'covariance', 'variance']
            else: keys = []

            for i, key in enumerate(keys):
                 if key in gathered_losses:
                     epoch_separate_loss[i] += gathered_losses[key]

            # Print batch-wise loss information (using gathered values on main process)
            if self.accelerator.is_main_process:
                component_str = ", ".join([f"{k}: {v:.4f}" for k, v in gathered_losses.items()])
                print(f"Train Batch Loss: {loss.item():.4f} | Components: [{component_str}]")
        
        # Average losses over batches
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        epoch_separate_loss /= num_batches

        return epoch_loss, epoch_separate_loss, res_dict # res_dict is from the last batch

    def eval(self, epoch):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_separate_loss = np.zeros(13)
        num_batches = len(self.val_loader)
        with torch.no_grad():  # Disable gradient computation
            for batch_data in self.val_loader:
                X = batch_data.to(self.device)
                X = X.flatten(0, 1) # Flatten the batch and time dimensions
                with self.accelerator.autocast():
                    batch_size, seq_len, _, _, _ = X.size()
                    X = self.VAE.quant_conv(self.VAE.encoder(X.flatten(0, 1)))
                    X = self.VAE.quantize.f_to_idxBl_or_fhat(X, to_fhat=True, v_patch_nums=self.patch_nums)
                    X = X[-1] # only use the largest scale
                    X = rearrange(X, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len)
                    res_dict = self.model(X)

                loss, loss_dict = self.accelerator.unwrap_model(self.model).total_loss(res_dict)
                loss = self.accelerator.gather_for_metrics(loss).mean()
                total_loss += loss.item()

                 # Gather and average separate losses
                gathered_losses = {}
                for key, value in loss_dict.items():
                    gathered_losses[key] = self.accelerator.gather_for_metrics(value).mean().item()

                # Map to the fixed-size array based on phase
                if self.phase == 1: keys = ['sensory_inf', 'place_inf', 'sensory_recon', 'covariance', 'variance', 'smoothing']
                elif self.phase == 2: keys = ['sensory_inf', 'sensory_gen', 'sensory_recon', 'place_inf', 'place_gen', 'grid', 'action', 'cycle', 'covariance', 'variance']
                elif self.phase == 3: keys = ['sensory_inf', 'sensory_gen', 'sensory_recon', 'place_inf', 'place_gen', 'grid', 'action', 'cycle', 'covariance', 'variance']
                else: keys = []

                for i, key in enumerate(keys):
                    if key in gathered_losses:
                        total_separate_loss[i] += gathered_losses[key]

                # Print batch-wise loss information (using gathered values on main process)
                if self.accelerator.is_main_process:
                    component_str = ", ".join([f"{k}: {v:.4f}" for k, v in gathered_losses.items()])
                    print(f"Eval Batch Loss: {loss.item():.4f} | Components: [{component_str}]")

        avg_loss = total_loss / num_batches
        avg_separate_loss = total_separate_loss / num_batches
        self.eval_loss = avg_loss
        return avg_loss, avg_separate_loss, res_dict

    def run(self):
        best_loss = float('inf')
        checkpoint_path = self.work_dir
        # Ensure checkpoint path exists (only on main process)
        if self.accelerator.is_main_process:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)
        self.accelerator.wait_for_everyone() # Ensure directory is created before proceeding

        for epoch in tqdm(range(self.num_epochs)):
            # Training phase
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            train_loss, train_separate_loss, _ = self.train(epoch) # Don't need train_res_dict here

            # Evaluation phase
            print("Validation phase")
            val_loss, val_separate_loss, _ = self.eval(epoch) # Don't need test_res_dict here

            # Update schedulers
            self.wm_scheduler.step()

            # Log losses and other metrics (on main process)
            if self.accelerator.is_main_process:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rates": self.wm_scheduler.get_last_lr()[0],
                }
                # Add phase-specific losses
                if self.phase == 1: keys = ['sensory_inf', 'place_inf', 'sensory_recon', 'covariance', 'variance', 'smoothing']
                elif self.phase == 2: keys = ['sensory_inf', 'sensory_gen', 'sensory_recon', 'place_inf', 'place_gen', 'grid', 'action', 'cycle', 'covariance', 'variance']
                elif self.phase == 3: keys = ['sensory_inf', 'sensory_gen', 'sensory_recon', 'place_inf', 'place_gen', 'grid', 'action', 'cycle', 'covariance', 'variance']
                else: keys = []

                for i, key in enumerate(keys):
                    log_dict[f"train_{key}_loss"] = train_separate_loss[i]
                    log_dict[f"val_{key}_loss"] = val_separate_loss[i]

                self.accelerator.log(log_dict, step=epoch)

                # Save model checkpoint using Accelerator for distributed safety
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = os.path.join(checkpoint_path, f'phase{self.phase}_best_model')
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "best.pth"))

                    if self.accelerator.is_main_process:
                        logger.info(f"Saved best model state at epoch {epoch} to {save_path}")
                
                # Save the latest model and optimizer state for resuming training
                save_path = os.path.join(checkpoint_path, f'phase{self.phase}_latest_model')
                os.makedirs(save_path, exist_ok=True)
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "latest.pth"))
                if self.accelerator.is_main_process:
                    logger.info(f"Saved latest model state at epoch {epoch} to {save_path}")

        self.accelerator.end_training() # End tracking
        return float(self.eval_loss)


if __name__ == "__main__":
    trainer = Trainer()
    final_loss = trainer.run()
    print(f"Training completed with final validation loss: {final_loss:.4f}")
