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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from einops import rearrange
from utils import SomethingSomethingV2Dataset
from models.hierarchical_model import HPC_model, MEC_model, JointHPCMEC, Inverse_World_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test hierarchical model")
    
    parser.add_argument('--model_ckpt', type=str, default='./checkpoints/model.pth', help='Path to the model checkpoint')
    
    return parser.parse_args()

def setup_loaders(batch_size=64, num_workers=4):
    root_dir = './dataset/20bn-something-something-v2/rawframes'
    test_file = './dataset/labels/test.json'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    test_dataset = SomethingSomethingV2Dataset(root_dir=root_dir, annotations_file=test_file, transform=transform, frames_per_clip=8, sliding_window=8, sample_downsample_rate=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dataset, test_loader

def init_model(model_ckpt):
    MODEL_DEPTH = 16   
    assert MODEL_DEPTH in {16, 20, 24, 30}
    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt = osp.join("VAR", "checkpoints", 'vae_ch160v4096z32.pth')
    if not osp.exists(vae_ckpt):
        os.makedirs(osp.dirname(vae_ckpt), exist_ok=True)
        os.system(f'wget -O {vae_ckpt} {hf_home}/{osp.basename(vae_ckpt)}')

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    if 'vae' not in globals():
        # build_vae_var returns (VQVAE, VAR); only the VQVAE tokenizer is needed
        vae, _var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    hpc_model = HPC_model().to(device)
    mec_model = MEC_model().to(device)
    JointHPCMEC_model = JointHPCMEC(hpc_model, mec_model).to(device)
    model = Inverse_World_model(JointHPCMEC_model).to(device)
    
    # load pre-trained weights
    model.load_state_dict(torch.load(model_ckpt, map_location='cpu'), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    print(f'load pre-trained weights finished.') 

    return model, vae



if __name__ == "__main__":
    args = parse_args()

    # load test dataset
    test_dataset, test_loader = setup_loaders(batch_size=24, num_workers=4)
    data = iter(test_loader)

    # load model
    model, vae = init_model(args.model_ckpt)
    model = model.to(device)
    vae = vae.to(device)

    with torch.no_grad():
        X = next(data).to(device)
        X = X.flatten(0, 1)  # Flatten the batch and sequence dimensions
        batch_size, seq_len, _, _, _ = X.size()
        z = vae.quant_conv(vae.encoder(X.flatten(0, 1)))
        z = vae.quantize.f_to_idxBl_or_fhat(z, to_fhat=True, v_patch_nums=vae.quantize.v_patch_nums)
        z = z[-1] # only use the largest scale
        z = rearrange(z, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len)
        res_dict = model(z)
        loss, loss_dict = model.total_loss(res_dict)
        # Print batch-wise loss information
        component_str = ", ".join([f"{i}: {loss_dict[i].item():.4f}" for i in loss_dict.keys()])
        print(f"Test Batch Loss: {loss.item():.4f} | Components: [{component_str}]")   

    x = res_dict['x']
    s_inf = res_dict['s_inf']

    s_int = res_dict['s_recon']
    p_inf = res_dict['p_inf']
    p_int = res_dict['p_int']
    g_inf = res_dict['g_inf']

    s_gen = res_dict['s_gen']
    p_gen = res_dict['p_gen']
    g_gen = res_dict['g_gen']
    a_low = res_dict['a_low']