"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def exists(val):
    return val is not None

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((512,512))
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def set_save_attn_map(model):
    def new_forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        # attention, what we cannot get enough of
        attn_mask = sim.softmax(dim=-1)
        
        #########################################
        ## new bookkeeping stuff
        #########################################
        self.attn = attn_mask#.clone()
        
        out = einsum('b i j, b j d -> b i d', attn_mask, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    for name, module in model.model.diffusion_model.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.forward = new_forward.__get__(module, type(module))

def save_cross_attention(inter, ts, token_idx, save_path):
    # ts = 591
    # token_idx = 9
    xa_maps_edit = inter['d_ts_to_name_to_M']
    l_layers = list(xa_maps_edit[ts].keys())
    l_maps = [F.interpolate(xa_maps_edit[ts][lname][:,:,:,:,token_idx], 64) for lname in l_layers]
    # print(l_maps[0].shape)
    viz_maps = torch.cat(l_maps, 1).sum((0,1)).detach().cpu().numpy()
    # img = Image.fromarray((viz_maps * 255).clip(0.0, 255.0).round().astype(np.uint8))
    # img.save(save_path)
    # print(viz_maps.shape)
    viz_maps_min = viz_maps.min(axis=(0, 1))
    viz_maps_max = viz_maps.max(axis=(0, 1))
    viz_maps = (viz_maps - viz_maps_min) / (viz_maps_max - viz_maps_min)
    # pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))

    plt.imshow(viz_maps, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(save_path, bbox_inches='tight')

def save_cross_attention_t(inter, ts, token_idx, save_path):
    # ts = 591
    # token_idx = 9
    xa_maps_edit = inter['d_ts_to_name_to_M']
    t_steps = list(xa_maps_edit.keys())
    l_layers = list(xa_maps_edit[ts].keys())
    l_maps = [F.interpolate(xa_maps_edit[ts_][lname][:,:,:,:,token_idx], 64) for ts_ in t_steps for lname in l_layers]
    # print(l_maps[0].shape)
    viz_maps = torch.cat(l_maps, 1).sum((0,1)).detach().cpu().numpy()
    # img = Image.fromarray((viz_maps * 255).clip(0.0, 255.0).round().astype(np.uint8))
    # img.save(save_path)
    # print(viz_maps.shape)
    # plt.imshow(viz_maps, cmap="Blues")
    plt.imshow(viz_maps, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(save_path, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../dreambooth/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # sample_path = os.path.join(outpath, "samples")
    # os.makedirs(sample_path, exist_ok=True)
    # attn_path = os.path.join(outpath, "attn_maps")
    # os.makedirs(attn_path, exist_ok=True)

    # base_count = len(os.listdir(sample_path))
    # grid_count = len(os.listdir(outpath)) - 1
    base_count = 0
    grid_count = 0
    attn_count = 0

    # assert os.path.isfile(opt.init_img)
    # init_image = load_img(opt.init_img).to(device)
    # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    # init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    img_files = glob.glob(opt.init_img + '/*.png')
    img_files.sort()

    print(img_files)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    opt.H = opt.W = 512
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        for file in img_files:
                            init_image = load_img(file).to(device)
                            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            # z_enc, _, = sampler.ddim_encode(S=opt.ddim_steps,
                            #                                  conditioning=c,
                            #                                  batch_size=opt.n_samples,
                            #                                  shape=shape,
                            #                                  verbose=False,
                            #                                  unconditional_guidance_scale=opt.scale,
                            #                                  unconditional_conditioning=uc,
                            #                                  eta=opt.ddim_eta,
                            #                                  x_0=init_latent,
                            #                                  t_enc=t_enc)
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,)

                            # print(inter['d_ts_to_name_to_M'].keys())
                            # pdb.set_trace()
                            # len_prompt = len(prompts[0].split(' '))
                            # for idx in range(len_prompt):
                            #     save_cross_attention_t(inter, ts=281, token_idx=idx+1, save_path=os.path.join(attn_path, f"{attn_count:05}_{idx:02}.jpg"))
                            # attn_count += 1

                            x_samples = model.decode_first_stage(samples)
                            # x_samples = model.decode_first_stage(z_enc)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(outpath, f"{base_count:05}.png"))
                                    base_count += 1
                            all_samples.append(x_samples)

                    # if not opt.skip_grid:
                    #     # additionally, save as grid
                    #     grid = torch.stack(all_samples, 0)
                    #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    #     grid = make_grid(grid, nrow=n_rows)

                    #     # to image
                    #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    #     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    #     grid_count += 1

                    toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
