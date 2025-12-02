"""
Train the diffusion UNet on Flickr30k captions + images,
using pretrained CLIP and VAE weights from your existing repo.

Save this as: train_flickr30k_unet.py
Run in Colab inside your repo folder.
"""

import os
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import CLIPTokenizer
from tqdm import tqdm

import model_loader  # from your repo

# -----------------------------
# Config
# -----------------------------

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

MAX_TEXT_LENGTH = 77        # CLIP tokenizer sequence length
NUM_TRAIN_TIMESTEPS = 1000  # DDPM training steps

@dataclass
class TrainConfig:
    vocab_path: str = "data/tokenizer_vocab.json"    # ### TODO: adjust if paths differ
    merges_path: str = "data/tokenizer_merges.txt"   # ### TODO: adjust if paths differ
    model_ckpt_path: str = "data/v1-5-pruned-emaonly.ckpt"  # ### TODO: your SD checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-5
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS
    num_workers: int = 2
    output_dir: str = "checkpoints"
    max_train_steps_per_epoch: int | None = None  # set to int to limit per-epoch steps


# -----------------------------
# Utility: Rescale + time embedding
# -----------------------------

def rescale(x: torch.Tensor, old_range, new_range, clamp: bool = False) -> torch.Tensor:
    """
    Same rescale function as in your pipeline:
    maps from old_range to new_range linearly.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = x.clone()
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timesteps: torch.LongTensor) -> torch.Tensor:
    """
    Vectorised version of your get_time_embedding for a batch of timesteps.
    Input: timesteps (B,) long
    Output: (B, 320) time embeddings
    """
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (B, 1)
    x = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B,160)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (B,320)


# -----------------------------
# Flickr30k Dataset wrapper
# -----------------------------

class Flickr30kDiffusionDataset(Dataset):
    """
    Wraps HF dataset `nlphuji/flickr30k` into a PyTorch Dataset that returns:
      - pixel_values: (3, H, W) in [-1, 1]
      - input_ids: (MAX_TEXT_LENGTH,)
    """

    def __init__(self, hf_split, tokenizer: CLIPTokenizer, image_size: int = 512, max_length: int = 77):
        self.data = hf_split
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # ----- Image preprocessing -----
        # 1) Resize to (image_size, image_size)
        image = example["image"].resize((self.image_size, self.image_size))
        # 2) PIL -> NumPy -> Torch Tensor (H, W, C), values in [0,255]
        import numpy as np
        image = np.array(image).astype("float32")
        image = torch.from_numpy(image)  # (H,W,C)
        # 3) Map [0,255] -> [-1,1] exactly like your pipeline
        image = rescale(image, (0, 255), (-1, 1))
        # 4) (H,W,C) -> (C,H,W)
        image = image.permute(2, 0, 1)

        # ----- Caption preprocessing -----
        # Flickr30k has multiple captions per image in "raw". Take a random one.
        captions = example["raw"]
        caption = random.choice(captions)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)  # (77,)

        return {
            "pixel_values": image,   # (3,H,W), float32 in [-1,1]
            "input_ids": input_ids,  # (77,), long
        }


def create_dataloader(cfg: TrainConfig, tokenizer: CLIPTokenizer) -> DataLoader:
    # Load HF dataset
    ds = load_dataset("nlphuji/flickr30k")

    # We'll just use train split here; you can add val if you want.
    hf_train = ds["train"]

    train_dataset = Flickr30kDiffusionDataset(
        hf_split=hf_train,
        tokenizer=tokenizer,
        image_size=WIDTH,
        max_length=MAX_TEXT_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


# -----------------------------
# Noise schedule for DDPM training
# -----------------------------

class NoiseScheduler:
    """
    A simple DDPM-style noise scheduler, independent of your DDPMSampler.
    This is used only for training (q(x_t | x_0)).
    """

    def __init__(self, num_train_timesteps: int, device: str):
        self.num_train_timesteps = num_train_timesteps
        self.device = device

        # Standard-ish beta schedule (you can adjust to match your sampler more precisely)
        beta_start = 0.00085
        beta_end = 0.012
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Draw x_t ~ q(x_t | x_0, t)
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * noise
        """
        # t: (B,)
        # gather per-sample coefficients
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise


# -----------------------------
# Model loading
# -----------------------------

def load_models_and_tokenizer(cfg: TrainConfig):
    """
    Load tokenizer and models (clip, encoder, diffusion, decoder) using your existing code.
    """

    # CLIP tokenizer (same as your generate.py)
    tokenizer = CLIPTokenizer(cfg.vocab_path, merges_file=cfg.merges_path)

    # Pretrained SD-like weights
    models = model_loader.preload_models_from_standard_weights(cfg.model_ckpt_path, cfg.device)

    clip = models["clip"]
    encoder = models["encoder"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]

    # Move to device
    clip.to(cfg.device)
    encoder.to(cfg.device)
    diffusion.to(cfg.device)
    decoder.to(cfg.device)

    # Freeze CLIP and VAE
    for p in clip.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    clip.eval()
    encoder.eval()
    decoder.eval()
    diffusion.train()  # we're training only the UNet

    return models, tokenizer


# -----------------------------
# Training loop
# -----------------------------

def train_unet_on_flickr30k(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Using device: {cfg.device}")
    print("Loading models and tokenizer...")
    models, tokenizer = load_models_and_tokenizer(cfg)

    clip = models["clip"]
    encoder = models["encoder"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]  # not used in training, but loaded

    print("Creating dataloader...")
    train_loader = create_dataloader(cfg, tokenizer)

    print("Building noise scheduler...")
    noise_scheduler = NoiseScheduler(cfg.num_train_timesteps, cfg.device)

    # Optimizer only on diffusion (UNet) params
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg.learning_rate)

    global_step = 0

    for epoch in range(cfg.num_epochs):
        diffusion.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=True)
        for step, batch in enumerate(pbar):
            if cfg.max_train_steps_per_epoch is not None and step >= cfg.max_train_steps_per_epoch:
                break

            pixel_values = batch["pixel_values"].to(cfg.device)  # (B,3,H,W), [-1,1]
            input_ids = batch["input_ids"].to(cfg.device)        # (B,77)

            batch_size = pixel_values.shape[0]

            # -------------------------
            # 1. Encode text with CLIP (frozen)
            # -------------------------
            with torch.no_grad():
                # (B,77) -> (B,77,Dim)
                context = clip(input_ids)

            # -------------------------
            # 2. Encode image into latents with VAE encoder (frozen)
            # -------------------------
            with torch.no_grad():
                # (B,3,H,W) -> (B,3,H,W) already, but encoder expects (B,C,H,W) in [-1,1]
                encoder_noise = torch.randn(
                    (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH),
                    device=cfg.device,
                )
                # (B,4,LatH,LatW)
                latents = encoder(pixel_values, encoder_noise)

            # -------------------------
            # 3. Sample random timesteps + noise, create noisy latents
            # -------------------------
            timesteps = torch.randint(
                low=0,
                high=cfg.num_train_timesteps,
                size=(batch_size,),
                device=cfg.device,
                dtype=torch.long,
            )

            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.q_sample(latents, timesteps, noise)

            # -------------------------
            # 4. Time embeddings
            # -------------------------
            time_embedding = get_time_embedding(timesteps).to(cfg.device)  # (B,320)

            # -------------------------
            # 5. Predict noise with diffusion UNet
            # -------------------------
            # diffusion(x_t, context, time_embedding) -> predicted noise
            model_pred = diffusion(noisy_latents, context, time_embedding)

            # -------------------------
            # 6. Loss = MSE(predicted_noise, true_noise)
            # -------------------------
            loss = F.mse_loss(model_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # -------------------------
        # Save UNet checkpoint each epoch
        # -------------------------
        ckpt_path = os.path.join(cfg.output_dir, f"diffusion_flickr30k_epoch{epoch+1}.pt")
        torch.save(diffusion.state_dict(), ckpt_path)
        print(f"Saved UNet checkpoint to {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train_unet_on_flickr30k(cfg)
