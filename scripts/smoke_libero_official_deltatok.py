#!/usr/bin/env python
"""Smoke-train or evaluate the official DeltaTok model on LIBERO frame pairs.

This intentionally imports the official ``models.deltatok.DeltaTok`` and
``models.dinov3.DINOv3`` implementations from this repository. The surrounding
training loop is minimal so we can run a quick LIBERO sanity check without
requiring Kinetics-700 or the full Lightning/W&B stack.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.machinery
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import cv2
import decord
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

VIDEO_CACHE: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()


def install_import_shims(repo_root: Path) -> None:
    """Avoid local environment conflicts while importing official modules."""
    if not hasattr(torch._C, "_IncompatibleKeys"):
        torch._C._IncompatibleKeys = object

    # ``models.deltatok`` imports ``training.base.load_sd`` for optional
    # checkpoint loading. Keep the official strict shape/missing-param behavior
    # while avoiding Lightning/W&B/task-head deps from ``training.base``.
    training_pkg = types.ModuleType("training")
    training_pkg.__path__ = [str(repo_root / "training")]
    training_pkg.__spec__ = importlib.machinery.ModuleSpec("training", loader=None, is_package=True)
    sys.modules["training"] = training_pkg
    training_base = types.ModuleType("training.base")

    def load_sd(module: torch.nn.Module, sd: dict) -> torch._C._IncompatibleKeys:
        sd = sd.get("state_dict", sd.get("model", sd))
        module = getattr(module, "_orig_mod", module)
        module_sd = module.state_dict()

        used = {}
        unmapped = []
        for ckpt_key, tensor in sd.items():
            if ckpt_key not in module_sd:
                continue
            if tensor.shape != module_sd[ckpt_key].shape:
                unmapped.append(ckpt_key)
                continue
            used[ckpt_key] = tensor

        missing = [
            name
            for name, param in module.named_parameters(remove_duplicate=False)
            if param.requires_grad and name not in used
        ]
        if missing or unmapped:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unmapped={unmapped}")
        return module.load_state_dict(used, strict=False)

    training_base.load_sd = load_sd
    training_base.__spec__ = importlib.machinery.ModuleSpec("training.base", loader=None)
    sys.modules["training.base"] = training_base

    # Prefer DeltaTok's local ``datasets`` folder over HuggingFace datasets.
    datasets = types.ModuleType("datasets")
    datasets.__path__ = [str(repo_root / "datasets")]
    datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None, is_package=True)
    sys.modules["datasets"] = datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--extra-site-packages",
        default=str(Path(__file__).resolve().parents[1] / ".venv-smoke" / "lib" / "python3.11" / "site-packages"),
        help="Optional overlay site-packages path used for Transformers 5 without changing the base CUDA env.",
    )
    parser.add_argument("--dataset-root", default="/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot")
    parser.add_argument("--view", default="observation.images.image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backbone-path",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "dinov3-vitb16-pretrain-lvd1689m"),
        help="Local HuggingFace-format DINOv3 path. Official DeltaTok-Kinetics is ViT-B / 768-D.",
    )
    parser.add_argument(
        "--deltatok-ckpt",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "deltatok-kinetics" / "pytorch_model.bin"),
        help="Official DeltaTok tokenizer checkpoint. Required for --mode eval.",
    )
    parser.add_argument(
        "--init-from-deltatok-ckpt",
        action="store_true",
        help="In train mode, initialize trainable DeltaTok weights from --deltatok-ckpt before finetuning.",
    )
    parser.add_argument(
        "--rgb-head-path",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "rgb-head-imagenet" / "pytorch_model.bin"),
        help="Optional official RGBHead checkpoint for DINO-feature-to-RGB visualizations.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=("rgb_head", "raev2", "none"),
        default="rgb_head",
        help="Feature-to-RGB visualization backend.",
    )
    parser.add_argument(
        "--raev2-root",
        default=str(Path(__file__).resolve().parents[2] / "RAEv2"),
        help="Local RAEv2 repository root.",
    )
    parser.add_argument(
        "--raev2-decoder-config",
        default=str(Path(__file__).resolve().parents[2] / "RAEv2" / "configs" / "decoder" / "ViTXL"),
        help="RAEv2 decoder config directory.",
    )
    parser.add_argument(
        "--raev2-decoder-ckpt",
        default=str(
            Path(__file__).resolve().parents[2]
            / "RAEv2"
            / "pretrained_models"
            / "stage1"
            / "imagenet"
            / "dinov3b-k1"
            / "decoder.pt"
        ),
        help="RAEv2 DINOv3-B/K=1 decoder checkpoint.",
    )
    parser.add_argument(
        "--raev2-stats-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "RAEv2"
            / "pretrained_models"
            / "stage1"
            / "imagenet"
            / "dinov3b-k1"
            / "stats.pt"
        ),
        help="RAEv2 DINOv3-B/K=1 latent mean/variance stats.",
    )
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument(
        "--video-cache-size",
        type=int,
        default=512,
        help="Number of decoded/resized videos to keep in RAM. Set 0 to disable.",
    )
    parser.add_argument(
        "--preload-video-cache",
        action="store_true",
        help="Decode and resize videos into RAM before training/eval. Useful for AV1 mp4 datasets.",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=1e-2)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--vis-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_hf_backbone(path: Path) -> None:
    missing = [name for name in ("config.json", "preprocessor_config.json") if not (path / name).exists()]
    has_weights = any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    if not has_weights:
        missing.append("model.safetensors or pytorch_model.bin")
    if missing:
        raise FileNotFoundError(
            f"Backbone path is incomplete: {path}. Missing: {', '.join(missing)}. "
            "Official DeltaTok-Kinetics needs DINOv3 ViT-B, not the cached ViT-S checkpoint."
        )


def collect_videos(dataset_root: Path, view: str) -> list[Path]:
    paths = sorted((dataset_root / "videos").glob(f"*/{view}/episode_*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No videos found under {dataset_root}/videos/*/{view}/episode_*.mp4")
    return paths


def _get_cached_video(path: Path, frame_size: int, cache_size: int) -> np.ndarray | None:
    if cache_size <= 0:
        return None
    key = (path, frame_size)
    frames = VIDEO_CACHE.get(key)
    if frames is not None:
        VIDEO_CACHE.move_to_end(key)
    return frames


def _put_cached_video(path: Path, frame_size: int, cache_size: int, frames: np.ndarray) -> None:
    if cache_size <= 0:
        return
    key = (path, frame_size)
    VIDEO_CACHE[key] = frames
    VIDEO_CACHE.move_to_end(key)
    while len(VIDEO_CACHE) > cache_size:
        VIDEO_CACHE.popitem(last=False)


def _read_full_video(path: Path, frame_size: int) -> np.ndarray:
    frames = iio.imread(path)
    return np.stack(
        [cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA) for frame in frames]
    )


def read_pair(path: Path, stride: int, frame_size: int, video_cache_size: int) -> torch.Tensor:
    cached = _get_cached_video(path, frame_size, video_cache_size)
    if cached is not None:
        if len(cached) <= stride:
            raise ValueError(f"Video too short for stride={stride}: {path}")
        start = random.randint(0, len(cached) - stride - 1)
        frames = cached[[start, start + stride]]
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

    try:
        vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        if len(vr) <= stride:
            raise ValueError(f"Video too short for stride={stride}: {path}")
        start = random.randint(0, len(vr) - stride - 1)
        frames = vr.get_batch([start, start + stride]).asnumpy()
        resized = np.stack(
            [cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA) for frame in frames]
        )
    except Exception:
        # Some local FFmpeg/decord builds cannot decode the LIBERO AV1 mp4s.
        # imageio's ffmpeg path is slower but works for smoke/eval visualization.
        all_frames = _read_full_video(path, frame_size)
        _put_cached_video(path, frame_size, video_cache_size, all_frames)
        if len(all_frames) <= stride:
            raise ValueError(f"Video too short for stride={stride}: {path}")
        start = random.randint(0, len(all_frames) - stride - 1)
        resized = all_frames[[start, start + stride]]
    return torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()


def sample_batch(
    paths: list[Path],
    batch_size: int,
    stride: int,
    frame_size: int,
    video_cache_size: int,
    device: torch.device,
) -> torch.Tensor:
    samples = []
    max_attempts = max(100, batch_size * 50)
    attempts = 0
    while len(samples) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            samples.append(read_pair(random.choice(paths), stride, frame_size, video_cache_size))
        except Exception:
            continue
    if len(samples) < batch_size:
        raise RuntimeError(
            f"Only decoded {len(samples)}/{batch_size} samples after {attempts} attempts. "
            "Check video codec support and dataset paths."
        )
    return torch.stack(samples, dim=0).to(device, non_blocking=True)


def preload_video_cache(paths: list[Path], frame_size: int, cache_size: int) -> None:
    if cache_size <= 0:
        print("video_cache=disabled; skipping preload", flush=True)
        return
    max_items = min(len(paths), cache_size)
    start_time = time.time()
    loaded = 0
    for path in paths[:max_items]:
        if _get_cached_video(path, frame_size, cache_size) is not None:
            continue
        try:
            frames = _read_full_video(path, frame_size)
            _put_cached_video(path, frame_size, cache_size, frames)
            loaded += 1
            if loaded % 50 == 0 or loaded == max_items:
                elapsed = time.time() - start_time
                print(
                    f"video_cache_preload={loaded}/{max_items} cached={len(VIDEO_CACHE)} elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"video_cache_preload_skip={path} error={exc}", flush=True)


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred.float() - target.detach().float()
    return (diff + F.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, device=diff.device))).mean()


def pca_images(feature_sets: list[torch.Tensor], size: int = 256) -> list[Image.Image]:
    tokens = [features.detach().float().cpu() for features in feature_sets]
    combined = torch.cat(tokens, dim=0)
    centered = combined - combined.mean(dim=0, keepdim=True)
    try:
        _, _, components = torch.pca_lowrank(centered, q=3, center=False)
        projected = centered @ components[:, :3]
    except RuntimeError:
        projected = centered[:, :3]
    lo = projected.quantile(0.01, dim=0, keepdim=True)
    hi = projected.quantile(0.99, dim=0, keepdim=True)
    projected = ((projected - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

    images = []
    offset = 0
    for features in tokens:
        count = features.shape[0]
        grid = int(count**0.5)
        if grid * grid != count:
            raise ValueError(f"Expected square token grid, got {count} tokens.")
        image = projected[offset : offset + count].reshape(grid, grid, 3).numpy()
        images.append(Image.fromarray((image * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR))
        offset += count
    return images


def rgb_image(frame: torch.Tensor, size: int = 256) -> Image.Image:
    array = frame.detach().cpu().permute(1, 2, 0).numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).resize((size, size), Image.Resampling.BILINEAR)


def decoded_rgb_image(image: torch.Tensor, size: int = 256) -> Image.Image:
    array = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)


def decode_rgb_feature(rgb_head: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    count = features.shape[0]
    grid = int(count**0.5)
    if grid * grid != count:
        raise ValueError(f"Expected square token grid, got {count} tokens.")
    feat_map = features.detach().float().transpose(0, 1).reshape(1, features.shape[-1], grid, grid)
    return rgb_head(feat_map)[0]


def _load_raev2_decoder_config(config_path: str, hidden_size: int, patch_size: int, num_patches: int):
    from transformers import AutoConfig, ViTMAEConfig

    path = Path(config_path)
    try:
        config = AutoConfig.from_pretrained(config_path)
    except Exception:
        config_json = path / "config.json"
        if not config_json.exists():
            raise
        data = json.loads(config_json.read_text())
        if data.get("patch_size") == "SHOULD BE RELOADED":
            data["patch_size"] = patch_size
        config = ViTMAEConfig(**data)
    config.hidden_size = hidden_size
    config.patch_size = patch_size
    config.image_size = int(patch_size * math.sqrt(num_patches))
    return config


class RAEv2FeatureDecoder(nn.Module):
    """Decode DeltaTok/HF DINOv3-B patch features with RAEv2 dinov3b-k1 decoder."""

    def __init__(
        self,
        raev2_root: Path,
        decoder_config_path: Path,
        decoder_ckpt_path: Path,
        stats_path: Path,
        dino_norm_weight: torch.Tensor,
        dino_norm_bias: torch.Tensor,
        hidden_size: int,
        patch_size: int = 16,
        eps: float = 1e-5,
    ):
        super().__init__()
        src_dir = raev2_root / "src"
        if not src_dir.exists():
            raise FileNotFoundError(f"RAEv2 src directory not found: {src_dir}")
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from stage1.decoders import GeneralDecoder

        if not decoder_ckpt_path.exists():
            raise FileNotFoundError(f"RAEv2 decoder checkpoint not found: {decoder_ckpt_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"RAEv2 stats not found: {stats_path}")

        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        mean = stats["mean"].float()
        var = stats["var"].float()
        if mean.ndim != 3 or var.shape != mean.shape:
            raise ValueError(f"Expected RAEv2 stats [C,H,W], got mean={tuple(mean.shape)} var={tuple(var.shape)}")

        num_patches = mean.shape[-1] * mean.shape[-2]
        config = _load_raev2_decoder_config(str(decoder_config_path), hidden_size, patch_size, num_patches)
        decoder = GeneralDecoder(config, num_patches=num_patches)
        state_dict = torch.load(decoder_ckpt_path, map_location="cpu", weights_only=False)
        missing = decoder.load_state_dict(state_dict, strict=False).missing_keys
        if missing:
            print(f"raev2_decoder_missing_keys={missing}", flush=True)

        self.decoder = decoder
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_var", var)
        self.register_buffer("dino_norm_weight", dino_norm_weight.detach().float().reshape(1, 1, -1).cpu())
        self.register_buffer("dino_norm_bias", dino_norm_bias.detach().float().reshape(1, 1, -1).cpu())
        self.eps = eps

    def feature_to_latent(self, features: torch.Tensor) -> torch.Tensor:
        # DeltaTok uses HuggingFace DINOv3 final LayerNorm with affine. RAEv2
        # dinov3b-k1 decoder expects the no-affine final-normalized feature.
        weight = self.dino_norm_weight.to(features.device, dtype=features.dtype)
        bias = self.dino_norm_bias.to(features.device, dtype=features.dtype)
        default_features = (features.float() - bias.float()) / weight.float()

        batch, tokens, channels = default_features.shape
        grid = int(tokens**0.5)
        if grid * grid != tokens:
            raise ValueError(f"Expected square token grid, got {tokens} tokens.")
        z = default_features.transpose(1, 2).reshape(batch, channels, grid, grid)
        mean = self.latent_mean.to(z.device, dtype=z.dtype)
        var = self.latent_var.to(z.device, dtype=z.dtype)
        return (z - mean) / torch.sqrt(var + self.eps)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.to(z.device, dtype=z.dtype)
        var = self.latent_var.to(z.device, dtype=z.dtype)
        z = z * torch.sqrt(var + self.eps) + mean
        batch, channels, height, width = z.shape
        tokens = z.reshape(batch, channels, height * width).transpose(1, 2)
        output = self.decoder(tokens, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decode_latent(self.feature_to_latent(features))


def decode_raev2_feature(decoder: RAEv2FeatureDecoder, features: torch.Tensor) -> torch.Tensor:
    return decoder(features.unsqueeze(0))[0]


def save_visualization(
    path: Path,
    frames: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    rgb_head: torch.nn.Module | None = None,
    raev2_decoder: RAEv2FeatureDecoder | None = None,
) -> None:
    panels = [rgb_image(frames[0]), rgb_image(frames[1])]
    labels = ["rgb_t", "rgb_t+k"]
    if raev2_decoder is not None:
        with torch.no_grad():
            current_rgb = decode_raev2_feature(raev2_decoder, current)
            oracle_rgb = decode_raev2_feature(raev2_decoder, target)
            pred_rgb = decode_raev2_feature(raev2_decoder, pred)
        panels.extend([decoded_rgb_image(current_rgb), decoded_rgb_image(oracle_rgb), decoded_rgb_image(pred_rgb)])
        labels.extend(["raev2_gt_t", "raev2_gt_t+k", "raev2_pred_t+k"])
    elif rgb_head is not None:
        with torch.no_grad():
            current_rgb = decode_rgb_feature(rgb_head, current)
            oracle_rgb = decode_rgb_feature(rgb_head, target)
            pred_rgb = decode_rgb_feature(rgb_head, pred)
        panels.extend([decoded_rgb_image(current_rgb), decoded_rgb_image(oracle_rgb), decoded_rgb_image(pred_rgb)])
        labels.extend(["decode_gt_t", "decode_gt_t+k", "decode_pred_t+k"])
    panels.extend(pca_images([current, target, pred]))
    labels.extend(["feat_t", "feat_t+k", "pred_feat"])
    width, height = 256 * len(panels), 286
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x = idx * 256
        canvas.paste(panel, (x, 0))
        draw.text((x + 8, 264), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def maybe_load_rgb_head(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module | None:
    if not path:
        return None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"rgb_head=missing path={ckpt_path}; skipping RGB decode visualization", flush=True)
        return None
    from models.task_heads import RGBHead
    from training.base import load_sd

    head = RGBHead(
        model.backbone.hidden_size,
        norm_weight=model.backbone.backbone.norm.weight.detach().cpu(),
        norm_bias=model.backbone.backbone.norm.bias.detach().cpu(),
        img_mean=model.backbone.processor.image_mean,
        img_std=model.backbone.processor.image_std,
    )
    incompatible = load_sd(head, torch.load(ckpt_path, map_location="cpu"))
    head = head.to(device).requires_grad_(False).eval()
    print(f"loaded_rgb_head={ckpt_path} incompatible={incompatible}", flush=True)
    return head


def maybe_load_raev2_decoder(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> RAEv2FeatureDecoder | None:
    if args.decode_backend != "raev2":
        return None
    decoder = RAEv2FeatureDecoder(
        raev2_root=Path(args.raev2_root),
        decoder_config_path=Path(args.raev2_decoder_config),
        decoder_ckpt_path=Path(args.raev2_decoder_ckpt),
        stats_path=Path(args.raev2_stats_path),
        dino_norm_weight=model.backbone.backbone.norm.weight.detach().cpu(),
        dino_norm_bias=model.backbone.backbone.norm.bias.detach().cpu(),
        hidden_size=model.backbone.hidden_size,
        patch_size=model.backbone.patch_size,
    )
    decoder = decoder.to(device).requires_grad_(False).eval()
    print(
        f"loaded_raev2_decoder={args.raev2_decoder_ckpt} stats={args.raev2_stats_path} "
        f"config={args.raev2_decoder_config}",
        flush=True,
    )
    return decoder


def delta_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Save trainable DeltaTok weights without the frozen DINOv3 backbone."""
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith("backbone.")}


def compute_pair_metrics(
    model: torch.nn.Module,
    frames: torch.Tensor,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        rope = model._rope(frames)
        features = model.backbone(frames)
        z = model.encode(features[:, 0], features[:, 1], rope)
        pred = model.decode(z, features[:, 0], rope)

    mse = F.mse_loss(pred.float(), features[:, 1].float())
    copy_mse = F.mse_loss(features[:, 0].float(), features[:, 1].float())
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        zero_pred = model.decode(torch.zeros_like(z), features[:, 0], rope)
        shuffled = z[torch.randperm(z.shape[0], device=z.device)]
        shuffle_pred = model.decode(shuffled, features[:, 0], rope)
    zero_mse = F.mse_loss(zero_pred.float(), features[:, 1].float())
    shuffle_mse = F.mse_loss(shuffle_pred.float(), features[:, 1].float())
    target_delta = features[:, 1].float() - features[:, 0].float()
    pred_delta = pred.float() - features[:, 0].float()
    delta_ratio = pred_delta.norm() / target_delta.norm().clamp_min(1e-6)
    metrics = {
        "mse": float(mse.detach().cpu()),
        "copy_mse": float(copy_mse.detach().cpu()),
        "zero_mse": float(zero_mse.detach().cpu()),
        "shuffle_mse": float(shuffle_mse.detach().cpu()),
        "delta_ratio": float(delta_ratio.detach().cpu()),
    }
    return z, metrics, features, pred, rope[0]


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(rows[0])
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    extra_site = Path(args.extra_site_packages)
    if extra_site.exists():
        sys.path.insert(1, str(extra_site))
    install_import_shims(repo_root)

    import models.deltatok as deltatok_module
    import models.predictor as predictor_module
    from models.dinov3 import DINOv3

    # The official source defaults this template to DINOv3 ViT-B. For an
    # offline smoke with a local backbone path, point the config lookup at the
    # same local DINOv3 config used by the frozen backbone.
    predictor_module.DINOV3_TEMPLATE = args.backbone_path
    deltatok_module.DINOV3_TEMPLATE = args.backbone_path
    DeltaTok = deltatok_module.DeltaTok

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = collect_videos(Path(args.dataset_root), args.view)
    print(f"videos={len(paths)} output_dir={output_dir}", flush=True)
    if args.preload_video_cache:
        preload_video_cache(paths, args.frame_size, args.video_cache_size)

    require_hf_backbone(Path(args.backbone_path))
    backbone = DINOv3(args.backbone_path)
    model = DeltaTok(backbone=backbone, num_hidden_layers=args.num_hidden_layers).to(device)
    if args.mode == "eval":
        if not args.deltatok_ckpt:
            raise ValueError("--mode eval requires --deltatok-ckpt")
        ckpt_path = Path(args.deltatok_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DeltaTok checkpoint not found: {ckpt_path}")
        from training.base import load_sd

        incompatible = load_sd(model, torch.load(ckpt_path, map_location="cpu"))
        print(f"loaded_ckpt={ckpt_path} incompatible={incompatible}", flush=True)
        model.eval()
        rgb_head = maybe_load_rgb_head(model, args.rgb_head_path, device) if args.decode_backend == "rgb_head" else None
        raev2_decoder = maybe_load_raev2_decoder(model, args, device)
        use_amp = args.precision != "float32" and device.type == "cuda"
        all_metrics = []
        start_time = time.time()
        with torch.no_grad():
            for batch_idx in range(args.eval_batches):
                frames = sample_batch(
                    paths, args.batch_size, args.stride, args.frame_size, args.video_cache_size, device
                )
                _, metrics, features, pred, _ = compute_pair_metrics(model, frames, use_amp)
                all_metrics.append(metrics)
                print(
                    "eval_batch={batch}/{total} mse={mse:.6f} copy_mse={copy_mse:.6f} "
                    "zero_mse={zero_mse:.6f} shuffle_mse={shuffle_mse:.6f} delta_ratio={delta_ratio:.3f}".format(
                        batch=batch_idx + 1,
                        total=args.eval_batches,
                        **metrics,
                    ),
                    flush=True,
                )
                if args.vis_interval > 0 and (batch_idx == 0 or (batch_idx + 1) % args.vis_interval == 0):
                    save_visualization(
                        output_dir / "vis" / f"eval_batch_{batch_idx + 1:04d}.png",
                        frames[0],
                        features[0, 0],
                        features[0, 1],
                        pred[0],
                        rgb_head=rgb_head,
                        raev2_decoder=raev2_decoder,
                    )
        summary = average_metrics(all_metrics)
        summary["eval_batches"] = args.eval_batches
        summary["samples"] = args.eval_batches * args.batch_size
        summary["minutes"] = (time.time() - start_time) / 60.0
        (output_dir / "metrics_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("eval_summary=" + json.dumps(summary, sort_keys=True), flush=True)
        return

    model.train()
    if args.init_from_deltatok_ckpt:
        ckpt_path = Path(args.deltatok_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DeltaTok checkpoint not found: {ckpt_path}")
        from training.base import load_sd

        incompatible = load_sd(model, torch.load(ckpt_path, map_location="cpu"))
        print(f"initialized_from_ckpt={ckpt_path} incompatible={incompatible}", flush=True)
        model.train()

    rgb_head = maybe_load_rgb_head(model, args.rgb_head_path, device) if args.decode_backend == "rgb_head" else None
    raev2_decoder = maybe_load_raev2_decoder(model, args, device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"backbone_hidden={model.backbone.hidden_size} patch={model.backbone.patch_size} "
        f"heads={model.backbone.num_heads} layers={args.num_hidden_layers} trainable_params={sum(p.numel() for p in trainable)}",
        flush=True,
    )

    use_amp = args.precision != "float32" and device.type == "cuda"
    start_time = time.time()
    last_log_time = start_time
    last_metrics = {}
    for step in range(1, args.max_steps + 1):
        frames = sample_batch(paths, args.batch_size, args.stride, args.frame_size, args.video_cache_size, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            rope = model._rope(frames)
            with torch.no_grad():
                features = model.backbone(frames)
            z = model.encode(features[:, 0], features[:, 1], rope)
            pred = model.decode(z, features[:, 0], rope)
            loss = log_cosh_loss(pred, features[:, 1])

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.max_steps:
            now = time.time()
            steps_per_sec = args.log_interval / max(now - last_log_time, 1e-6)
            last_log_time = now
            with torch.no_grad():
                mse = F.mse_loss(pred.float(), features[:, 1].float())
                copy_mse = F.mse_loss(features[:, 0].float(), features[:, 1].float())
                zero_pred = model.decode(torch.zeros_like(z), features[:, 0], rope)
                zero_mse = F.mse_loss(zero_pred.float(), features[:, 1].float())
                shuffled = z[torch.randperm(z.shape[0], device=z.device)]
                shuffle_pred = model.decode(shuffled, features[:, 0], rope)
                shuffle_mse = F.mse_loss(shuffle_pred.float(), features[:, 1].float())
                target_delta = features[:, 1].float() - features[:, 0].float()
                pred_delta = pred.float() - features[:, 0].float()
                delta_ratio = pred_delta.norm() / target_delta.norm().clamp_min(1e-6)
            last_metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "copy_mse": float(copy_mse.detach().cpu()),
                "zero_mse": float(zero_mse.detach().cpu()),
                "shuffle_mse": float(shuffle_mse.detach().cpu()),
                "delta_ratio": float(delta_ratio.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "steps_per_sec": steps_per_sec,
            }
            print(
                "step={step} loss={loss:.6f} mse={mse:.6f} copy_mse={copy_mse:.6f} "
                "zero_mse={zero_mse:.6f} shuffle_mse={shuffle_mse:.6f} "
                "delta_ratio={delta_ratio:.3f} grad={grad_norm:.4f} steps/s={steps_per_sec:.3f}".format(
                    **last_metrics
                ),
                flush=True,
            )

        if args.vis_interval > 0 and step % args.vis_interval == 0:
            save_visualization(
                output_dir / "vis" / f"step_{step:06d}.png",
                frames[0],
                features[0, 0],
                features[0, 1],
                pred[0],
                rgb_head=rgb_head,
                raev2_decoder=raev2_decoder,
            )

        if args.save_interval > 0 and step % args.save_interval == 0:
            torch.save(
                {
                    "step": step,
                    "state_dict": delta_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "last_metrics": last_metrics,
                },
                output_dir / "last.pt",
            )

    (output_dir / "metrics_last.json").write_text(json.dumps(last_metrics, indent=2, sort_keys=True))
    print(f"finished minutes={(time.time() - start_time) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
