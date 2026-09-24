# Adapted from lumalabs/tvm under the license in https://github.com/lumalabs/tvm/blob/main/LICENSE
# Source: https://github.com/lumalabs/tvm/blob/main/training/encoders.py
# Adjusted for this project needs.

"""Converting between pixel and latent representations of image data."""

from collections.abc import Mapping
import json
import os
import warnings
import numpy as np
import torch
from utils import artifacts
from utils import persistence
from utils import tensors

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.serializable_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass
    
    def decode_float(self, x): # final latents => raw pixels (float, no quantization)
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.serializable_class
class IdentityEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x 
    def encode(self, x):
        return x
    def decode(self, x): # final latents => raw pixels
        return x
    
    def decode_float(self, x): # final latents => raw pixels (float, no quantization)
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255.0
        return x
#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into a configurable range.

@persistence.serializable_class
class StandardRGBEncoder(Encoder):
    def __init__(self, output_range="-1_1", **unused_kwargs):
        super().__init__()
        self.output_range = str(output_range).lower().replace("-", "_")
        if self.output_range in {"0,1", "[0,1]", "zero_one"}:
            self.output_range = "0_1"
        elif self.output_range in {"_1_1", "minus_one_one", "neg_one_one", "[-1,1]"}:
            self.output_range = "-1_1"
        if self.output_range not in {"-1_1", "0_1"}:
            raise ValueError(f"Unsupported StandardRGBEncoder output_range={output_range!r}")

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        if self.output_range == "0_1":
            return x.to(torch.float32) / 255.0
        return x.to(torch.float32) / 127.5 - 1

    def encode(self, x): 
        return self.encode_latents(self.encode_pixels(x))
    
    
    def decode(self, x): # final latents => raw pixels
        if self.output_range == "0_1":
            return (x.to(torch.float32) * 255.0).round().clip(0, 255).to(torch.uint8)
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
    def decode_float(self, x): # final latents => raw pixels (float, no quantization)
        if self.output_range == "0_1":
            return x.to(torch.float32).clamp(0, 1)
        return (x.to(torch.float32) * 127.5 + 128).clamp(0, 255) / 255.0

#----------------------------------------------------------------------------
# Pre-trained VAE encoder from Stability AI.

@persistence.serializable_class
class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name    = 'stabilityai/sd-vae-ft-mse',  # Name of the VAE to use.
        autoencoder_cls = 'AutoencoderKL',          # Autoencoder class to use (e.g., 'AutoencoderTiny').
        raw_mean    = [1.56, -0.695, 0.483, 0.729],    # Assumed mean of the raw latents.
        raw_std     = [5.27, 5.91, 4.21, 4.31],     # Assumed standard deviation of the raw latents.
        scale   = None,
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
        use_fp16    = False,                # Data type to use for the latents.
        use_compile     = False,                          # Whether to compile the VAE with torch.compile().
    ):
        super().__init__()
        self.vae_name = vae_name
        self.autoencoder_cls = autoencoder_cls
        if scale is None:
            self.scale = np.float32(final_std) / np.float32(raw_std)
        else:
            self.scale = scale 
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.use_compile = use_compile

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_vae(
                self.vae_name,
                device=device,
                dtype=self.dtype,
                use_compile=self.use_compile,
                autoencoder_cls=self.autoencoder_cls,
            )
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        dtype = x.dtype
        d = self._vae.encode(x.to(self.dtype))['latent_dist']
        return torch.cat([d.mean, d.std], dim=1).to(dtype)

    def _run_vae_decoder(self, x):
        dtype = x.dtype
        return self._vae.decode(x.to(self.dtype))['sample'].to(dtype)
 
    def encode_pixels(self, x): # raw pixels => raw latents
        self.init(x.device) 
        x = x.to(torch.float32) / 127.5 - 1
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

    def encode_latents(self, x): # raw latents => final latents
        mean, std = x.to(torch.float32).chunk(2, dim=1)
        x = mean + torch.randn_like(mean) * std
        x = x * tensors.constant_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + tensors.constant_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def encode(self, x): 
        if x.shape[1] == 2 * 4:
            return self.encode_latents(x)
        elif x.shape[1] == 3:
            return self.encode_latents(self.encode_pixels(x))
        else:
            raise ValueError(f'Invalid number of channels: {x.shape[1]}')
     
    def decode_latents_to_pixels(self, x): 
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - tensors.constant_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / tensors.constant_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = (x * 0.5 + 0.5).clamp(0,1).mul(255).to(torch.uint8)
        return x
        
    def decode_latents_to_pixels_float(self, x):
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - tensors.constant_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / tensors.constant_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = (x * 0.5 + 0.5).clamp(0, 1)
        return x
        
    def decode(self, x): # final latents => raw pixels
        if x.shape[1] == 2 * 4:
            mean, std = x.to(torch.float32).chunk(2, dim=1)
            x = mean + torch.randn_like(mean) * std
            return self.decode_latents_to_pixels(x)
        elif x.shape[1] == 4:
            return self.decode_latents_to_pixels(x)
        elif x.shape[1] == 3:
            return x
        else:
            raise ValueError(f'Invalid number of channels: {x.shape[1]}')

    def decode_float(self, x): # final latents => raw pixels (float, no quantization)
        if x.shape[1] == 2 * 4:
            mean, std = x.to(torch.float32).chunk(2, dim=1)
            x = mean + torch.randn_like(mean) * std
            return self.decode_latents_to_pixels_float(x)
        elif x.shape[1] == 4:
            return self.decode_latents_to_pixels_float(x)
        elif x.shape[1] == 3:
            if x.dtype == torch.uint8:
                return x.to(torch.float32) / 255.0
            return x
        else:
            raise ValueError(f'Invalid number of channels: {x.shape[1]}')

#----------------------------------------------------------------------------
# Pre-trained VAE encoder using diffusers AutoencoderTiny.

@persistence.serializable_class
class AutoencoderTiny(StabilityVAEEncoder):
    def __init__(self,
        vae_name    = 'madebyollin/taesd3',  # Name of the VAE to use.
        raw_mean    = [1.56, -0.695, 0.483, 0.729],    # Assumed mean of the raw latents.
        raw_std     = [5.27, 5.91, 4.21, 4.31],     # Assumed standard deviation of the raw latents.
        scale   = None,
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
        use_fp16    = False,                # Data type to use for the latents.
        use_compile     = False,                          # Whether to compile the VAE with torch.compile().
        latent_norm = None,                          # Optional fixed TAESD3 latent normalization stats.
        vae_checkpoint = None,                        # Optional checkpoint to load TinyAE encoder and decoder weights from.
        vae_checkpoint_strict = True,                 # Require all requested VAE component weights to be present.
        encoder_checkpoint = None,                    # Optional checkpoint to load only TinyAE encoder weights from.
        encoder_checkpoint_strict = True,             # Require all encoder weights to be present.
        decoder_checkpoint = None,                    # Optional checkpoint to load only TinyAE decoder weights from.
        decoder_checkpoint_strict = True,              # Require all decoder weights to be present.
    ):
        super().__init__(
            vae_name=vae_name,
            autoencoder_cls='AutoencoderTiny',
            raw_mean=raw_mean,
            raw_std=raw_std,
            scale=scale,
            final_mean=final_mean,
            final_std=final_std,
            batch_size=batch_size,
            use_fp16=use_fp16,
            use_compile=use_compile,
        )

        if isinstance(vae_checkpoint, Mapping):
            vae_checkpoint_strict = vae_checkpoint.get('strict', vae_checkpoint_strict)
            vae_checkpoint = vae_checkpoint.get('path', None)
        if isinstance(encoder_checkpoint, Mapping):
            encoder_checkpoint_strict = encoder_checkpoint.get('strict', encoder_checkpoint_strict)
            encoder_checkpoint = encoder_checkpoint.get('path', None)
        if isinstance(decoder_checkpoint, Mapping):
            decoder_checkpoint_strict = decoder_checkpoint.get('strict', decoder_checkpoint_strict)
            decoder_checkpoint = decoder_checkpoint.get('path', None)
        self.vae_checkpoint = vae_checkpoint
        self.vae_checkpoint_strict = bool(vae_checkpoint_strict)
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_checkpoint_strict = bool(encoder_checkpoint_strict)
        self.decoder_checkpoint = decoder_checkpoint
        self.decoder_checkpoint_strict = bool(decoder_checkpoint_strict)
        self.scale = final_std * torch.ones(16)
        self.bias = torch.zeros(16)
        self.latent_channels = 16
        self.latent_norm_enabled = False
        self.latent_norm_stats_path = None
        self.latent_norm_eps = 1e-6
        self.latent_norm_mean = None
        self.latent_norm_std = None
        self._configure_latent_norm(latent_norm)

    def init(self, device): # force lazy init to happen now
        Encoder.init(self, device)
        if self._vae is None:
            for label, checkpoint_path in (
                ('VAE', self.vae_checkpoint),
                ('encoder', self.encoder_checkpoint),
                ('decoder', self.decoder_checkpoint),
            ):
                if checkpoint_path and not os.path.exists(os.fspath(checkpoint_path)):
                    raise FileNotFoundError(f'AutoencoderTiny {label} checkpoint not found: {checkpoint_path}')
            self._vae = load_stability_vae(
                self.vae_name,
                device=device,
                dtype=self.dtype,
                use_compile=False,
                autoencoder_cls=self.autoencoder_cls,
            )
            if self.vae_checkpoint:
                load_tinyae_checkpoint_into_vae(
                    self._vae,
                    self.vae_checkpoint,
                    components=('encoder', 'decoder'),
                    strict=self.vae_checkpoint_strict,
                    map_location=device,
                    dtype=self.dtype,
                )
            if self.encoder_checkpoint:
                load_tinyae_checkpoint_into_vae(
                    self._vae,
                    self.encoder_checkpoint,
                    components=('encoder',),
                    strict=self.encoder_checkpoint_strict,
                    map_location=device,
                    dtype=self.dtype,
                )
            if self.decoder_checkpoint:
                load_tinyae_checkpoint_into_vae(
                    self._vae,
                    self.decoder_checkpoint,
                    components=('decoder',),
                    strict=self.decoder_checkpoint_strict,
                    map_location=device,
                    dtype=self.dtype,
                )
            if self.use_compile:
                self._vae = torch.compile(self._vae.eval().requires_grad_(False).to(device), mode="max-autotune", fullgraph=True)
        else:
            self._vae.to(device)

    def _configure_latent_norm(self, latent_norm):
        if latent_norm is None:
            return
        enabled = bool(latent_norm.get('enabled', False))
        if not enabled:
            return
        stats_path = latent_norm.get('stats_path', None)
        if not stats_path:
            raise ValueError('encoder.latent_norm.stats_path must be set when latent_norm is enabled')
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f'encoder.latent_norm.stats_path not found: {stats_path}')

        eps = float(latent_norm.get('eps', 1e-6))
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        latent_channels = int(stats.get('latent_channels', self.latent_channels))
        if latent_channels != self.latent_channels:
            raise ValueError(
                f'AutoencoderTiny latent_norm requires latent_channels={self.latent_channels}, '
                f'got {latent_channels} in {stats_path}'
            )
        mean = stats.get('mean', None)
        std = stats.get('std', None)
        if mean is None or std is None:
            raise ValueError(f'latent norm stats must contain mean and std arrays: {stats_path}')
        if len(mean) != self.latent_channels or len(std) != self.latent_channels:
            raise ValueError(
                f'AutoencoderTiny latent_norm expects mean/std length {self.latent_channels}, '
                f'got mean={len(mean)} std={len(std)} from {stats_path}'
            )

        self.latent_norm_enabled = True
        self.latent_norm_stats_path = stats_path
        self.latent_norm_eps = eps
        self.latent_norm_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.latent_norm_std = torch.as_tensor(std, dtype=torch.float32).clamp_min(eps)
        print(
            f'[AutoencoderTiny] latent_norm enabled: stats_path={stats_path} '
            f'mean_len={len(mean)} std_len={len(std)} eps={eps}',
            flush=True,
        )

    def _latent_norm_mean_std_like(self, x):
        mean = tensors.constant_like(x, self.latent_norm_mean).reshape(1, -1, 1, 1)
        std = tensors.constant_like(x, self.latent_norm_std).reshape(1, -1, 1, 1)
        return mean, std

    def _apply_latent_norm(self, x):
        if not self.latent_norm_enabled:
            return x
        if x.shape[1] != self.latent_channels:
            raise ValueError(
                f'AutoencoderTiny latent_norm expected {self.latent_channels} latent channels, got {x.shape[1]}'
            )
        mean, std = self._latent_norm_mean_std_like(x)
        return (x.to(torch.float32) - mean) / std

    def _invert_latent_norm(self, x):
        if not self.latent_norm_enabled:
            return x
        if x.shape[1] != self.latent_channels:
            raise ValueError(
                f'AutoencoderTiny latent_norm expected {self.latent_channels} latent channels, got {x.shape[1]}'
            )
        mean, std = self._latent_norm_mean_std_like(x)
        return x.to(torch.float32) * std + mean

    def latent_mean_std(self, x):
        """
        Compute per-sample, per-channel mean/std of VAE latents from AutoencoderTiny.
        Returns:
            mean: [B,C,1,1] float32
            std:  [B,C,1,1] float32
        """
        self.init(x.device)
        latents = self._vae.encode(x.to(self.dtype))["latents"]
        latents = latents.to(torch.float32)
        mean = latents.mean(dim=(2, 3), keepdim=True)
        std = latents.std(dim=(2, 3), keepdim=True, unbiased=False)
        return mean, std

    @torch.no_grad()
    def estimate_tiny_latent_stats(vae, dataloader, device, dtype=torch.float32, max_batches=64):
        ch_sum = None
        ch_sq_sum = None
        count = 0

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            # adapt this line to your dataloader
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device=device, dtype=dtype)

            z = vae.encode(imgs).latents.float()  # [B,16,H,W]

            s = z.sum(dim=(0, 2, 3))  # [16]
            ss = (z * z).sum(dim=(0, 2, 3))  # [16]
            n = z.shape[0] * z.shape[2] * z.shape[3]

            if ch_sum is None:
                ch_sum = s
                ch_sq_sum = ss
            else:
                ch_sum += s
                ch_sq_sum += ss

            count += n

        raw_mean = ch_sum / count
        raw_var = ch_sq_sum / count - raw_mean.square()
        raw_std = raw_var.clamp_min(1e-12).sqrt()
        return raw_mean, raw_std

    def encode_latents(self, x): # raw latents => final latents
        x = self._apply_latent_norm(x)
        x = x * tensors.constant_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + tensors.constant_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def _run_vae_encoder(self, x):
        self.init(x.device)
        latents = self._vae.encode(x.to(self.dtype))["latents"]
        latents = latents.to(torch.float32)
        return latents

    def encode(self, x):
        if x.shape[1] == 2 * self.latent_channels:
            mean, std = x.to(torch.float32).chunk(2, dim=1)
            x = mean + torch.randn_like(mean) * std
            return self.encode_latents(x)
        elif x.shape[1] == self.latent_channels:
            return self.encode_latents(x)
        elif x.shape[1] == 3:
            return self.encode_latents(self.encode_pixels(x))
        else:
            raise ValueError(f'Invalid number of channels: {x.shape[1]}')

    def decode_latents_to_pixels(self, x):
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - tensors.constant_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / tensors.constant_like(x, self.scale).reshape(1, -1, 1, 1)
        x = self._invert_latent_norm(x)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = (x * 0.5 + 0.5).clamp(0,1).mul(255).to(torch.uint8)
        return x

    def decode_latents_to_pixels_float(self, x):
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - tensors.constant_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / tensors.constant_like(x, self.scale).reshape(1, -1, 1, 1)
        x = self._invert_latent_norm(x)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = (x * 0.5 + 0.5).clamp(0, 1)
        return x

    def decode(self, x):  # final latents => raw pixels
        if x.shape[1] == 2 * 16:
            mean, std = x.to(torch.float32).chunk(2, dim=1)
            x = mean + torch.randn_like(mean) * std
            return self.decode_latents_to_pixels(x)
        elif x.shape[1] == 16:
            return self.decode_latents_to_pixels(x)
        elif x.shape[1] == 3:
            return x
        else:
            raise ValueError(f'Invalid number of channels: {x.shape[1]}')

    def decode_float(self, x):  # final latents => raw pixels (float, no quantization)
        if x.shape[1] == 2 * 16:
            mean, std = x.to(torch.float32).chunk(2, dim=1)
            x = mean + torch.randn_like(mean) * std
            return self.decode_latents_to_pixels_float(x)
        elif x.shape[1] == 16:
            return self.decode_latents_to_pixels_float(x)
        elif x.shape[1] == 3:
            if x.dtype == torch.uint8:
                return x.to(torch.float32) / 255.0
            return x
        else:
            raise ValueError(f'Invalid number of channels: {x.shape[1]}')
#----------------------------------------------------------------------------

def load_stability_vae(
    vae_name='stabilityai/sd-vae-ft-mse',
    device=torch.device('cpu'),
    dtype=torch.float32,
    use_compile=False,
    autoencoder_cls='AutoencoderKL',
):
    cache_dir = artifacts.cache_path('diffusers')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HOME'] = cache_dir
    
    import diffusers # pip install diffusers # pyright: ignore [reportMissingImports]
    diffusers_logging = diffusers.utils.logging
    prev_verbosity = diffusers_logging.get_verbosity()
    diffusers_logging.set_verbosity_error()
    if isinstance(autoencoder_cls, str):
        try:
            autoencoder_cls = getattr(diffusers.models, autoencoder_cls)
        except AttributeError as e:
            raise ValueError(f'Unknown autoencoder class: {autoencoder_cls}') from e
    try:
        try:
            # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
            vae = autoencoder_cls.from_pretrained(
                vae_name, cache_dir=cache_dir, local_files_only=True, torch_dtype=dtype
            )
        except Exception:
            # Could not load the model from cache; try without local_files_only.
            vae = autoencoder_cls.from_pretrained(vae_name, cache_dir=cache_dir, torch_dtype=dtype)
    finally:
        diffusers_logging.set_verbosity(prev_verbosity)
    if use_compile:
        return torch.compile(vae.eval().requires_grad_(False).to(device), mode="max-autotune", fullgraph=True)
    else:
        return vae.eval().requires_grad_(False).to(device)  # without compile

#----------------------------------------------------------------------------

def _torch_load_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _looks_like_state_dict(value):
    return isinstance(value, Mapping) and any(torch.is_tensor(v) for v in value.values())


def _extract_state_dict_from_checkpoint(checkpoint, component=None):
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f'TinyAE checkpoint must be a state dict or mapping, got {type(checkpoint).__name__}')
    if component is not None:
        value = checkpoint.get(component, None)
        if _looks_like_state_dict(value):
            return value
    for key in ('state_dict', 'model_state_dict', 'ema_state_dict', 'model', 'module', 'vae', 'autoencoder', 'taesd3'):
        value = checkpoint.get(key, None)
        if _looks_like_state_dict(value):
            return value
    raise ValueError('TinyAE checkpoint did not contain a tensor state_dict')


def _component_state_dict_from_checkpoint_state(state_dict, component, component_keys=None):
    component_keys = set() if component_keys is None else set(component_keys)
    component_prefix = f'{component}.'
    component_state = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith(component_prefix):
            component_key = key
        elif f'.{component_prefix}' in key:
            component_key = key[key.index(component_prefix):]
        elif key in component_keys:
            component_key = f'{component}.{key}'
        else:
            continue
        component_state[component_key] = value
    return component_state


def _load_tinyae_folder_state_dict(vae, folder_path, components, dtype=torch.float32, device='cpu'):
    if not hasattr(vae.__class__, 'from_pretrained'):
        raise ValueError(f'AutoencoderTiny VAE class cannot load Diffusers folder: {vae.__class__.__name__}')
    loaded_vae = vae.__class__.from_pretrained(folder_path, torch_dtype=dtype).eval().to(device)
    load_state = {}
    for component in components:
        if not hasattr(loaded_vae, component):
            raise ValueError(f'Diffusers TinyAE folder has no {component} module: {folder_path}')
        for key, value in getattr(loaded_vae, component).state_dict().items():
            load_state[f'{component}.{key}'] = value
    return load_state


def load_tinyae_checkpoint_into_vae(vae, checkpoint_path, components=('encoder', 'decoder'), strict=True, map_location='cpu', dtype=torch.float32):
    checkpoint_path = os.fspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'AutoencoderTiny checkpoint not found: {checkpoint_path}')
    components = tuple(components)
    if os.path.isdir(checkpoint_path):
        load_state = _load_tinyae_folder_state_dict(vae, checkpoint_path, components, dtype=dtype, device=map_location)
    else:
        checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=map_location)
        load_state = {}
        for component in components:
            if not hasattr(vae, component):
                raise ValueError(f'AutoencoderTiny VAE has no {component} module')
            component_module = getattr(vae, component)
            state_dict = _extract_state_dict_from_checkpoint(checkpoint, component=component)
            component_state = _component_state_dict_from_checkpoint_state(
                state_dict,
                component,
                component_module.state_dict().keys(),
            )
            if not component_state:
                raise ValueError(f'No loadable {component} weights found in AutoencoderTiny checkpoint: {checkpoint_path}')
            load_state.update(component_state)

    result = vae.load_state_dict(load_state, strict=False)
    missing_requested = [
        key for key in result.missing_keys
        if any(key.startswith(f'{component}.') for component in components)
    ]
    if result.unexpected_keys:
        preview = ', '.join(result.unexpected_keys[:8])
        raise ValueError(f'Unexpected TinyAE checkpoint keys: {preview}')
    if strict and missing_requested:
        preview = ', '.join(missing_requested[:8])
        component_label = ','.join(components)
        raise ValueError(
            f'AutoencoderTiny checkpoint is missing {len(missing_requested)} requested {component_label} weights; '
            f'first missing keys: {preview}'
        )
    print(
        f'[AutoencoderTiny] loaded checkpoint: path={checkpoint_path} '
        f'components={",".join(components)} keys={len(load_state)} strict={strict}',
        flush=True,
    )
    return result


def load_decoder_checkpoint_into_vae(vae, checkpoint_path, strict=True, map_location='cpu'):
    return load_tinyae_checkpoint_into_vae(
        vae,
        checkpoint_path,
        components=('decoder',),
        strict=strict,
        map_location=map_location,
    )


def load_encoder_checkpoint_into_vae(vae, checkpoint_path, strict=True, map_location='cpu'):
    return load_tinyae_checkpoint_into_vae(
        vae,
        checkpoint_path,
        components=('encoder',),
        strict=strict,
        map_location=map_location,
    )

#----------------------------------------------------------------------------
