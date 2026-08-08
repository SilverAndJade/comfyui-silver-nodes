"""Shared (cached) model loaders for the Silver node pack.

The core pain this solves: when jobs are submitted to ComfyUI over the HTTP
API, loader nodes run once per prompt.  A plain CLIPLoader/UNetLoader creates
a brand new ModelPatcher every time, so its weights get garbage-collected once
the prompt finishes.  On the next prompt the full file is re-read from disk --
the 8.2 GB flux2 Qwen text encoder takes ~3.5 minutes here.

By keeping a strong reference to the ModelPatcher in a module-level cache
keyed by a cache key, the object survives across prompts.  The weights stay in
RAM, so a later prompt only re-transfers to VRAM (seconds) instead of reading
the file from disk (minutes).
"""

import logging
import uuid

from nodes import CLIPLoader, UNETLoader
import folder_paths


_CACHE = {}
_CACHE_COUNT = {}


def _cache_key(key_opt, *parts):
    """Compute the cache key, honouring an explicit key_opt."""
    key_opt = key_opt.strip() if isinstance(key_opt, str) else ""
    if key_opt:
        return key_opt
    return "_".join(parts)


def _cache_put(key, tag, value):
    _CACHE[key] = (tag, value)
    _CACHE_COUNT[key] = _CACHE_COUNT.get(key, 0) + 1


def _cache_get(key):
    return _CACHE.get(key)


def _cache_weak_hash(key):
    return (key, _CACHE_COUNT.get(key, 0))


def clear_all():
    _CACHE.clear()
    _CACHE_COUNT.clear()


class SilverDiffusionModelShared:
    """A shared diffusion (UNet/Flux) model loader that is cached across prompts.

    key_opt defines the cache key so several jobs can reuse one loaded model
    instead of re-reading the file each time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_type": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "key_opt": ("STRING", {"default": "", "multiline": False}),
                "mode": (["Auto", "Override Cache", "Read Only"],),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "cache key")
    FUNCTION = "doit"
    CATEGORY = "silver/Backend"

    def doit(self, model_name, weight_type="default", key_opt="", mode="Auto"):
        key = _cache_key(key_opt, model_name, weight_type)

        if mode == "Read Only":
            entry = _cache_get(key)
            if entry is None:
                raise RuntimeError(f"[SilverDiffusionModelShared] cache key '{key}' is not loaded")
            _, model = entry
            return model, key

        if key not in _CACHE or mode == "Override Cache":
            loader = UNETLoader()
            model = loader.load_unet(model_name, weight_type)[0]
            _cache_put(key, "diffusion", model)
            logging.info(f"[Silver] diffusion model '{model_name}' cached under key '{key}'")
        else:
            _, model = _CACHE[key]
            logging.info(f"[Silver] diffusion model '{key}' reused (loading skip)")

        return model, key

    @staticmethod
    def IS_CHANGED(model_name, weight_type="default", key_opt="", mode="Auto"):
        key = _cache_key(key_opt, model_name, weight_type)
        if mode == "Override Cache":
            return model_name
        return _cache_weak_hash(key)


class SilverTextEncoderShared:
    """A shared text-encoder (CLIP) loader that is cached across API prompts.

    Drop-in caching wrapper around the stock CLIPLoader. Supports any type the
    stock node does, including the flux2 Qwen text encoder used by the app.
    """

    _TYPES = CLIPLoader.INPUT_TYPES()["required"]["type"][0]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (cls._TYPES,),
                "key_opt": ("STRING", {"default": "", "multiline": False}),
                "mode": (["Auto", "Override Cache", "Read Only"],),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "cache key")
    FUNCTION = "doit"
    CATEGORY = "silver/Backend"

    def doit(self, clip_name, type="flux2", key_opt="", mode="Auto", device="default"):
        key = _cache_key(key_opt, clip_name, type, device)

        if mode == "Read Only":
            entry = _cache_get(key)
            if entry is None:
                raise RuntimeError(f"[SilverTextEncoderShared] cache key '{key}' is not loaded")
            _, clip = entry
            return clip, key

        if key not in _CACHE or mode == "Override Cache":
            loader = CLIPLoader()
            clip = loader.load_clip(clip_name, type=type, device=device)[0]
            _cache_put(key, "text", clip)
            logging.info(f"[Silver] text encoder '{clip_name}' cached under key '{key}'")
        else:
            logging.info(f"[Silver] text encoder '{clip_name}' reused from '{key}' (loading skip)")
            clip = _CACHE[key][1]

        return clip, key

    @staticmethod
    def IS_CHANGED(clip_name, type, key_opt="", mode="Auto", device="default"):
        key = _cache_key(key_opt, clip_name, type, device)
        if mode == "Override Cache":
            return _cache_weak_hash(key) + (uuid.uuid4().hex,)
        return _cache_weak_hash(key)


class SilverClearModelCache:
    """Drops all cached diffusion/text-encoder models to free RAM on the node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "doit"
    CATEGORY = "silver/Backend"
    OUTPUT_NODE = True

    def doit(self):
        cleared = len(_CACHE)
        clear_all()
        return {"ui": {"cleared": [cleared]}}


NODE_CLASS_MAPPINGS = {
    "SilverDiffusionModelShared": SilverDiffusionModelShared,
    "SilverTextEncoderShared": SilverTextEncoderShared,
    "SilverClearModelCache": SilverClearModelCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverDiffusionModelShared": "Silver Shared Diffusion Model",
    "SilverTextEncoderShared": "Silver Shared Text Encoder",
    "SilverClearModelCache": "Silver Clear Model Cache",
}