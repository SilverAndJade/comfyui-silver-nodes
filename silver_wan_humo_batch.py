"""Silver WAN HuMo Batch - N audio-synced talking-head clips, staged.

WAN HuMo (Human Motion / audio-conditioned video) does NOT generate an audio
track - it consumes audio as conditioning and renders lips/expression that
follow it. So the guide audio IS the clip's audio: it is passed to the model
(through ComfyUI's native Audio Encoder -> audio_embed conditioning) and it is
what we save/output alongside each clip.

Same swap-minimising discipline as SilverH3Batch: the text encoder, audio
encoder, DiT and VAE each load once for the whole batch, so N clips cost the
same model swaps as 1:

    Phase A  ENCODE  text encoder + audio encoder hot:
                     every prompt text-encoded in one TE session; the guide
                     audio is encoded once and windowed per clip; optional
                     ref image(s) VAE-encoded once per unique image.
    Phase B  SAMPLE  DiT hot: every clip sampled back-to-back with WAN's
                     NATIVE audio conditioning in the latent (no H3-style
                     per-step spine hack needed).
    Phase C  DECODE  VAE hot: every stored latent decoded together.

Net: 2 model swaps for the whole batch (TE/AE -> DiT, DiT -> VAE) instead of
~3 x N queuing the generator once per clip. Uses the standard WAN sampler
(SamplerCustom), the WAN CLIP text encoder and ComfyUI's Audio Encoder.
"""

from .silver_h3_staged import (
    _sampler_names,
    _scheduler_names,
)
import os
import time

import torch

from comfy_extras.nodes_video import CreateVideo
from comfy_api.latest._util.video_types import VideoContainer, VideoCodec
import folder_paths

# WAN HuMo audio-embed helpers (the native conditioning math).
from comfy_extras.nodes_wan import linear_interpolation, get_audio_emb_window


class SilverWanHuMoBatch:
    """Render N audio-synced WAN HuMo clips with one TE/AE/DiT/VAE load each.

    Same fast-staging idea as SilverH3Batch, but for the WAN HuMo (audio
    driven) model. The guide_audio is REQUIRED and is BOTH the conditioning
    the model follows and the audio track emitted with each clip - WAN HuMo
    does not generate its own audio. Prompts are independent clips, no
    inter-clip continuity; speed win is purely from amortising the model loads.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "audio_encoder": ("AUDIO_ENCODER",),
            "guide_audio": ("AUDIO", {
                "tooltip": "REQUIRED audio drive for the WHOLE batch (voice, "
                           "music, ...). WAN HuMo CONSUMES audio as "
                           "conditioning - it does not generate a track - so "
                           "this audio is what the lips/expression follow AND "
                           "what gets saved alongside each clip. Encoded once, "
                           "then windowed per clip (step_audio advances the "
                           "window, so clip i follows seconds "
                           "[i*clip, (i+1)*clip])."}),
            "prompts": ("STRING_LIST", {
                "tooltip": "A single LIST of prompt strings - one per clip "
                           "(e.g. the output of a prompt builder). Every "
                           "prompt is text-encoded in ONE text-encoder session "
                           "before any sampling. Empty entries are skipped."}),
            "width": ("INT", {"default": 832, "min": 16, "max": 4096,
                              "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": 4096,
                               "step": 16}),
            "length": ("INT", {
                "default": 97, "min": 5, "max": 4096, "step": 4,
                "tooltip": "Frames per clip (HuMo default 97 = ~4s @24fps). "
                           "Latent temporal length is (length-1)//4 + 1."}),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (_sampler_names(), {"default": "res_multistep"}),
            "scheduler": (_scheduler_names(), {"default": "simple"}),
        }, "optional": {
            "overall_prompt": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "Optional SCENE-WIDE prompt prepended to EVERY "
                           "clip's prompt before encoding (overall + clip)."}),
            "ref_image": ("IMAGE", {
                "tooltip": "Optional CHARACTER / REFERENCE frame (the HuMo "
                           "ref_image). IMAGE batch: image i is the identity "
                           "for clip i; a single image is reused for every "
                           "clip. Encoded once per unique image and applied as "
                           "the model's reference_latent (cross-attention "
                           "identity) with no temporal claim."}),
            "combine_videos": ("BOOLEAN", {
                "default": False, "label_on": "one master file",
                "label_off": "one file per clip",
                "tooltip": "SAVED FILES only. ON = stitch every clip into ONE "
                           "master video (frames concatenated, guide audio laid "
                           "over the whole take) and save that. OFF (default) "
                           "= save one video file per clip, plus its per-clip "
                           "audio slice. Tensor outputs always carry the "
                           "per-clip lists."}),
            "step_audio": ("BOOLEAN", {
                "default": False, "label_on": "advance per clip",
                "label_off": "same head",
                "tooltip": "Only used with guide_audio. OFF = every clip locks "
                           "to the FIRST clip-length window of audio (all reuse "
                           "the opening ~4s). ON = clip i uses window "
                           "[i*clip, (i+1)*clip]. Automatically enabled whenever "
                           "combine_videos is ON, which implies a continuous "
                           "long-form take."}),
            "seed_per_shot": ("BOOLEAN", {
                "default": True, "label_on": "vary per clip",
                "label_off": "same seed every clip",
                "tooltip": "Varying the seed per clip gives each clip its own "
                           "take; off reuses one seed for all rendered clips."}),
            "save_video": ("BOOLEAN", {
                "default": True, "label_on": "save video files",
                "label_off": "frames only",
                "tooltip": "Create + save one video file per clip using "
                           "ComfyUI's base Create Video + Save Video code "
                           "paths (frames + audio muxed). The frames/audio are "
                           "still emitted as list outputs."}),
            "filename_prefix": ("STRING", {
                "default": "SilverWanHuMoBatch",
                "tooltip": "Output filename prefix for the saved videos."}),
            "frame_rate": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                "tooltip": "Playback fps of the saved video. Marches the audio "
                           "window slicing (clip duration = length/frame_rate)."}),
            "format": (["mp4", "auto"], {"default": "mp4"}),
            "frame_compression": ("FLOAT", {
                "default": 23.0, "min": 0.0, "max": 32.0, "step": 1.0,
                "tooltip": "Constant Rate Factor (CRF) - H.264 quality knob "
                           "for the saved video. 0 = lossless (huge file), "
                           "23 = visually clean (default), 32 = smallest "
                           "file. Range 0-32."}),
        }}

    RETURN_TYPES = ("IMAGE_LIST", "AUDIO_LIST", "STRING", "INT")
    RETURN_NAMES = ("video_frames", "audio", "saved_videos", "videos_rendered")
    OUTPUT_TOOLTIPS = (
        "List of frame tensors, one per rendered clip: "
        "[[clip1 frames], [clip2 frames], ...] (each is [F,H,W,C]).",
        "List of audio dicts, one per rendered clip: the guide audio window "
        "that clip followed (['waveform','sample_rate']). WAN HuMo produces "
        "no audio, so this is sliced from guide_audio.",
        "Newline-separated paths of the saved video files.",
        "How many clips actually rendered.")
    FUNCTION = "run"
    CATEGORY = "silver_nodes/sampling"

    @classmethod
    def VALIDATE_INPUTS(cls, prompts=None, overall_prompt=""):
        if isinstance(prompts, (list, tuple)) and not (
                len(prompts) == 2 and all(isinstance(v, int) for v in prompts)):
            if not any((p or "").strip() for p in prompts):
                return ("The prompts list is empty - feed at least one "
                        "non-blank prompt.")
        return True

    @staticmethod
    def _assemble_frames(frames_parts):
        shapes = {tuple(p.shape[1:]) for p in frames_parts}
        if len(shapes) == 1:
            total = sum(int(p.shape[0]) for p in frames_parts)
            master = torch.empty((total,) + tuple(frames_parts[0].shape[1:]),
                                 dtype=frames_parts[0].dtype)
            o = 0
            for p in frames_parts:
                master[o:o + p.shape[0]].copy_(p)
                o += int(p.shape[0])
            return master
        print(f"[SilverWanHuMoBatch] clips do not share a grid "
              f"({sorted(shapes)}) - master falls back to clip 1 frames only.",
              flush=True)
        return frames_parts[0]

    def run(self, model, clip, vae, audio_encoder, guide_audio, prompts,
            width, height, length, seed, steps, cfg=7.0,
            sampler_name="res_multistep", scheduler="simple",
            overall_prompt="", ref_image=None, combine_videos=False,
            step_audio=False, seed_per_shot=True, save_video=True,
            filename_prefix="SilverWanHuMoBatch", frame_rate=24.0,
            format="mp4", frame_compression=23.0):
        import node_helpers
        import comfy.model_management as mm
        import comfy.samplers as sf
        from comfy_extras import nodes_custom_sampler as ncs

        t_all = time.time()

        def _dur(sec):
            s = int(round(sec))
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            return (f"{h}h{m}m{s}s" if h else
                    f"{m}m{s}s" if m else f"{s}s")

        def _el():
            return time.strftime("[+%H:%M:%S]",
                                  time.gmtime(time.time() - t_all))

        prompts = [str(p).strip() for p in (prompts or []) if (p or "").strip()]
        if not prompts:
            raise ValueError("No non-empty prompt in the prompts list.")
        if str(overall_prompt or "").strip():
            _ov = str(overall_prompt).strip()
            prompts = [f"{_ov} {p}" for p in prompts]
            print(f"[SilverWanHuMoBatch] overall prompt ON: prepended to "
                  f"every clip ({len(prompts)} combined).", flush=True)
        n = len(prompts)
        if guide_audio is None:
            raise ValueError("WAN HuMo REQUIRES guide_audio - it drives the "
                             "lips/expression (the model consumes audio as "
                             "conditioning).")
        _step = bool(step_audio) or bool(combine_videos)

        latent_t = ((length - 1) // 4) + 1
        latent_h = height // 8
        latent_w = width // 8
        print(f"[SilverWanHuMoBatch] batch: {n} clip(s), {length}f @ "
              f"{width}x{height} (latent {latent_t}x{latent_h}x{latent_w}), "
              f"{steps} steps {sampler_name}/{scheduler}.", flush=True)

        sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler, steps,
                                                 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]

        def _evict(tag):
            try:
                dev = mm.get_torch_device()
                b4 = mm.get_free_memory(dev) / (1024 ** 3)
                try:
                    mm.unload_all_models()
                except Exception:
                    pass
                mm.free_memory(mm.get_total_memory(dev) * 0.9, dev)
                mm.soft_empty_cache()
                af = mm.get_free_memory(dev) / (1024 ** 3)
                print(f"[SilverWanHuMoBatch] {tag}: VRAM swept "
                      f"({b4:.1f} -> {af:.1f} GB free).", flush=True)
            except Exception as e:
                print(f"[SilverWanHuMoBatch] {tag}: sweep failed ({e}) - "
                      "continuing.", flush=True)

        # ==========================================================
        # PHASE A - ENCODE: prompts + audio + refs, one session
        # ==========================================================
        t0 = time.time()

        # --- guide audio conditioning (HuMo audio_embed) encoded ONCE.
        # audio_encoder_output = audio_encoder.encode_audio(waveform, sr) gives
        # {encoded_audio_all_layers, encoded_audio, audio_samples} at 16 kHz.
        _w = guide_audio["waveform"]
        _sr = int(guide_audio["sample_rate"])
        _w3 = _w if _w.ndim == 3 else _w.unsqueeze(0)
        _w3 = _w3.detach().cpu()
        if _w3.shape[1] == 1:
            _w3 = _w3.repeat(1, 2, 1)
        elif _w3.shape[1] > 2:
            _w3 = _w3[:, :2]
        _full_source = {"waveform": _w3, "sample_rate": _sr}
        _aeo = audio_encoder.encode_audio(_w3, _sr)   # -> audio_embed source
        _ae = torch.stack(_aeo["encoded_audio_all_layers"], dim=2)  # [1,T,33,C]
        _al = _aeo["audio_samples"] // 640
        _ae = _ae[:, :_al * 2]
        _feat = [linear_interpolation(_ae[:, :, a:b].mean(dim=2), 50, 25)
                 if b is not None else linear_interpolation(_ae[:, :, a], 50, 25)
                 for a, b in [(0, 8), (8, 16), (16, 24), (24, 32), (32, None)]]
        _audio_emb_t = torch.stack(_feat, dim=2)[0]   # [T, 5, 1280]
        print(f"[SilverWanHuMoBatch] guide audio: {_al} frames encoded once "
              f"(window {'step' if _step else 'head'} per clip).", flush=True)

        # clip audio windows (output audio slices) and audio_embed per clip
        _clip_samp = max(1, int(length / float(frame_rate) * _sr))
        _wins = []
        for i in range(n):
            s = (i * _clip_samp) if _step else 0
            w = _w3[..., s:s + _clip_samp]
            if w.shape[-1] < _clip_samp:
                w = torch.cat([w, _w3[..., :_clip_samp - w.shape[-1]]],
                              dim=-1)
            _wins.append(w)

        # --- ref image latents, cached per image index (per-clip identity) --
        _ref_lat = []
        if ref_image is not None:
            R = int(ref_image.shape[0])
            for k in range(R):
                img = torch.nn.functional.interpolate(
                    ref_image[k:k + 1].movedim(-1, 1), size=(height, width),
                    mode="bilinear", align_corners=False).movedim(1, -1)
                _ref_lat.append(vae.encode(img[:, :, :, :3]).detach().cpu())
            print(f"[SilverWanHuMoBatch] ref image(s): {R} encoded once "
                  f"(cropped/stretched to {width}x{height}), per clip.",
                  flush=True)

        # --- text-encode every prompt in one TE session; build pos+neg -------
        # negative is the EMPTY prompt, reconditioned per clip with zero audio
        # embed + zero reference latent (mirrors the native HuMo node).
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        conds = []
        for i, text in enumerate(prompts):
            tok = clip.tokenize(text)
            pos = clip.encode_from_tokens_scheduled(tok)
            f0 = 0 if not _step else i * length
            # (frame0 advances by the CLIP length each step)
            _wnd, _ = get_audio_emb_window(_audio_emb_t, length, frame0_idx=f0)
            _emb = _wnd.unsqueeze(0)
            _emb_neg = torch.zeros_like(_emb)
            pos = node_helpers.conditioning_set_values(
                pos, {"audio_embed": _emb.to(mm.get_torch_device())})
            negc = node_helpers.conditioning_set_values(
                neg, {"audio_embed": _emb_neg.to(mm.get_torch_device())})
            if _ref_lat:
                _rl = _ref_lat[min(i, R - 1)]
                zl = torch.zeros(_rl.shape, dtype=_rl.dtype)
                pos = node_helpers.conditioning_set_values(
                    pos, {"reference_latents": [_rl.to(mm.get_torch_device())]})
                negc = node_helpers.conditioning_set_values(
                    negc, {"reference_latents": [zl.to(mm.get_torch_device())]})
            else:
                zr = torch.zeros([1, 16, 1, latent_h, latent_w],
                                 device=mm.get_torch_device())
                pos = node_helpers.conditioning_set_values(
                    pos, {"reference_latents": [zr]})
                negc = node_helpers.conditioning_set_values(
                    negc, {"reference_latents": [zr]})
            conds.append((pos, negc))
        print(f"[SilverWanHuMoBatch] ENCODE done: {n} POS+NEG conditioning "
              f"pairs in one TE session ({_dur(time.time() - t0)}) {_el()}.",
              flush=True)
        _evict("ENCODE -> SAMPLE")

        # ==========================================================
        # PHASE B - SAMPLE: DiT hot; every clip sampled standalone
        # ==========================================================
        t0 = time.time()
        stored = []
        for si, (pos, negc) in enumerate(conds):
            latent = {"samples": torch.zeros(
                [1, 16, latent_t, latent_h, latent_w],
                device=mm.intermediate_device())}
            shot_seed = (seed + si) if seed_per_shot else seed
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            guider = sf.CFGGuider(model)
            guider.set_conds(pos, negc)
            guider.set_cfg(cfg)
            dt = time.time()
            out, _dup = ncs.SamplerCustomAdvanced().sample(
                noise, guider, sampler, sigmas, latent)
            sdt = time.time() - dt
            stored.append(out["samples"].detach().cpu())
            del out, latent
            print(f"[SilverWanHuMoBatch] clip {si + 1}/{n} sampled in "
                  f"{_dur(sdt)} {_el()}.", flush=True)
        print(f"[SilverWanHuMoBatch] SAMPLE done: {n} clip(s) back-to-back, "
              f"DiT loaded ONCE ({_dur(time.time() - t0)}) {_el()}.", flush=True)
        _evict("SAMPLE -> DECODE")

        # ==========================================================
        # PHASE C - DECODE: VAE hot; every stored latent decoded
        # ==========================================================
        t0 = time.time()
        frames_out = []
        for si, s in enumerate(stored):
            imgs = vae.decode(s)
            if imgs.ndim == 5:
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2],
                                    imgs.shape[-1])
            frames_out.append(imgs.detach().cpu().half())
            print(f"[SilverWanHuMoBatch] decoded clip {si + 1}/{n} "
                  f"({imgs.shape[0]}f).", flush=True)
        print(f"[SilverWanHuMoBatch] DECODE done: {n} clip(s) in one VAE "
              f"session ({_dur(time.time() - t0)}) {_el()}.", flush=True)

        # audio per clip = the guide window that clip followed
        audio_out = [{"waveform": w, "sample_rate": _sr} for w in _wins]

        # ---- optional: create + save video file(s) -------------------------
        saved_files = []
        if save_video:
            try:
                vc = VideoContainer(format)
            except ValueError:
                vc = VideoContainer.MP4
            codec = VideoCodec.H264
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = folder_paths.get_output_directory()
            if combine_videos:
                f = self._assemble_frames(frames_out).float().clamp(0.0, 1.0)
                video = CreateVideo.execute(f, float(frame_rate),
                                            _full_source, 8).args[0]
                H, W = f.shape[-3], f.shape[-2]
                full_output_folder, filename, counter, subfolder, _ = \
                    folder_paths.get_save_image_path(
                        filename_prefix, out_dir, H, W)
                file = f"{filename}_{stamp}_master.mp4"
                path = os.path.join(full_output_folder, file)
                video.save_to(path, format=vc, codec=codec, metadata=None,
                              crf=float(frame_compression))
                saved_files.append(path)
                print(f"[SilverWanHuMoBatch] saved MASTER video "
                      f"({f.shape[0]}f): {path}", flush=True)
            else:
                for si in range(n):
                    f = frames_out[si].float().clamp(0.0, 1.0).cpu()
                    a = audio_out[si]
                    video = CreateVideo.execute(f, float(frame_rate), a, 8)\
                        .args[0]
                    H, W = f.shape[-3], f.shape[-2]
                    full_output_folder, filename, counter, subfolder, _ = \
                        folder_paths.get_save_image_path(
                            filename_prefix, out_dir, H, W)
                    file = f"{filename}_{stamp}_{si + 1:02d}.mp4"
                    path = os.path.join(full_output_folder, file)
                    video.save_to(path, format=vc, codec=codec, metadata=None,
                                  crf=float(frame_compression))
                    saved_files.append(path)
                    print(f"[SilverWanHuMoBatch] saved clip {si + 1}/{n}: "
                          f"{path}", flush=True)

        saved_videos = "\n".join(saved_files)

        print(f"[SilverWanHuMoBatch] done: {n} clips, total "
              f"{_dur(time.time() - t_all)} - 2 model swaps for the whole "
              f"batch instead of ~{3 * n}.", flush=True)
        return (frames_out[:n], audio_out[:n], saved_videos, n)


NODE_CLASS_MAPPINGS = {
    "SilverWanHuMoBatch": SilverWanHuMoBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverWanHuMoBatch":
        "Silver WAN HuMo Batch (N clips / encode once / sample once / decode once)",
}
