"""Silver H3 Video Editor - regenerate a section of an existing video via ref2va.

Takes a source video, lets the user pick a section (capped at 15 s, the
model's limit) in the preview panel, and feeds the WHOLE section - frames AND
its soundtrack - to MiniMax-H3's Ref2VA checkpoint as a single reference video
(<Video 1> with its paired <Audio 1>). The model re-renders the section from
the edit prompt, producing a brand-new video. There is no splicing: the output
is exactly the regenerated section.

Key differences from SilverH3SegmentEditor (fl2v "regenerate in place"):

  - this is REFERENCE-TO-VIDEO editing, not keyframe regeneration. The
    selected section is a reference, so the edit is a re-render of the whole
    window guided by it, not a pixel-exact splice between start/end anchors.
  - keyframes and refs are mutually exclusive by construction
    (model_base.py:2173 replaces cond_video_latents with the refs' latents),
    so there are no first/last keyframes here - hence a ref2va-style
    checkpoint is required, and fl2v models will ignore the reference.
  - the output audio is whatever the model generates (natively guided by the
    reference soundtrack), so no audio spine/lock mechanism is needed.

Phases (the same 2-swap discipline as the other H3 nodes, ENCODE -> SAMPLE ->
DECODE, one model swap between phases):

  Phase A  ENCODE  text encoder + video VAE (section frames) + audio VAE
                   (section soundtrack) in one session -> cond + ref block.
  Phase B  SAMPLE  DiT hot: the section regenerated from the ref.
  Phase C  DECODE  VAE hot: frames + generated audio muxed to mp4.
"""

# The video-picker + preview + EDIT WINDOW range handles live in
# js/silver_h3_segment_editor.js (WEB_DIRECTORY "js"), shared with the segment
# editor node via a per-node config block.

import av
import folder_paths
import json
import math
import os
import time
import torch

from .silver_h3_staged import (
    _sampler_names,
    _scheduler_names,
    _FPS,
)
from .silver_h3_segment_editor import (
    _input_videos,
    _largest_grid_le,
    SilverH3SegmentEditor,
)

# The model's trained range (frames @ 24 fps): ~5.17 s - ~15.08 s. A 15 s
# selection (360 frames) snaps DOWN to 345, the largest on-grid value <= it,
# so the reference video and the output always share the exact same length.
MAX_SECTION_SEC = 15.0
MIN_SECTION_FRAMES = 124
MAX_SECTION_FRAMES = 362

# The prompt preamble: names <Video 1> as the clip the edit is based on. The
# tokenizer presents the soundtrack as <Audio 1> right before <Video 1>, so
# the model sees them as one paired reference.
_EDIT_ALIGN_LINE = ("For the target video, <Video 1> is the source clip "
                    "(with its soundtrack) that this edit is based on.\n\n")


def _encode_ref_audio(audio_vae, audio):
    """Encode a reference soundtrack the same way the stock ref2va node does.

    `audio` = {"waveform": [B, C, L], "sample_rate": sr} -> encoded latent
    [1, 32, 2, T] on CPU, plus the audio-column count (ref_audio_t).
    """
    import torchaudio
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    w = waveform[:1]
    if w.shape[1] == 1:
        w = w.repeat(1, 2, 1)
    elif w.shape[1] > 2:
        w = w[:, :2]
    z = audio_vae.encode(w.movedim(1, -1)).detach()
    return z.cpu(), int(z.shape[-1])


class SilverH3VideoEditor:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "source_video": (_input_videos(), {
                "tooltip": "SOURCE video from the input folder (browse the "
                           "dropdown, drag-and-drop, or use the Open button "
                           "in the node's preview panel to upload). The "
                           "section between source_start and source_end "
                           "(capped at 15s) is fed to the model as the "
                           "reference video <Video 1> - frames and its "
                           "soundtrack - and regenerated as a brand new "
                           "video."}),
            "video_prompt": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "Edit instruction for the regenerated section. "
                           "<Video 1> is the source clip (its soundtrack "
                           "rides as the paired audio reference). State what "
                           "to CHANGE (subject, clothing, scene, style...) "
                           "and what to KEEP (motion, camera, timing, "
                           "environment, lighting). Requires a ref2va-style "
                           "checkpoint."}),
            "source_start": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 7200.0, "step": 0.1,
                "tooltip": "Start (seconds) of the section to regenerate. "
                           "Use the preview panel's EDIT WINDOW handles to "
                           "set it visually."}),
            "source_end": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 7200.0, "step": 0.1,
                "tooltip": "End (seconds) of the section to regenerate. 0 = "
                           "to the end of the video. Capped at 15 seconds "
                           "(the model's limit) and snapped down to H3's "
                           "frame grid (5 + 17k at 24 fps); the reference "
                           "and output share exactly that length."}),
            "megapixels": ("FLOAT", {
                "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.05,
                "tooltip": "Resolution of the OUTPUT and the reference video "
                           "in megapixels: the whole section is resized to "
                           "this total pixel count, keeping the source's "
                           "aspect ratio as closely as the H3 canvas allows "
                           "(both dimensions divisible by 32), re-sampled "
                           "to 24 fps."}),
        }, "optional": {
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "sampler_name": (_sampler_names(), {"default": "res_multistep"}),
            "scheduler": (_scheduler_names(), {"default": "beta"}),
            "filename_prefix": ("STRING", {
                "default": "SilverH3VideoEditor",
                "tooltip": "Output filename prefix."}),
            "format": (["mp4", "auto"], {"default": "mp4"}),
            "frame_compression": ("FLOAT", {
                "default": 23.0, "min": 0.0, "max": 51.0,
                "tooltip": "H.264 CRF for the saved video. 0 = lossless, "
                           "23 = default good quality."}),
        },
        "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_video",)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # source_video is a browse/upload combo: strict "value not in list"
        # validation would reject freshly uploaded / workflow-pinned files.
        # Every input is re-checked defensively inside run().
        return True
    OUTPUT_TOOLTIPS = (
        "Path of the saved output video: the selected section of the source "
        "regenerated from <Video 1> as a brand-new 24 fps clip at the "
        "megapixel resolution, with the model's generated audio.",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "silver_nodes/sampling"

    @staticmethod
    def _extract_source_audio(video_path, crop_start_s, crop_end_s):
        """Decode the source video's audio track and crop to the section.

        Returns {"waveform": [1, C, L] float, "sample_rate": sr} or None
        (no / undecodable audio track). Mirrors the segment editor's proven
        extraction, cropped to the [crop_start_s, crop_end_s] section so the
        soundtrack stays paired with the section's video frames.
        """
        _src_audio = None
        try:
            import numpy as np
            cnt = av.open(video_path)
            astream = next((s for s in cnt.streams if s.type == "audio"), None)
            if astream is not None:
                aframes = [f.to_ndarray() for f in cnt.decode(audio=0)]
                if aframes:
                    a0 = aframes[0]
                    if a0.ndim == 1:
                        afull = np.concatenate([a[None, :] for a in aframes],
                                               axis=1)
                    elif a0.ndim == 2:
                        if a0.shape[0] <= a0.shape[1]:
                            afull = np.concatenate(aframes, axis=1)
                        else:
                            afull = np.concatenate([a.T for a in aframes],
                                                   axis=1)
                    else:
                        afull = np.concatenate(
                            [a.reshape(a.shape[0], -1) for a in aframes],
                            axis=1)
                    at = torch.from_numpy(np.ascontiguousarray(afull)).float()
                    if at.shape[0] == 1:
                        at = at.repeat(2, 1)
                    elif at.shape[0] > 2:
                        at = at[:2]
                    _sr = int(astream.sample_rate or 32000)
                    _cs = int(round(crop_start_s * _sr))
                    _ce = int(round(crop_end_s * _sr))
                    _ce = min(at.shape[-1], _ce)
                    _cs = max(0, min(at.shape[-1], _cs))
                    if _ce - _cs < 1:
                        _cs, _ce = 0, at.shape[-1]
                    at_cropped = at[..., _cs:_ce]
                    if at_cropped.shape[-1] >= _sr // 50:   # >= 20 ms
                        _src_audio = {
                            "waveform": at_cropped.unsqueeze(0),
                            "sample_rate": _sr}
            cnt.close()
        except Exception as e:
            print(f"[SilverH3VideoEditor] source audio extraction failed: {e} "
                  "- the reference will carry no soundtrack.", flush=True)
        return _src_audio

    def run(self, model, clip, video_vae, audio_vae,
            source_video, video_prompt,
            source_start=0.0, source_end=0.0,
            megapixels=1.0,
            seed=0, steps=20,
            sampler_name="res_multistep", scheduler="beta",
            filename_prefix="SilverH3VideoEditor",
            format="mp4", frame_compression=23.0,
            prompt=None, extra_pnginfo=None):
        import node_helpers
        import comfy.model_management as mm
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3

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
                print(f"[SilverH3VideoEditor] {tag}: VRAM swept "
                      f"({b4:.1f} -> {af:.1f} GB free).", flush=True)
            except Exception as e:
                print(f"[SilverH3VideoEditor] {tag}: sweep failed ({e}) "
                      "- continuing.", flush=True)

        # ----------------------------------------------------------------
        # Resolve source + metadata
        # ----------------------------------------------------------------
        video_path = SilverH3SegmentEditor._resolve_video_path(source_video)
        src_frames, src_fps, src_w, src_h = SilverH3SegmentEditor._source_meta(
            video_path)
        if src_frames < 1:
            raise ValueError("Could not determine the source video's length.")

        # ----------------------------------------------------------------
        # The 15 s section on the SOURCE grid, capped at the model's limit.
        # ----------------------------------------------------------------
        src_total_sec = src_frames / src_fps
        crop_start_s = max(0.0, min(float(source_start or 0.0),
                                    src_total_sec))
        crop_end_s = (src_total_sec
                      if source_end is None or float(source_end) <= 0
                      else min(src_total_sec, float(source_end)))
        crop_end_s = max(crop_start_s, crop_end_s)
        crop_end_s = min(crop_end_s, crop_start_s + MAX_SECTION_SEC)
        crop_a_f = int(round(crop_start_s * src_fps))
        crop_b_f = max(crop_a_f + 1, int(round(crop_end_s * src_fps)))
        crop_b_f = max(crop_a_f + 1, min(src_frames, crop_b_f))
        sec_out = max(1, int(round((crop_b_f - crop_a_f) / src_fps * _FPS)))

        # Snap the section DOWN to H3's 5+17k grid so the reference video and
        # the output are the exact same length (a length-matched reference
        # keeps the video-to-video re-render faithful to the very last frame).
        gen_frames = _largest_grid_le(sec_out)
        if gen_frames < MIN_SECTION_FRAMES:
            raise ValueError(
                f"The selected section ({sec_out / _FPS:.1f}s) is shorter "
                f"than the model's trained minimum "
                f"({MIN_SECTION_FRAMES / _FPS:.1f}s at 24 fps).")
        if gen_frames > MAX_SECTION_FRAMES:
            gen_frames = MAX_SECTION_FRAMES

        # ----------------------------------------------------------------
        # Output / reference canvas from the megapixel budget (32-divisible,
        # aspect kept as close to the source as the H3 canvas allows).
        # ----------------------------------------------------------------
        width, height = SilverH3SegmentEditor._compute_dims(
            float(megapixels), src_w, src_h)
        print(f"[SilverH3VideoEditor] source: {src_frames}f "
              f"({src_total_sec:.1f}s) @ {src_fps:.1f}fps {src_w}x{src_h}; "
              f"edit window [{crop_start_s:.1f}s:{crop_start_s + crop_end_s - crop_start_s:.1f}s] "
              f"-> {gen_frames}f ({gen_frames / _FPS:.1f}s) @ {_FPS:.0f}fps "
              f"{width}x{height} (megapixels={megapixels}).", flush=True)

        # ----------------------------------------------------------------
        # Pre-resize the section ONCE to the gen geometry at 24 fps (the same
        # pool the segment editor uses, so the reference and output canvases
        # are identical).
        # ----------------------------------------------------------------
        t0 = time.time()
        section_frames = []
        for fr in SilverH3SegmentEditor()._iter_source_frames(
                video_path, width, height, out_fps=_FPS,
                start_out=0, end_out=gen_frames,
                crop_a_f=crop_a_f, crop_b_f=crop_b_f):
            section_frames.append(fr[0])
        if len(section_frames) != gen_frames:
            raise ValueError(f"Source too short: requested {gen_frames}f of "
                             f"the section, decoded {len(section_frames)}.")
        ref_video_px = torch.stack(section_frames)   # [F, H, W, 3] 0..1
        del section_frames
        print(f"[SilverH3VideoEditor] pre-resized {gen_frames}f of the "
              f"section to {width}x{height} "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)

        # ----------------------------------------------------------------
        # Source audio cropped to the exact section (the soundtrack of
        # <Video 1>).
        # ----------------------------------------------------------------
        gen_dur_s = gen_frames / _FPS
        src_audio = self._extract_source_audio(
            video_path, crop_start_s, crop_start_s + gen_dur_s)
        if src_audio is not None:
            print(f"[SilverH3VideoEditor] source audio: "
                  f"{src_audio['waveform'].shape[-1]} samples @ "
                  f"{src_audio['sample_rate']}Hz (cropped to the section).",
                  flush=True)
        else:
            print("[SilverH3VideoEditor] no source audio: <Video 1> carries "
                  "no soundtrack - the model still generates one.",
                  flush=True)

        # ----------------------------------------------------------------
        # PHASE A - ENCODE: reference video + soundtrack + prompt tokens in
        # one TE/VAE session.
        # ----------------------------------------------------------------
        t0 = time.time()
        z = video_vae.encode(ref_video_px).detach()     # [1, 24, T, H/16, W/16]
        latent_t = int(z.shape[2])

        audio_latent, ref_audio_t = None, 0
        if src_audio is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, src_audio)

        # Qwen sees the reference at 2 fps with 0.5 s timestamps (stock
        # ref2va presentation): an <Audio 1> label precedes <Video 1>.
        sample_idx = list(range(0, gen_frames, mmh3.FPS // 2))
        qwen_frames = ref_video_px[sample_idx]
        ref_items = []
        if audio_latent is not None:
            ref_items.append({"type": "audio"})
        ref_items.append({
            "type": "video",
            "data": qwen_frames,
            "timestamps": [i / 2.0 for i in range(len(sample_idx))],
        })

        tokens = clip.tokenize(_EDIT_ALIGN_LINE + video_prompt,
                               minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        # The DiT-side ref block: video rows + its paired audio rows, packed
        # in the same block (stock MiniMaxH3ReferenceToVideo layout).
        ref_block = {
            "kind": "video_audio" if ref_audio_t else "video",
            "latent_t": latent_t,
            "latent_h": height // 16,
            "latent_w": width // 16,
            "ref_audio_t": ref_audio_t,
            "latent": z.cpu(),
            "audio_latent": (audio_latent.cpu()
                             if audio_latent is not None else None),
        }
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_refs": [ref_block]})
        del z, ref_video_px, qwen_frames
        if src_audio is not None:
            del src_audio
        print(f"[SilverH3VideoEditor] ENCODE done (ref video "
              f"{gen_frames}f {width}x{height}, "
              f"{'with' if ref_audio_t else 'without'} soundtrack) "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)
        _evict("ENCODE -> SAMPLE")

        # ----------------------------------------------------------------
        # PHASE B - SAMPLE: single shot, ref2va (no keyframes, no spine).
        # ----------------------------------------------------------------
        t0 = time.time()
        latent, _fc = mmh3._empty_av_latent(width, height, gen_frames)
        noise = ncs.RandomNoise().get_noise(seed)[0]
        guider = ncs.BasicGuider().get_guider(model, cond)[0]
        sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler, steps,
                                                 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]

        out, _dup = ncs.SamplerCustomAdvanced().sample(
            noise, guider, sampler, sigmas, latent)

        lat = out["samples"]
        parts = (list(lat.unbind()) if getattr(lat, "is_nested", False)
                 else [lat])
        v = parts[0]
        a = parts[1] if len(parts) > 1 else None
        seg_video_lat = v.detach().cpu()
        seg_audio_lat = (a.detach().cpu() if a is not None else None)
        del out, latent, lat, parts, v, a, noise, guider
        print(f"[SilverH3VideoEditor] SAMPLE done ({_dur(time.time() - t0)}) "
              f"{_el()}.", flush=True)
        _evict("SAMPLE -> DECODE")

        # ----------------------------------------------------------------
        # PHASE C - DECODE + save the regenerated section as a new video.
        # ----------------------------------------------------------------
        t0 = time.time()

        decoded_video = video_vae.decode(seg_video_lat).detach().cpu().half()
        if decoded_video.ndim == 5:
            decoded_video = decoded_video.reshape(
                -1, decoded_video.shape[-3],
                decoded_video.shape[-2], decoded_video.shape[-1])
        decoded_video = decoded_video[:gen_frames]
        print(f"[SilverH3VideoEditor] decoded {decoded_video.shape[0]}f.",
              flush=True)

        out_audio = None
        if seg_audio_lat is not None:
            try:
                from comfy_extras.nodes_audio import vae_decode_audio
                aud = vae_decode_audio(audio_vae, {"samples": seg_audio_lat})
                gw = aud["waveform"].detach().cpu()
                if gw.ndim == 2:
                    gw = gw.unsqueeze(0)
                if gw.shape[0] > 1:
                    gw = gw[:1]
                out_audio = {"waveform": gw,
                             "sample_rate": int(aud["sample_rate"])}
                print(f"[SilverH3VideoEditor] generated audio: "
                      f"{gw.shape[-1]} samples @ {out_audio['sample_rate']}Hz.",
                      flush=True)
            except Exception as e:
                print(f"[SilverH3VideoEditor] generated-audio decode failed "
                      f"({e}) - output video will be silent.", flush=True)
        del seg_audio_lat

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                f"{filename_prefix}/{stamp}", out_dir, height, width)
        output_path = os.path.join(
            full_output_folder, f"{filename}_{stamp}_edit.mp4")

        _meta = {}
        if prompt is not None:
            _meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for _k in extra_pnginfo:
                _meta[_k] = json.dumps(extra_pnginfo[_k])

        writer = SilverH3SegmentEditor._MP4Stream(
            output_path, _FPS, audio=out_audio,
            crf=float(frame_compression), metadata=_meta)
        total_written = int(decoded_video.shape[0])
        writer.add_frames(decoded_video)
        writer.close()

        print(f"[SilverH3VideoEditor] DECODE + SAVE done "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)
        print(f"[SilverH3VideoEditor] saved: {output_path} "
              f"({total_written}f, ~{total_written / _FPS:.1f}s).", flush=True)
        print(f"[SilverH3VideoEditor] done in {_dur(time.time() - t_all)}.",
              flush=True)
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "SilverH3VideoEditor": SilverH3VideoEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverH3VideoEditor":
        "Silver H3 Video Editor (edit a section from a video reference)",
}