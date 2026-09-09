"""Silver H3 Batch - N independent videos, staged like the multishot.

The staged multishot (silver_h3_staged.py) chains its shots into ONE long
video (continuity / handoff). This node instead renders N
independent videos from N independent prompts, but keeps the SAME
swap-minimising discipline: the text encoder, DiT and VAE each load exactly
ONCE for the whole batch, so producing 5 (or however many) clips costs the
same model swaps as producing 1 - versus queuing the minimax node N times,
each of which pays the full TE -> DiT -> VAE load cycle:

    Phase A  ENCODE  text encoder (and video VAE for refs) hot:
                      every prompt text-encoded in one TE session; the
                      optional start frame, character image ref and character
                      voice ref are encoded once and reused for every video.
    Phase B  SAMPLE  DiT hot:  every video sampled back-to-back, each its
own standalone clip - NO chaining, NO handoff (videos share nothing but the
                       shared settings / seed policy).
    Phase C  DECODE + SAVE  VAEs hot: every stored latent decoded (audio
                      pre-pass, then each video decoded, streamed to its own
                      mp4 and freed immediately). At most ONE clip's frames
                      are resident; when combine_videos is ON the same frames
                      are also streamed into a single master mp4 (no full
                      master tensor). RAM stays ~one clip regardless of N.

Net: 2 model swaps for the whole batch (TE -> DiT, DiT -> VAE) instead of
~3 x N if you ran the generator once per clip. Requires
comfy_extras.nodes_minimax_h3 (MiniMax-H3 support). Reuses the prompt /
sampler helpers from silver_h3_staged.
"""

from .silver_h3_staged import (
    _sampler_names,
    _scheduler_names,
    _ALIGN_LINE,
    _FPS,
)
import os

import time

import torch

# Base ComfyUI video codec/path helpers (no VideoHelperSuite).
from comfy_api.latest._util.video_types import VideoContainer, VideoCodec
from comfy_api.latest._input_impl.video_types import mp4_output_open_kwargs
import av
import json
import math
from fractions import Fraction
import folder_paths


class SilverH3Batch:
    """Render N independent MiniMax-H3 videos with one TE/DiT/VAE load each.

    Same fast-staging idea as SilverH3StagedMultishot, but the prompts are
    UNRELATED clips: each entry in the prompts list becomes its own video, there is
    no inter-video continuity or handoff. The speed win is purely
    from amortising the model loads across the batch - encode every prompt in
    one text-encoder session, sample every video with the DiT resident, decode
    every video with the VAE resident.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompts": ("STRING_LIST", {
                "tooltip": "A single LIST of prompt strings - one per video "
                           "(e.g. the output of the Silver H3 Prompts Builder). "
                           "Every prompt is text-encoded in ONE text-encoder "
                           "session before any sampling starts, so the text "
                           "encoder loads once for the whole batch. Empty "
                           "entries are skipped."}),
            "width": ("INT", {"default": 768, "min": 32, "max": 4096,
                              "step": 32}),
            "height": ("INT", {"default": 1344, "min": 32, "max": 4096,
                               "step": 32}),
            "frames_per_shot": ("INT", {
                "default": 243, "min": 39, "max": 1450, "step": 17,
                "tooltip": "Frames at 24fps on H3's 17k+5 grid "
                           "(243 = ~10.1s per clip). Trained range is "
                           "~124-362; beyond is RoPE-extrapolation territory."}),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "sampler_name": (_sampler_names(), {"default": "res_multistep"}),
            "scheduler": (_scheduler_names(), {"default": "simple"}),
        }, "optional": {
            "overall_prompt": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "Optional SCENE-WIDE prompt prepended to EVERY "
                           "video's prompt before encoding (overall + shot)."}),
            "start_image": ("IMAGE", {
                "tooltip": "Optional FIRST FRAME (I2V). IMAGE batch: image i "
                           "opens clip i, so N+1 images chain N clips "
                           "frame-to-frame. A single image is reused as every "
                           "clip's frame-0 keyframe. Encoded ONCE per unique "
                           "image and applied via H3's native first/last "
                           "keyframe channel. Requires an fl2v-capable model. "
                           "When combine_videos is ON, clip i's LAST frame is "
                           "auto-taken as image i+1 (so the next clip's start "
                           "frame closes this clip) - no separate end input."}),
            "combine_videos": ("BOOLEAN", {
                "default": False, "label_on": "one master file",
                "label_off": "one file per clip",
                "tooltip": "SAVED FILES only. ON = stitch every clip into ONE "
                           "master video (full guide audio over the whole "
                           "stitch when a guide_audio is supplied, else "
                           "per-clip audio crossfaded at the seams) saved "
                           "into a DATED subfolder "
                           "prefix_yyyymmddHHMMSS/ as clip_01.mp4, clip_02..." 
                           "plus _master.mp4 - so the clips are on disk for "
                           "manual re-combining. OFF (default) = save one "
                           "flat file per clip, the original behaviour. "
                           "Streamed: only one clip is ever in RAM, so long "
                           "batches fit a small system."}) ,
"char_refs_all": ("IMAGE", {
                 "tooltip": "Optional CHARACTER REFERENCE list: EVERY image "
                            "here rides EVERY clip (<Picture N> labels stay "
                            "stable across the whole batch). 1..k images, "
                            "one per reference (character, prop, "
                            "location...). Each is encoded ONCE - a DiT-side "
                            "image ref row plus <Picture N> vision tokens: "
                            "outfit/face carry with no temporal claim, so it "
                            "never fights the start keyframes. Combined with "
                            "char_refs_index (deduped) when both ride. "
                            "Mutually exclusive with start_image (refs are "
                            "DROPPED, with a warning, if start_image is "
                            "given)."}),
             "char_refs_index": ("IMAGE", {
                 "tooltip": "Optional CHARACTER REFERENCE list INDEXED BY "
                            "CLIP: image i is ridden by CLIP i (a list "
                            "shorter than the prompts reuses its last image "
                            "for the remaining clips). When combine_videos "
                            "is ON with a batch, clip i also receives image "
                            "i+1 so zoom/pose transitions read as "
                            "continuous. Each clip references a DIFFERENT "
                            "<Picture N>, so the labels shift per clip. "
                            "Leave unconnected to disable indexed refs (see "
                            "char_refs_all for refs on every clip)."}),
            "char_audio": ("AUDIO", {
                "tooltip": "Optional CHARACTER VOICE REFERENCE (ref2va). "
                           "ONE voice, ridden by EVERY clip as a DiT-side "
                           "audio ref row plus <Audio N> tokens - "
                           "voice/style identity with no timing claim, so "
                           "it never fights guide_audio (the driving "
                           "spine) or the generated speech. Encoded once "
                           "in the ENCODE phase. Single track only (the "
                           "first audio is used if a batch is supplied). "
                           "Any sample rate; mono is upmixed. Dropped with "
                           "a warning when start_image rides."}),
            "guide_audio": ("AUDIO", {
                "tooltip": "Optional AUDIO SPINE for the WHOLE batch: a "
                           "continuous track (voice, music, ...) encoded ONCE "
                           "in the ENCODE phase. Each video's audio stream is "
                           "locked to a slice of it at every sampling step so "
                           "the clip follows the audio. With step_guide_audio "
                           "OFF every video uses the track's HEAD (first "
                           "clip-length); with it ON each video advances to "
                           "the next clip-length slice."}),
            "step_guide_audio": ("BOOLEAN", {
                "default": False, "label_on": "advance per video",
                "label_off": "same head",
                "tooltip": "Only used when guide_audio is set. OFF = every "
                           "video locks to the first clip-length of the guide "
                           "(so all 5 reuse the opening 5s). ON = video i "
                           "locks to guide audio seconds [i*clip, "
                           "(i+1)*clip], so each clip gets its own slice. "
                           "NOTE: stepping is AUTO-ENABLED whenever "
                           "combine_videos is ON - combining implies a "
                           "continuous long-form take, so each clip must "
                           "advance through the spine."}),
            "seed_per_shot": ("BOOLEAN", {
                "default": True, "label_on": "vary per video",
                "label_off": "same seed every video",
                "tooltip": "Varying the seed per video gives each clip its "
                           "own take; off reuses one seed for all rendered "
                           "videos."}),
            "filename_prefix": ("STRING", {
                "default": "SilverH3Batch",
                "tooltip": "Output filename prefix for the saved videos."}),
            "format": (["mp4", "auto"], {"default": "mp4"}),
            "frame_compression": ("FLOAT", {
                "default": 23.0, "min": 0.0, "max": 32.0, "step": 1.0,
                "tooltip": "Constant Rate Factor (CRF) - H.264 quality knob "
                           "for the saved video. 0 = lossless (huge file), "
                           "23 = visually clean (default), 32 = smallest "
                           "file. Lower = higher quality / larger file, "
                           "higher = lower quality / smaller file. Range 0-32 "
                           "is the useful span; above ~30 quality drops "
                           "noticeably."}),
            "apply_diffaid": ("BOOLEAN", {
                "default": False, "label_on": "on", "label_off": "off",
                "tooltip": "Applies the ComfyUI-DiffAid-Patches MiniMax H3 "
                           "sparse patch (the diffaid-patches custom node "
                           "must be installed) to the model BEFORE sampling. "
                           "The patch clones the model once and is shared by "
                           "every video (one clone, applied in Phase A, so the "
                           "DiT still loads once). Raises a hard error if the "
                           "diffaid-patches node is missing."}),
            "diffaid_strength": ("FLOAT", {
                "default": 0.20, "min": -1.0, "max": 1.0, "step": 0.01,
                "advanced": True,
                "tooltip": "Diff-Aid strength (the sparse patch's strength "
                           "parameter). 0.20 is the node/paper default; "
                           "stronger = harder conditioning focus on the "
                           "patched blocks."}),
            "double_pass": ("BOOLEAN", {
                "default": False, "label_on": "refine pass",
                "label_off": "single pass",
                "tooltip": "Second sampling pass over the FULL latent at a "
                           "reduced denoise (Motion-Director style refine). "
                           "Re-sampling from the already-sampled latent with "
                           "only the bottom of the sigma schedule regenerates "
                           "fine detail and cleans up the image. Costs one "
                           "extra DiT sample per video (n extra total)."}),
            "double_pass_denoise": ("FLOAT", {
                "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                "tooltip": "How much of the first-pass latent is re-noised "
                           "for the refine pass. 0.25 = only the bottom 25% "
                           "of the sigma schedule - a light refine that keeps "
                           "composition, 0.5+ visibly rewrites it, 1.0 = a "
                           "full re-roll of that clip."}),
            "double_pass_steps": ("INT", {
                "default": 8, "min": 1, "max": 50,
                "tooltip": "Sampling steps for the refine pass. 8 is plenty "
                           "for a 0.25-denoise refine; raise only when "
                           "denoise is pushed higher or quality demands it."}),
            "upscale_factor": ("FLOAT", {
                "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05,
                "tooltip": "Spatial latent upscale applied to every video "
                           "BEFORE decoding: the video latent is trilinearly "
                           "resampled to this factor BEFORE any double_pass "
                           "refine (recommended combo: upscale 1.25x + "
                           "double_pass to fill the new detail). 1.0 = native "
                           "resolution. The upscaled latent is decoded at the "
                           "larger size; audio is unaffected. The refine "
                           "pass's VRAM cost scales with the upscaled "
                           "area."}),
        },
        "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    RETURN_TYPES = ("AUDIO_LIST", "STRING", "INT")
    RETURN_NAMES = ("audio", "saved_videos", "videos_rendered")
    OUTPUT_TOOLTIPS = (
        "List of audio dicts, one per rendered video: "
        "[{video1 audio}, {video2 audio}, ...] (each is "
        "{'waveform','sample_rate'}); None for silent clips.",
        "Newline-separated paths of the saved video files (one per clip). "
        "When combine_videos is ON, the clips are saved into a dated "
        "subfolder SilverH3Batch_yyyymmddHHMMSS/ alongside a _master.mp4, "
        "so you can re-combine them manually if needed.",
        "How many videos actually rendered (one per entry in the prompts "
        "list).")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "silver_nodes/sampling"

    @classmethod
    def VALIDATE_INPUTS(cls, prompts=None, overall_prompt=""):
        # At validation time, LINKED inputs are NOT resolved yet - they arrive
        # as None or as a [node_id, slot] reference, so we cannot inspect their
        # contents here. Emptiness is enforced at runtime in run() instead. We
        # only validate when we are genuinely handed a concrete list/tuple.
        if isinstance(prompts, (list, tuple)) and not (
                len(prompts) == 2 and all(isinstance(v, int) for v in prompts)):
            if not any((p or "").strip() for p in prompts):
                return ("The prompts list is empty - feed at least one "
                        "non-blank prompt (e.g. from the Silver H3 Prompts "
                        "Builder).")
        return True

    # ---- assemble per-clip audio into one crossfaded master --------------
    @staticmethod
    def _assemble_audio(audio_parts, sr):
        # audio_parts is a list of dicts {"waveform","sample_rate"} or None
        # (matching the AUDIO_LIST output). Seed the master with the FIRST
        # non-silent clip, then merge each SUBSEQUENT clip in with a short
        # cosine crossfade at the seam. Skipping the seed clip matters - if we
        # looped over every waveform we would crossfade clip 1 against ITSELF
        # and duplicate a span of audio at the head. The waveforms are small
        # (a few MB), so keeping every clip's audio in RAM is fine.
        waveforms = [(a["waveform"] if a is not None else None)
                     for a in audio_parts]
        waveform = None
        seed_idx = None
        for idx, w in enumerate(waveforms):
            if w is not None:
                waveform = w
                seed_idx = idx
                break
        if waveform is not None:
            xfade = max(1, int((sr or 32000) * 40 / 1000.0))
            for w in waveforms[seed_idx + 1:]:
                if w is None:
                    continue
                nxt = w
                k = min(xfade, waveform.shape[-1], nxt.shape[-1])
                if k < 8:
                    waveform = torch.cat([waveform, nxt], dim=-1)
                    continue
                t = torch.linspace(0, 1, k, dtype=waveform.dtype)
                seam = (waveform[..., -k:] * torch.cos(t * 3.14159265 / 2)
                        + nxt[..., :k] * torch.sin(t * 3.14159265 / 2))
                waveform = torch.cat([waveform[..., :-k], seam,
                                      nxt[..., k:]], dim=-1)
        return waveform

    # ---- stream clip tensors into a single mp4 without a full master ----
    class _MP4Stream:
        """Incremental H.264 mp4 writer that holds at most ONE clip's frames.

        ComfyUI's SaveVideo.save_to needs the whole image stack up front, but
        that would mean materialising the full combined master (~27GB fp16 for
        12 clips of 768x1344x243) - too much for a 25GB box. Instead this
        writer opens the file once and we feed clip-by-clip, muxing frames as
        they arrive so RAM stays at ~one clip. Mirrors save_to's encode +
        mux logic (8-bit H.264, optional AAC audio truncated to frame count).

        IMPORTANT (PyAV quirk): the audio stream MUST be registered in __init__
        - adding it after any video frame has been muxed raises
        "Cannot rebase to zero time". So the final audio dict (guide or
        crossfaded per-clip audio) is passed up front and its stream created
        there; close() just flushes, slices the waveform to the total frame
        count and encodes it.
        """
        def __init__(self, path, frame_rate, audio=None, crf=23.0,
                     bit_depth=8, metadata=None):
            self._path = path
            self._frame_rate = float(frame_rate)
            self._out = av.open(path, **mp4_output_open_kwargs(
                path, VideoContainer.MP4, VideoCodec.H264))
            if metadata:
                # embed the generating workflow/prompt as container tags so
                # the saved videos can be dragged back into the UI / read by
                # prompt-extraction tools (movflags=use_metadata_tags is set
                # in mp4_output_open_kwargs).
                for _k in metadata:
                    self._out.metadata[_k] = metadata[_k]
            is_10bit = bit_depth >= 10
            fr = Fraction(round(self._frame_rate * 1000), 1000)
            self._pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
            self._is_10bit = is_10bit
            self._vstream = self._out.add_stream('h264', rate=fr)
            if crf is not None:
                self._vstream.options = {"crf": str(crf)}
            self._pix_fmt_set = False
            self._total_frames = 0
            self._astream = None
            self._astream_rate = None
            self._astream_layout = None
            if audio is not None:
                self._astream_rate = int(audio["sample_rate"])
                ch = int(audio["waveform"].shape[1])
                self._astream_layout = ({1: 'mono', 2: 'stereo', 6: '5.1'}
                                        .get(ch, 'stereo'))
                self._astream = self._out.add_stream(
                    'aac', rate=self._astream_rate,
                    layout=self._astream_layout)
                self._audio = audio["waveform"]
            else:
                self._audio = None

        def _ensure_geometry(self, H, W):
            if not self._pix_fmt_set:
                self._vstream.width = int(W)
                self._vstream.height = int(H)
                self._vstream.pix_fmt = self._pix_fmt
                self._pix_fmt_set = True

        def add_frames(self, tensor):
            """Encode + mux one clip's frame tensor ([F,H,W,C], 0..1)."""
            H, W = int(tensor.shape[1]), int(tensor.shape[2])
            self._ensure_geometry(H, W)
            for i in range(int(tensor.shape[0])):
                fr_t = tensor[i]
                if self._is_10bit:
                    img = (fr_t.float() * 65535).clamp(0, 65535).cpu() \
                        .numpy().astype("uint16")
                    af = av.VideoFrame.from_ndarray(img, format="rgb48le")
                else:
                    img = (fr_t * 255).clamp(0, 255).byte().cpu().numpy()
                    af = av.VideoFrame.from_ndarray(img, format="rgb24")
                af = af.reformat(format=self._pix_fmt)
                self._out.mux(self._vstream.encode(af))
            self._total_frames += int(tensor.shape[0])

        def close(self):
            """Flush the video, then mux the pre-registered audio (sliced to
            the total frame count) and close the container."""
            self._out.mux(self._vstream.encode(None))
            if self._audio is not None:
                wav = self._audio
                wav = wav[0, :, :math.ceil(
                    (self._astream_rate / self._frame_rate)
                    * self._total_frames)]
                wf = av.AudioFrame.from_ndarray(
                    wav.float().cpu().contiguous().numpy(),
                    format='fltp', layout=self._astream_layout)
                wf.sample_rate = self._astream_rate
                wf.pts = 0
                self._out.mux(self._astream.encode(wf))
                self._out.mux(self._astream.encode(None))
            self._out.close()

    def run(self, model, clip, video_vae, audio_vae, prompts,
            width, height, frames_per_shot, seed, steps,
            sampler_name="res_multistep", scheduler="simple",
            overall_prompt="", start_image=None, combine_videos=False,
            char_refs_all=None, char_refs_index=None,
            seed_per_shot=True,
            filename_prefix="SilverH3Batch",
            format="mp4", frame_compression=23.0,
            guide_audio=None, step_guide_audio=False, char_audio=None,
            apply_diffaid=False, diffaid_strength=0.20,
            double_pass=False, double_pass_denoise=0.25, double_pass_steps=8,
            upscale_factor=1.0, prompt=None, extra_pnginfo=None):
        import time
        import torch
        import node_helpers
        import comfy.model_management as mm
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio
        from torch.nn import functional as F
        import comfy.nested_tensor

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

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(p).strip() for p in (prompts or []) if (p or "").strip()]
        if not prompts:
            raise ValueError("No non-empty prompt in the prompts list.")
        if str(overall_prompt or "").strip():
            _ov = str(overall_prompt).strip()
            prompts = [f"{_ov} {p}" for p in prompts]
            print(f"[SilverH3Batch] overall prompt ON: prepended to every "
                  f"video ({len(prompts)} combined).", flush=True)
        n = len(prompts)
        print(f"[SilverH3Batch] batch: {n} video(s), {frames_per_shot}f @ "
              f"{width}x{height}, {steps} steps {sampler_name}/{scheduler}.",
              flush=True)

        if apply_diffaid:
            import importlib
            _diffaid_cls = None
            try:
                _diffaid_pkg = importlib.import_module("diffaid_x_patches")
                _diffaid_cls = getattr(_diffaid_pkg, "NODE_CLASS_MAPPINGS", {}) \
                    .get("MiniMaxH3DiffAidSparsePatch")
            except Exception as e:
                raise ValueError(
                    "apply_diffaid is ON but the diffaid-patches custom node "
                    "is not installed/loadable in this ComfyUI. Install it "
                    "under custom_nodes/, or turn apply_diffaid off "
                    f"(import error: {e!r}).")
            if _diffaid_cls is None:
                raise ValueError(
                    "apply_diffaid is ON but MiniMaxH3DiffAidSparsePatch was "
                    "not found in the diffaid-patches node mappings.")
            model, _dsum = _diffaid_cls().patch(
                model, enabled=True, block_indices="1,13,25,37,50",
                strength=float(diffaid_strength),
                sigma_start=0.0, sigma_end=1.0, sigma_ramp=0.0,
                token_weight_mode="none", token_tail=0.35, cond_only=True)
            print(f"[SilverH3Batch] Diff-Aid ON (model patched once, shared "
                  f"by every video): {_dsum}", flush=True)

        sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler, steps,
                                                 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]

        def _evict(tag):
            """One controlled swap at a PHASE BOUNDARY - never mid-phase."""
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
                print(f"[SilverH3Batch] {tag}: VRAM swept "
                      f"({b4:.1f} -> {af:.1f} GB free).", flush=True)
            except Exception as e:
                print(f"[SilverH3Batch] {tag}: sweep failed ({e}) - "
                      "continuing.", flush=True)

        # ==========================================================
        # PHASE A - ENCODE: every prompt in ONE text-encoder session
        # ==========================================================
        t0 = time.time()
        frame_count = mmh3.temporal_shape(frames_per_shot)[0]
        # --- character references: encoded once per unique image, into ONE
        # merged sheet. char_refs_all fills the head (rides EVERY clip);
        # char_refs_index rides CLIP i (its offset starts after the all-set).
        _char_ref, _char_px = [], []
        _n_all, _n_idx = 0, 0
        def _encode_sheet(imgs, kind):
            nonlocal _n_all, _n_idx
            if imgs is None:
                return 0
            C = int(imgs.shape[0])
            for k in range(C):
                _cimg = mmh3._resize(imgs[k:k + 1], width, height,
                                     "center")    # crop-to-fit, like start
                _cz = video_vae.encode(_cimg).detach()
                _char_ref.append({"kind": "image",
                                  "latent_h": _cz.shape[-2],
                                  "latent_w": _cz.shape[-1],
                                  "latent": _cz.cpu()})
                _char_px.append(_cimg.detach().cpu())
            if kind == "all":
                _n_all = C
            else:
                _n_idx = C
            return C
        # start_image keyframes + character refs are a HARD conflict: when
        # minimax_refs is set, comfy/model_base.py:2173 replaces
        # cond_video_latents with ONLY the refs' "latent" blocks - the
        # keyframe latent would be wiped and the packed cond rows
        # mis-filled. When start_image is given, ALL character refs (image
        # AND voice) are dropped with a warning rather than sent alongside
        # keyframes. Decided HERE from the inputs so the dropped refs are
        # never VAE-encoded in the first place.
        _drop_refs = start_image is not None \
            and (char_refs_all is not None or char_refs_index is not None
                 or char_audio is not None)
        if _drop_refs:
            print("[SilverH3Batch] start_image + character refs are "
                  "mutually exclusive (model_base.py refs override the "
                  "keyframe latents): DROPPING the character refs (images "
                  "+ voice) and keying clips from start_image only.",
                  flush=True)
        if not _drop_refs:
            _encode_sheet(char_refs_all, "all")
            _encode_sheet(char_refs_index, "index")
            if bool(_char_ref):
                print(f"[SilverH3Batch] character references ON: {_n_all} "
                      f"apply-to-all + {_n_idx} indexed image(s), encoded "
                      f"once (cropped to fit {width}x{height}, like the "
                      f"start frames).", flush=True)
        # --- character VOICE reference: ONE voice, encoded once, ridden by
        # every clip (one audio ref at a time). If a batch is supplied only
        # the first entry is used.
        _char_audio = None       # dit-side audio ref latent [1,32,2,T]
        _char_audio_t = 0        # ref_audio_t: audio columns
        if char_audio is not None and not _drop_refs:
            _wav = char_audio["waveform"]
            _vsr = int(char_audio["sample_rate"])
            _vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if _wav.shape[0] > 1:
                print(f"[SilverH3Batch] character VOICE: {_wav.shape[0]} "
                      f"audios supplied - only the FIRST is used (one "
                      f"voice at a time).", flush=True)
            _wav = _wav[:1]
            if _vsr != _vae_sr:
                import torchaudio
                _wav = torchaudio.functional.resample(_wav, _vsr, _vae_sr)
                print(f"[SilverH3Batch] character voice reference: "
                      f"resampled {_vsr} -> {_vae_sr} Hz.", flush=True)
            _seg = _wav[:1]
            if _seg.shape[1] == 1:
                _seg = _seg.repeat(1, 2, 1)
            _za = audio_vae.encode(_seg.movedim(1, -1)).detach().cpu()
            _char_audio = _za
            _char_audio_t = int(_za.shape[-1])
            print(f"[SilverH3Batch] character VOICE reference ON: "
                  f"{_char_audio_t} audio cols "
                  f"({_char_audio_t / float(getattr(mmh3, 'AUDIO_LATENT_FPS', 40)):.1f}s), "
                  f"rides every clip as an audio ref row + <Audio N> "
                  f"tokens.", flush=True)
        # --- start keyframe latents + pixels, cached per image index. A batch
        # shorter than n reuses its last image for the remaining clips; longer
        # is trimmed. The clip COUNT is the prompts length (n), never the image
        # count. When combine_videos is ON, clip i's LAST frame is auto-taken as
        # image i+1 (the next clip's start) - that is what stitches the clips
        # frame-to-frame, so there is no separate end_image input.
        _start_lat, _start_px = [], []
        if start_image is not None:
            S = int(start_image.shape[0])
            for k in range(S):
                img = mmh3._resize(start_image[k:k + 1], width, height,
                                   "center")    # crop-to-fit, aspect-preserving
                _start_px.append(img.detach().cpu())
                _start_lat.append(video_vae.encode(img).detach().cpu())
            print(f"[SilverH3Batch] I2V start frames: {S} image(s) encoded "
                  f"once (each reused as clip frame-0 keyframe).", flush=True)
        conds = []
        for i, text in enumerate(prompts):
            ref_items = []
            kf = []
            if start_image is not None:
                si = min(i, S - 1)
                text = _ALIGN_LINE + text
                ref_items.append({"type": "image", "data": _start_px[si]})
                kf.append({"resolved_frame_index": 0,
                           "latent": _start_lat[si].to(mm.get_torch_device())})
                if combine_videos:
                    ei = min(i + 1, S - 1)
                    ref_items.append({"type": "image", "data": _start_px[ei]})
                    kf.append({"resolved_frame_index": frame_count - 1,
                               "latent": _start_lat[ei]
                               .to(mm.get_torch_device())})
            # character references for clip i: char_refs_all rides the whole
            # list on EVERY clip; char_refs_index rides image i (+ image i+1
            # when combining, so the zoom/pose transition reads as
            # continuous - same chaining policy as start_image's end frame).
            # Both can ride at once (deduped against each other). Nested
            # helper: leave either side out and it contributes nothing.
            cis = []
            if _char_px and not _drop_refs:
                if _n_all:
                    cis += list(range(_n_all))
                if _n_idx:
                    cis.append(_n_all + min(i, _n_idx - 1))
                    if combine_videos and _n_idx > 1:
                        cis.append(_n_all + min(i + 1, _n_idx - 1))
                _seen = set()
                cis = [_x for _x in cis
                       if not (_x in _seen or _seen.add(_x))]
                for ci in cis:
                    ref_items.append({"type": "image", "data": _char_px[ci]})
            # character VOICE reference: ONE voice rides every clip, in the
            # fixed order the reference node uses (images, then standalone
            # audio), so <Picture N> / <Audio N> labels stay in sync with
            # the DiT's ref rows. Dropped when the keyframes win (_drop_refs).
            if _char_audio is not None and not _drop_refs:
                ref_items.append({"type": "audio"})
            tokens = (clip.tokenize(text, minimax_ref_items=ref_items)
                      if ref_items else clip.tokenize(text))
            cond = clip.encode_from_tokens_scheduled(tokens)
            if kf:
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": kf,
                    "minimax_frame_count": frame_count,
                })
            ref_blocks = ([
                {"kind": "image",
                 "latent_h": _char_ref[ci]["latent_h"],
                 "latent_w": _char_ref[ci]["latent_w"],
                 "latent": _char_ref[ci]["latent"]
                 .to(mm.get_torch_device())}
                for ci in cis
            ] if _char_px else [])
            if _char_audio is not None and not _drop_refs:
                ref_blocks.append({
                    "kind": "audio",
                    "ref_audio_t": _char_audio_t,
                    "audio_latent": _char_audio.to(mm.get_torch_device()),
                })
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_refs": ref_blocks,
                })
            conds.append(cond)
        del _char_px, _char_audio, _char_audio_t, _start_px

        # ---- guide-audio spine (encoded ONCE, still in ENCODE phase) ------
        _spine = None
        _guide_source = None   # raw full guide audio for the COMBINED master
        _AFPS = float(getattr(mmh3, "AUDIO_LATENT_FPS", 40))
        if guide_audio is not None:
            _w = guide_audio["waveform"]
            _sr = int(guide_audio["sample_rate"])
            _w3 = _w if _w.ndim == 3 else _w.unsqueeze(0)
            _vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if _sr != _vae_sr:
                import torchaudio
                _w3 = torchaudio.functional.resample(_w3, _sr, _vae_sr)
                print(f"[SilverH3Batch] guide audio: resampled {_sr} -> "
                      f"{_vae_sr} Hz.", flush=True)
            if _w3.shape[1] == 1:
                _w3 = _w3.repeat(1, 2, 1)
            elif _w3.shape[1] > 2:
                _w3 = _w3[:, :2]
            # raw full guide audio for the COMBINED master's audio track ONLY
            # (combine_videos ON + guide_audio supplied). Kept entire and
            # unprocessed (no tiling/crossfade); save_to truncates it to the
            # video length when longer.
            _guide_source = {"waveform": _w3.detach().cpu(),
                             "sample_rate": int(_vae_sr)}
            # Encode the spine in small chunks. The H3 audio VAE (latent_dim 2,
            # latent [B,32,2,T]) has NO working tiled fallback in ComfyUI - a
            # long guide OOMs the non-tiled encode, falls back to the 2D tiler
            # (sd.py encode_tiled_) and dies on waveform.shape[3]. Chunking
            # keeps every piece inside VRAM so the broken path is never hit.
            # The chunks are concatenated in latent time; the spine is only
            # ever re-injected as conditioning, never decoded, so the seams are
            # harmless.
            _spine_parts = []
            _s_chunk = max(1, int(10.0 * _vae_sr))
            for _s0 in range(0, _w3.shape[-1], _s_chunk):
                _seg = _w3[..., _s0:_s0 + _s_chunk]
                _spine_parts.append(audio_vae.encode(
                    _seg[:1].movedim(1, -1)).detach())
            _spine = torch.cat(_spine_parts, dim=-1)
            if len(_spine_parts) > 1:
                print(f"[SilverH3Batch] guide audio: encoded in "
                      f"{len(_spine_parts)} x ~10s chunks "
                      f"({_s_chunk} samples) - the H3 audio VAE has no tiled "
                      f"path, so long spines are chunked instead.", flush=True)
            _guide_s = _w3.shape[-1] / _vae_sr
            _a_lat_w = mmh3.temporal_shape(frames_per_shot)[1]
            # combining clips into one master implies stepping through the
            # spine per clip (otherwise every clip replays the head and the
            # master audio is wrong). _step honours the explicit toggle OR
            # the combine intent.
            _step = bool(step_guide_audio) or bool(combine_videos)
            _need = _a_lat_w * n if _step else _a_lat_w
            if _spine.shape[-1] < _need:
                _tiles = -(-_need // _spine.shape[-1])
                _spine = _spine.repeat(1, 1, _tiles)[..., :_need]
                print(f"[SilverH3Batch] guide audio: {_a_lat_w} cols "
                      f"({_guide_s:.1f}s guide) tiled x{_tiles} -> "
                      f"{_spine.shape[-1]} cols.", flush=True)
            else:
                _mode = "per-video (step)" if _step else \
                    "to its head for every video"
                print(f"[SilverH3Batch] guide audio: {_spine.shape[-1]} cols "
                      f"({_guide_s:.1f}s) locked {_mode}.", flush=True)

        print(f"[SilverH3Batch] ENCODE done: {n} conditionings in one TE "
              f"session ({_dur(time.time() - t0)}) {_el()}.", flush=True)
        _evict("ENCODE -> SAMPLE")

        # audio-lock plumbing (mirrors the staged multishot's spine arm)
        ms_obj = model.get_model_object("model_sampling")
        try:
            _asc = float(getattr(ms_obj, "audio_scale", 1.0) or 1.0)
        except Exception:
            _asc = 1.0
        _tss = None
        _shv, _sha = 12.0, 3.0
        try:
            from comfy.ldm.minimax.model import time_shift_sigma as _tss
            _dm = getattr(getattr(model, "model", None), "diffusion_model", None)
            _shv = float(getattr(_dm, "sigma_shift_video", 12.0))
            _sha = float(getattr(_dm, "sigma_shift_audio", 3.0))
        except Exception:
            _tss = None
        if abs(_asc - 1.0) > 1e-6:
            print(f"[SilverH3Batch] audio injections scaled x{_asc:.3f} onto "
                  f"the sampler's audio domain (ModelSamplingAV).", flush=True)

        _DBG = False
        _HO_ROWS = 5
        _HO_R0 = 2

        def _arm(guider, ho_v, ho_a, ho_guard, ho_tsrc, release, taper,
                 vshape, ashape, spine_seg=None):
            """Lock a guide-audio spine slice into THIS trajectory."""
            orig = guider.predict_noise
            nv = 1
            for d in vshape[1:]:
                nv *= d
            na = 0
            if ashape is not None:
                na = 1
                for d in ashape[1:]:
                    na *= d
            state = {"logged": False}

            def _pn(x, timestep, model_options={}, seed=None,
                    _o=orig, _hv=ho_v, _ha=ho_a, _hg=ho_guard,
                    _ts=ho_tsrc, _hs=spine_seg,
                    _ms=ms_obj, _tss=_tss, _shv=_shv,
                    _sha=_sha, _asc=_asc, _rel=float(release),
                    _tp=int(taper), _vs=vshape, _as=ashape,
                    _Nv=nv, _Na=na, _st=state):
                try:
                    sig = float(timestep.flatten()[0])
                except Exception:
                    sig = float(timestep)
                if (x.ndim == 3 and x.shape[1] == 1
                        and x.shape[2] >= _Nv + _Na):
                    try:
                        x = x.clone()
                        st = torch.tensor([sig], device=x.device,
                                          dtype=x.dtype)
                        sa = (_tss(sig, _shv, _sha) if _tss is not None
                              else sig)
                        sta = torch.tensor([sa], device=x.device,
                                           dtype=x.dtype)
                        if sig > _rel and _hv is not None:
                            xv = x[:, 0, :_Nv].reshape(
                                (x.shape[0],) + _vs[1:])
                            tv = _hv.to(device=x.device, dtype=x.dtype)
                            xv[:, :, _HO_R0:_HO_R0 + tv.shape[2]] = \
                                _ms.noise_scaling(st, torch.randn_like(tv),
                                                  tv)
                            if _tp > 0 and _ts is not None:
                                r0 = _HO_R0 + tv.shape[2]
                                tn = min(_tp, xv.shape[2] - r0)
                                if tn > 0:
                                    ts_ = _ts.to(device=x.device,
                                                 dtype=x.dtype)
                                    for j in range(tn):
                                        w = (tn - j) / (tn + 1.0)
                                        tgt = _ms.noise_scaling(
                                            st, torch.randn_like(ts_),
                                            ts_)[:, :, 0]
                                        xv[:, :, r0 + j] = \
                                            (1.0 - w) * xv[:, :, r0 + j] \
                                            + w * tgt
                        # AUDIO locks through EVERY step: content must be
                        # exact, and late releases shift speech onsets.
                        if _Na and _hs is not None:
                            xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                                (x.shape[0],) + _as[1:])
                            ts = _hs.to(device=x.device,
                                        dtype=x.dtype) * _asc
                            _nn = min(ts.shape[-1], xa.shape[-1])
                            xa[..., :_nn] = _ms.noise_scaling(
                                sta, torch.randn_like(ts[..., :_nn]),
                                ts[..., :_nn])
                            if not _st["logged"]:
                                _st["logged"] = True
                                print(f"[SilverH3Batch] SPINE inject: "
                                      f"{_nn}/{_as[-1]} audio cols "
                                      f"(sigma={sig:.3f} audio_sigma={sa:.3f} "
                                      f"asc={_asc:.2f}).", flush=True)
                        elif _Na and _ha is not None:
                            xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                                (x.shape[0],) + _as[1:])
                            ta = _ha.to(device=x.device, dtype=x.dtype) * _asc
                            if _hg is not None:
                                ta = torch.cat(
                                    [ta, _hg.to(device=x.device,
                                                dtype=x.dtype) * _asc],
                                    dim=-1)
                            xa[..., :ta.shape[-1]] = _ms.noise_scaling(
                                sta, torch.randn_like(ta), ta)
                        if not _st["logged"]:
                            _st["logged"] = True
                            print(f"[SilverH3Batch] audio lock ACTIVE "
                                  f"(sigma {sig:.3f}): "
                                  f"{(_hs.shape[-1] if _hs is not None else (_ha.shape[-1] if _ha is not None else 0))}"
                                  f" audio cols locked"
                                  f"{' (SPINE)' if _hs is not None else ''}"
                                  f"{' + guard' if _hg is not None else ''}.",
                                  flush=True)
                    except Exception as e:
                        print(f"[SilverH3Batch] audio injection FAILED: "
                              f"{e!r}", flush=True)
                return _o(x, timestep, model_options=model_options, seed=seed)

            guider.predict_noise = _pn

        # ==========================================================
        # PHASE B - SAMPLE: DiT hot; every video sampled standalone
        # ==========================================================
        t0 = time.time()
        stored = []
        for si, cond in enumerate(conds):
            latent, _fc = mmh3._empty_av_latent(width, height,
                                                frames_per_shot)
            comps = latent["samples"].unbind()
            vshape = tuple(comps[0].shape)
            ashape = tuple(comps[1].shape) if len(comps) > 1 else None
            shot_seed = (seed + si) if seed_per_shot else seed
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            spine_seg = None
            if _spine is not None and ashape is not None:
                _w = ashape[-1]
                _start = (si * _w) if _step else 0
                spine_seg = _spine[..., _start:_start + _w]
                print(f"[SilverH3Batch] video {si + 1}: guide-audio lock "
                      f"{spine_seg.shape[-1]}/{_w} cols "
                      f"({'step' if _step else 'head'}).",
                      flush=True)
            if spine_seg is not None:
                _arm(guider, None, None, None, None, 0.0, 0,
                     vshape, ashape, spine_seg=spine_seg)
            dt = time.time()
            out, _dup = ncs.SamplerCustomAdvanced().sample(
                noise, guider, sampler, sigmas, latent)
            sdt = time.time() - dt
            _up = float(upscale_factor)
            if _up > 1.0:
                _lat = out["samples"]
                _parts = (list(_lat.unbind()) if getattr(_lat, "is_nested", False)
                          else [_lat])
                _v = _parts[0]
                _nh = max(1, round(_v.shape[-2] * _up))
                _nw = max(1, round(_v.shape[-1] * _up))
                _v = F.interpolate(_v, size=(_v.shape[-3], _nh, _nw),
                                   mode="trilinear", align_corners=False)
                vshape = tuple(_v.shape)
                if len(_parts) > 1:
                    out["samples"] = comfy.nested_tensor.NestedTensor(
                        (_v, _parts[1]))
                else:
                    out["samples"] = _v
                print(f"[SilverH3Batch] video {si + 1}: video latent upscaled "
                      f"{_v.shape[-2]}x{_v.shape[-1]} (x{_up:.2f}); audio "
                      f"unchanged.", flush=True)
                del _lat, _parts, _v
            if double_pass:
                _sigmas2 = ncs.BasicScheduler().get_sigmas(
                    model, scheduler, double_pass_steps,
                    double_pass_denoise)[0]
                _noise2 = ncs.RandomNoise().get_noise(shot_seed + 10000000)[0]
                _guider2 = ncs.BasicGuider().get_guider(model, cond)[0]
                if spine_seg is not None:
                    _arm(_guider2, None, None, None, None, 0.0, 0,
                         vshape, ashape, spine_seg=spine_seg)
                _dt2 = time.time()
                out, _dup = ncs.SamplerCustomAdvanced().sample(
                    _noise2, _guider2, sampler, _sigmas2, out)
                print(f"[SilverH3Batch] video {si + 1}: second pass done - "
                      f"{double_pass_steps} steps @ "
                      f"denoise {double_pass_denoise:.2f} "
                      f"({_dur(time.time() - _dt2)}).", flush=True)
                del _sigmas2, _noise2, _guider2
            lat = out["samples"]
            parts = (list(lat.unbind()) if getattr(lat, "is_nested", False)
                     else [lat])
            v = parts[0]
            a = parts[1] if len(parts) > 1 else None
            stored.append({
                "video": v.detach().cpu(),
                "audio": a.detach().cpu() if a is not None else None,
            })
            del out, latent, lat, parts, v, a
            print(f"[SilverH3Batch] video {si + 1}/{n} sampled in "
                  f"{_dur(sdt)} {_el()}.", flush=True)
        print(f"[SilverH3Batch] SAMPLE done: {n} video(s) back-to-back, DiT "
              f"loaded ONCE ({_dur(time.time() - t0)}) {_el()}.", flush=True)
        _evict("SAMPLE -> DECODE")

        # ==========================================================
        # PHASE C - DECODE + SAVE: VAE hot; each latent decoded, its clip
        # saved to its own mp4 immediately, then freed. At most ONE clip's
        # frames (~2.3GB at 768x1344x243 fp16) is ever resident, so the
        # whole batch fits the ~25GB box regardless of how many clips there
        # are. When combine_videos is ON, the same frames are also streamed
        # into a single master mp4 (still one clip at a time - no full
        # master tensor is ever materialised).
        # ==========================================================
        t0 = time.time()
        audio_out = []
        saved_files = []

        # --- pre-pass: decode every per-clip AUDIO first. The audio latents
        # are tiny, so this costs little, but it means the master writer can
        # register its aac stream up front (PyAV cannot add an audio stream
        # after video frames are already muxed). Runs the audio VAE for ALL
        # clips consecutively so it stays resident exactly once, then frees
        # each audio latent as it goes.
        for s in stored:
            aud_dict = None
            if s["audio"] is not None:
                aud = vae_decode_audio(audio_vae, {"samples": s["audio"]})
                wav = aud["waveform"].detach().cpu()
                if wav.ndim == 2:
                    wav = wav.unsqueeze(0)
                aud_dict = {"waveform": wav,
                            "sample_rate": aud["sample_rate"]}
                s["audio"] = None    # release the audio latent immediately
            audio_out.append(aud_dict)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = folder_paths.get_output_directory()

        master_writer = None
        master_path = None
        full_output_folder = None
        filename = None
        master_total_frames = 0
        # workflow/prompt metadata, embedded in every mp4 like the stock save
        # nodes (prompt = API prompt, workflow = UI graph). Built once for the
        # master + every per-clip file.
        _meta = {}
        if prompt is not None:
            _meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for _k in extra_pnginfo:
                _meta[_k] = json.dumps(extra_pnginfo[_k])
        if combine_videos:
            # MASTER AUDIO: when a guide_audio was supplied, lay the FULL
            # original guide track, unprocessed, over the whole combined video
            # (user's choice: the raw spine - cleanest, no per-clip assembly).
            # The streaming writer truncates it to the video length. Without a
            # guide, fall back to the assembled per-clip video audio (each
            # clip's own AV audio, crossfaded at the seams).
            if _guide_source is not None:
                master_audio = _guide_source
            else:
                _sr = None
                for a in audio_out:
                    if a is not None:
                        _sr = a["sample_rate"]
                        break
                master_wave = self._assemble_audio(audio_out, _sr)
                master_audio = ({"waveform": master_wave,
                                 "sample_rate": _sr}
                                if master_wave is not None and _sr else None)
            full_output_folder, filename, counter, subfolder, _ = \
                folder_paths.get_save_image_path(
                    f"{filename_prefix}/{stamp}", out_dir, width, height)
            master_path = os.path.join(full_output_folder,
                                       f"{filename}_{stamp}_master.mp4")
            master_writer = self._MP4Stream(
                master_path, _FPS, audio=master_audio,
                crf=float(frame_compression), metadata=_meta)

        for si, s in enumerate(stored):
            imgs = video_vae.decode(s["video"])
            if imgs.ndim == 5:
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2],
                                    imgs.shape[-1])
            imgs = imgs.detach().cpu().half()
            aud_dict = audio_out[si]

            if combine_videos and master_writer is not None:
                # stream this clip's frames into the (still open) master,
                # AND write its own copy into the same dated subfolder so
                # the user can re-combine manually if needed. Frames are
                # released after each clip; only the (small) audio dicts
                # are retained.
                master_writer.add_frames(imgs)
                master_total_frames += int(imgs.shape[0])
                clip_path = os.path.join(
                    full_output_folder,
                    f"{filename}_{stamp}_clip_{si + 1:02d}.mp4")
                cw = self._MP4Stream(clip_path, _FPS, audio=aud_dict,
                                     crf=float(frame_compression),
                                     metadata=_meta)
                cw.add_frames(imgs)
                cw.close()
                saved_files.append(clip_path)
            else:
                H, W = imgs.shape[-3], imgs.shape[-2]
                full_output_folder, filename, counter, subfolder, _ = \
                    folder_paths.get_save_image_path(
                        filename_prefix, out_dir, H, W)
                path = os.path.join(full_output_folder,
                                    f"{filename}_{stamp}_{si + 1:02d}.mp4")
                cw = self._MP4Stream(path, _FPS, audio=aud_dict,
                                     crf=float(frame_compression),
                                     metadata=_meta)
                cw.add_frames(imgs)
                cw.close()
                saved_files.append(path)
            del imgs, s
            stored[si] = None    # release this latent the moment it's saved
            print(f"[SilverH3Batch] saved video {si + 1}/{n}.", flush=True)

        print(f"[SilverH3Batch] DECODE done: {n} video(s) in one VAE session "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)

        if combine_videos and master_writer is not None:
            master_writer.close()
            saved_files.append(master_path)
            print(f"[SilverH3Batch] saved MASTER video "
                  f"({master_total_frames}f): {master_path}", flush=True)

        saved_videos = "\n".join(saved_files)

        print(f"[SilverH3Batch] done: {n} videos, total "
              f"{_dur(time.time() - t_all)} - 2 model swaps for the whole "
              f"batch instead of ~{3 * n}.", flush=True)
        return (audio_out[:n], saved_videos, n)


NODE_CLASS_MAPPINGS = {
    "SilverH3Batch": SilverH3Batch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverH3Batch":
        "Silver H3 Batch (N videos / encode once / sample once / decode once)",
}
