"""Silver H3 Staged Multishot - encode once, sample once, decode once.

A long-form MiniMax-H3 pipeline built around one observation: a per-shot
chain (encode -> sample -> decode per clip) pays THREE model swaps per
shot, because ComfyUI loads the text encoder for conditioning, evicts it
for the DiT, then evicts that for the VAE - over and over. This node
reorders the work into three phases so the load count is CONSTANT
regardless of shot count - strictly 2 swaps in latent_handoff mode
(TE -> DiT -> audio-VAE); first_frame chaining additionally decodes ONLY
each finished shot's LAST FRAME (a 1-frame plain decode) through Phase B
to seed the next join's keyframe:

    Phase A  ENCODE  text encoder hot:   every prompt's conditioning,
                                         built in one TE session.
    Phase B  SAMPLE  DiT hot:            every shot sampled back-to-back,
                                         chained IN LATENT SPACE - each
                                         shot's first video rows / audio
                                         columns are locked to the previous
                                         shot's actual tail latents at
                                         every sampling step (latent
                                         handoff). No VAE round trip, no
                                         keyframes, no re-encoding: the
                                         hand-off material is passed from
                                         latent space to latent space.
    Phase C  DECODE  VAEs hot:           every stored latent decoded,
                                         replayed join heads trimmed,
                                         and the whole take STREAMED out -
                                         ONE mp4 per shot (that shot's
                                         audio) plus the combined
                                         _master.mp4, one shot's frames
                                         in RAM at a time (never a full
                                         master tensor).

An optional guide_audio AUDIO SPINE rides the ENCODE phase (encoded once)
and locks every shot's ENTIRE audio stream to its time-slice of one
continuous track at every sampling step - voice and lipsync held across
the whole piece, still with zero mid-chain model swaps. An optional
char_audio CHARACTER VOICE REFERENCE is encoded once too and ridden by
every shot as a DiT-side audio ref row plus <Audio N> tokens.

Phase C does NOT assemble a giant master tensor in RAM: each shot is
decoded, trimmed and streamed to its OWN mp4 as it goes, then into the
master (the same incremental writers the Batch node uses), so at most one
shot's frames are ever resident no matter how long the take.

The latent handoff answers "can I pass a frame from one latent space to
another?" - yes: the previous shot's last video latent rows and audio
latent columns are re-noised to the CURRENT sigma inside the sampler's
packed stream at every step, so the new shot literally wakes up inside
its own previous state and continues it (motion, exposure and speech
carry; the regenerated overlap is trimmed on decode).

Mechanics mirror the proven `latent_handoff` path of
ComfyUI-H3-Multishot (block depth), reduced to what survives staging:
no pixel-domain guards, nothing that would force a model swap
mid-phase. Continuity is selected by the `ref2v` toggle: OFF (fl2v)
chains shots 2+ via H3's native first_frame keyframe channel (last
frame is first frame) with motion carry; ON (ref2v) chains shots 2+
by feeding the previous shot's last frame as the FIRST minimax_refs
reference (before char_ref) with motion carry. Both modes use latent
handoff for motion continuity (head clone + per-step locks + taper).
Requires comfy_extras.nodes_minimax_h3 (MiniMax-H3 support).
"""

import av
import folder_paths
import json
import math
import os
import torch
from fractions import Fraction

from comfy_api.latest._util.video_types import VideoContainer, VideoCodec
from comfy_api.latest._input_impl.video_types import mp4_output_open_kwargs

# ---- H3 Motion Context patches (port from ComfyUI-H3-Motion-Director) -----
# These monkey-patch PackedLayout.__init__ and MiniMaxH3.extra_conds to
# support multiple keyframes at arbitrary temporal positions (not just
# first/last). Safe to apply globally — idempotent, versioned.
_MC_KEY = "motion_context_index"
_MC_AUDIO_KEY = "motion_context_audio_end_frame"
_CONTEXT_GRID = (39, 22, 5, 1)   # allowed context window sizes (pixel frames)
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # H3 temporal grid: row 0=1f, rows 1-4=4f each
_mc_layout_ok = False
_mc_payload_ok = False
try:
    import comfy.ldm.minimax.model as _mm

    def _mc_apply_layout_patch():
        global _mc_layout_ok
        if _mc_layout_ok:
            return
        import inspect as _ins
        _orig = _mm.PackedLayout.__init__
        _native = "frame_count" in _ins.signature(_orig).parameters

        def _frame_count(lt):
            p = _FRAME_PER_TOKEN
            return sum(p[i % len(p)] for i in range(int(lt)))

        def _target_origin(layout):
            a, b, kind = layout.segments[-1]
            return float(layout.position_ids[a, 0])

        def _patched(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
            fc = int(frame_count) if frame_count is not None else _frame_count(latent_t)
            # stock layout accepts only resolved_frame_index 0 or fc-1. MC_KEY
            # keyframes are repositioned AFTER _orig via the position_ids fixup
            # below, so translate them to a stock-valid frame here (in copies,
            # never mutating the caller's dicts).
            kfs_in = None
            if keyframes:
                kfs_in = []
                for kf in keyframes:
                    if kf.get(_MC_KEY) is not None:
                        kc = dict(kf)
                        kc["resolved_frame_index"] = int(fc) - 1
                        kfs_in.append(kc)
                    else:
                        kfs_in.append(kf)
            if _native:
                _orig(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=kfs_in, refs=refs, frame_count=fc)
            else:
                _orig(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=kfs_in, refs=refs)
            # fixup: rewrite position_ids for MC_KEY-marked keyframes
            if keyframes and any(k.get(_MC_KEY) is not None for k in keyframes):
                spans = [(a, b) for a, b, kind in self.segments if kind == "cond"]
                if len(spans) == len(keyframes):
                    off = _target_origin(self) - float(text_len)
                    for (a, b), kf in zip(spans, keyframes):
                        marker = kf.get(_MC_KEY)
                        if marker is None:
                            continue
                        p = int(marker)
                        if p == 0:
                            t = float(text_len)
                        elif p == int(fc) - 1:
                            t = float(text_len) + sum(
                                _mm._video_t_spans(latent_t)) - _mm.FRAME_RESCALE
                        else:
                            t = float(text_len) + _mm.FRAME_RESCALE * float(p)
                        self.position_ids[a:b, 0] = t + off
            # fixup: reposition audio context ref
            if refs and any(r.get(_MC_AUDIO_KEY) is not None for r in refs):
                marked = [i for i, r in enumerate(refs)
                          if r.get(_MC_AUDIO_KEY) is not None]
                if len(marked) == 1:
                    idx = marked[0]
                    ref = refs[idx]
                    steps = int(ref.get("ref_audio_t", 0))
                    if steps > 0 and ref.get("kind") == "audio":
                        actual = [(a, b, k) for a, b, k in self.segments
                                  if k == "ref_audio"]
                        if len(actual) > idx:
                            a, b = actual[idx][0], actual[idx][1]
                            want = (_target_origin(self)
                                    + _mm.FRAME_RESCALE * float(ref[_MC_AUDIO_KEY])
                                    - steps)
                            cur = float(self.position_ids[a, 0])
                            self.position_ids[a:b, 0] += want - cur

        _mm.PackedLayout.__init__ = _patched
        _mc_layout_ok = True

    def _mc_apply_payload_patch():
        global _mc_payload_ok
        if _mc_payload_ok:
            return
        import comfy.model_base as _mb
        cls = getattr(_mb, "MiniMaxH3", None)
        if cls is None or not hasattr(cls, "extra_conds"):
            return
        _orig_ec = cls.extra_conds

        def _patched_ec(self, **kwargs):
            out = _orig_ec(self, **kwargs)
            kfs = kwargs.get("minimax_keyframes", None)
            refs = kwargs.get("minimax_refs", None)
            if not kfs or not refs:
                return out
            cond = out.get("minimax_payload", None)
            payload = getattr(cond, "cond", None) if cond is not None else None
            if not isinstance(payload, dict):
                return out
            kf_v = [k["latent"] for k in kfs if k.get("latent") is not None]
            ref_v = [r["latent"] for r in refs if r.get("latent") is not None]
            kf_a = [k["audio_latent"] for k in kfs if k.get("audio_latent") is not None]
            ref_a = [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
            payload["cond_video_latents"] = kf_v + ref_v
            payload["cond_audio_latents"] = kf_a + ref_a
            fc = kwargs.get("minimax_frame_count", None)
            if fc is not None:
                payload["frame_count"] = fc
            return out

        cls.extra_conds = _patched_ec
        _mc_payload_ok = True

    _mc_apply_layout_patch()
    _mc_apply_payload_patch()
except Exception:
    pass   # non-H3 environment; patches not needed


def _sampler_names():
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        return ["res_multistep", "euler", "dpmpp_2m"]


def _scheduler_names():
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return ["simple", "normal", "beta"]


# ---- latent_handoff geometry (H3 block depth, verbatim constants) --------
_DBG = True       # runtime forensics for shot-to-shot diagnostics
_HO_ROWS = 5      # video latent rows locked = the first FULL 17-frame block
_HO_R0 = 2        # first two rows are the bootstrap (5 frames); lock starts here
_HO_ACOLS = 37    # audio latent cols locked (~0.92 s at 40 latent-fps)
_GUARD_COLS = 16  # room-tone cols appended past the lock (~0.4 s)
_OV_HO = 22       # reference overlap depth (unused; _TRIM=0 for latent_handoff)
_FPS = 24.0       # fixed playback rate - H3's audio sync requires 24 fps
_HO_RELEASE = 0.30  # sigma below which the locked/tapered head is released
_ALIGN_LINE = ("For the target video, at 0.00 seconds into the target "
               "video, <Picture 1> (from [Shot 1]) is fully "
               "referenced.\n\n")


# ---- Motion Context helpers (Phase 1-3) -----------------------------------
def _pixel_frames_for_latent_steps(latent_t):
    """H3 temporal grid: how many pixel frames N latent steps decode to."""
    return sum(_FRAME_PER_TOKEN[i % len(_FRAME_PER_TOKEN)]
               for i in range(int(latent_t)))


def _latent_step_offsets(latent_t):
    """Pixel-frame offset for each latent step (for MC_KEY placement)."""
    out, cursor = [], 0
    for i in range(int(latent_t)):
        out.append(cursor)
        cursor += _FRAME_PER_TOKEN[i % len(_FRAME_PER_TOKEN)]
    return out


def _select_context_span(requested, available):
    """Snap requested context frames to the nearest H3 grid value."""
    n = min(max(1, int(requested)), max(0, int(available)))
    for run in _CONTEXT_GRID:
        if run <= n:
            return run
    return 1


def _video_context_from_latent(video_latent, span, context_end_frame=None):
    """Extract last N latent blocks from a video latent for motion context.

    Returns (blocks, pixel_offsets, selected_end).
    """
    total_steps = int(video_latent.shape[2])
    # find needed steps
    needed = 0
    for s in range(1, 512):
        if _pixel_frames_for_latent_steps(s) >= span:
            needed = s
            break
    if needed == 0:
        needed = total_steps
    sample_frames = _pixel_frames_for_latent_steps(total_steps)
    end_limit = (sample_frames if context_end_frame is None
                 else min(sample_frames, max(0, int(context_end_frame))))
    end_step, selected_end = 0, 0
    for s in range(1, total_steps + 1):
        ep = _pixel_frames_for_latent_steps(s)
        if ep > end_limit:
            break
        end_step, selected_end = s, ep
    if end_step < needed:
        needed = end_step
    start_step = end_step - needed
    blocks = [video_latent[:1, :, step:step + 1].clone()
              for step in range(start_step, end_step)]
    return blocks, _latent_step_offsets(needed), selected_end


def _pin_renorm(latent_video, baseline_std, eps=1e-6):
    """Normalize a latent tensor's std to match the chain baseline.

    Returns (adjusted_tensor, current_std, scale). baseline_std=None is
    a no-op that returns the current std as the new baseline.
    """
    v = latent_video.float()
    cur_std = v.std(unbiased=False)
    if baseline_std is None:
        return latent_video, float(cur_std), 1.0
    scale = float(baseline_std) / (float(cur_std) + eps)
    mean = v.mean()
    adj = (v - mean) * scale + mean
    return adj.to(latent_video.dtype), float(cur_std), scale


def _color_reanchor(frames, anchor_mean, anchor_std, strength=0.5, eps=1e-6):
    """Nudge per-frame RGB mean/std toward the chain anchor.

    frames: [F, H, W, C] float in [0,1]. anchor_*: [3] per-channel.
    """
    if anchor_mean is None or anchor_std is None:
        return frames
    rgb = frames[..., :3].float()
    fm = rgb.mean(dim=(1, 2))           # [F, 3]
    fs = rgb.std(dim=(1, 2), unbiased=False)  # [F, 3]
    am = torch.as_tensor(anchor_mean, dtype=torch.float32,
                         device=frames.device).view(1, 3)
    ast = torch.as_tensor(anchor_std, dtype=torch.float32,
                          device=frames.device).view(1, 3)
    matched = (rgb - fm.unsqueeze(1).unsqueeze(1)) / (fs.unsqueeze(1).unsqueeze(1) + eps)
    matched = matched * ast.unsqueeze(1).unsqueeze(1) + am.unsqueeze(1).unsqueeze(1)
    out = rgb * (1.0 - strength) + matched * strength
    frames = torch.cat([out.clamp(0, 1), frames[..., 3:]], dim=-1) if frames.shape[-1] > 3 else out.clamp(0, 1)
    return frames


def _seam_color_match(new_frames, prev_frames, strength=0.65,
                      fade_frames=24, ref_frames=4, eps=1e-6):
    """Affine color match on new segment's head, faded out linearly."""
    if prev_frames is None or prev_frames.shape[0] < 1:
        return new_frames
    prev_n = min(ref_frames, prev_frames.shape[0])
    new_n = min(ref_frames, new_frames.shape[0])
    if prev_n < 1 or new_n < 1:
        return new_frames
    prgb = prev_frames[-prev_n:, ..., :3].float()
    nrgb = new_frames[:new_n, ..., :3].float()
    pm, ps = prgb.mean(dim=(0, 1, 2)), prgb.std(dim=(0, 1, 2), unbiased=False)
    nm, ns = nrgb.mean(dim=(0, 1, 2)), nrgb.std(dim=(0, 1, 2), unbiased=False)
    scale = (ps / (ns + eps)).clamp(0.67, 1.5)
    bias = pm - nm * scale
    fc = min(fade_frames, new_frames.shape[0])
    head = new_frames[:fc, ..., :3].float()
    matched = (head * scale + bias).clamp(0, 1)
    w = torch.linspace(strength, 0.0, fc, device=new_frames.device)
    w = w.view(fc, 1, 1, 1)
    blended = head * (1 - w) + matched * w
    out = new_frames.clone()
    out[:fc, ..., :3] = blended.clamp(0, 1)
    return out


def _compute_color_anchor(frame):
    """Compute per-channel RGB mean/std from a single [H,W,C] or [1,H,W,C] frame."""
    if frame is None:
        return None, None
    f = frame[:1, ..., :3].float() if frame.ndim == 4 else frame[..., :3].float()
    return f.mean(dim=(0, 1, 2)).tolist(), f.std(dim=(0, 1, 2), unbiased=False).tolist()


class SilverH3StagedMultishot:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompts": ("STRING_LIST", {
                "tooltip": "A single LIST of prompt strings - one PER SHOT "
                           "(e.g. the output of the Silver H3 Prompts "
                           "Builder or Prompt List Builder, or a plain "
                           "string turned into a 1-entry list). Every "
                           "prompt is text-encoded in ONE text-encoder "
                           "session before any sampling starts. Empty "
                           "entries are skipped."}),
            "overall_prompt": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "Optional SCENE-WIDE MASTER prompt prepended to "
                           "EVERY shot's prompt before encoding "
                           "(overall + shot). Put the shared scene "
"description here and the per-shot specifics in "
                            "the prompts list."}),
            "ref2v": ("BOOLEAN", {
                "default": False, "label_on": "ref2v (char refs + motion)",
                "label_off": "fl2v (first-to-video)",
                "tooltip": "Continuity MODE (must match your checkpoint):\n"
                           "OFF / fl2v - start_image seeds shot 1 (I2V); shots "
                           "2+ chain on the previous shot's ACTUAL LAST FRAME "
                           "via H3's native first_frame keyframe channel (last "
                           "frame is first frame). Character refs are ignored "
                           "in this mode.\n"
                           "ON / ref2v - start_image is ignored. Shots 2+ "
                           "inject the previous shot's last frame as the FIRST "
                           "reference block (before any char_ref) - the shot "
                           "may re-angle around that anchor. Character refs "
                           "ride on every shot for identity.\n"
                           "BOTH modes carry motion into the next shot via "
                           "latent handoff (head clone + per-step locks + "
                           "taper). Use an fl2v model when OFF, a ref2v/ref2va "
                           "model when ON."}),
            "width": ("INT", {"default": 768, "min": 32, "max": 4096,
                              "step": 32}),
            "height": ("INT", {"default": 1344, "min": 32, "max": 4096,
                               "step": 32}),
            "frames_per_shot": ("INT", {
                "default": 243, "min": 39, "max": 1450, "step": 17,
                "tooltip": "Frames at 24fps on H3's 17k+5 grid "
                           "(243 = ~10.1s per shot). Trained range is "
                           "~124-362; beyond is RoPE-extrapolation territory."}),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "sampler_name": (_sampler_names(), {"default": "res_multistep"}),
            "scheduler": (_scheduler_names(), {"default": "simple"}),
        }, "optional": {
"start_image": ("IMAGE", {
                "tooltip": "fl2v mode ONLY: the FIRST FRAME of shot 1 (I2V). "
                           "Encoded once during the ENCODE phase and handed "
                           "over the way stock Image-to-Video does it - "
                           "vision tokens through the text encoder plus the "
                           "frame-0 keyframe latent. Only shot 1 is seeded "
                           "by this; shots 2+ always chain on the previous "
                           "shot's ACTUAL last frame via the keyframe "
                           "channel (last frame is first frame). IGNORED in "
                           "ref2v mode (the ref2v toggle). In fl2v mode "
                           "character refs are ignored - fl2v and refs are "
                           "contradictory, char refs are DROPPED with a "
                           "warning if given while fl2v."}),
             "latent_handoff_taper": ("FLOAT", {
                "default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0,
                "tooltip": "LATENT handoff bleeding: percent of THIS shot "
                           "that carries the previous "
                           "shot's motion as a linear fade - makes the seam "
                           "CONTINUE (the cut stays invisible); it does NOT "
                           "create a transition. Transitions are "
                           "prompt-authored: if this shot's prompt opens "
                           "with one, drop toward 0-5% so stale motion "
                           "doesn't fight it. 15% of a 10s clip = 1.5s of "
                           "carry-over, a good default. 0 = hard seam. "
                           "Saved videos are always 24 fps."}),
            "audio_onset_guard": ("BOOLEAN", {
                "default": True, "label_on": "lock room tone past the join",
                "label_off": "off",
                "tooltip": "AUDIO onset guard: appends the quietest ~0.4s of "
                           "shot's AUDIO LATENT just past the locked span, "
                           "so the model cannot plan speech under the replay "
                           "- otherwise the free region resumes mid-word and "
                           "the trim chops the line's opening. Latent-domain "
                           "version: no decode involved."}),
            "audio_lock": ("BOOLEAN", {
                "default": True, "label_on": "locked (replay)",
                "label_off": "free (silent-join)",
                "tooltip": "ON: the next shot's audio head is a locked "
                           "replay of the previous tail and the replayed "
                           "span is dropped on decode (symmetric trim). OFF: "
                           "audio generates freely - the new head's audio is "
                           "kept in full and the PREVIOUS tail is trimmed "
                           "instead, so script each shot to land its line "
                           "and hold still for the last beat."}),
            "context_frames": ("INT", {
                "default": 5, "min": 1, "max": 39,
                "tooltip": "Motion context: how many decoded frames from the "
                           "previous shot's tail are used as temporal context "
                           "for the next shot. 1 = single keyframe (stock "
                           "behaviour); 5 = one latent block; 22 = two "
                           "blocks; 39 = three blocks (maximum). Higher values "
                           "give stronger shot-to-shot continuity at the cost "
                           "of one extra VAE encode per join. Snaps to the "
                           "H3 temporal grid."}),
            "pin_renorm": ("BOOLEAN", {
                "default": True, "label_on": "on",
                "label_off": "off",
                "tooltip": "Pin Renorm: normalizes the std of each shot's "
                           "latent tail to match the chain baseline (shot 0). "
                           "Prevents brightness/contrast drift between shots "
                           "at zero cost."}),
            "color_reanchor": ("BOOLEAN", {
                "default": True, "label_on": "on",
                "label_off": "off",
                "tooltip": "Color Re-anchor: stabilizes per-frame RGB "
                           "statistics across shots (nudges each context "
                           "frame's mean/std toward the chain root) and "
                           "applies a seam color match at each cut boundary "
                           "(affine transform faded over 24 frames). Reduces "
                           "exposure/white-balance jumps between shots."}),
            "seed_per_shot": ("BOOLEAN", {
                "default": True, "label_on": "vary per shot",
                "label_off": "same seed every shot",
                "tooltip": "Varying the seed per shot holds identity across "
                           "the chain better than one shared seed."}),
            # APPEND-ONLY: sockets take no widget slot, but keep new inputs at
            # the end anyway so saved graphs stay stable.
            "guide_audio": ("AUDIO", {
                "tooltip": "AUDIO SPINE: a continuous audio track for the "
                           "WHOLE take (a voice recording, a song, a "
                           "low-res long pass). Encoded ONCE in the ENCODE "
                           "phase; each shot's audio stream is then locked "
                           "to its time-slice of the spine at every sampling "
                           "step - the video follows the locked audio (lips "
                           "included) and the voice cannot change between "
                           "shots. Shot stride accounts for the 22-frame "
                           "join trim, so the spine stays in sync across the "
                            "whole piece. Any sample rate; mono is upmixed."}),
"char_refs_all": ("IMAGE", {
                 "tooltip": "Optional CHARACTER REFERENCE list: EVERY image "
                            "here rides EVERY shot (stable <Picture N> "
                            "labels across the whole take). 1..k images, one "
                            "per reference (character, prop, location...). "
                            "Encoded ONCE in the ENCODE phase - image ref "
                            "rows PLUS <Picture N> vision tokens in every "
                            "shot's text conditioning: outfit/face carry "
                            "with NO temporal claim, so it never fights the "
                            "join keyframes. Aspect preserved (downscaled "
                            "toward the generation's pixel area, never "
"stretched). Combined with char_refs_index "
                             "(deduped) when both ride. IGNORED in fl2v "
                             "mode (the ref2v toggle OFF): char refs and "
                             "fl2v keyframes contradict, so fl2v drops "
                             "refs with a warning. In ref2v mode (toggle "
                             "ON) they are the identity anchor, second only "
                             "to the chained last-frame reference."}),
             "char_refs_index": ("IMAGE", {
                 "tooltip": "Optional CHARACTER REFERENCE list INDEXED BY "
                            "SHOT: image i is ridden by SHOT i "
                            "(per-position; a list shorter than the shots "
                            "reuses its last image for the rest). Each shot "
                            "references a DIFFERENT <Picture N>, so the "
                            "labels shift per shot. Leave unconnected to "
                            "disable indexed refs (see char_refs_all for "
                            "refs on every shot)."}),
            "char_audio": ("AUDIO", {
                "tooltip": "Optional CHARACTER VOICE REFERENCE (ref2va). "
                           "Encoded ONCE in the ENCODE phase and ridden by "
                           "EVERY shot as a DiT-side audio ref row plus "
                           "<Audio N> tokens - voice/style identity with no "
                           "timing claim, so it never fights guide_audio "
                           "(the driving spine) or the generated speech. "
"Single track only (one voice at a time). Any "
                            "sample rate; mono is upmixed. IGNORED in fl2v "
                            "mode (toggle OFF): fl2v drops refs with a "
                            "warning."}),
            "filename_prefix": ("STRING", {
                "default": "SilverH3",
                "tooltip": "Output filename prefix. Saved into a dated "
                           "subfolder prefix_yyyymmddHHMMSS/ as "
                           "shot_01.mp4, shot_02, ... plus _master.mp4 "
                           "(the combined take)."}),
            "format": (["mp4", "auto"], {
                "default": "mp4",
                "tooltip": "Container/codec for the saved video. mp4 = H.264 "
                           "in an mp4 container; auto picks the usual "
                           "pairing."}),
            "frame_compression": ("FLOAT", {
                "default": 23.0, "min": 0.0, "max": 51.0,
                "tooltip": "H.264 CRF for the saved video. 0 = lossless "
                           "(huge file), 23 = default good quality."}),
            # ---- output-refine group (same semantics as the Batch node) ----
            "double_pass": ("BOOLEAN", {
                "default": False, "label_on": "refine pass",
                "label_off": "single pass",
                "tooltip": "Second sampling pass over EACH shot's FULL latent "
                           "at a reduced denoise (Motion-Director style "
                           "refine). Re-sampling from the already-sampled "
                           "latent with only the bottom of the sigma "
                           "schedule regenerates fine detail. The shot "
                           "chain (latent handoff / first frame + audio "
                           "lock) is re-applied during the refine so the "
                           "continuity locks survive it. Costs one extra "
                           "DiT sample per shot."}),
            "double_pass_denoise": ("FLOAT", {
                "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                "tooltip": "How much of the first-pass latent is re-noised "
                           "for the refine pass. 0.25 = only the bottom 25% "
                           "of the sigma schedule - a light refine that "
                           "keeps composition, 0.5+ visibly rewrites it, "
                           "1.0 = a full re-roll of that shot."}),
            "double_pass_steps": ("INT", {
                "default": 8, "min": 1, "max": 50,
                "tooltip": "Sampling steps for the refine pass. 8 is plenty "
                           "for a 0.25-denoise refine; raise only when "
                           "denoise is pushed higher or quality demands it."}),
            "upscale_factor": ("FLOAT", {
                "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05,
                "tooltip": "Spatial latent upscale applied to every shot's "
                           "video BEFORE the refine pass / decode: the video "
                           "latent is trilinearly resampled to this factor "
                           "and decoded at the larger size (recommended "
                           "combo: upscale 1.25x + double_pass to fill the "
                           "new detail). 1.0 = native resolution. The "
                           "upscaled latent is decoded at the larger size; "
                           "audio is unaffected. The refine pass's VRAM "
                           "cost scales with the upscaled area. The latent "
                           "handoff chain is kept at native resolution."}),
        },
        "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("master_audio", "saved_video")
    OUTPUT_TOOLTIPS = (
        "Master audio at the audio VAE's rate (per-shot audios assembled "
        "and crossfaded at the seams).",
        "Path of the combined streamed take (_master.mp4) that Phase C "
        "wrote direct to disk, with ONE mp4 per shot saved alongside in "
        "the dated subfolder (each shot streamed out as it decodes - one "
        "shot in RAM at a time - so progress survives a long run and the "
        "shots can be re-combined manually).")
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

    # ---- stream shot tensors into one mp4 per shot + the combined master ----
    class _MP4Stream:
        """Incremental H.264 mp4 writer that holds at most ONE shot's frames.

        Same class the Batch node uses: this node used to assemble the whole
        take's frames into a single master tensor, which ran out of host RAM
        on long takes (every shot's fp16 frames plus the master copy). This
        writer opens the file once and each shot's frames are muxed as they
        are decoded, then freed.

        IMPORTANT (PyAV quirk): the audio stream MUST be registered in
        __init__ - adding it after any video frame has been muxed raises
        "Cannot rebase to zero time". So the assembled master audio dict is
        passed up front and its stream created there; close() just flushes,
        slices the waveform to the total frame count and encodes it.
        """
        def __init__(self, path, frame_rate, audio=None, crf=23.0,
                     bit_depth=8, metadata=None):
            self._path = path
            self._frame_rate = float(frame_rate)
            self._out = av.open(path, **mp4_output_open_kwargs(
                path, VideoContainer.MP4, VideoCodec.H264))
            if metadata:
                # ComfyUI save nodes embed the generating workflow/prompt as
                # container tags (movflags=use_metadata_tags is already in
                # mp4_output_open_kwargs), so the master + per-shot mp4s can be
                # dragged back into the UI or read by prompt-extraction tools.
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
            """Encode + mux one shot's frame tensor ([F,H,W,C], 0..1)."""
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

    # ---- main entry point ------------------------------------------------
    def run(self, model, clip, video_vae, audio_vae, prompts,
            width, height, frames_per_shot, seed, steps,
            sampler_name="res_multistep", scheduler="simple",
            overall_prompt="", ref2v=False,
            start_image=None,
            latent_handoff_taper=15.0, audio_onset_guard=True, audio_lock=True,
            context_frames=5, pin_renorm=True, color_reanchor=True,
            seed_per_shot=True,
            guide_audio=None, char_refs_all=None, char_refs_index=None,
            char_audio=None,
            filename_prefix="SilverH3",
            format="mp4", frame_compression=23.0,
            double_pass=False, double_pass_denoise=0.25,
            double_pass_steps=8, upscale_factor=1.0,
            prompt=None, extra_pnginfo=None):
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
            # compact human duration: 42s / 13m45s / 1h2m30s
            s = int(round(sec))
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            return (f"{h}h{m}m{s}s" if h else
                    f"{m}m{s}s" if m else f"{s}s")

        def _el():
            # elapsed since run start as H:MM:SS, cross-checkable against
            # the log's own line timestamps
            return time.strftime("[+%H:%M:%S]", time.gmtime(time.time() - t_all))
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(p).strip() for p in (prompts or []) if (p or "").strip()]
        if not prompts:
            raise ValueError("No non-empty prompt in the prompts list.")
        if str(overall_prompt or "").strip():
            _ov = str(overall_prompt).strip()
            prompts = [f"{_ov} {p}" for p in prompts]
            print(f"[SilverH3] overall scene prompt ON: prepended to every "
                  f"shot ({len(prompts)} combined).", flush=True)
        n = len(prompts)
        print(f"[SilverH3] staged multishot: {n} prompt(s), "
              f"{frames_per_shot}f @ {width}x{height}, {steps} steps "
              f"{sampler_name}/{scheduler}.", flush=True)

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
                print(f"[SilverH3] {tag}: VRAM swept "
                      f"({b4:.1f} -> {af:.1f} GB free).", flush=True)
            except Exception as e:
                print(f"[SilverH3] {tag}: sweep failed ({e}) - continuing.",
                      flush=True)

        # ==============================================================
        # PHASE A - ENCODE: every prompt through ONE text-encoder session
        # ==============================================================
        t0 = time.time()
        conds = []
        kf_latent = None
        frame_count = mmh3.temporal_shape(frames_per_shot)[0]
        # continuity SELECTED BY THE ref2v TOGGLE (no longer by the inputs):
        #   OFF / fl2v  (_FF=True):   start_image seeds shot 1 (I2V); shots 2+
        #                             chain on the previous shot's ACTUAL LAST
        #                             FRAME via H3's native first_frame keyframe
        #                             channel (last frame is first frame).
        #                             Character refs are ignored (dropped).
        #   ON  / ref2v (_REF2V=True): start_image is ignored. Shots 2+ inject
        #                             the previous shot's last frame as the
        #                             FIRST minimax_refs block (before any
        #                             char_ref); the shot may re-angle around
        #                             that anchor. Character refs ride every
        #                             shot for identity.
        # BOTH chained path turns off in-mode the previous "refs overwrite the
        # keyframe" contradiction (model_base.py:2173): in fl2v the refs are
        # dropped, in ref2v the chain frame is itself a ref so it coexists with
        # char refs and nothing is wiped.
        _REF2V = bool(ref2v)
        _REFS_RIDE = (char_refs_all is not None
                         or char_refs_index is not None
                         or char_audio is not None)
        _DROP_REFS = _REFS_RIDE and not _REF2V      # fl2v drops the char refs
        if _DROP_REFS:
            _what = ("character references are dropped (fl2v uses the "
                     "keyframe channel; refs and keyframes contradict via "
                     "model_base.py refs-overwrite) and the chain runs from "
                     "start_image+last-frame instead")
            print(f"[SilverH3] ref2v toggle is OFF (fl2v): {_what}.",
                  flush=True)
        if _REFS_RIDE and _REF2V and start_image is not None:
            print("[SilverH3] ref2v toggle ON + char refs given: start_image "
                  "is IGNORED (ref2v chains via the last-frame reference, "
                  "not the first-frame keyframe); char refs ride.", flush=True)
        _FF = not _REF2V    # fl2v always uses the keyframe channel
        # join trim in FRAMES, applied to the head of every shot AFTER the
        # first in Phase C (video AND audio both trimmed, so they stay
        # sample-aligned). Both modes clone the previous tail's latent head
        # into shot 2+ (motion carry), so the trim is the latent_handoff
        # geometry: the cloned rows (rows 1..4 = 17*4 = 68 frames) are
        # trimmed; row 0 (5 frames, the keyframe/first-frame anchor) is
        # PRESERVED so the visual continuity pin survives into the master.
        _hok = min(_HO_ROWS, frame_count - 1)
        _HOF = (17 * (_hok - 1)) if _hok > 1 else 0
        _TRIM = _HOF
        # join bleed length: latent_handoff_taper is a PERCENT of this shot's
        # duration (10% of a 10s clip = 1s). Every latent row covers ~4
        # decoded frames in the steady region, so percent -> rows = /4.
        _TR = int(round(float(frame_count)
                        * (float(latent_handoff_taper) / 100.0))) // 4
        # motion context: how many decoded tail frames to pass as temporal
        # conditioning for the next shot (snapped to H3 grid).
        _ctx_span = _select_context_span(context_frames, frame_count - 1)
        # motion context state: updated each shot
        _ctx_latent_cache = None   # cached AV-latent tail for next shot (Phase 2)
        _ctx_pixel_cache = None    # decoded pixel tail for next shot
        _pin_baseline = None       # pin renorm baseline std (set on shot 0)
        _color_anchor_m = None     # chain-root RGB mean
        _color_anchor_s = None     # chain-root RGB std
        # character reference LISTS: encoded ONCE here in the ENCODE phase while
        # the video VAE is hot (encoding after the phase sweep would force a
        # VAE reload mid-SAMPLE). Stock ref2va resize policy: aspect-preserving
        # downscale toward the generation's pixel area, never stretched onto
        # the canvas grid. char_refs_all fills the head (rides EVERY shot);
        # char_refs_index is indexed by SHOT (its offset starts after the
        # all-set). Gated on the start_image drop so a dropped sheet is never
        # VAE-encoded.
        _char_refs = []        # dit-side image ref blocks, one per image
        _char_px = []          # TE-side <Picture N> presentation pixels
        _n_all, _n_idx = 0, 0
        def _enc_sheet(imgs, kind):
            nonlocal _n_all, _n_idx
            if imgs is None:
                return 0
            _kI = int(imgs.shape[0])
            for _j in range(_kI):
                _im = imgs[_j:_j + 1]
                ch_, cw_ = int(_im.shape[1]), int(_im.shape[2])
                scale = min(1.0, ((width * height) / float(cw_ * ch_)) ** 0.5)
                _cm = getattr(mmh3, "CANVAS_MULTIPLE", 32)
                tw = max(_cm, round(cw_ * scale / _cm) * _cm)
                th = max(_cm, round(ch_ * scale / _cm) * _cm)
                _cimg = mmh3._resize(_im, tw, th, "disabled")
                _cz = video_vae.encode(_cimg).detach()
                _char_refs.append({
                    "kind": "image",
                    "latent_h": _cz.shape[-2],
                    "latent_w": _cz.shape[-1],
                    "latent": _cz.cpu(),
                })
                _char_px.append(_cimg.detach().cpu())
            if kind == "all":
                _n_all = _kI
            else:
                _n_idx = _kI
            return _kI
        if not _DROP_REFS:
            _enc_sheet(char_refs_all, "all")
            _enc_sheet(char_refs_index, "index")
            if bool(_char_refs):
                print(f"[SilverH3] character references ON: {_n_all} "
                      f"apply-to-all + {_n_idx} indexed image(s) -> ref "
                      f"rows + <Picture N> vision tokens.", flush=True)
        # per-SHOT character-image subsets, decided ONCE here so the ENCODE
        # tokens and the SAMPLE ref rows agree. char_refs_all: every image
        # rides every shot. char_refs_index: image i rides shot i. Both can
        # ride at once (deduped).
        _ref_idx = [[] for _ in range(n)]
        if not _DROP_REFS:
            for _si in range(n):
                _idxs = []
                if _n_all:
                    _idxs += list(range(_n_all))
                if _n_idx:
                    _idxs.append(_n_all + min(_si, _n_idx - 1))
                _seen = set()
                _ref_idx[_si] = [_x for _x in _idxs
                                 if not (_x in _seen or _seen.add(_x))]
        # character VOICE reference: encoded once, AFTER the prompt loop so
        # the audio VAE load does not evict the text encoder mid-ENCODE (the
        # spine below rides the same audio-VAE session). Single track [:1],
        # reused by every shot as an audio ref row + <Audio N> tokens.
        _char_audio = None       # dit-side audio ref block latent [1,32,2,T]
        _char_audio_t = 0        # ref_audio_t: audio columns
        for i, text in enumerate(prompts):
            ref_items = []    # TE presentation, request order
            if _FF and i == 0 and start_image is not None:
                img = mmh3._resize(start_image[:1], width, height, "disabled")
                kf_latent = [{"resolved_frame_index": 0,
                              "latent": video_vae.encode(img)}]
                text = _ALIGN_LINE + text
                # presented via ref_items (not images=) because the two
                # channels are exclusive in the tokenizer and shot 1 may
                # need BOTH pictures; start stays Picture 1 either way
                ref_items.append({"type": "image", "data": img})
                print("[SilverH3] I2V: shot 1 opens on the supplied image "
                      "(vision tokens + frame-0 keyframe latent).", flush=True)
            if _ref_idx[i]:
                for _j in _ref_idx[i]:
                    ref_items.append({"type": "image",
                                      "data": _char_px[_j]})
            if _char_audio is not None:
                # char_audio was dropped in Phase A whenever start_image
                # keyframes ride, so no keyframe-wipe guard is needed here
                ref_items.append({"type": "audio"})
            tokens = (clip.tokenize(text, minimax_ref_items=ref_items)
                      if ref_items else clip.tokenize(text))
            cond = clip.encode_from_tokens_scheduled(tokens)
            if i == 0 and kf_latent is not None:
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": kf_latent,
                    "minimax_frame_count": frame_count,
                })
            conds.append(cond)
        del _char_px   # TE presentation pixels not needed past ENCODE
        print(f"[SilverH3] ENCODE done: {n} conditionings in one TE session "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)

        # audio spine: encoded ONCE here, still inside the ENCODE phase, so
        # the swap count stays at 2 for the whole run
        _spine = None
        _AFPS = float(getattr(mmh3, "AUDIO_LATENT_FPS", 40))
        if guide_audio is not None:
            _w = guide_audio["waveform"]
            _sr = int(guide_audio["sample_rate"])
            _w3 = _w if _w.ndim == 3 else _w.unsqueeze(0)
            _vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if _sr != _vae_sr:
                import torchaudio
                _w3 = torchaudio.functional.resample(_w3, _sr, _vae_sr)
                print(f"[SilverH3] audio spine: resampled {_sr} -> "
                      f"{_vae_sr} Hz.", flush=True)
            if _w3.shape[1] == 1:
                _w3 = _w3.repeat(1, 2, 1)      # encoder wants two channels
            elif _w3.shape[1] > 2:
                _w3 = _w3[:, :2]
            _spine = audio_vae.encode(
                _w3[:1].movedim(1, -1)).detach()
            _guide_s = _w3.shape[-1] / _vae_sr
            # tile the spine if it is shorter than the total run so every
            # shot gets a full-length lock instead of partial/exhausted
            _a_lat_w = mmh3.temporal_shape(frames_per_shot)[1]
            _s_cols = (frames_per_shot - _TRIM) / 24.0 * _AFPS
            _total_a = int(round(_a_lat_w + (n - 1)
                                 * (_a_lat_w - round(_s_cols * 33.0
                                                     / 451.0))))
            if _spine.shape[-1] < _total_a:
                _tiles = -(-_total_a // _spine.shape[-1])
                _spine = _spine.repeat(1, 1, _tiles)[..., :_total_a]
                print(f"[SilverH3] audio spine: {_a_lat_w} cols "
                      f"({_guide_s:.1f}s guide) tiled x{_tiles} -> "
                      f"{_spine.shape[-1]} cols for {n} shots.",
                      flush=True)
            else:
                print(f"[SilverH3] audio spine: {_spine.shape[-1]} cols "
                      f"({_guide_s:.1f}s) - every shot's audio is locked "
                      f"to its time-slice of it, so the voice cannot "
                      f"change between shots.", flush=True)
        # character VOICE reference (same audio-VAE session as the spine)
        if char_audio is not None and not _DROP_REFS:
            _w = char_audio["waveform"]
            _asr = int(char_audio["sample_rate"])
            _w3 = _w if _w.ndim == 3 else _w.unsqueeze(0)
            _avr = getattr(audio_vae, "audio_sample_rate", 32000)
            if _asr != _avr:
                import torchaudio
                _w3 = torchaudio.functional.resample(_w3, _asr, _avr)
                print(f"[SilverH3] character voice: resampled {_asr} -> "
                      f"{_avr} Hz.", flush=True)
            if _w3.shape[1] == 1:
                _w3 = _w3.repeat(1, 2, 1)      # encoder wants two channels
            elif _w3.shape[1] > 2:
                _w3 = _w3[:, :2]
            _ca = audio_vae.encode(_w3[:1].movedim(1, -1)).detach().cpu()
            _char_audio = _ca
            _char_audio_t = int(_ca.shape[-1])
            print(f"[SilverH3] character VOICE reference ON: "
                  f"{_char_audio_t} audio cols "
                  f"({_char_audio_t / _AFPS:.1f}s), rides every shot as an "
                  f"audio ref row + <Audio N> tokens.", flush=True)
        _evict("ENCODE -> SAMPLE")

        # ==============================================================
        # PHASE B - SAMPLE: DiT stays hot; chains pass through LATENT space
        # ==============================================================
        ms_obj = model.get_model_object("model_sampling")
        try:
            _asc = float(getattr(ms_obj, "audio_scale", 1.0) or 1.0)
        except Exception:
            _asc = 1.0
        _tss = None
        _shv, _sha = 12.0, 3.0
        try:
            from comfy.ldm.minimax.model import time_shift_sigma as _tss
            _dm = getattr(getattr(model, "model", None),
                          "diffusion_model", None)
            _shv = float(getattr(_dm, "sigma_shift_video", 12.0))
            _sha = float(getattr(_dm, "sigma_shift_audio", 3.0))
        except Exception:
            _tss = None
        if abs(_asc - 1.0) > 1e-6:
            print(f"[SilverH3] audio injections scaled x{_asc:.3f} onto the "
                  f"sampler's audio domain (ModelSamplingAV).", flush=True)

        def _arm(guider, ho_v, ho_a, ho_guard, ho_tsrc, release, taper,
                 vshape, ashape, spine_seg=None):
            """Lock the previous shot's tail latents into THIS trajectory."""
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
                # comfy packs the nested AV latent into flat [B,1,N] before
                # sampling - the lock is written through reshaped views.
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
                            if _DBG and not _st["logged"]:
                                _wv = xv[:, :, _HO_R0:_HO_R0
                                         + tv.shape[2]]
                                print(f"[SilverH3][dbg] inject: "
                                      f"x{tuple(x.shape)} vs Nv+Na="
                                      f"{_Nv + _Na} wrote "
                                      f"rows[{_HO_R0}:"
                                      f"{_HO_R0 + tv.shape[2]}] "
                                      f"wmean={_wv.float().mean():.4f} "
                                      f"wstd={_wv.float().std():.4f} | "
                                      f"bootstrap std="
                                      f"{xv[:, :, :_HO_R0].float().std():.4f}",
                                      flush=True)
                            if _tp > 0 and _ts is not None:
                                # graded taper: rows after the hard lock are
                                # biased toward the previous motion at
                                # linearly decaying strength, so the pose has
                                # a ramp instead of a cliff
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
                            # spine mode: the WHOLE audio stream is a locked
                            # slice of one continuous track - nothing left
                            # for the model to plan. Injected at the AUDIO
                            # clock (the two-clocks scaling below).
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
                                print(f"[SilverH3] SPINE inject: "
                                      f"{_nn}/{_as[-1]} audio cols "
                                      f"from spine[{0}:{_nn}] "
                                      f"(guide {0}/{_hs.shape[-1]} cols), "
                                      f"sigma={sig:.3f} "
                                      f"audio_sigma={sa:.3f} "
                                      f"asc={_asc:.2f}.", flush=True)
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
                            print(f"[SilverH3] latent handoff ACTIVE "
                                  f"(sigma {sig:.3f}): {_hv.shape[2] if _hv is not None else 0}"
                                  f" video rows + "
                                  f"{(_hs.shape[-1] if _hs is not None else (_ha.shape[-1] if _ha is not None else 0))}"
                                  f" audio cols locked"
                                  f"{' (SPINE)' if _hs is not None else ''}"
                                  f"{' + guard' if _hg is not None else ''}"
                                  f", release below {_rel:.2f}.", flush=True)
                    except Exception as e:
                        print(f"[SilverH3] handoff injection FAILED: {e!r}",
                              flush=True)
                return _o(x, timestep, model_options=model_options,
                          seed=seed)

            guider.predict_noise = _pn

        def _latent_guard(a_lat):
            """Quietest ~0.4s window of the audio latent = room-tone guard."""
            try:
                e = a_lat.detach().float().pow(2).mean(dim=(0, 1, 2))
                t = e.shape[0]
                if t < _GUARD_COLS * 2:
                    return None
                cs = torch.cumsum(torch.cat([torch.zeros(1, device=e.device),
                                             e]), 0)
                k = int((cs[_GUARD_COLS:] - cs[:-_GUARD_COLS]).argmin())
                return a_lat[..., k:k + _GUARD_COLS].detach().clone()
            except Exception:
                return None

        stored = []          # {"video","audio","trim","alocked"} per shot
        ho_v = ho_a = ho_guard = ho_tsrc = None
        _chain_img = None    # both modes decode prev shot's last frame for anchor
        if _FF:
            print(f"[SilverH3] continuity: fl2v (keyframe) - shots 2+ chain "
                  f"on the previous shot's ACTUAL last frame via H3's native "
                  f"keyframe channel; latent handoff carries motion "
                  f"({_TRIM} decoded-frame head trimmed per join).",
                  flush=True)
        else:
            print(f"[SilverH3] continuity: ref2v (reference) - shots 2+ "
                  f"inject the previous shot's last frame as the FIRST "
                  f"reference block; latent handoff carries motion "
                  f"({_TRIM} decoded-frame head trimmed per join).",
                  flush=True)
        t_sample = 0.0
        for si, cond in enumerate(conds):
            latent, _fc = mmh3._empty_av_latent(width, height,
                                                frames_per_shot)
            comps = latent["samples"].unbind()
            vshape = tuple(comps[0].shape)
            ashape = tuple(comps[1].shape) if len(comps) > 1 else None
            if si > 0 and stored[si - 1]["video"] is not None:
                import comfy.nested_tensor
                _dev = mm.get_torch_device()
                _pv = stored[si - 1]["video"]
                _pa = stored[si - 1]["audio"]
                _nv = comps[0].clone().to(_dev)
                _na = comps[1].clone().to(_dev) if len(comps) > 1 else None
                _tv = min(_HO_ROWS, _pv.shape[2], _nv.shape[2])
                _nv[:, :, :_tv] = _pv[:, :, -_tv:].to(_dev)
                if _pa is not None and _na is not None:
                    _ta = min(_HO_ACOLS, _pa.shape[-1], _na.shape[-1])
                    _na[..., :_ta] = _pa[..., -_ta:].to(_dev)
                latent = {"samples": comfy.nested_tensor.NestedTensor(
                    (_nv, _na) if _na is not None else (_nv,))}
                comps = latent["samples"].unbind()
                print(f"[SilverH3] latent_handoff: shot {si+1} seeded "
                      f"({_tv}V + {_ta if _pa is not None else 0}"
                      f"A head rows from prev tail).", flush=True)
            if _FF and si > 0 and _chain_img is not None:
                # fl2v: native keyframe channel - the model consumes this
                # encoded frame as ground truth; vision-token re-encoding
                # would need a TE session per join, which would break
                # one-shot TE staging. The keyframe is anchored at the FIRST
                # FRAME THAT SURVIVES THE JOIN TRIM (_TRIM), not frame 0:
                # frame 0 is inside the discarded overlap head, so a frame-0
                # keyframe never reaches the visible output. Placing it at
                # _TRIM makes the shot open exactly on the previous shot's
                # last frame (a crisp match-cut resume). Falls back to frame
                # 0 (stock behaviour) if the layout patch is unavailable.
                img = mmh3._resize(_chain_img[:1], width, height, "disabled")
                kf = {
                    "resolved_frame_index": 0,
                    "latent": video_vae.encode(img),
                }
                if _mc_layout_ok:
                    kf[_MC_KEY] = _TRIM
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": [kf],
                    "minimax_frame_count": _fc,
                })
            ref_blocks = []
            # ref2v: the previous shot's last frame becomes the FIRST
            # reference block - inserted before char refs so the model sees
            # the scene anchor (same canvas, can re-angle) before picking up
            # the stable <Picture N> identity set. This is the ref2v
            # equivalent of the fl2v keyframe channel above.
            if (not _FF) and si > 0 and _chain_img is not None:
                _cimg = mmh3._resize(_chain_img[:1], width, height,
                                     "disabled")
                _cz = video_vae.encode(_cimg)
                ref_blocks.append({
                    "kind": "image",
                    "latent_h": _cz.shape[-2],
                    "latent_w": _cz.shape[-1],
                    "latent": _cz.to(mm.get_torch_device()),
                })
                del _cimg, _cz
                print(f"[SilverH3] ref2v: shot {si+1} chains via prev last "
                      f"frame as FIRST reference (before char refs).",
                      flush=True)
            # fl2v: refs are always dropped (_DROP_REFS in Phase A made
            # _ref_idx/_char_refs empty, so nothing is appended here even
            # if _refs_ok were true). ref2v: char refs ride on every shot.
            _refs_ok = not _FF
            if _refs_ok and _ref_idx[si]:
                for _j in _ref_idx[si]:
                    ref_blocks.append({
                        "kind": "image",
                        "latent_h": _char_refs[_j]["latent_h"],
                        "latent_w": _char_refs[_j]["latent_w"],
                        "latent": _char_refs[_j]["latent"].to(
                            mm.get_torch_device()),
                    })
            if _refs_ok and _char_audio is not None:
                ref_blocks.append({
                    "kind": "audio",
                    "ref_audio_t": _char_audio_t,
                    "audio_latent": _char_audio.to(
                        mm.get_torch_device()),
                })
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(
                    cond, {"minimax_refs": ref_blocks})
            # this shot's time-slice of the spine: shots advance by
            # (frames_per_shot - join trim) in output time. A spine shorter
            # than the run degrades gracefully: partial coverage locks the
            # covered prefix, exhausted coverage improvises freely.
            _spine_seg = None
            if _spine is not None and ashape is not None:
                _a0 = int(round(si * (frames_per_shot - _TRIM)
                                / 24.0 * _AFPS))
                _a0 = max(0, min(_a0, max(0, _spine.shape[-1] - 1)))
                _spine_seg = _spine[..., _a0:_a0 + ashape[-1]]
                print(f"[SilverH3] shot {si+1}: spine_seg cols "
                      f"{_spine_seg.shape[-1]}/{ashape[-1]} "
                      f"(guide[{_a0}:{_a0 + ashape[-1]}] of "
                      f"{_spine.shape[-1]}).", flush=True)
            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            armed = False
            if _spine_seg is not None or si > 0:
                # both modes: the head was CLONED from the prev tail at init
                # (rows 0-4 = the first 17-frame block; the 2 bootstrap rows
                # are left to breathe). The per-step lock re-asserts that
                # tail through the structure phase (sigma > release) so the
                # identity and motion are CARRIED, not just seeded once.
                _arm(guider,
                     ho_v,                 # per-step video lock, rows [2:7]
                     None if _spine_seg is not None else ho_a,
                     None if _spine_seg is not None else ho_guard,
                     ho_tsrc, _HO_RELEASE, _TR,
                     vshape, ashape, spine_seg=_spine_seg)
                armed = True
            shot_seed = (seed + si) if seed_per_shot else seed
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            t0 = time.time()
            out, _dup = ncs.SamplerCustomAdvanced().sample(
                noise, guider, sampler, sigmas, latent)
            dt = time.time() - t0
            t_sample += dt
            lat = out["samples"]
            parts = (list(lat.unbind()) if getattr(lat, "is_nested", False)
                     else [lat])
            v = parts[0]
            a = parts[1] if len(parts) > 1 else None
            stored.append({
                "video": v.detach().cpu(),
                "audio": a.detach().cpu() if a is not None else None,
                "trim": (_TRIM if si > 0 else 0),
                "alocked": (_spine_seg is not None or audio_lock),
            })
            # ---- output-refine: UPSCALE then REFINE, same order as the
            # Batch node. The stored "video"/"audio" (what seeds the NEXT
            # shot's handoff head) stays at NATIVE resolution - the chained
            # pipeline must never see a spatial change - while the final
            # latent (post-upscale / post-refine) is kept separately in
            # "video_final" for Phase C's decode and the chain capture.
            _up = float(upscale_factor)
            if _up > 1.0:
                _lat = out["samples"]
                _parts = (list(_lat.unbind()) if getattr(_lat, "is_nested", False)
                          else [_lat])
                _v = _parts[0]
                _nw = max(1, round(_v.shape[-1] * _up))
                _nh = max(1, round(_v.shape[-2] * _up))
                _v = F.interpolate(_v, size=(_v.shape[-3], _nh, _nw),
                                   mode="trilinear", align_corners=False)
                vshape = tuple(_v.shape)
                if len(_parts) > 1:
                    out["samples"] = comfy.nested_tensor.NestedTensor(
                        (_v, _parts[1]))
                else:
                    out["samples"] = _v
                print(f"[SilverH3] shot {si + 1}: video latent upscaled "
                      f"{_v.shape[-2]}x{_v.shape[-1]} (x{_up:.2f}); audio "
                      f"unchanged.", flush=True)
                del _lat, _parts, _v
            if double_pass:
                _sigmas2 = ncs.BasicScheduler().get_sigmas(
                    model, scheduler, double_pass_steps,
                    double_pass_denoise)[0]
                _noise2 = ncs.RandomNoise().get_noise(shot_seed + 10000000)[0]
                _guider2 = ncs.BasicGuider().get_guider(model, cond)[0]
                # the SAME lock rides the refine so it cannot melt the
                # audio/video continuity the first pass just established.
                if armed:
                    _hv_lock = ho_v
                    _ts_lock = ho_tsrc
                    if (_up > 1.0 and _hv_lock is not None
                            and tuple(_hv_lock.shape[-2:])
                            != tuple(vshape[3:])):
                        _NAR = (_hv_lock.shape[-3],
                                vshape[-2], vshape[-1])
                        _hv_lock = F.interpolate(
                            _hv_lock, size=_NAR, mode="trilinear",
                            align_corners=False)
                        if (_ts_lock is not None
                                and tuple(_ts_lock.shape[-2:])
                                != tuple(vshape[3:])):
                            _tsNAR = (_ts_lock.shape[-3],
                                      vshape[-2], vshape[-1])
                            _ts_lock = F.interpolate(
                                _ts_lock, size=_tsNAR, mode="trilinear",
                                align_corners=False)
                        print(f"[SilverH3] shot {si + 1}: refine lock "
                              f"rescaled to {vshape[-2]}x{vshape[-1]}.",
                              flush=True)
                    _arm(_guider2, _hv_lock,
                         None if _spine_seg is not None else ho_a,
                         None if _spine_seg is not None else ho_guard,
                         _ts_lock, _HO_RELEASE, _TR,
                         vshape, ashape, spine_seg=_spine_seg)
                _dt2 = time.time()
                out, _dup = ncs.SamplerCustomAdvanced().sample(
                    _noise2, _guider2, sampler, _sigmas2, out)
                print(f"[SilverH3] shot {si + 1}: refine pass done - "
                      f"{double_pass_steps} steps @ "
                      f"denoise {double_pass_denoise:.2f} "
                      f"({_dur(time.time() - _dt2)}).", flush=True)
                del _sigmas2, _noise2, _guider2
            if _up > 1.0 or double_pass:
                _lat = out["samples"]
                _parts = (list(_lat.unbind()) if getattr(_lat, "is_nested", False)
                          else [_lat])
                _vf = _parts[0]
                _af = _parts[1] if len(_parts) > 1 else None
                stored[-1]["video_final"] = _vf.detach().cpu()
                stored[-1]["audio"] = (_af.detach().cpu()
                                       if _af is not None else None)
                del _lat, _parts, _vf, _af
            # ---- capture for the NEXT shot (both modes need it) ----
            # Chain frame: decoded tail for keyframe (fl2v) or ref (ref2v).
            # For multi-frame context we decode the full tail window, not
            # just the last frame. The latent tail is cached for Phase 2's
            # AV-latent path (skips VAE re-encode on subsequent runs).
            _video_lat = stored[-1].get("video_final",
                                 stored[-1]["video"])
            _n_tail = min(_ctx_span, int(_video_lat.shape[2]))
            _tail_lat = _video_lat[:, :, -_n_tail:].to(
                mm.get_torch_device())
            _clean = video_vae.decode(_tail_lat)
            if _clean.ndim == 5:
                _clean = _clean.reshape(-1, _clean.shape[-3],
                                        _clean.shape[-2],
                                        _clean.shape[-1])
            _ctx_pixel_cache = _clean.detach().cpu()   # [F, H, W, C]
            _chain_img = _clean[-1:].detach().cpu()     # last frame for ref2v
            # Phase 2: cache the raw latent tail (skip VAE re-encode later)
            _ctx_latent_cache = {"samples": _tail_lat.detach().cpu()}
            # Phase 2: pin renorm — normalize latent tail std to chain baseline
            if pin_renorm:
                _lat_v = _ctx_latent_cache["samples"]
                _lat_v, _cur_std, _scale = _pin_renorm(
                    _lat_v, _pin_baseline)
                _ctx_latent_cache["samples"] = _lat_v
                if _pin_baseline is None:
                    _pin_baseline = _cur_std
                    print(f"[SilverH3] pin renorm: baseline std="
                          f"{_cur_std:.4f} (shot {si + 1}).", flush=True)
                elif abs(_scale - 1.0) > 0.001:
                    print(f"[SilverH3] pin renorm: shot {si + 1} std="
                          f"{_cur_std:.4f} -> scale={_scale:.4f} "
                          f"(baseline={_pin_baseline:.4f}).", flush=True)
            del _tail_lat, _video_lat
            # Phase 3: color re-anchor — nudge context frames toward chain root
            if color_reanchor and _ctx_pixel_cache is not None:
                if _color_anchor_m is None:
                    _color_anchor_m, _color_anchor_s = _compute_color_anchor(
                        _ctx_pixel_cache[-1:])
                    if _color_anchor_m is not None:
                        print(f"[SilverH3] color anchor: mean="
                              f"[{','.join(f'{x:.3f}' for x in _color_anchor_m)}] "
                              f"std=[{','.join(f'{x:.3f}' for x in _color_anchor_s)}] "
                              f"(shot {si + 1}).", flush=True)
                _ctx_pixel_cache = _color_reanchor(
                    _ctx_pixel_cache, _color_anchor_m, _color_anchor_s)
            # Hand-off state for the NEXT shot: raw tail latents, taken
            # exactly as sampled - the latent-space frame pass that carries
            # motion through the seam (both modes).
            ho_v = _ctx_latent_cache["samples"][:, :, -_HO_ROWS:].clone()
            ho_a = (a[..., -_HO_ACOLS:].detach().clone()
                    if a is not None and audio_lock else None)
            ho_guard = (_latent_guard(a)
                        if a is not None and audio_lock and audio_onset_guard
                        else None)
            ho_tsrc = (v[:, :, -1:].detach().clone()
                       if _TR > 0 else None)
            if _DBG:
                _hv = _ctx_latent_cache["samples"]
                print(f"[SilverH3][dbg] capture: nested="
                      f"{getattr(lat, 'is_nested', False)} "
                      f"parts={len(parts)} vshape={tuple(v.shape)} "
                      f"ashape={None if a is None else tuple(a.shape)} "
                      f"ctx_frames={int(_ctx_pixel_cache.shape[0])}",
                      flush=True)
                print(f"[SilverH3][dbg] ho_v mean="
                      f"{_hv.float().mean():.4f} "
                      f"std={_hv.float().std():.4f} "
                      f"min={_hv.float().min():.3f} "
                      f"max={_hv.float().max():.3f} "
                      f"nan={torch.isnan(_hv).any().item()} "
                      f"inf={torch.isinf(_hv).any().item()}", flush=True)
            del out, latent, lat, parts, v, a
            # NO soft_empty_cache() here: it runs torch.cuda.empty_cache(),
            # which on the DynamicVRAM path releases the comfy_aimdo arena's
            # warm weight pages - the next shot then pays a full re-map
            # ("model initializing" + ~1min first step). The caching
            # allocator reuses freed activation blocks on its own.
            print(f"[SilverH3] shot {si + 1}/{n} sampled in {_dur(dt)} "
                  f"{_el()} "
                  f"({'handoff armed' if armed else 'fresh take'}).",
                  flush=True)
        del ho_v, ho_a, ho_guard, ho_tsrc
        print(f"[SilverH3] SAMPLE done: {n} shot(s) back-to-back, DiT loaded "
              f"ONCE ({_dur(t_sample)} total sampling) {_el()}.", flush=True)
        _evict("SAMPLE -> DECODE")

        # ==============================================================
        # PHASE C - DECODE + SAVE: VAEs hot, one mp4 per shot + master streamed
        # ==============================================================
        t0 = time.time()
        # ---- pre-pass: decode every per-shot AUDIO first. Audio latents
        # are tiny, but the master writer must register its aac stream up
        # front (PyAV cannot add an audio stream after video frames are
        # already muxed), so the assembled master waveform is complete
        # before a single video frame is muxed.
        sr = None
        xfade = max(1, int((sr or 32000) * 40 / 1000.0))
        audio_parts = []       # per-shot {"waveform","sample_rate"} or None
        for s in stored:
            wav = None
            _sr = None
            if s["audio"] is not None:
                aud = vae_decode_audio(audio_vae, {"samples": s["audio"]})
                _sr = aud["sample_rate"]
                sr = _sr
                wav = aud["waveform"].detach().cpu()
                if wav.ndim == 2:
                    wav = wav.unsqueeze(0)
                xfade = max(1, int(_sr * 40 / 1000.0)) if _sr else xfade
            audio_parts.append({"waveform": wav, "sample_rate": _sr})

        # the join-trim math needs each shot's decoded frame count; every
        # shot shares the same sampled length, so decode the FIRST one up
        # front to learn it (one shot in RAM - the streaming budget allows
        # exactly that). Then replay the same trim logic the old decode
        # loop used, but on the pre-passed audio so the master waveform is
        # ready before the writer opens.
        first_imgs = video_vae.decode(
            stored[0].get("video_final",
                          stored[0]["video"])).detach().cpu().half()
        if first_imgs.ndim == 5:
            first_imgs = first_imgs.reshape(-1, first_imgs.shape[-3],
                                            first_imgs.shape[-2],
                                            first_imgs.shape[-1])
        _F = int(first_imgs.shape[0])
        _last_aud = -1    # most recent NON-silent shot, for the join math
        for si in range(len(stored)):
            if audio_parts[si]["waveform"] is None:
                continue
            _sr = audio_parts[si]["sample_rate"]
            if si > 0:
                ov = min(stored[si]["trim"], _F - 1)
                if stored[si].get("alocked"):
                    # symmetric trim: the head audio is a replay of the
                    # previous tail - drop it with its frames, weld-
                    # compensated so any residual step lands together
                    keep = int(round(_sr * ov / 24.0)) - xfade
                    if keep > 0:
                        audio_parts[si]["waveform"] = \
                            audio_parts[si]["waveform"][..., keep:]
                elif _last_aud >= 0:
                    # silent-join: keep the new head in full, trim the
                    # PREVIOUS audio part instead (+100ms sqrt fade-out)
                    cut = int(round(_sr * ov / 24.0)) - xfade
                    prev = audio_parts[_last_aud]["waveform"]
                    if cut > 0 and prev.shape[-1] > cut:
                        fn = min(int(_sr * 0.10), prev.shape[-1])
                        if fn > 8:
                            fade = (torch.linspace(1.0, 0.0, fn) ** 0.5
                                    ).to(prev.dtype)
                            prev[..., -fn:] = prev[..., -fn:] * fade
                        audio_parts[_last_aud]["waveform"] = prev[..., :-cut]
            _last_aud = si

        # assemble the master waveform (cosine crossfade at the seams). Seed on
        # the FIRST non-silent part and merge only from the one AFTER it -
        # looping from [1:] would crossfade the seed against ITSELF when a
        # leading shot is silent
        waveform = None
        _seed = -1
        for _i, p in enumerate(audio_parts):
            if p["waveform"] is not None:
                waveform = p["waveform"]
                _seed = _i
                break
        if waveform is not None:
            for p in audio_parts[_seed + 1:]:
                nxt = p["waveform"]
                if nxt is None:
                    continue
                k = min(xfade, waveform.shape[-1], nxt.shape[-1])
                if k < 8:
                    waveform = torch.cat([waveform, nxt], dim=-1)
                    continue
                t = torch.linspace(0, 1, k, dtype=waveform.dtype)
                seam = (waveform[..., -k:] * torch.cos(t * 3.14159265 / 2)
                        + nxt[..., :k] * torch.sin(t * 3.14159265 / 2))
                waveform = torch.cat([waveform[..., :-k], seam,
                                      nxt[..., k:]], dim=-1)

        master_audio = ({"waveform": waveform, "sample_rate": sr}
                        if waveform is not None and sr else None)
        audio_out = (master_audio if master_audio is not None else
                     {"waveform": torch.zeros(1, 1, 1),
                      "sample_rate": 32000})

        # ---- streamed save: ONE mp4 per shot + the combined master,
        # one shot in RAM. Mirrors the Batch node's save process: each shot
        # is written to its OWN clip file in the dated subfolder as it
        # decodes (so a very long take's progress survives even if a later
        # shot fails), and the same trimmed frames stream into the
        # still-open master. Only one shot's frames are ever resident.
        # workflow/prompt metadata, embedded in every mp4 like the stock
        # save nodes (prompt = API prompt, workflow = UI graph). Build once,
        # share across the master and each per-shot file.
        _meta = {}
        if prompt is not None:
            _meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for _k in extra_pnginfo:
                _meta[_k] = json.dumps(extra_pnginfo[_k])

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                f"{filename_prefix}/{stamp}", out_dir,
                int(first_imgs.shape[-3]), int(first_imgs.shape[-2]))
        master_path = os.path.join(
            full_output_folder, f"{filename}_{stamp}_master.mp4")
        master_writer = self._MP4Stream(
            master_path, _FPS, audio=master_audio,
            crf=float(frame_compression), metadata=_meta)
        master_total_frames = 0
        imgs = first_imgs
        _prev_tail_frames = None  # for seam color match (Phase 3b)
        for si, s in enumerate(stored):
            if si > 0:
                imgs = video_vae.decode(s.get("video_final",
                                      s["video"])).detach().cpu().half()
                if imgs.ndim == 5:
                    imgs = imgs.reshape(-1, imgs.shape[-3],
                                        imgs.shape[-2], imgs.shape[-1])
            if _DBG and s["video"] is not None:
                _lv = s.get("video_final", s["video"]).float().std(
                    dim=(0, 2, 3, 4)).flatten()
                _fs = imgs.float().std(dim=(1, 2, 3))
                print(f"[SilverH3][dbg] stream shot {si + 1}: "
                      f"lat rows std[:7]="
                      f"{[round(float(x), 3) for x in _lv[:7]]} "
                      f"lat rows std[-3:]="
                      f"{[round(float(x), 3) for x in _lv[-3:]]}",
                      flush=True)
                print(f"[SilverH3][dbg]   frames std[:24:4]="
                      f"{[round(float(_fs[i]), 4) for i in range(0, min(24, len(_fs)), 4)]} "
                      f"mid={round(float(_fs[len(_fs) // 2]), 4)} "
                      f"nan={torch.isnan(imgs).any().item()}", flush=True)
            kept = int(imgs.shape[0])
            if si > 0:
                ov = min(s["trim"], kept - 1)
                imgs = imgs[ov:]
                kept = int(imgs.shape[0])
            # Phase 3b: seam color match at cut boundary
            if color_reanchor and _prev_tail_frames is not None and kept > 0:
                imgs = _seam_color_match(
                    imgs, _prev_tail_frames,
                    strength=0.65, fade_frames=min(24, kept),
                    ref_frames=4)
                # update chain anchor with matched frames
                if _color_anchor_m is not None:
                    _new_m, _new_s = _compute_color_anchor(
                        imgs[:1].float())
                    if _new_m is not None:
                        _color_anchor_m = [
                            a * 0.95 + b * 0.05
                            for a, b in zip(_color_anchor_m, _new_m)]
                        _color_anchor_s = [
                            a * 0.95 + b * 0.05
                            for a, b in zip(_color_anchor_s, _new_s)]
                # Phase 3a: color re-anchor each decoded frame
                imgs_f = imgs.float()
                if _color_anchor_m is not None:
                    imgs_f = _color_reanchor(
                        imgs_f, _color_anchor_m, _color_anchor_s,
                        strength=0.35)
                imgs = imgs_f.to(imgs.dtype)
            # store tail for next shot's seam match
            _prev_tail_frames = imgs[-4:].clone() if kept > 4 else imgs.clone()
            master_writer.add_frames(imgs)
            master_total_frames += kept
            cw_audio = (audio_parts[si]
                        if audio_parts[si]["waveform"] is not None
                        else None)
            clip_path = os.path.join(
                full_output_folder,
                f"{filename}_{stamp}_shot_{si + 1:02d}.mp4")
            cw = self._MP4Stream(clip_path, _FPS,
                                 audio=cw_audio,
                                 crf=float(frame_compression),
                                 metadata=_meta)
            cw.add_frames(imgs)
            cw.close()
            del imgs
            stored[si] = None    # release this shot's latent immediately
            print(f"[SilverH3] decoded + saved shot {si + 1}/{n} "
                  f"({kept}f kept).", flush=True)
        master_writer.close()

        print(f"[SilverH3] DECODE done: {n} shot(s) in one VAE session "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)
        print(f"[SilverH3] saved master ({master_total_frames}f): "
              f"{master_path}", flush=True)
        print(f"[SilverH3] done: {n} shots, {master_total_frames} frames "
              f"(~{master_total_frames / 24.0:.1f}s), total "
              f"{_dur(time.time() - t_all)} - 2 model swaps for the whole "
              f"run instead of ~{3 * n}.", flush=True)
        return (audio_out, master_path)


_BUILDER_MAX_SLOTS = 4096


class SilverH3PromptsBuilder:
    """Line-batch prompt splitter for the Silver H3 Batch / Staged nodes.

    Paste a list of prompts (one per line) into the multiline box. The node
    emits a sliding window of ``batch`` lines, addressed by ``index`` x
    ``batch``:

        batch=1,  index=0  -> line 1
        batch=5,  index=0  -> lines 1..5
        batch=10, index=1  -> lines 11..20   (and so on)

    ``batch`` is essentially unlimited (capped at _BUILDER_MAX_SLOTS), so one
    window can carry a long continuous video's worth of clips (each entry is
    one clip in the H3 Batch / Staged nodes). Blank lines are ignored when
    building the prompt list, so the window is measured in PROMPTS, not
    physical lines. The single list output wires straight into the H3 Batch /
    Staged nodes' prompts input, which renders one video per entry, so one
    text box can drive a long queue of renders just by bumping ``index`` each
    run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "One prompt per line. Blank lines are skipped, so "
                           "every non-empty line is one video prompt."}),
            "index": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffff,
                "tooltip": "Which window to emit. Window starts at "
                           "index * batch (in prompts, after blank lines are "
                           "removed). Bump this by 1 each run to advance to "
                           "the next batch of lines."}),
            "batch": ("INT", {
                "default": 5, "min": 1, "max": _BUILDER_MAX_SLOTS, "step": 1,
                "tooltip": f"How many prompts per window (1-{_BUILDER_MAX_SLOTS}"
                           "). Each entry becomes one clip in the H3 Batch / "
                           "Staged nodes, so a large batch = one long "
                           "combined video."}),
        }}

    RETURN_TYPES = ("STRING_LIST",)
    RETURN_NAMES = ("prompts",)
    OUTPUT_TOOLTIPS = (
        "A single list of prompt strings for this window "
        "(e.g. ['video 1 prompt', 'video 2 prompt', ...]). Feed this ONE "
        "output straight into the Silver H3 Batch / Staged node's prompts "
        "input - it renders one video per entry.",)
    FUNCTION = "build"
    CATEGORY = "silver_nodes/sampling"
    DESCRIPTION = ("Split a multiline prompt list into a sliding window of "
                   "prompts addressed by index x batch, output as ONE list "
                   "so it wires into the Silver H3 Batch / Staged node's "
                   "prompts input. One prompt per rendered clip.")

    def build(self, text, index, batch):
        prompts = [ln.strip() for ln in (text or "").splitlines()
                   if ln.strip()]
        b = max(1, min(_BUILDER_MAX_SLOTS, int(batch) if batch is not None
                       else _BUILDER_MAX_SLOTS))
        idx = max(0, int(index) if index is not None else 0)
        if not prompts:
            return ([],)
        start = (idx * b) % len(prompts)
        window = [prompts[(start + i) % len(prompts)] for i in range(b)]
        return (list(window),)


class SilverH3PromptListBuilder:
    """Convert / append a string into a STRING_LIST for the Silver H3 Batch/Staged.

    Primary `string` tolerates BOTH a plain string (converted into a 1-entry
    list) and an existing STRING_LIST (passed through, blank entries dropped).
    The optional `append` string is added as a NEW final entry - so wiring a
    single prompt + a secondary prompt here gives the node two videos.
    Emits STRING_LIST, ready for the prompts input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STRING_LIST", {
                "tooltip": "Primary input: a STRING or STRING_LIST. A plain "
                           "string is converted into a 1-entry list; an "
                           "existing list is passed through with blank "
                           "entries dropped."}),
            "append": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "Optional secondary string APPENDED as a NEW final "
                           "entry - each entry becomes its own video in the "
                           "Silver H3 Batch / Staged, so this adds one extra "
                           "clip. Leave blank to skip."}),
        }}

    RETURN_TYPES = ("STRING_LIST",)
    RETURN_NAMES = ("prompts",)
    OUTPUT_TOOLTIPS = (
        "A single list of prompt strings: the primary input (as a list) plus "
        "the appended secondary string as a final entry. Feed this ONE "
        "output straight into the Silver H3 Batch / Staged node's prompts "
        "input - it renders one video per entry.",)
    FUNCTION = "build"
    CATEGORY = "silver_nodes/sampling"
    DESCRIPTION = ("Convert a plain string into a STRING_LIST (one video per "
                   "entry) and optionally append a secondary string as an "
                   "extra entry, wired into the Silver H3 Batch / Staged "
                   "node's prompts input.")

    def build(self, string, append=""):
        items = [string] if isinstance(string, str) else list(string or [])
        prompts = [str(p).strip() for p in items if (p or "").strip()]
        ap = str(append or "").strip()
        if ap:
            prompts.append(ap)
        return (prompts,)


NODE_CLASS_MAPPINGS = {
    "SilverH3StagedMultishot": SilverH3StagedMultishot,
    "SilverH3PromptsBuilder": SilverH3PromptsBuilder,
    "SilverH3PromptListBuilder": SilverH3PromptListBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverH3StagedMultishot":
        "Silver H3 Staged Multishot (encode once / sample once / decode once)",
    "SilverH3PromptsBuilder":
        "Silver H3 Line Batch",
    "SilverH3PromptListBuilder":
        "Silver H3 Prompt List (string -> string list)",
}
