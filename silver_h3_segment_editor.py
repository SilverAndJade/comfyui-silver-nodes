"""Silver H3 Segment Editor - regenerate a segment of an existing video.

Takes a source video, selects a time range, and regenerates that segment
with MiniMax-H3 while keeping continuity at BOTH boundaries:

  - the frame at the START of the segment becomes H3's frame-0 keyframe,
  - the frame at the END of the segment becomes H3's last-frame keyframe,
  - by default the source segment's AUDIO is encoded as a spine and LOCKED
    into the generation at every sampling step (predict_noise injection),
    so the regenerated video lip-syncs to / recreates the existing audio.

This is the audio *lock* (sampler-level, works with an fl2v model), NOT the
ref2va audio reference (``<Audio N>`` + audio ref block, which is a
ref2va-checkpoint-only feature and would conflict with the keyframes this
node relies on). With regenerate_audio=True the model generates fresh
segment audio; audio_lock_strength then controls a SOFT lock (every step
injects w*spine + (1-w)*model latents) so the fresh audio keeps the clip's
voice/room tone. The generated audio is spliced into the output over the
original.

The source video can be CROPPED first: only the section between
source_start and source_end is used, and the output is exactly that section
(the selected segment is regenerated inside it). This lets you take a long
video, keep only the part you care about, and edit a small segment at the
right place within it without re-encoding or stepping over the rest.

The whole OUTPUT is normalised to one resolution and 24 fps, because that
is what H3 generates at: the megapixels input picks a W x H (both divisible
by 32) that keeps the source's aspect ratio as closely as the grid allows,
every source frame is Lanczos resized to it, and frames are re-sampled to
24 fps (frame aligned by timestamp).

Phase A  ENCODE  text encoder (and video VAE for the two keyframes) hot:
                  prompt + start/end keyframes encoded in one session.
Phase B  SAMPLE  DiT hot: the replacement segment sampled once, audio
                  stream optionally locked to the source audio spine.
Phase C  DECODE  VAE hot: segment decoded and streamed out with the source
                  video's (resampled) frames and audio re-muxed.
"""

# The video-picker + preview + SECTION/SEGMENT range handles live in
# js/silver_h3_segment_editor.js (WEB_DIRECTORY "js").

import av
import folder_paths
import json
import math
import os
import torch
from fractions import Fraction

from comfy_api.latest._util.video_types import VideoContainer, VideoCodec
from comfy_api.latest._input_impl.video_types import mp4_output_open_kwargs

from .silver_h3_staged import (
    _sampler_names,
    _scheduler_names,
    _FPS,
    _ALIGN_LINE,
    _seam_color_match,
)


def clamp01(x):
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def _largest_grid_le(n):
    """Largest H3-valid frame count (5 + 17*k) that is <= n."""
    n = max(22, int(n))
    while n % 17 != 5:
        n -= 1
    return n


def _read_source_frame(video_path, src_idx, src_fps, width, height):
    """Read exact source frame src_idx, resized to width x height.

    PyAV's seek lands on a KEYFRAME (usually before the target), so the
    decoder is walked forward until a frame's pts matches the target index.
    Returns [1,H,W,3] float in 0..1.
    """
    from comfy_extras.nodes_minimax_h3 import _resize
    cnt = av.open(video_path)
    st = next(s for s in cnt.streams if s.type == "video")
    tb = float(st.time_base)
    cnt.seek(int(round(max(0, src_idx - 2) / (src_fps * tb))), stream=st)
    for fr in cnt.decode(video=0):
        fi = int(round(fr.pts * src_fps * tb))
        if fi >= src_idx:
            arr = fr.to_ndarray(format="rgb24")
            t = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0)
            cnt.close()
            return _resize(t, width, height, "center")
    cnt.close()
    raise IndexError(f"Could not read source frame {src_idx}")


def _input_videos():
    """Video files available from the input folder (browse/upload backend).

    Mirrors LoadImage: this ComfyUI fork never registers an "input" entry in
    folder_names_and_paths, so get_filename_list("input") raises KeyError.
    We list the input directory directly instead, recursively (subfolders are
    returned as "subdir/file.mp4", matching get_filename_list's convention).
    """
    try:
        input_dir = folder_paths.get_input_directory()
        if not os.path.isdir(input_dir):
            return [""]
        found = []
        for root, _dirs, names in os.walk(input_dir):
            for name in names:
                if name.startswith("."):
                    continue
                if not is_supported_video(name):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, input_dir).replace(os.sep, "/")
                found.append(rel)
    except Exception:
        found = []
    return sorted(found) if found else [""]


def is_supported_video(name):
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    return ext in ("mp4", "mov", "mkv", "avi", "webm", "m4v", "mpeg", "mpg")


class SilverH3SegmentEditor:

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
                           "section between source_start and source_end is "
                           "used for the output; the segment is regenerated "
                           "inside it and the whole cropped section is "
                           "re-saved at 24 fps / the megapixel resolution."}),
            "segment_prompt": ("STRING", {
                "multiline": True, "dynamicPrompts": False, "default": "",
                "tooltip": "Prompt describing the replacement segment "
                           "content. The start and end frames are injected "
                           "as first/last keyframes for continuity, and the "
                           "source audio is locked in so the lipsync "
                           "matches."}),
            "timeline_start": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                "tooltip": "Start time of the segment to regenerate, in "
                           "seconds RELATIVE to the cropped section (0 = the "
                           "cropped section's start frame)."}),
            "timeline_duration": ("FLOAT", {
                "default": 10.0, "min": 1.0, "max": 100.0, "step": 0.1,
                "tooltip": "Duration of the segment to regenerate, in "
                           "seconds. Snapped to H3's frame grid "
                           "(5+17*k frames at 24 fps). Use the preview "
                           "panel's SEGMENT handles to set it visually."}),
            # --- TEST: motion lead-in (may be removed if it does not help) ---
            "pre_roll_frames": ("INT", {
                "default": 12, "min": 0, "max": 200, "step": 1,
                "tooltip": "TEST: number of extra lead-in frames generated "
                           "BEFORE the edit so the camera motion has runway "
                           "to establish before the actual edit starts. "
                           "Snapped UP to a multiple of 17 (the H3 grid): "
                           "12 -> 17 frames (~0.7s at 24fps). The regenerated "
                           "region widens backward by this amount and "
                           "REPLACES the source in place (keyframes give a "
                           "pixel-exact seam both sides). Experimental - "
                           "remove if the motion is not improved."}),
        }, "optional": {
            "source_start": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 7200.0, "step": 0.1,
                "tooltip": "Start (seconds) of the SECTION of the source "
                           "video to use. Everything before this is ignored "
                           "and NOT included in the output - e.g. crop a "
                           "10min video down to the 1min you want to work "
                           "on, then edit a 10s segment inside it. Use the "
                           "preview panel's SECTION handles to set it "
                           "visually."}),
            "source_end": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 7200.0, "step": 0.1,
                "tooltip": "End (seconds) of the SECTION of the source "
                           "video to use. 0 = until the end of the video. "
                           "The FULL output is exactly this cropped section "
                           "(normalised to 24 fps / megapixels)."}),
            "megapixels": ("FLOAT", {
                "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.05,
                "tooltip": "The WHOLE output resolution in megapixels: the "
                           "entire video (source + regenerated segment) is "
                           "resized DOWN to this total pixel count, keeping "
                           "the source's aspect ratio as closely as the H3 "
                           "canvas allows and both dimensions divisible by "
                           "32. If the source is already smaller it is "
                           "snapped to the nearest 32-divisible size "
                           "instead of upscaled."}),
            "regenerate_audio": ("BOOLEAN", {
                "default": False, "label_on": "generate new audio",
                "label_off": "keep original audio",
                "tooltip": "OFF (default): keep the original audio - the "
                           "source segment's audio is encoded and LOCKED "
                           "into the generation at every sampling step, so "
                           "the regenerated video lip-syncs to / recreates "
                           "the existing audio, and the output keeps the "
                           "original audio track untouched. ON: the model "
                           "generates fresh audio for the segment; the "
                           "generated audio is spliced into the output over "
                           "the original, which is kept for the rest of the "
                           "video. Use audio_lock_strength to pull the fresh "
                           "audio toward the clip's voice/room tone."}),
            "audio_lock_strength": ("FLOAT", {
                "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Only used when regenerate_audio is ON. How "
                           "strongly the regenerated segment audio is pulled "
                           "toward the SOURCE audio at every sampling step: "
                           "0.0 = fully fresh (the source is ignored, the "
                           "model invents a new voice/room), 1.0 = fully "
                           "re-locked (the generation reproduces the source "
                           "audio). A middle value keeps the new audio "
                           "consistent with the clip's environment/speech "
                           "while still letting the model vary it."}),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "sampler_name": (_sampler_names(), {"default": "res_multistep"}),
            "scheduler": (_scheduler_names(), {"default": "simple"}),
            "filename_prefix": ("STRING", {
                "default": "SilverH3SegmentEditor",
                "tooltip": "Output filename prefix."}),
            "format": (["mp4", "auto"], {"default": "mp4"}),
            "frame_compression": ("FLOAT", {
                "default": 23.0, "min": 0.0, "max": 51.0,
                "tooltip": "H.264 CRF for the saved video. 0 = lossless, "
                           "23 = default good quality."}),
            "color_match_seam": ("BOOLEAN", {
                "default": True, "label_on": "on", "label_off": "off",
                "tooltip": "Apply affine color matching at the before/after "
                           "seams so the regenerated segment blends "
                           "visually with the original footage."}),
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
        # validation would reject freshly uploaded / workflow-pinned files
        # (the server captures the option list at node-registration time).
        # Every input is re-checked defensively inside run().
        return True
    OUTPUT_TOOLTIPS = (
        "Path of the saved output video: the whole source video re-sampled "
        "to 24 fps / the megapixel resolution, with the selected segment "
        "regenerated (start + end keyframes, audio kept or regenerated "
        "per the toggle).",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "silver_nodes/sampling"

    # ---- MP4 writer (same pattern as Batch / Staged) ----------------------
    class _MP4Stream:
        def __init__(self, path, frame_rate, audio=None, crf=23.0,
                     bit_depth=8, metadata=None):
            self._path = path
            self._frame_rate = float(frame_rate)
            self._out = av.open(path, **mp4_output_open_kwargs(
                path, VideoContainer.MP4, VideoCodec.H264))
            if metadata:
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
            """Encode + mux a frame tensor ([F,H,W,C], 0..1)."""
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

    @staticmethod
    def _resolve_video_path(value):
        """Turn a combo/upload value (or an absolute path) into a file path."""
        if isinstance(value, dict):
            value = (value.get("video_path") or value.get("path")
                     or value.get("filename", ""))
        value = str(value or "").strip()
        if os.path.isfile(value):
            return value
        resolved = folder_paths.get_annotated_filepath(value)
        if resolved and os.path.isfile(resolved):
            return resolved
        raise ValueError(
            f"Source video not found: {value!r}. Pick a file from the "
            "input folder in the node, upload one, or give an absolute "
            "path to an existing video file.")

    # ---- source-video -> 24 fps / gen-geometry streaming helpers ----------
    @staticmethod
    def _count_video_frames(video_path):
        """Count decoded video frames when the container has no frame count."""
        import av as _av
        count = 0
        with _av.open(video_path) as src:
            for packet in src.demux(video=0):
                for _frame in packet.decode():
                    count += 1
        return count

    @staticmethod
    def _source_meta(video_path):
        """Return (frames, fps, width, height) for a source video.

        `st.frames` is trusted when present (it is exact CFR). When a trim /
        copy edit leaves it missing or zero, the duration fields are also
        unreliable, so the frames are counted by DECODING - the only number
        that reflects what is actually in the file.
        """
        import av as _av
        src = _av.open(video_path)
        try:
            st = next(s for s in src.streams if s.type == "video")
            fps = float(st.average_rate or st.rate or 24.0)
            frames = st.frames or 0
            w = st.codec_context.width
            h = st.codec_context.height
        finally:
            src.close()
        if frames < 1:
            frames = SilverH3SegmentEditor._count_video_frames(video_path)
            if frames > 0:
                print(f"[SilverH3SegmentEditor] frame metadata missing - "
                      f"counted {frames} frames by decoding.", flush=True)
        return max(1, frames), fps, w, h

    @staticmethod
    def _src_frame_at(src_fps, limit_frames, out_idx, out_fps=24.0,
                      offset=0):
        """Source frame index that lands at output frame out_idx's time.

        offset/limit define the CROPPED source-frame range, so the returned
        index stays inside [offset, limit_frames-1].
        """
        return min(limit_frames - 1,
                   max(offset, offset + int(round(out_idx / out_fps
                                                  * src_fps))))

    @staticmethod
    def _compute_dims(mp, src_w, src_h):
        """Pick output W x H from a megapixel budget for a given source AR.

        Both dims are multiples of 32 (H3's canvas multiple). If the source
        already fits in the budget its size is kept (snapped up to 32) -
        never upscaled. Otherwise the LARGEST W x H that stays within the
        budget and keeps the closest aspect ratio is chosen.
        """
        cm = 32
        max_d = 4096
        target = max(0.01, float(mp)) * 1_000_000
        aspect = src_w / src_h
        if src_w * src_h <= target:
            w = max(cm, min(max_d, int(math.ceil(src_w / cm) * cm)))
            h = max(cm, min(max_d, int(math.ceil(src_h / cm) * cm)))
            return w, h
        best = None
        for kh in range(1, max_d // cm + 1):
            h = kh * cm
            w = int(round(h * aspect / cm) * cm)
            w = max(cm, min(max_d, w))
            area = w * h
            if area > target:
                continue
            ratio = w / h
            score = abs(ratio - aspect)
            if (best is None or area > best[0]
                    or (area == best[0] and score < best[1])):
                best = (area, score, w, h)
        if best is None:
            return cm, cm
        return best[2], best[3]

    def _iter_source_frames(self, video_path, width, height, out_fps=24.0,
                            start_out=0, end_out=None,
                            crop_a_f=0, crop_b_f=None):
        """Yield source frames re-sampled to out_fps and center-cropped to
        width x height, for output indices [start_out, end_out).

        crop_a_f/crop_b_f select a sub-range of the SOURCE frames (frames
        outside it are never read); output index 0 corresponds to the first
        cropped source frame. Frames are selected by timestamp (nearest
        source frame to each output frame's time), reading ahead only as
        needed. Yields [1,H,W,3] float.
        """
        from comfy_extras.nodes_minimax_h3 import _resize
        src = av.open(video_path)
        st = next(s for s in src.streams if s.type == "video")
        src_frames = st.frames or 0
        if src_frames <= 0:
            src_frames, src_fps, __, __ = self._source_meta(video_path)
        else:
            src_fps = float(st.average_rate or st.rate or 24.0)
        limit = (src_frames if crop_b_f is None
                 else max(0, min(src_frames, int(crop_b_f))))
        a = max(0, min(limit - 1, int(crop_a_f)))
        if end_out is None:
            end_out = max(0, int(round((limit - a) / src_fps * out_fps)))
        if end_out <= start_out:
            src.close()
            return
        tb = float(st.time_base)

        def _fr_index(fr):
            return int(round(fr.pts * src_fps * tb))

        need = self._src_frame_at(src_fps, limit, start_out, out_fps, a)
        # pyav seek lands on a KEYFRAME at/before the target; rewind a couple
        # of frames and walk forward so frame indexing is exact (CFR).
        seek_back = max(a, need - 2)
        src.seek(int(round(seek_back / (src_fps * tb))), stream=st)
        gen = src.decode(video=0)
        cur_idx = seek_back - 1
        cur = None
        for oi in range(start_out, end_out):
            need = self._src_frame_at(src_fps, limit, oi, out_fps, a)
            while cur_idx < need:
                try:
                    cur = next(gen)
                except StopIteration:
                    break
                cur_idx = _fr_index(cur)
            if cur is not None:
                arr = cur.to_ndarray(format="rgb24")
                t = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0)
                yield _resize(t, width, height, "center")
        src.close()

    # ---- main entry point --------------------------------------------------
    def run(self, model, clip, video_vae, audio_vae,
            source_video, segment_prompt,
            source_start=0.0, source_end=0.0,
            timeline_start=0.0, timeline_duration=10.0,
            pre_roll_frames=12,
            megapixels=1.0, regenerate_audio=False,
            audio_lock_strength=0.35,
            seed=0, steps=20,
            sampler_name="res_multistep", scheduler="simple",
            filename_prefix="SilverH3SegmentEditor",
            format="mp4", frame_compression=23.0,
            color_match_seam=True,
            prompt=None, extra_pnginfo=None):
        import time
        import node_helpers
        import comfy.model_management as mm
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy.ldm.minimax.model import time_shift_sigma as _tss

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
                print(f"[SilverH3SegmentEditor] {tag}: VRAM swept "
                      f"({b4:.1f} -> {af:.1f} GB free).", flush=True)
            except Exception as e:
                print(f"[SilverH3SegmentEditor] {tag}: sweep failed ({e}) "
                      "- continuing.", flush=True)

        # ----------------------------------------------------------------
        # Resolve source video path (input-folder name, uploaded file,
        # or an absolute path)
        # ----------------------------------------------------------------
        video_path = self._resolve_video_path(source_video)

        # ----------------------------------------------------------------
        # Source metadata + total output length at 24 fps
        # ----------------------------------------------------------------
        src_frames, src_fps, src_w, src_h = self._source_meta(video_path)
        if src_frames < 1:
            raise ValueError("Could not determine the source video's length.")

        # ----------------------------------------------------------------
        # Crop the source video to the [source_start, source_end] section.
        # The WHOLE output (segment + preview + audio) uses ONLY this
        # sub-range; timeline_start/duration are relative to it.
        # ----------------------------------------------------------------
        src_total_sec = src_frames / src_fps
        crop_start_s = max(0.0, min(float(source_start or 0.0),
                                    src_total_sec))
        crop_end_s = (src_total_sec
                      if source_end is None or float(source_end) <= 0
                      else min(src_total_sec, float(source_end)))
        crop_end_s = max(crop_start_s, crop_end_s)
        crop_a_f = int(round(crop_start_s * src_fps))
        crop_b_f = int(round(crop_end_s * src_fps))
        crop_a_f = max(0, min(crop_a_f, src_frames - 1))
        crop_b_f = max(crop_a_f + 1, min(src_frames, crop_b_f))
        crop_frames = crop_b_f - crop_a_f
        total_out = max(1, int(round(crop_frames / src_fps * _FPS)))

        # ----------------------------------------------------------------
        # Output resolution from the megapixel budget (32-divisible, aspect
        # kept as close to the source as the H3 canvas allows)
        # ----------------------------------------------------------------
        width, height = self._compute_dims(float(megapixels), src_w, src_h)
        print(f"[SilverH3SegmentEditor] source: {src_frames}f "
              f"({src_frames / src_fps:.1f}s) @ {src_fps:.1f}fps "
              f"{src_w}x{src_h}; using section "
              f"[{crop_start_s:.1f}s:{crop_end_s:.1f}s] "
              f"({crop_frames}f -> {total_out}f @ {_FPS:.0f}fps "
              f"{width}x{height} (megapixels={megapixels}).", flush=True)

        # ----------------------------------------------------------------
        # Pre-resize the WHOLE selected section ONCE, up front, to the
        # gen geometry. Every frame (keyframes included) then comes from
        # this same pool, so the source frames written before/after the
        # edit and the H3 keyframe anchors are pixel-identical - no
        # double Lanczos pass, no decoder-path variance between the
        # single-frame keyframe read and the streaming read.
        # ----------------------------------------------------------------
        t0 = time.time()
        section_frames = []
        for fr in self._iter_source_frames(
                video_path, width, height, out_fps=_FPS,
                start_out=0, end_out=total_out,
                crop_a_f=crop_a_f, crop_b_f=crop_b_f):
            section_frames.append(fr[0])
        print(f"[SilverH3SegmentEditor] pre-resized {len(section_frames)}f "
              f"of the section to {width}x{height} "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)

        # ----------------------------------------------------------------
        # Segment range ON the 24fps output grid, snapped to H3's 5+17k grid
        # ----------------------------------------------------------------
        seg_start_out = max(0, int(round(timeline_start * _FPS)))
        if seg_start_out >= total_out:
            raise ValueError(f"timeline_start {timeline_start}s is past the "
                             f"end of the video ({src_frames / src_fps:.1f}s).")
        user_frames = max(1, int(round(timeline_duration * _FPS)))
        max_frames = total_out - seg_start_out
        # Does the user's requested span reach (or cover) the end of the
        # section? Judge by INTENT, before grid snapping: a segment dragged to
        # the last frame must be regenerated TO the last frame - the end
        # keyframe is excluded (no footage after it) and no original tail is
        # left. A middle edit with footage after keeps the end keyframe anchor.
        edit_to_end = seg_start_out + user_frames >= total_out
        if edit_to_end:
            # smallest on-grid value covering the remainder: the regenerated
            # clip runs past the last output frame by up to 16, truncated on
            # write so the output stays exactly total_out.
            gen_frames = mmh3.align_frame_count(max_frames)
            if gen_frames < 22:
                raise ValueError("The remaining segment to the end of the "
                                 "video is too short (<22 frames).")
        else:
            gen_frames = mmh3.align_frame_count(user_frames)
            if gen_frames > max_frames:
                gen_frames = _largest_grid_le(max_frames)
                if gen_frames < 22:
                    raise ValueError("The segment is longer than the video at "
                                     "the requested start.")
        seg_end_out = seg_start_out + gen_frames
        print(f"[SilverH3SegmentEditor] segment: out frames "
              f"[{seg_start_out}:{seg_end_out}] "
              f"({seg_start_out / _FPS:.1f}s + {gen_frames / _FPS:.1f}s) "
              f"snapped to H3 grid ({gen_frames}f) "
              f"{'TO THE END' if edit_to_end else ''}.", flush=True)
        # Editing to the very end: there are no source frames AFTER the last
        # output frame, so the last-frame keyframe would only pull the
        # regenerated ending back to the pre-edit frame (e.g. a suit reverting
        # after a onesie change). Exclude it and let the generation keep the
        # edit to the last frame. Middle edits keep it - it anchors the segment
        # to the footage that follows.
        assert seg_end_out >= total_out if edit_to_end else seg_end_out <= total_out

        # --- TEST: motion lead-in (may be removed if it does not help) -----
        # Widen the regenerated region BACKWARD by a motion lead-in: the
        # model now starts from the source frame BEFORE the edit, giving it
        # runway to establish the camera motion so the visible seam is a
        # pixel-exact cut into continuing (not freshly started) motion.
        # gen_frames is on H3's 5+17k grid, so keeping the TOTAL on-grid
        # forces `lead` to a multiple of 17 (12 -> 17 frames ~0.7s).
        lead = 0
        if pre_roll_frames:
            _req = max(0, int(pre_roll_frames or 0))
            if _req > 0 and seg_start_out > 0:
                # gen_total = lead + gen_frames must stay on H3's 5+17k grid;
                # align_frame_count(_req+gen_frames) - gen_frames is already a
                # multiple of 17, and lead must not push reg_start < 0.
                lead = mmh3.align_frame_count(gen_frames + _req) - gen_frames
                while lead > seg_start_out:
                    lead -= 17
                if lead < 17:
                    lead = 0
        reg_start = seg_start_out - lead
        gen_total = gen_frames + lead               # H3-grid aligned
        # REPLACE model: the regenerated clip spans [reg_start, seg_end_out)
        # in place; that span is exactly gen_total frames wide, so the output
        # stays total_out frames (no insert, no crossfade). The generated
        # head is keyframed to src[reg_start] (= the frame before the tail
        # region) and its tail to src[seg_end_out - 1], so both seams are
        # pixel-exact cuts into continuing footage.
        if lead:
            print(f"[SilverH3SegmentEditor] TEST pre-roll: regenerated "
                  f"region widened backward by {lead}f "
                  f"({lead / _FPS:.2f}s) of motion lead-in to "
                  f"[{reg_start}:{seg_end_out}] "
                  f"({gen_total}f generated, replaces source in place).",
                  flush=True)

        # ----------------------------------------------------------------
        # Keyframes at the segment's start frame and end frame (the start
        # moves back by the lead-in so camera motion has a runway)
        # ----------------------------------------------------------------
        start_frame = section_frames[reg_start].unsqueeze(0)
        if edit_to_end:
            # No frames after the edit: no end keyframe, the regenerated
            # segment keeps the edit (onesie) right through the last frame.
            end_frame = None
            print(f"[SilverH3SegmentEditor] edit to end: no end keyframe, "
                  f"start=frame {start_frame.shape}.", flush=True)
        else:
            end_frame = section_frames[seg_end_out - 1].unsqueeze(0)
            print(f"[SilverH3SegmentEditor] keyframes: start=frame "
                  f"{start_frame.shape} end={end_frame.shape}.", flush=True)

        # ----------------------------------------------------------------
        # Source audio (full) + the segment slice as the audio spine
        # ----------------------------------------------------------------
        _src_audio = None
        try:
            import numpy as np
            cnt = av.open(video_path)
            astream = next((s for s in cnt.streams if s.type == "audio"), None)
            if astream is not None:
                aframes = [f.to_ndarray() for f in cnt.decode(audio=0)]
                if aframes:
                    a0 = aframes[0]
                    if aframes[0].ndim == 1:
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
                    _src_audio = {"waveform": at_cropped.unsqueeze(0),
                                  "sample_rate": _sr}
                    print(f"[SilverH3SegmentEditor] source audio: "
                          f"{at_cropped.shape[-1]} samples @ {_sr}Hz "
                          f"(cropped to the selected section).", flush=True)
            cnt.close()
        except Exception as e:
            print(f"[SilverH3SegmentEditor] source audio extraction "
                  f"failed: {e} - output will be silent.", flush=True)

        # ----------------------------------------------------------------
        # PHASE A - ENCODE: prompt + start/end keyframes in one TE session
        # ----------------------------------------------------------------
        t0 = time.time()
        frame_count = mmh3.temporal_shape(gen_total)[0]

        start_img = start_frame
        start_lat = video_vae.encode(start_img).detach()

        if edit_to_end:
            ref_items = [{"type": "image", "data": start_img}]
            kf = [{"resolved_frame_index": 0, "latent": start_lat}]
            print(f"[SilverH3SegmentEditor] single start keyframe, "
                  f"no end anchor ({frame_count} gen frames).", flush=True)
        else:
            end_img = end_frame
            end_lat = video_vae.encode(end_img).detach()
            ref_items = [{"type": "image", "data": start_img},
                         {"type": "image", "data": end_img}]
            kf = [{"resolved_frame_index": 0, "latent": start_lat},
                  {"resolved_frame_index": frame_count - 1, "latent": end_lat}]
        tokens = clip.tokenize(_ALIGN_LINE + segment_prompt,
                               minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": kf,
            "minimax_frame_count": frame_count,
        })
        print(f"[SilverH3SegmentEditor] ENCODE done ({_dur(time.time() - t0)}) "
              f"{_el()}.", flush=True)
        del start_img, start_lat
        if not edit_to_end:
            del end_img, end_lat

        # --- audio spine: the source segment's audio, encoded once ----------
        # regenerate_audio=False -> HARD lock (weight 1.0, previous
        # behaviour). regenerate_audio=True -> SOFT lock at the chosen
        # strength: every step injects w*spine + (1-w)*model, so the fresh
        # audio keeps the clip's voice/room tone without being reproduced.
        spine = None
        _lock_strength = 1.0 if not regenerate_audio else clamp01(audio_lock_strength)
        if _src_audio is not None and _lock_strength > 0.0:
            _w = _src_audio["waveform"][0:1]          # [1,C,T]
            _sr = _src_audio["sample_rate"]
            _vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if _sr != _vae_sr:
                import torchaudio
                _w = torchaudio.functional.resample(_w, _sr, _vae_sr)
                print(f"[SilverH3SegmentEditor] audio spine: resampled "
                      f"{_sr} -> {_vae_sr} Hz.", flush=True)
            if _w.shape[1] == 1:
                _w = _w.repeat(1, 2, 1)
            # segment slice, in seconds on the OUTPUT timeline (covers the
            # pre-roll lead-in too, so motion + audio stay continuous)
            a0 = int(round((reg_start / _FPS) * _vae_sr))
            a1 = int(round((seg_end_out / _FPS) * _vae_sr))
            _seg_w = _w[..., max(0, a0):a1]
            if _seg_w.shape[-1] < 1:
                _seg_w = _w
            spine = audio_vae.encode(
                _seg_w[:1].movedim(1, -1)).detach()
            if _lock_strength >= 1.0:
                _mode = "LOCKED"
            else:
                _mode = f"soft-locked at x{_lock_strength:.2f}"
            print(f"[SilverH3SegmentEditor] audio spine: "
                  f"{spine.shape[-1]} audio cols "
                  f"({_seg_w.shape[-1] / _vae_sr:.1f}s) {_mode} into "
                  f"the regenerated segment.", flush=True)
        elif _src_audio is None:
            print("[SilverH3SegmentEditor] no source audio: no spine - "
                  "the model generates fresh segment audio.", flush=True)
        else:
            print("[SilverH3SegmentEditor] regenerate_audio ON with "
                  f"audio_lock_strength={_lock_strength:.2f}: no spine "
                  "lock - the model generates fresh segment audio.",
                  flush=True)
        _evict("ENCODE -> SAMPLE")

        # ----------------------------------------------------------------
        # PHASE B - SAMPLE: single shot, audio locked to the spine
        # ----------------------------------------------------------------
        ms_obj = model.get_model_object("model_sampling")
        try:
            _asc = float(getattr(ms_obj, "audio_scale", 1.0) or 1.0)
        except Exception:
            _asc = 1.0
        _shv, _sha = 12.0, 3.0
        try:
            _dm = getattr(getattr(model, "model", None),
                          "diffusion_model", None)
            _shv = float(getattr(_dm, "sigma_shift_video", 12.0))
            _sha = float(getattr(_dm, "sigma_shift_audio", 3.0))
        except Exception:
            pass
        if abs(_asc - 1.0) > 1e-6:
            print(f"[SilverH3SegmentEditor] audio injections scaled "
                  f"x{_asc:.3f} (ModelSamplingAV).", flush=True)

        def _arm_spine(guider, spine_seg, vshape, ashape, strength=1.0):
            """Blend THIS trajectory's audio stream toward the spine.

            strength=1.0 hard-locks (spine replaces the stream every step).
            strength<1.0 soft-locks: w*spine + (1-w)*model latents, so the
            fresh audio keeps the source voice/room tone without reproducing
            it exactly.
            """
            orig = guider.predict_noise
            nv = 1
            for d in vshape[1:]:
                nv *= d
            na = 1
            for d in ashape[1:]:
                na *= d
            logged = {"did": False}

            def _pn(x, timestep, model_options={}, seed=None,
                    _o=orig, _hs=spine_seg, _ms=ms_obj,
                    _ts=_tss, _shv=_shv, _sha=_sha, _asc=_asc,
                    _w=float(strength), _vs=vshape, _as=ashape,
                    _Nv=nv, _Na=na):
                try:
                    sig = float(timestep.flatten()[0])
                except Exception:
                    sig = float(timestep)
                if (x.ndim == 3 and x.shape[1] == 1
                        and x.shape[2] >= _Nv + _Na):
                    x = x.clone()
                    st = torch.tensor([sig], device=x.device, dtype=x.dtype)
                    sa = (_ts(sig, _shv, _sha) if _ts is not None else sig)
                    sta = torch.tensor([sa], device=x.device, dtype=x.dtype)
                    if _Na and _hs is not None:
                        xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                            (x.shape[0],) + _as[1:])
                        ts = _hs.to(device=x.device, dtype=x.dtype) * _asc
                        _nn = min(ts.shape[-1], xa.shape[-1])
                        cur = xa[..., :_nn]
                        lock = _ms.noise_scaling(
                            sta, torch.randn_like(ts[..., :_nn]),
                            ts[..., :_nn])
                        xa[..., :_nn] = _w * lock + (1.0 - _w) * cur
                        if not logged["did"]:
                            logged["did"] = True
                            _mode = "SOFT" if _w < 1.0 else "SPINE"
                            print(f"[SilverH3SegmentEditor] {_mode} inject: "
                                  f"{_nn}/{_as[-1]} audio cols x{_w:.2f} "
                                  f"(sigma={sig:.3f} audio_sigma={sa:.3f}).",
                                  flush=True)
                return _o(x, timestep, model_options=model_options,
                          seed=seed)

            guider.predict_noise = _pn

        t0 = time.time()
        latent, _fc = mmh3._empty_av_latent(width, height, gen_total)
        comps = latent["samples"].unbind()
        vshape = tuple(comps[0].shape)
        ashape = tuple(comps[1].shape)
        noise = ncs.RandomNoise().get_noise(seed)[0]
        guider = ncs.BasicGuider().get_guider(model, cond)[0]
        if spine is not None:
            _spine_seg = spine
            if _spine_seg.shape[-1] < ashape[-1]:
                _tiles = -(-ashape[-1] // _spine_seg.shape[-1])
                _spine_seg = _spine_seg.repeat(1, 1, 1, _tiles)[..., :ashape[-1]]
            elif _spine_seg.shape[-1] > ashape[-1]:
                _spine_seg = _spine_seg[..., :ashape[-1]]
            _arm_spine(guider, _spine_seg, vshape, ashape,
                       strength=_lock_strength)
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
        seg_audio_lat = (a.detach().cpu() if a is not None and regenerate_audio
                         else None)
        del out, latent, lat, parts, v, a, noise, guider
        print(f"[SilverH3SegmentEditor] SAMPLE done ({_dur(time.time() - t0)}) "
              f"{_el()}.", flush=True)
        _evict("SAMPLE -> DECODE")

        # ----------------------------------------------------------------
        # PHASE C - DECODE + streamed reassembly + save
        # ----------------------------------------------------------------
        t0 = time.time()

        decoded_seg = video_vae.decode(seg_video_lat).detach().cpu().half()
        if decoded_seg.ndim == 5:
            decoded_seg = decoded_seg.reshape(
                -1, decoded_seg.shape[-3],
                decoded_seg.shape[-2], decoded_seg.shape[-1])
        decoded_seg = decoded_seg[:gen_total]
        print(f"[SilverH3SegmentEditor] segment decoded: "
              f"{decoded_seg.shape[0]} frames.", flush=True)

        if color_match_seam:
            decoded_seg = _seam_color_match(
                decoded_seg, start_frame, strength=0.65,
                fade_frames=min(24, decoded_seg.shape[0]), ref_frames=4)
            if end_frame is not None:
                decoded_seg_rev = _seam_color_match(
                    decoded_seg.flip(0), end_frame.flip(0), strength=0.65,
                    fade_frames=min(24, decoded_seg.shape[0]), ref_frames=4)
                decoded_seg = decoded_seg_rev.flip(0)
        del start_frame, end_frame

        # ----------------------------------------------------------------
        # Output audio: keep full source (the replace model preserves the
        # total_out-frame timeline, so source audio stays in sync), or
        # splice generated audio over [reg_start, seg_end_out) to match the
        # video replace exactly.
        # ----------------------------------------------------------------
        _out_audio = _src_audio
        if regenerate_audio and _src_audio is not None \
                and seg_audio_lat is not None:
            try:
                from comfy_extras.nodes_audio import vae_decode_audio
                aud = vae_decode_audio(audio_vae, {"samples": seg_audio_lat})
                gw = aud["waveform"].detach().cpu()
                gsr = int(aud["sample_rate"])
                if gw.ndim == 2:
                    gw = gw.unsqueeze(0)
                if gw.shape[0] > 1:
                    gw = gw[:1]
                sw = _src_audio["waveform"]
                sr = _src_audio["sample_rate"]
                if gsr != sr:
                    import torchaudio
                    gw = torchaudio.functional.resample(gw, gsr, sr)
                if gw.shape[1] == 1 and sw.shape[1] == 2:
                    gw = gw.repeat(1, 2, 1)
                elif gw.shape[1] == 2 and sw.shape[1] == 1:
                    sw = sw[:1].mean(dim=1, keepdim=True).repeat(1, 2, 1)
                    _src_audio["waveform"] = sw
                # generate audio for the WHOLE generated clip (gen_total),
                # spliced in at the replaced span [reg_start, seg_end_out),
                # matching the video replace model exactly. When the edit
                # runs to the end, cap at the frames actually written.
                _span = min(gen_total, total_out - reg_start)
                _os = int(round((reg_start / _FPS) * sr))
                _os = max(0, min(_os, sw.shape[-1]))
                _need = int(round((_span / _FPS) * sr))
                _need = max(1, _need)
                if gw.shape[-1] < _need:
                    pad = torch.zeros(gw.shape[0], gw.shape[1],
                                      _need - gw.shape[-1])
                    gw = torch.cat([gw, pad], dim=-1)
                else:
                    gw = gw[..., :_need]
                _ae = int(round((seg_end_out / _FPS) * sr))
                _ae = max(_os, min(sw.shape[-1], _ae))
                out_wav = torch.cat([sw[..., :_os], gw, sw[..., _ae:]],
                                    dim=-1)
                _out_audio = {"waveform": out_wav, "sample_rate": sr}
                print(f"[SilverH3SegmentEditor] generated audio spliced "
                      f"over [reg_start, seg_end_out) "
                      f"({_os}-{_ae} samples), replacing the source.",
                      flush=True)
            except Exception as e:
                print(f"[SilverH3SegmentEditor] generated-audio splice "
                      f"failed ({e}) - falling back to source audio.",
                      flush=True)
        del seg_audio_lat
        if _out_audio is not None:
            print(f"[SilverH3SegmentEditor] output audio: "
                  f"{_out_audio['waveform'].shape[-1]} samples @ "
                  f"{_out_audio['sample_rate']}Hz.", flush=True)

        # save output (streamed: before chunks -> segment -> after chunks)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                f"{filename_prefix}/{stamp}", out_dir, height, width)
        output_path = os.path.join(
            full_output_folder, f"{filename}_{stamp}_master.mp4")

        _meta = {}
        if prompt is not None:
            _meta["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for _k in extra_pnginfo:
                _meta[_k] = json.dumps(extra_pnginfo[_k])

        writer = self._MP4Stream(
            output_path, _FPS, audio=_out_audio,
            crf=float(frame_compression), metadata=_meta)

        total_written = 0

        # --- TEST: motion lead-in splice (may be removed if it does not
        # help) - REPLACE model: the generated clip (gen_total frames,
        # lead-in + edit) replaces the source in place over
        # [reg_start, seg_end_out) (exactly gen_total frames wide), so the
        # output stays total_out frames. The generated head/tail are
        # keyframed to src[reg_start]/src[seg_end_out - 1], giving a
        # pixel-exact cut into continuing footage on both seams.

        # before: source [0, reg_start)
        chunk = []
        for i in range(0, reg_start):
            chunk.append(section_frames[i])
            if len(chunk) >= 32:
                writer.add_frames(torch.stack(chunk))
                total_written += len(chunk)
                chunk = []
        if chunk:
            writer.add_frames(torch.stack(chunk))
            total_written += len(chunk)

        # regenerated segment (lead-in + edit, replaces [reg_start, seg_end_out)).
        # When the segment runs to the very end, gen_frames may over-snap past
        # the last output frame by up to 16; cap so the output is total_out.
        _fit = min(int(decoded_seg.shape[0]), total_out - reg_start)
        writer.add_frames(decoded_seg[:_fit])
        total_written += _fit
        del decoded_seg

        # after
        chunk = []
        for i in range(seg_end_out, total_out):
            chunk.append(section_frames[i])
            if len(chunk) >= 32:
                writer.add_frames(torch.stack(chunk))
                total_written += len(chunk)
                chunk = []
        if chunk:
            writer.add_frames(torch.stack(chunk))
            total_written += len(chunk)

        writer.close()

        print(f"[SilverH3SegmentEditor] DECODE + SAVE done "
              f"({_dur(time.time() - t0)}) {_el()}.", flush=True)
        print(f"[SilverH3SegmentEditor] saved: {output_path} "
              f"({total_written}f, ~{total_written / _FPS:.1f}s).", flush=True)
        print(f"[SilverH3SegmentEditor] done in "
              f"{_dur(time.time() - t_all)}.", flush=True)
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "SilverH3SegmentEditor": SilverH3SegmentEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverH3SegmentEditor":
        "Silver H3 Segment Editor (regenerate a video segment)",
}