# Changelog

All notable changes to the comfyui-silver-nodes project will be documented in this file.

## [Unreleased]

### Added
- **Seam bleed control** (`handoff_taper`): percent of each shot that
  carries the previous shot's motion as a linear fade (10% of a 10s clip =
  1s; 15% default; 0 = hard seam). The join is a relay: a few seconds of
  the previous action carries into the new shot, then the new description
  takes over. It makes the seam CONTINUE (invisible) - it is NOT a
  transition control; scene transitions are prompt-authored (drop toward
  0-5% when a shot's prompt opens with one, so stale motion doesn't fight
  it).
- **Character reference LISTS on both H3 nodes**: two IMAGE inputs replace
  the old single sheet + toggle pair (the toggles are gone - connecting an
  input IS the toggle):
  - `char_refs_all` — EVERY image rides EVERY clip/shot with stable
    `<Picture N>` labels (covers the old `apply_to_all_refs` semantics).
  - `char_refs_index` — image `i` rides clip/shot `i` (covers the old
    `index_refs` semantics; batch still rides `i+1` too when combining).
  - Both inputs can be connected at once, deduped per clip/shot. Neither
    connected = no refs at all (the old footgun of "two false toggles =
    silently no refs" is gone by construction).
  - `char_audio` is now **one voice shared by every clip/shot** (batch used
    to index per clip); if an audio batch is supplied only the first entry
    is used.

### Fixed
- **Join timing controls simplified on the staged node**: `handoff_release`
  removed (threshold fixed at `_HO_RELEASE = 0.30` - the sigma below which
  the lock/taper stops injecting, so late detail steps reconcile freely).
  `handoff_taper` changed from latent rows to a **percent of the shot's
  duration** (converted backend-side via the ~4 frames/row mapping).
- **Playback rate fixed at 24 fps on both nodes**: `frame_rate` input
  removed - H3's audio sync requires 24 fps, and any other rate desynced
  the muxed audio. Writers now use the shared `_FPS = 24.0` constant.
- **Staged `latent_handoff` join-replay**: shots 2+ cloned the previous
  tail's last rows but never trimmed the replayed head, so a new scene opened
  by replaying the previous scene (~3.5s). Handoff `_TRIM` is now
  `5 + 17*(_HO_ROWS-1)` = 73 decoded frames, applied to shot 2+ video AND
  audio (sample-aligned).
- **Refs + start_image mutually exclusive, enforced with a warning on both
  nodes**: `start_image` (keyframes) + character refs share `cond_video_latents`
  (`model_base.py:2173` lets refs overwrite the keyframe latents). When both
  are given, the character refs are DROPPED with a warning. Staged re-tokenizes
  without the orphan `<Picture N>` items on that path.
- **Staged continuity is now AUTO-SELECTED** (the `continuity` widget is
  removed): a `start_image` provided -> `first_frame` keyframe chaining;
  WITHOUT one the shots join by `latent_handoff` (the default, refs or not).
- **Staged memory bank is now two stills from the previous TWO shots**: when
  ON, a shot references the **START and MIDDLE frames** (1 latent row each)
  of the last 2 prior shots - people/outfits/locations stay consistent as
  the camera moves, and a later clip can return to a setting that appeared
  up to two clips ago. Depth (2) and still size (5 frames = 1 latent row)
  are FIXED constants in the backend - `bank_frames` / `bank_depth` are no
  longer inputs; `memory_bank` is the only on/off switch. The old end
  anchor is gone: latent handoff already carries the previous tail's motion
  + identity into the next shot by construction.

### Changed
- **Silver H3 Staged prompts input now matches the Batch node exactly:** one
  MASTER prompt in `overall_prompt` (prepended to EVERY shot => `"overall
  shot"`) plus a `prompts` STRING_LIST with one prompt PER SHOT. The old
  `prompts_json` input is gone (along with its JSON / `---` parser - a blobs-
  of-text prompt never encoded again silently). A plain string is accepted as
  a 1-shot list; blank entries are skipped; an all-empty list errors at run
  time. Feed the list from the **Silver H3 Line Batch** /
  **Silver H3 Prompt List** builders, which now advertise Batch / Staged.
- **Silver H3 Staged decode/save rewritten to stream** so peak host RAM stays at
  ONE shot regardless of take length. Phase C previously decoded every shot and
  assembled a giant master tensor (every shot's fp16 frames plus the master
  copy) - running a small machine out of RAM on long chains. It now:
  - decodes each shot, trims the join head and **saves ONE mp4 PER SHOT plus
    the combined `_master.mp4`** into a dated `prefix_yyyymmddHHMMSS/` folder
    (the same incremental H.264+AAC writer as Batch: aac stream registered up
    front, audio sliced to the frame count on close) - matching the Batch save
    process so a very long take's progress survives even if a later shot
    fails, instead of the old single-file path that could overload RAM and
    lose everything (saving is always on - no toggle).
  - trims the node's OUTPUTS to `master_audio` + `saved_video` (the master
    mp4 path). The `master_frames` (IMAGE), `shots_rendered`, `video_latents`,
    `audio_latents` and `head_frames` outputs are gone - only the trimmed
    decoded mp4s are correct to use (the untrimmed latent outputs carried
    the replayed join head).
  - drops the Phase B full-shot tiled decode in `first_frame` mode: only the
    1-frame chain decode remains (full shots are decoded once, streamed and
    freed in Phase C), so a take never holds every shot's frames at once.
  - adds `filename_prefix`, `frame_rate`, `format`, `frame_compression` save
    controls.
- **Silver H3 Batch save path rewritten to stream** so peak RAM stays at ONE
  clip regardless of batch size (was ~54GB thrash for 12 clips on a 25GB box).
  Each clip is encoded to its own mp4 as it is decoded, then freed; when
  `combine_videos` is ON the same frames are also streamed into a single
  master mp4 (no full master tensor is ever materialised).
  - `combine_videos` now writes the clips into a **dated subfolder**
    `prefix_yyyymmddHHMMSS/` (`clip_01.mp4`, `clip_02.mp4`, ... plus
    `_master.mp4`) so clips are on disk for manual re-combining. When OFF,
    files stay flat as before.
  - The `video_frames` (IMAGE_LIST) output was **removed** (unused; frames are
    saved to disk). The node now returns `audio`, `saved_videos`,
    `videos_rendered`. Master audio is unchanged: full raw guide over the
    combined take when `guide_audio` supplied, else per-clip audio crossfaded.
  - The `save_video` toggle is **gone** - saving is always on (each clip is
    written to disk as it decodes; the frames-only mode served no workflow).
  - Streaming writer registers its AAC stream up front (PyAV cannot add an
    audio stream after video frames are muxed) and truncates the soundtrack to
    the total frame count exactly like ComfyUI's save_to.
  - Long `guide_audio` spines are encoded in ~10s chunks to avoid ComfyUI's
    broken tiled fallback for the H3 audio VAE (IndexError on 120s+ guides).

### Added
- **Silver H3 Batch**: upgraded with per-clip **start keyframes** (H3's native
  first/last channel). `start_image` accepts an IMAGE batch - image `i` opens
  clip `i`, so N+1 images chain N clips frame-to-frame. There is no `end_image`
  input: when `combine_videos` is ON, clip `i`'s LAST frame is auto-taken as
  image `i+1` (the next clip's start) and stitched. A single image is reused for
  every clip; the clip count is driven by the prompts list, never the image
  count. Added `combine_videos` (default OFF) to save one stitched master vs one
  file per clip. Behaviour is unchanged when no image inputs are supplied.
  - **Model note**: supply an **fl2v** model for start frames and a **ref2v**
    model for `char_image`.
  - `char_image` now accepts an IMAGE batch (image `i` = clip `i`'s identity
    sheet; a single image is reused for every clip), matching the per-clip
    indexing of `start_image`.
  - NEW `char_audio`: optional **character voice reference** (ref2va). AUDIO
    batch (waveform entry `i` = clip `i`'s voice sheet; a single audio is
    reused for every clip), encoded once in Phase A and ridden as `<Audio N>`
    tokens + a DiT-side audio ref row. Voice/style identity with **no timing
    claim**, so it composes with `guide_audio` (the driving spine) instead of
    fighting it. Works alongside `char_image` for face + voice identity.
  - `char_audio` + `start_image` WITHOUT `char_image` is now guarded: an audio
    ref block has no `latent`, so `minimax_refs` would overwrite
    `cond_video_latents` and WIPE the keyframes. The voice ref is skipped with
    a clear warning in that combo (add `char_image` to ride both).
  - `prompts` now tolerates a **single string** (wraps it as one prompt);
    previously a plain `str` was iterated character-by-character, producing a
    ~Nvideos-per-character batch.
- **Silver H3 Prompt List**: small helper that turns a **plain string** (or an
  existing STRING_LIST) into a STRING_LIST and optionally appends a secondary
  `append` string as a new final entry - wire the output straight into the
  Batch node's `prompts` input (one video per entry).
  - `combine_videos` + `guide_audio` supplied: the saved master file's audio is
    the **raw full guide track, unprocessed**, laid over the whole combined
    video (truncated to its length). Without a guide, the master uses the
    assembled per-clip audio (each clip's own AV audio, crossfaded at the
    seams).
- **Silver WAN HuMo Batch**: new node wrapping the **WAN HuMo** audio-conditioned
  video model with the same staging discipline - text encoder, audio encoder,
  DiT and VAE each load once for the whole batch. `guide_audio` and
  `audio_encoder` are required (HuMo consumes audio as conditioning and
  generates no track, so the guide IS the clip's audio). Optional per-clip
  `ref_image` identity, `length` (default 97f), `cfg`, `step_audio` (auto-ON
  with `combine_videos`), plus the shared save/combine path.
- **Silver H3 Staged Multishot**: New long-form MiniMax-H3 sampler that
  reorders multishot work into three phases so model swaps happen twice per
  run instead of three times per shot:
  - Phase A (ENCODE): every prompt's conditioning is text-encoded in one
    text-encoder session (optional `start_image` becomes shot 1's I2V first
    frame here too)
  - Phase B (SAMPLE): every shot is sampled back-to-back with the DiT loaded
    once; consecutive shots are chained IN LATENT SPACE - the previous
    shot's tail video rows / audio columns are locked into the next shot's
    trajectory at every sampling step (latent handoff, no VAE round trip)
  - Phase C (DECODE): every stored latent is decoded in one VAE session,
    replayed join heads trimmed, master assembled
  - Prompts accepted as JSON `{"prompts": [...]}` (or a bare array, or
    `---` separated blocks); malformed JSON fails before any sampling
  - Optional `guide_audio` AUDIO SPINE: one continuous track encoded once
    in the ENCODE phase; every shot's full audio stream is locked to its
    time-slice of it at every sampling step (voice/lipsync held across the
    whole take, shot stride compensates the join trim)
  - `continuity` modes: **first_frame** (default) chains shots through H3's
    NATIVE keyframe channel - each shot opens on the previous shot's
    decoded last frame (stock-I2V mechanics, 1 duplicate frame trimmed per
    join, pixels decoded hot during Phase B and cached so every frame is
    VAE-decoded exactly once); **latent_handoff** (legacy) keeps the
    per-step noise-domain tail lock (22-frame overlap), kept for comparison
    after it rendered planner-conflict artifacts on current checkpoints
  - MEMORY BANK (`memory_bank`): each shot's conditioning
    carries two reference STILLS (the previous shot's opening + mid rows)
    sliced straight from prior shots' raw latents (the ref2va `minimax_refs`
    mechanism, latent-only - context_pin style, no VAE round trip, no TE
    sessions), keeping faces/outfits AND whole-scene identity/location
    consistent across scene changes - every non-ref-imaged character and
    prop in frame rides along; fully optional
  - CHARACTER REFERENCE (`char_image`): one image encoded once in the
    ENCODE phase and ridden by EVERY shot through BOTH ref2va channels -
    DiT-side `minimax_refs` rows plus `<Picture N>` vision tokens in every
    shot's text conditioning; aspect-preserving downscale (stock policy,
    never stretched), encoded before the phase sweep so no mid-run VAE
    reload
  - CHARACTER VOICE REFERENCE (`char_audio`): one voice sheet encoded once
    in the ENCODE phase (same audio-VAE session as the spine) and ridden
    by every shot as a DiT-side audio ref row plus `<Audio N>` tokens -
    voice/style identity with no timing claim, so it composes with
    `guide_audio`. Mirrors the Batch node's per-clip policy (images, then
    audio, in the reference node's fixed order). Guarded against the
    audio-only-refs-wipe-a-keyframe conflict exactly like Batch
  - Graceful `guide_audio` exhaustion: a spine shorter than the run no
    longer fails - shots partially covered lock the covered prefix and
    improvise the rest, fully exhausted shots generate freely (per-shot
    audio-lock tracking drives the join trim/fade choice in assembly)
- **Silver URL Image Loader**: New node for loading images directly from URLs
  - Added cache busting to prevent browser caching issues
  - Implemented duplicate detection to avoid loading the same image multiple times
  - Added support for graceful error handling with `InterruptProcessingException`
  - Included retry mechanism for failed downloads

- **Silver Web Image Loader**: New node to extract the largest image from a webpage
  - Parses HTML to find the largest image on a webpage
  - Includes options to exclude specific image types or classes
  - Handles relative URLs and various image formats

- **Enhanced Error Handling**: Improved error handling across all nodes
  - Added `InterruptProcessingException` for graceful workflow interruption
  - Implemented retry mechanisms for network operations
  - Added detailed error messages for troubleshooting

### Fixed
- **Silver H3 Staged Multishot**: `prompts_json` whose outer braces were
  stripped upstream (e.g. by a text widget node that drops `{}`) no longer
  silently renders the whole blob as ONE shot's prompt - it now fails
  validation with a message naming the likely cause
- **Silver H3 Staged Multishot**: `prompts_json` that parses as JSON but has
  an unrecognized shape (e.g. a `{"prompt": ...}` key typo, a quoted string)
  now fails with a clear error instead of silently rendering its own source
  text as a single-shot prompt
- **Silver H3 Staged Multishot**: first_frame hot-decodes now use tiled
  decoding - a plain full-clip decode makes model management reserve the
  whole clip's footprint, which on tight VRAM evicts the resident DiT and
  forces a full reload on the next shot; tile-bounded reservations keep the
  DiT in place through Phase B (docstring swap-count claims corrected too)

### Changed
- **Silver Folder Video Loader**: 
  - Updated to be compatible with moviepy 2.2.1
  - Added proper resource management with context managers
  - Improved audio extraction reliability
  - Enhanced error handling for corrupt video files

### Fixed
- **Silver Lora Model Loader**:
  - Fixed issue with model caching
  - Improved regex filter performance
  - Fixed edge cases in model cycling logic

- **Silver Folder Image Loader**:
  - Fixed sorting by date modified
  - Improved handling of unsupported image formats
  - Fixed batch processing with empty folders

## [1.0.0] - 2025-08-17

### Added
- **Initial Release**: First public release of Silver Nodes
  - Core set of nodes for image, video, and text processing
  - Support for batch processing and sequential workflows
  - Comprehensive documentation and examples

### Changed
- Updated `load_videos_from_folder` method to handle audio data alongside video frames
- Improved error handling and logging for audio-related operations
- Modified return structure to include audio in the result tuple

### Dependencies
- Added `moviepy>=2.2.1` as a required dependency for audio processing

## How to Revert Changes

To revert the audio extraction changes, you would need to:

1. Revert the changes to the following methods in `node.py`:
   - `_extract_frames`: Remove audio extraction logic and restore original return values
   - `load_videos_from_folder`: Remove audio handling and restore original return structure
   - Revert the `RETURN_TYPES` and `RETURN_NAMES` to their original values

2. Remove `moviepy` from your dependencies if it's not needed elsewhere.

3. Update any nodes that might be consuming the audio output from the `FolderVideoLoader`.

## Versioning

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-07-28
### Added
- Initial release of comfyui-silver-nodes
- Basic video loading functionality with `FolderVideoLoader` class
