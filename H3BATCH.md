# Silver H3 Batch / Staged — design & usage reference

Living notes on the two H3 long-form nodes (own file so batch/staged design
decisions have a home without bloating the README). Kept deliberately short;
update as the nodes evolve.

## The two nodes

Both render MiniMax-H3 video + audio (24 fps, 17k+5 frame grid, 2 model swaps
total for a whole run: TE -> DiT -> VRAM queue). The dividing line is **what a
prompt produces**:

- **Staged** = ONE continuous take. N consecutive shots are chained
  (first_frame keyframes or latent_handoff tail clones) so the output reads as
  a single long continuous shot as much as the model allows. This is its whole
  purpose — batch's per-shot independence is deliberately absent.
- **Batch** = N INDIVIDUAL shots. Perfect frame-by-frame independence, no
  fake continuity.

Either node can still CONCATENATE its clips into one output video when asked
(batch via `combine_videos`, staged always writes one master) — but that is
stitching, not continuity: a batch master is clips joined at cuts, a staged
master is one continuous piece.

| | Silver H3 Batch | Silver H3 Staged Multishot |
|---|---|---|
| Output | N **independent** clips (+ optional stitched master) | ONE long **continuous** take from N shots |
| Chaining | none (memory-bank/handoff deliberately excluded) | first_frame or latent_handoff |
| Save | per-clip mp4s + master on demand | one streamed `_master.mp4`, always |
| Keyframe source | user `start_image` sheet, one per clip | user sheet OR prior shot's decoded last frame |
| Prompt inputs | `prompts` STRING_LIST + `overall_prompt` | **same** `prompts` + `overall_prompt` |

Shared helpers: `SilverH3LineBatch` (index x batch sliding-window prompt list),
`SilverH3PromptList` (string -> STRING_LIST + optional append). Inputs and
runtime handling of `prompts` are identical between the two nodes (a plain
string wraps to a 1-shot list, blanks are dropped, `overall_prompt` is
prepended to every prompt as the scene-wide master).

## Model requirement

`model_base.py:2170-2174` — when `minimax_refs` is set, `cond_video_latents`
is **replaced by the refs' `"latent"` blocks only**. Consequences (both nodes):

- **Keyframes + refs are mutually exclusive by construction.** Any audio-only
  ref block (no `"latent"`) reduces `cond_video_latents` to empty and the
  keyframe is silently wiped — packed cond rows then misalign. Audio-only
  character voices are therefore skipped (with a loud warning) whenever
  start-image keyframes are present without an image ref to keep a latent in
  the list.
- **End-of-clip (last-frame) keyframes require an fl2v-capable (hybrid)
  model.** A plain t2v model ignores them.

Presentation order must stay fixed: images first, then videos, then standalone
audio (`<Picture N>` / `<Video K>` / `<Audio N>`), matching both the DiT ref
rows and the tokenizer's media labels.

## Silver H3 Batch

Use when the clips are **independent** (each video = its own subject/story) or
when you want a fast short-form batch. No inter-video continuity.

- `start_image` — per-clip keyframes: image `i` -> clip `i` (batch shorter
  than `n` reuses its last image; longer is trimmed; the clip count is always
  the `prompts` length). With `combine_videos` ON, clip `i`'s LAST frame is
  auto-taken as image `i+1`, which is what stitches the clips frame-to-frame,
  so there is no separate end-image input.
- Character refs (image + voice) are per-clip too; when combining, clip `i`
  sees BOTH sheet `i` and `i+1` for continuous zoom/pose transitions.
- `combine_videos` ON -> clips + `_master.mp4` in a dated subfolder; OFF ->
  one flat mp4 per clip. Video is streamed per clip (never a full master
  tensor in RAM).
- Not for: "one story, five scenes" — that's staged.

Supports (per user spec): N-per-index start images; a single start image
reused for **every** clip (spec #1/#4).

## Silver H3 Staged Multishot

Use for **one continuous story across N shots** (phase-staged: every prompt
text-encoded in ONE TE session, every shot sampled DiT-hot, every shot decoded
+ streamed in ONE VAE session).

### Continuity (auto-selected — the widget is gone)

The mode is implied by the inputs:

- **`start_image` provided** -> `first_frame`: every shot opens on the previous
  shot's DECODED last frame via H3's native keyframe channel; `_TRIM = 1`
  duplicate frame dropped per join. Shot 1 opens on the supplied image.
- **No `start_image`** -> `latent_handoff` (the default, refs or not): each
  shot's head rows are cloned from the previous shot's raw tail latents
  (5 video latent rows ~= 3.5s, 37 audio cols ~= 0.92s at 40 latent-fps) so
  the new shot "wakes up inside" the last shot's motion. The replayed decoded
  head is now TRIMMED (see join-trim fix, #2 closed). Character refs, the
  memory bank and `guide_audio` all ride in this mode.
- **`start_image` + character refs together**: refs are DROPPED with a warning
  (they wipe the keyframe latents); start-image chaining wins, and the TE
  tokenization is re-built WITHOUT the ref items so no orphan `<Picture N>`
  tokens survive (implemented in Phase A via `_DROP_REFS`).

### start_image semantics (staged, per spec — NOT YET BUILT beyond shot 1)

Currently staged keys **shot 1 only**. The following per-index spec is planned:

- **`len(start_image)` >= number of shots**: per-index — shot `i` opens on
  `start_image[i]`. Shot `i` ALSO gets `start_image[i+1]` as its END keyframe
  ("ease into the next shot's start") when that index exists; fl2v model
  required.
- **Single image**: classic chain — only shot 1 uses it; every later shot
  chains from the prior shot's decoded last frame.
- **Missing index** (`0 < len < n`): per-index up to the sheet's end, then
  FALL BACK to N's last decoded frame for N+1 (chain). Never reuse
  `start_image[last]` for the tail.

### Character references (both nodes, implemented)

- `char_refs_all` (IMAGE list — character, handbag, car, street,
  secondary character...) encoded ONCE; EVERY image rides EVERY clip/shot
  with stable `<Picture N>` labels.
- `char_refs_index` (IMAGE list) encoded ONCE; image `i` rides clip/shot `i`
  (a list shorter than the prompts reuses its last image). When combining,
  batch also rides `i+1` for continuous zoom/pose transitions.
- Both lists can be connected at once (deduped per clip/shot).
- Reference each image in the prompt as `<Picture N>` (e.g. "Picture 1 is
  the main character..."). With `char_refs_index`, labels shift per clip/shot.
- `char_audio` (ONE voice) encoded once and ridden by every clip/shot as an
  audio ref row + `<Audio N>` token. Only a single voice at a time.
- In handoff mode every ref block rides every shot (no FF keyframe guard).
- **Memory bank (staged)**: every shot banks the PREVIOUS shot only as two
  reference STILLS - its opening + mid rows, 1 latent row each - keeping
  identity AND location (every non-ref-imaged character/prop in frame)
  consistent when the camera pans between scenes; `memory_bank` = on/off.
  Latent handoff carries the tail's motion/identity separately; `bank_frames`
  is gone.

### guide_audio spine

One track encoded once, tiled if shorter than the run; each shot's audio
latents are locked to a time-slice (`int(round(si * (frames_per_shot - trim)
/ 24 * AUDIO_LATENT_FPS))`) through a per-step injection at the audio clock —
the video follows the locked audio, so the voice cannot drift between shots.

### Phase C streaming save

Always streamed save (no toggle): one mp4 per shot (`shot_01.mp4`, ...)
plus the combined master (`filename_prefix/yyyymmdd_HHMMSS/_master.mp4`),
audio pre-passed so the aac stream registers before video mux; each shot
decoded, join-trimmed (FF: 1 frame; handoff: the cloned head), freed
immediately. Outputs: `master_audio` (crossfaded master mix) and
`saved_video` (the master mp4 path). The old untrimmed latent / frame
outputs are gone — only the trimmed decoded mp4s are correct to use.

## Known bugs / findings (session)

1. **Batch node schema staleness**: after the node's inputs changed, OLD graphs
   keep the previous widget rows and misalign connected inputs. Fix in the
   workflow: recreate the node (or re-wire), not a code bug.
2. **Handoff join-replay (CLOSED)**: in `latent_handoff`, shot N+1's cloned
   head (~5 latent rows) was never trimmed in phase C (`_TRIM = 0` on the
   handoff path), so the master REPLAYED shot N's tail (scene-2 "driving"
   started by replaying scene-1 "getting in the car"). FIXED: handoff
   `_TRIM` is now `5 + 17*(_HO_ROWS-1)` = 73 decoded frames for a 5-row
clone, applied to shot 2+ heads (video AND audio stay sample-aligned). The
    join lands on the first row after the clone instead of replaying the scene.
3. **FF ref/vision-token asymmetry (CLOSED)**: in first_frame, shot 2+ dropped
   DiT ref rows to protect the keyframe but the TE still baked `<Picture N>`
   tokens into those shots' text. With the auto rule (refs -> handoff) this
   path only survives the start_image+refs "drop refs" branch, which now
   re-tokenizes without the ref items (`_DROP_REFS`, Phase A).
4. **Memory bank flooded new scenes (CLOSED)**: the bank used a sliding
   window over several prior shots' motion (up to ~18 ref rows at old
   defaults). Now banks the PREVIOUS shot only as two stills (opening + mid
   rows, 2 ref rows); `bank_frames` and `bank_depth` are gone with the end
   anchor (handoff covers the tail).

## Workflow matrix (agreed spec)

| You want... | Node | Inputs |
|---|---|---|
| 5 shots, each opens on its own start image | Batch | `start_image` 5 images, 5 prompts |
| Same one image reused for 5 clips | Batch | `start_image` 1 image, 5 prompts |
| 5 scenes, each eases into the next shot's start image | Staged (planned) | `start_image` >= 5 images, 5 prompts (fl2v) — NOT built yet; current staged keys shot 1 only |
| 1 start image, chain N's last frame to N+1 | Staged | `start_image` 1 image, 5 prompts |
| Same character refs across all scenes, continuous | Staged | `char_refs_all` (auto `latent_handoff`) |
| Every ref on every clip, stable labels | Both | `char_refs_all` |
| Per-index refs (clip i sees ref i) | Batch | `char_refs_index` |