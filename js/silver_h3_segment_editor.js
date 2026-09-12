// Silver H3 Segment Editor - in-node preview, playback and range picking.
//
// Adds a DOM panel to the SilverH3SegmentEditor node that:
//   - previews/plays the selected source video (served from the input folder),
//   - lets you UPLOAD a new video straight from the node,
//   - picks the SECTION (crop) of the main video via source_start/source_end,
//   - picks the SEGMENT to regenerate via timeline_start/timeline_duration.
//
// All values are pushed back into the real ComfyUI widgets, so the workflow
// serialises them and the Python node reads the exact same values.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "SilverH3SegmentEditor";
const RANGE_WIDGETS = [
    "source_video",
    "source_start",
    "source_end",
    "timeline_start",
    "timeline_duration",
];

const round1 = (v) => Math.round((Number(v) || 0) * 10) / 10;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (t) => {
    if (!isFinite(t) || t < 0) t = 0;
    const m = Math.floor(t / 60);
    const s = Math.round(t % 60);
    if (s === 60) return fmt(m * 60);
    return `${m}:${String(s).padStart(2, "0")}`;
};

// Build a /view URL for an input-folder name like "video.mp4" or
// "subdir/video.mp4" (get_filename_list uses "/" for subfolders).
function inputFileUrl(file) {
    if (!file || typeof file !== "string") return null;
    file = file.trim();
    if (!file) return null;
    const idx = file.lastIndexOf("/");
    let filename = file;
    let subfolder = "";
    if (idx > 0) {
        filename = file.slice(idx + 1);
        subfolder = file.slice(0, idx);
    }
    let url = `/view?filename=${encodeURIComponent(filename)}&type=input`;
    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
    return api.apiURL(url);
}

// ---- dual-thumb range slider (drag the two thumbs, no track jump) --------
function makeDualRange(options = {}) {
    const wrap = document.createElement("div");
    wrap.style.cssText =
        "position:relative;height:22px;width:100%;user-select:none;";

    const fill = document.createElement("div");
    fill.style.cssText =
        "position:absolute;top:9px;left:0;width:0;height:4px;" +
        "border-radius:2px;background:rgba(179,255,0,.55);" +
        "pointer-events:none;mix-blend-mode:screen;";

    const common =
        "position:absolute;inset:0;width:100%;height:100%;margin:0;" +
        "background:transparent;-webkit-appearance:none;appearance:none;" +
        "pointer-events:none;outline:none;";
    const thumb =
        "-webkit-appearance:none;appearance:none;height:14px;width:14px;" +
        "border-radius:50%;background:#b3ff00;border:1px solid #333;" +
        "cursor:ew-resize;pointer-events:auto;box-shadow:0 0 2px rgba(0,0,0,.6);";
    const styles = document.createElement("style");
    styles.textContent = `
.du-wrap input[type=range]::-webkit-slider-runnable-track{background:transparent;border:none;}
.du-wrap input[type=range]::-moz-range-track{background:transparent;border:none;}
.du-wrap input[type=range]::-webkit-slider-thumb{${thumb}}
.du-wrap input[type=range]::-moz-range-thumb{${thumb}}
.du-wrap input[type=range]::-moz-range-progress{background:transparent;}
`;

    const lo = document.createElement("input");
    lo.type = "range";
    lo.style.cssText = common + "z-index:3;";
    const hi = document.createElement("input");
    hi.type = "range";
    hi.style.cssText = common + "z-index:4;";

    wrap.className = "du-wrap";
    wrap.appendChild(styles);
    wrap.appendChild(lo);
    wrap.appendChild(hi);
    wrap.appendChild(fill);

    const state = { min: 0, max: 0, step: 0.1, oninput: null };

    const handlers = [lo, hi];
    for (const el of handlers) {
        el.addEventListener("input", () => {
            if (state.oninput) state.oninput(render());
        });
    }

    function render() {
        const lv = parseFloat(lo.value);
        const hv = parseFloat(hi.value);
        const span = Math.max(1e-6, state.max - state.min);
        const lp = ((lv - state.min) / span) * 100;
        const hp = ((hv - state.min) / span) * 100;
        fill.style.left = `${lp}%`;
        fill.style.width = `${Math.max(0, hp - lp)}%`;
        return [lv, hv];
    }

    return {
        el: wrap,
        setBounds(min, max, step = 0.1) {
            state.min = min;
            state.max = max;
            state.step = step;
            lo.min = hi.min = min;
            lo.max = hi.max = max;
            lo.step = hi.step = step;
            hi.value = max;
            lo.value = min;
            render();
        },
        set(l, h) {
            lo.value = clamp(l, state.min, state.max);
            hi.value = clamp(h, state.min, state.max);
            render();
        },
        get() {
            const [l, h] = render();
            return [clamp(l, state.min, state.max), clamp(h, state.min, state.max)];
        },
        refresh: render,
        onInput(cb) {
            state.oninput = cb;
        },
    };
}

// ---------------------------------------------------------------------------
app.registerExtension({
    name: "SilverH3.SegmentEditorPreview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.call(this);
            try {
                initPreview(this);
            } catch (err) {
                console.error("[SilverH3SegmentEditor] preview init failed:", err);
            }
        };
    },
});

function initPreview(node) {
    const W = {};
    for (const w of node.widgets || []) {
        if (RANGE_WIDGETS.includes(w.name)) W[w.name] = w;
    }
    if (!W.source_video) return;

    const root = document.createElement("div");
    root.style.cssText =
        "box-sizing:border-box;width:100%;padding:6px 8px 8px;" +
        "font-size:12px;color:#ddd;background:#1a1a1a;border-radius:4px;";

    // ---- video ------------------------------------------------------------
    const video = document.createElement("video");
    video.playsInline = true;
    video.preload = "metadata";
    video.style.cssText =
        "width:100%;max-height:190px;background:#000;display:block;" +
        "border-radius:3px;";
    root.appendChild(video);

    const status = document.createElement("div");
    status.style.cssText =
        "padding:2px 0;color:#999;font-size:11px;white-space:nowrap;" +
        "overflow:hidden;text-overflow:ellipsis;";
    root.appendChild(status);

    // ---- transport --------------------------------------------------------
    const transport = document.createElement("div");
    transport.style.cssText =
        "display:flex;align-items:center;gap:6px;margin:4px 0 10px;";

    const playBtn = document.createElement("button");
    playBtn.textContent = "\u25B6";
    playBtn.title = "Play / pause the source video";
    transport.appendChild(playBtn);

    const seek = document.createElement("input");
    seek.type = "range";
    seek.min = 0;
    seek.max = 0;
    seek.step = 0.1;
    seek.style.cssText = "flex:1;";
    transport.appendChild(seek);

    const timeLbl = document.createElement("span");
    timeLbl.textContent = "0:00 / 0:00";
    timeLbl.style.cssText = "min-width:70px;text-align:right;";
    transport.appendChild(timeLbl);
    root.appendChild(transport);

    // ---- SECTION (crop) + SEGMENT range rows ------------------------------
    function rangeRow(label, color) {
        const row = document.createElement("div");
        row.style.cssText = "margin:2px 0 6px;";

        const head = document.createElement("div");
        head.style.cssText =
            "display:flex;justify-content:space-between;align-items:center;" +
            "margin-bottom:1px;";
        const lab = document.createElement("span");
        lab.textContent = label;
        lab.style.cssText = `color:${color};font-weight:600;`;
        const nums = document.createElement("span");
        nums.style.cssText = "font-variant-numeric:tabular-nums;color:#bbb;";
        head.appendChild(lab);
        head.appendChild(nums);
        row.appendChild(head);

        const dual = makeDualRange();
        row.appendChild(dual.el);

        const loInput = document.createElement("input");
        loInput.type = "number";
        loInput.step = 0.1;
        loInput.style.cssText = "width:58px;margin-right:4px;";
        const hiInput = document.createElement("input");
        hiInput.type = "number";
        hiInput.step = 0.1;
        hiInput.style.cssText = "width:58px;";
        const numBar = document.createElement("div");
        numBar.style.cssText =
            "display:flex;align-items:center;margin-top:2px;color:#888;";
        numBar.append(loInput, document.createTextNode("  \u2013  "), hiInput);
        row.appendChild(numBar);
        return { row, dual, loInput, hiInput, nums };
    }

    const section = rangeRow("SECTION (main video used)", "#8fdfff");
    const segment = rangeRow("SEGMENT (regenerated)", "#b3ff00");
    root.appendChild(section.row);
    root.appendChild(segment.row);

    // ---- open / upload ----------------------------------------------------
    const openBtn = document.createElement("button");
    openBtn.textContent = "\uD83D\uDCC2 Open video \u2026";
    openBtn.title = "Upload a source video into the input folder";
    openBtn.style.cssText = "margin-top:6px;";
    root.appendChild(openBtn);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "video/*";
    fileInput.style.display = "none";
    root.appendChild(fileInput);

    const domWidget = node.addDOMWidget("segment_preview", "preview", root, {
        serialize: false,
        getMinHeight: () => 430,
        getMaxHeight: () => 620,
        hideOnZoom: false,
    });

    // Re-home the source_video picker so it is the LAST input, sitting
    // directly above the preview panel.
    const dwIdx = node.widgets.indexOf(domWidget);
    const wIdx = node.widgets.indexOf(W.source_video);
    if (dwIdx >= 0 && wIdx >= 0 && wIdx !== dwIdx - 1) {
        node.widgets.splice(wIdx, 1);
        node.widgets.splice(node.widgets.indexOf(domWidget), 0, W.source_video);
        node.graph?.setDirtyCanvas(true, true);
    }

    // ---- state + sync -----------------------------------------------------
    const S = {
        duration: 0,
        loaded: false,
        syncing: false,
    };

    const getDuration = () =>
        S.loaded && isFinite(video.duration) && video.duration > 0
            ? video.duration
            : 0;

    function sectionBounds() {
        const dur = getDuration();
        if (dur <= 0) return { lo: 0, hi: 0, dur: 0 };
        let lo = clamp(round1(W.source_start.value) || 0, 0, dur);
        let hi = round1(W.source_end.value);
        if (hi <= 0) hi = dur; // 0 = "to the end"
        hi = clamp(hi, lo, dur);
        return { lo, hi, dur };
    }

    function clampSegInSection(sec) {
        const dur = getDuration();
        if (dur <= 0) return [0, 0];
        let start = clamp(round1(W.timeline_start.value) || 0, sec.lo, sec.hi);
        let len = clamp(round1(W.timeline_duration.value) || 0, 0.1, sec.hi - start);
        return [round1(start), round1(len)];
    }

    function widgetsToUI() {
        if (S.syncing) return;
        const sec = sectionBounds();
        const dur = sec.dur;
        section.dual.setBounds(0, dur || 1, 0.1);
        section.dual.set(sec.lo, sec.hi);
        section.loInput.value = round1(sec.lo);
        section.hiInput.value = round1(sec.hi === dur && W.source_end.value <= 0 ? 0 : sec.hi);
        section.nums.textContent =
            dur > 0
                ? `[${round1(sec.lo)} \u2013 ${sec.hi >= dur ? "end" : round1(sec.hi)}]s`
                : "";

        const [s0, d0] = clampSegInSection(sec);
        segment.dual.setBounds(sec.lo, Math.max(sec.lo, sec.hi), 0.1);
        segment.dual.set(s0, s0 + d0);
        segment.loInput.value = s0;
        segment.hiInput.value = round1(d0);
        segment.nums.textContent =
            dur > 0 ? `${round1(s0)} \u2192 ${round1(s0 + d0)}s (+${round1(d0)}s)` : "";
    }

    function pushWidget(name, value) {
        const w = W[name];
        if (!w) return;
        w.value = value;
        if (typeof w.callback === "function") w.callback(w.value);
        if (node.graph?.setDirtyCanvas) node.graph.setDirtyCanvas(true, true);
    }

    function sliderState() {
        const dur = getDuration();
        const [slo, shi] = section.dual.get();
        const [glo, ghi] = segment.dual.get();
        return { dur, slo, shi, glo, ghi };
    }

    // Write the panel's number fields + labels straight from the slider
    // positions. Independent of the node widgets so a drag always shows.
    function writeFields(s) {
        section.loInput.value = round1(s.slo);
        section.hiInput.value =
            s.dur > 0 && s.shi >= s.dur - 1e-6 ? 0 : round1(s.shi);
        section.nums.textContent =
            s.dur > 0
                ? `[${round1(s.slo)} \u2013 ${s.shi >= s.dur ? "end" : round1(s.shi)}]s`
                : "";

        const segStart = round1(clamp(s.glo, s.slo, s.shi));
        const segEnd = round1(clamp(s.ghi, segStart, s.shi));
        segment.loInput.value = segStart;
        segment.hiInput.value = round1(segEnd - segStart);
        segment.nums.textContent =
            s.dur > 0 ? `${round1(segStart)} \u2192 ${round1(segEnd)}s (+${round1(segEnd - segStart)}s)` : "";
    }

    function uiToWidgets() {
        if (S.syncing) return;
        const s = sliderState();
        const dur = s.dur;
        if (dur <= 0) return;
        S.syncing = true;
        try {
            pushWidget("source_start", round1(s.slo));
            pushWidget("source_end", s.shi >= dur - 1e-6 ? 0 : round1(s.shi));

            const segStart = clamp(s.glo, s.slo, s.shi);
            const segEnd = clamp(s.ghi, segStart, s.shi);
            pushWidget("timeline_start", round1(segStart));
            pushWidget("timeline_duration", round1(Math.max(0.1, segEnd - segStart)));
        } finally {
            S.syncing = false;
        }
    }

    section.dual.onInput(() => {
        if (getDuration() <= 0) return;
        const [slo, shi] = section.dual.get();
        const [glo, ghi] = segment.dual.get();
        const len = Math.max(0.1, Math.min(shi - slo, ghi - glo));
        const s = clamp(glo, slo, shi - len);
        segment.dual.setBounds(slo, shi, 0.1);
        segment.dual.set(s, s + len);
        writeFields(sliderState());
        uiToWidgets();
        widgetsToUI();
    });

    segment.dual.onInput(() => {
        if (getDuration() <= 0) return;
        writeFields(sliderState());
        uiToWidgets();
        widgetsToUI();
    });

    for (const [name, dual, picker] of [
        ["source_start", section.dual, section.loInput],
        ["source_end", section.dual, section.hiInput],
        ["timeline_start", segment.dual, segment.loInput],
        ["timeline_duration", segment.dual, segment.hiInput],
    ]) {
        picker.addEventListener("change", () => {
            if (S.syncing) return;
            let v = round1(picker.value);
            const dur = getDuration();
            if (name === "source_start") v = clamp(v, 0, dur || 0);
            else if (name === "source_end")
                v = dur > 0 && (v <= 0 || v >= dur) ? 0 : clamp(v, 0, dur);
            else if (name === "timeline_start") {
                const sec = sectionBounds();
                v = clamp(v, sec.lo, sec.hi);
                const dlen = round1(W.timeline_duration.value) || 0;
                pushWidget(
                    "timeline_duration",
                    round1(clamp(dlen, 0.1, sec.hi - v))
                );
            } else if (name === "timeline_duration") {
                const sec = sectionBounds();
                const s0 = clamp(round1(W.timeline_start.value) || 0, sec.lo, sec.hi);
                v = Math.max(0.1, clamp(v, 0.1, sec.hi - s0));
            }
            pushWidget(name, v);
            widgetsToUI();
        });
    }

    // ---- video load/sync --------------------------------------------------
    function loadFromWidget() {
        const file = typeof W.source_video.value === "string" ? W.source_video.value : "";
        S.loaded = false;
        S.duration = 0;
        if (!file) {
            video.removeAttribute("src");
            status.textContent = "No source video selected.";
            widgetsToUI();
            return;
        }
        const url = inputFileUrl(file);
        if (!url) {
            status.textContent = `Cannot preview ${file} (must live in the input folder).`;
            widgetsToUI();
            return;
        }
        video.src = url;
        video.pause();
        playBtn.textContent = "\u25B6";
        seek.value = 0;
        timeLbl.textContent = "0:00 / 0:00";
        status.textContent = `Preview: ${file}`;
        video.onloadedmetadata = () => {
            S.loaded = true;
            S.duration = video.duration;
            seek.max = video.duration;
            seek.value = 0;
            segmentsSeekToStart();
            widgetsToUI();
        };
        video.onerror = () => {
            S.loaded = false;
            status.textContent = `Could not load preview for ${file}.`;
            widgetsToUI();
        };
    }

    function segmentsSeekToStart() {
        const sec = sectionBounds();
        if (sec.lo > 0) video.currentTime = sec.lo;
    }

    // transport
    playBtn.addEventListener("click", () => {
        if (!S.loaded) return;
        if (video.paused) {
            video.play().catch(() => {});
            playBtn.textContent = "\u23F8";
        } else {
            video.pause();
            playBtn.textContent = "\u25B6";
        }
    });
    video.addEventListener("ended", () => {
        playBtn.textContent = "\u25B6";
    });
    video.addEventListener("timeupdate", () => {
        if (!S.loaded) return;
        seek.value = video.currentTime;
        timeLbl.textContent = `${fmt(video.currentTime)} / ${fmt(video.duration || 0)}`;
    });
    seek.addEventListener("input", () => {
        if (!S.loaded) return;
        video.currentTime = parseFloat(seek.value) || 0;
        timeLbl.textContent = `${fmt(video.currentTime)} / ${fmt(video.duration || 0)}`;
    });

    // upload
    openBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async () => {
        const file = fileInput.files && fileInput.files[0];
        fileInput.value = "";
        if (!file) return;
        try {
            const body = new FormData();
            body.append("image", file, file.name);
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });
            if (!resp.ok) {
                status.textContent = `Upload failed (HTTP ${resp.status}).`;
                return;
            }
            const data = await resp.json();
            const value = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
            const comboOpts = Array.isArray(W.source_video.options?.values)
                ? W.source_video.options.values
                : null;
            if (comboOpts && !comboOpts.includes(value)) comboOpts.push(value);
            W.source_video.value = value;
            if (typeof W.source_video.callback === "function")
                W.source_video.callback(value);
            loadFromWidget();
        } catch (err) {
            console.error("[SilverH3SegmentEditor] upload failed:", err);
            status.textContent = "Upload failed - see browser console.";
        }
    });

    const srcCb = W.source_video.callback;
    W.source_video.callback = (v) => {
        srcCb?.call(W.source_video, v);
        loadFromWidget();
    };
    for (const name of ["source_start", "source_end", "timeline_start", "timeline_duration"]) {
        const widget = W[name];
        const old = widget.callback;
        widget.callback = (v) => {
            old?.call(widget, v);
            if (!S.syncing) widgetsToUI();
        };
    }

    loadFromWidget();
}