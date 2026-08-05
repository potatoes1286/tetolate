import { SCREEN } from "./screens.js";

const COLORS = { ocr_raw: "#e11d48", ocr_merged: "#2563eb", ocr_structured: "#059669", translations: "#7c3aed", placements: "#d97706" };
const DRAG_UNLOCK_PIXELS = 7;
const RESIZE_HANDLE_VISUAL_PIXELS = 10;
const RESIZE_HANDLE_HIT_RADIUS_PIXELS = 14;

function oppositeColour(fill) { return fill === "white" ? "black" : "white"; }

function splitWord(ctx, word, maxWidth) {
  const parts = [];
  let current = "";
  for (const character of word) {
    const candidate = current + character;
    if (current && ctx.measureText(`${candidate}-`).width > maxWidth) {
      parts.push(`${current}-`);
      current = character;
    } else {
      current = candidate;
    }
  }
  if (current) parts.push(current);
  return parts;
}

function wrapText(ctx, text, maxWidth) {
  const lines = [];
  for (const paragraph of String(text).split("\n")) {
    const words = paragraph.trim().split(/\s+/).filter(Boolean);
    if (!words.length) { lines.push(""); continue; }
    let line = "";
    for (const rawWord of words) {
      const parts = ctx.measureText(rawWord).width <= maxWidth
        ? [rawWord]
        : splitWord(ctx, rawWord, maxWidth);
      for (const word of parts) {
        const candidate = line ? `${line} ${word}` : word;
        if (line && ctx.measureText(candidate).width > maxWidth) {
          lines.push(line); line = word;
        } else {
          line = candidate;
        }
      }
    }
    if (line) lines.push(line);
  }
  return lines;
}

export class EditorCanvas {
  constructor(canvas, store, onStatus) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.store = store;
    this.onStatus = onStatus;
    this.image = new Image();
    this.showPlacementText = true;
    this.busy = false;
    this.drag = null;
    this.drawAction = null;
    this.image.onload = () => { this.resize(); this.draw(); };
    canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
    canvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
    store.addEventListener("change", () => this.draw());
  }

  load(url, options = {}) {
    this.showPlacementText = options.showPlacementText !== false;
    const separator = url.includes("?") ? "&" : "?";
    this.image.src = `${url}${separator}v=${Date.now()}`;
  }
  setBusy(busy) {
    this.busy = !!busy;
    this.canvas.style.cursor = this.busy ? "progress" : "";
  }
  resize() { this.canvas.width = this.image.naturalWidth; this.canvas.height = this.image.naturalHeight; }
  point(event) {
    const box = this.canvas.getBoundingClientRect();
    return { x: (event.clientX - box.left) * this.canvas.width / box.width, y: (event.clientY - box.top) * this.canvas.height / box.height };
  }
  imagePixelsForScreenPixels(screenPixels) {
    const box = this.canvas.getBoundingClientRect();
    return {
      x: screenPixels * this.canvas.width / Math.max(1, box.width),
      y: screenPixels * this.canvas.height / Math.max(1, box.height),
    };
  }
  region(record, artifact) {
    return (artifact === "placements" ? record.placementRegion : record.region) || record.region || record._displayRegion || null;
  }
  visibleArtifacts() { return SCREEN[this.store.stage].artifacts; }

  draw() {
    if (!this.image.complete || !this.canvas.width) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(this.image, 0, 0);
    for (const artifact of this.visibleArtifacts().slice().reverse()) {
      for (const record of this.store.records[artifact] || []) {
        const region = this.region(record, artifact);
        if (!region) continue;
        const selected = this.store.selection?.recordId === record.recordId;
        this.ctx.strokeStyle = selected ? "#f59e0b" : (COLORS[artifact] || "#2563eb");
        this.ctx.lineWidth = selected ? 5 : 3;
        this.ctx.strokeRect(region[0], region[1], region[2] - region[0], region[3] - region[1]);
        this.ctx.fillStyle = this.ctx.strokeStyle;
        this.ctx.fillRect(region[0], Math.max(0, region[1] - 24), 44, 24);
        this.ctx.fillStyle = "white";
        this.ctx.font = "bold 18px sans-serif";
        this.ctx.fillText(String(record.boxno ?? "?"), region[0] + 5, Math.max(18, region[1] - 5));
        if (selected && SCREEN[this.store.stage].movable) {
          const handle = this.imagePixelsForScreenPixels(RESIZE_HANDLE_VISUAL_PIXELS);
          this.ctx.fillStyle = "#f59e0b";
          this.ctx.fillRect(
            region[2] - handle.x / 2,
            region[3] - handle.y / 2,
            handle.x,
            handle.y,
          );
          this.ctx.strokeStyle = "#111";
          this.ctx.lineWidth = Math.max(
            this.imagePixelsForScreenPixels(1.5).x,
            this.imagePixelsForScreenPixels(1.5).y,
          );
          this.ctx.strokeRect(
            region[2] - handle.x / 2,
            region[3] - handle.y / 2,
            handle.x,
            handle.y,
          );
        }
        if (artifact === "placements" && this.showPlacementText) {
          const translation = (this.store.records.translations || []).find((item) => item.boxno === record.boxno);
          const text = record.manualLineBreaks || translation?.englishText || "";
          if (text) {
            const size = record.fontSizeWidthPercent
              ? Math.max(1, this.canvas.width * Number(record.fontSizeWidthPercent) / 100)
              : Math.max(1, Number(record._roughPointSize) || Math.min((region[2] - region[0]) / 8, (region[3] - region[1]) / 5));
            const fontFamily = "sans-serif";
            const maxWidth = Math.max(1, (region[2] - region[0]) * 0.9);
            const fill = record.fill || "black";
            this.ctx.font = `${size}px ${fontFamily}`;
            const lines = record.manualLineBreaks
              ? String(record.manualLineBreaks).split("\n")
              : (record._roughText ? String(record._roughText).split("\n") : wrapText(this.ctx, text, maxWidth));
            this.ctx.fillStyle = fill;
            this.ctx.strokeStyle = record.stroke || oppositeColour(fill);
            this.ctx.lineWidth = Math.max(1, Number(record.strokeWidth) || 2);
            this.ctx.lineJoin = "round";
            this.ctx.textAlign = "center";
            const lineHeight = size * 1.15;
            const startY = (region[1] + region[3] - lines.length * lineHeight) / 2 + size;
            lines.forEach((line, index) => {
              const x = (region[0] + region[2]) / 2;
              const y = startY + index * lineHeight;
              this.ctx.strokeText(line, x, y, maxWidth);
              this.ctx.fillText(line, x, y, maxWidth);
            });
            this.ctx.textAlign = "start";
          }
        }
      }
    }
    if (this.drag?.mode === "draw") {
      const { start, current } = this.drag;
      this.ctx.strokeStyle = "#f59e0b"; this.ctx.lineWidth = 4;
      this.ctx.strokeRect(start.x, start.y, current.x - start.x, current.y - start.y);
    }
  }

  startDraw(action) { this.drawAction = action; this.onStatus("Drag a region on the page."); }
  selectedResizeHit(point) {
    if (!SCREEN[this.store.stage].movable || !this.store.selection) return null;
    const artifact = this.store.selection.artifact;
    const record = (this.store.records[artifact] || []).find(
      (item) => item.recordId === this.store.selection.recordId,
    );
    const region = record ? this.region(record, artifact) : null;
    if (!region) return null;
    const radius = this.imagePixelsForScreenPixels(RESIZE_HANDLE_HIT_RADIUS_PIXELS);
    if (
      Math.abs(point.x - region[2]) <= radius.x
      && Math.abs(point.y - region[3]) <= radius.y
    ) {
      return { artifact, record, region };
    }
    return null;
  }
  hits(point) {
    const hits = [];
    for (const artifact of this.visibleArtifacts()) {
      for (const record of this.store.records[artifact] || []) {
        const region = this.region(record, artifact);
        if (region && point.x >= region[0] && point.x <= region[2] && point.y >= region[1] && point.y <= region[3]) hits.push({ artifact, record, region });
      }
    }
    return hits;
  }
  pointerDown(event) {
    if (this.busy) { this.onStatus("Wait for the current OCR request to finish."); return; }
    const point = this.point(event);
    if (this.drawAction) { this.drag = { mode: "draw", start: point, current: point, action: this.drawAction }; return; }
    const resizeHit = this.selectedResizeHit(point);
    const hit = resizeHit || this.hits(point)[0];
    if (!hit) { this.store.select(null, null); return; }
    this.store.select(hit.artifact, hit.record.recordId);
    if (!SCREEN[this.store.stage].movable) return;
    const radius = this.imagePixelsForScreenPixels(RESIZE_HANDLE_HIT_RADIUS_PIXELS);
    const resize = !!resizeHit || (
      Math.abs(point.x - hit.region[2]) <= radius.x
      && Math.abs(point.y - hit.region[3]) <= radius.y
    );
    this.drag = {
      mode: resize ? "resize" : "move",
      start: point,
      pointerStart: { x: event.clientX, y: event.clientY },
      unlocked: false,
      original: [...hit.region],
      artifact: hit.artifact,
      recordId: hit.record.recordId,
    };
    this.canvas.setPointerCapture?.(event.pointerId);
  }
  pointerMove(event) {
    if (!this.drag) return;
    const point = this.point(event);
    if (this.drag.mode === "draw") { this.drag.current = point; this.draw(); return; }
    if (!this.drag.unlocked) {
      const distance = Math.hypot(
        event.clientX - this.drag.pointerStart.x,
        event.clientY - this.drag.pointerStart.y,
      );
      if (distance < DRAG_UNLOCK_PIXELS) return;
      this.drag.unlocked = true;
      this.store.checkpoint();
    }
    const dx = point.x - this.drag.start.x, dy = point.y - this.drag.start.y;
    const [l, t, r, b] = this.drag.original;
    const region = this.drag.mode === "move" ? [l + dx, t + dy, r + dx, b + dy] : [l, t, Math.max(l + 2, r + dx), Math.max(t + 2, b + dy)];
    const key = this.drag.artifact === "placements" ? "placementRegion" : "region";
    this.store.updateSelected(
      { [key]: region.map(Math.round), _roughText: null, _roughPointSize: null },
      { history: false },
    );
  }
  pointerUp(event) {
    if (this.drag?.mode === "draw") {
      const point = this.point(event), start = this.drag.start, action = this.drag.action;
      const region = [Math.min(start.x, point.x), Math.min(start.y, point.y), Math.max(start.x, point.x), Math.max(start.y, point.y)].map(Math.round);
      this.drag = null; this.drawAction = null; this.draw();
      if (region[2] - region[0] > 4 && region[3] - region[1] > 4) action(region);
      return;
    }
    this.drag = null;
  }
}
