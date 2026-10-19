import { SCREEN } from "./screens.js";

const COLORS = { ocr_raw: "#e11d48", ocr_merged: "#2563eb", ocr_structured: "#059669", translations: "#7c3aed", placements: "#d97706" };
const DRAG_UNLOCK_PIXELS = 7;
const RESIZE_HANDLE_VISUAL_PIXELS = 10;
const RESIZE_HANDLE_HIT_RADIUS_PIXELS = 14;

function roundPixel(value) {
  const floor = Math.floor(value);
  return value - floor === 0.5 ? floor + Math.abs(floor % 2) : Math.round(value);
}

export class EditorCanvas {
  constructor(canvas, store, onStatus, renderTextLayers) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.store = store;
    this.onStatus = onStatus;
    this.image = new Image();
    this.showPlacementText = true;
    this.busy = false;
    this.drag = null;
    this.drawAction = null;
    this.layers = new Map();
    this.layerTimer = null;
    this.renderingLayers = false;
    this.renderTextLayers = renderTextLayers;
    this.image.onload = () => { this.resize(); this.draw(); };
    canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
    canvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
    store.addEventListener("change", () => this.draw());
  }

  load(url, options = {}) {
    this.showPlacementText = options.showPlacementText !== false;
    if (!this.showPlacementText) {
      clearTimeout(this.layerTimer);
      this.layerTimer = null;
      this.trimLayers(new Set());
    }
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

  drawTextLayer(record, required) {
    const region = this.region(record, "placements");
    if (!region) return;
    const text = record.manualLineBreaks || (this.store.records.translations || []).find((item) => item.boxno === record.boxno)?.englishText || "";
    if (!text.trim()) return;
    const [left, top, right, bottom] = region.map((value, index) => roundPixel(Math.max(0,
      Math.min(index % 2 ? this.canvas.height : this.canvas.width, value))));
    const width = right - left, height = bottom - top;
    if (width <= 0 || height <= 0) return;
    const entry = {placementRegion: [0, 0, width, height], englishText: text};
    for (const field of ["font", "fill", "stroke", "strokeWidth", "gravity", "fontSizeWidthPercent", "minPointSize", "maxPointSize", "renderBleed", "rotation", "skewX", "skewY"])
      if (record[field] !== undefined) entry[field] = record[field];
    const page = this.store.page;
    const key = JSON.stringify([page, this.canvas.width, this.canvas.height, this.store.document?.defaultFont, entry]);
    required.add(key);
    let layer = this.layers.get(key);
    if (!layer) {
      layer = {entry, page, state: "queued"};
      this.layers.set(key, layer);
    }
    if (layer.bitmap) this.ctx.drawImage(layer.bitmap,
      left + Math.floor((width - layer.bitmap.width) / 2), top + Math.floor((height - layer.bitmap.height) / 2));
  }

  scheduleLayers() {
    clearTimeout(this.layerTimer);
    this.layerTimer = null;
    if (this.renderingLayers || ![...this.layers.values()].some((layer) => layer.state === "queued")) return;
    this.layerTimer = setTimeout(() => this.flushLayers(), 200);
  }

  async flushLayers() {
    this.layerTimer = null;
    const batch = [...this.layers].filter(([_key, layer]) => layer.state === "queued").slice(0, 256);
    if (!batch.length || this.renderingLayers) return;
    this.renderingLayers = true;
    for (const [_key, layer] of batch) layer.state = "pending";
    this.onStatus(`Rendering ${batch.length} preview text layers...`);
    try {
      const bitmaps = await this.renderTextLayers(batch[0][1].page, batch.map(([_key, layer]) => layer.entry));
      if (!Array.isArray(bitmaps) || bitmaps.length !== batch.length) {
        for (const bitmap of bitmaps || []) bitmap.close();
        throw new Error("The preview batch returned an incorrect layer count.");
      }
      batch.forEach(([key, layer], index) => {
        if (this.layers.get(key) === layer) { layer.bitmap = bitmaps[index]; layer.state = "ready"; }
        else bitmaps[index].close();
      });
      if (batch.some(([key, layer]) => this.layers.get(key) === layer))
        this.onStatus("Preview text rendered with ImageMagick.");
    } catch (error) {
      let active = false;
      for (const [key, layer] of batch) {
        if (this.layers.get(key) === layer) { layer.state = "failed"; active = true; }
      }
      if (active) this.onStatus(`Could not render preview text: ${error.message}`, true);
    } finally {
      this.renderingLayers = false;
      this.draw();
    }
  }

  trimLayers(required) {
    for (const [key, layer] of this.layers) {
      if (!required.has(key) && (!layer.bitmap || this.layers.size > 128)) {
        layer.bitmap?.close();
        this.layers.delete(key);
      }
    }
  }

  draw() {
    if (this.layerDocument !== this.store.document) {
      for (const layer of this.layers.values()) {
        layer.bitmap?.close();
      }
      this.layers.clear();
      this.layerDocument = this.store.document;
    }
    if (!this.image.complete || !this.canvas.width) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(this.image, 0, 0);
    const required = new Set();
    if (this.store.stage === "placement" && this.showPlacementText)
      for (const record of this.store.records.placements || []) this.drawTextLayer(record, required);
    this.trimLayers(required);
    this.scheduleLayers();
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
      { [key]: region.map(Math.round) },
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
