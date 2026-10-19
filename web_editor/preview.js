// Text layers are prepared first. Cleaning never blocks page navigation.
export class PreviewController {
  constructor(store, canvas, api, status) {
    this.store = store;
    this.canvas = canvas;
    this.api = api;
    this.status = status;
    this.active = false;
    this.generation = 0;
    this.timer = null;
    this.request = null;
    this.key = null;
    store.addEventListener("change", () => { if (this.active) this.refresh(); });
  }

  hide() {
    this.active = false;
    this.generation++;
    clearTimeout(this.timer);
    this.timer = null;
    this.request?.abort();
    this.request = null;
    this.key = null;
  }

  show() {
    this.hide();
    this.active = true;
    this.refresh(true);
  }

  refresh(initial = false) {
    if (this.store.stage !== "placement" || !this.store.document) return;
    const records = (this.store.records.ocr_structured || []).filter((record) => !record.openLettering);
    const boxes = this.store.records.clean_boxes || [];
    const key = JSON.stringify([this.store.page, records.map((record) => record.region), boxes.map((record) => record.region)]);
    if (!initial && key === this.key && this.document === this.store.document) return;
    this.document = this.store.document;
    this.key = key;
    this.request?.abort();
    clearTimeout(this.timer);
    const generation = ++this.generation;
    const document = this.store.document;
    const page = this.store.page;
    const cleanRecords = structuredClone(records), cleanBoxes = structuredClone(boxes);
    const run = () => {
      this.timer = null;
      void this.render({generation, document, page, cleanRecords, cleanBoxes, initial});
    };
    if (initial) run();
    else this.timer = setTimeout(run, 250);
  }

  current(context) {
    return this.active && context.generation === this.generation
      && context.document === this.store.document && context.page === this.store.page;
  }

  async render(context) {
    const request = new AbortController();
    this.request = request;
    try {
      if (context.initial) {
        const loaded = await this.canvas.load(context.document.image.cleanedUrl || context.document.image.url);
        if (loaded === false) throw new Error("Could not load the page background.");
        if (!this.current(context)) return;
      }
      this.status("Preparing ImageMagick text layers...");
      await this.canvas.prepareTextLayers();
      if (!this.current(context)) return;
      this.status("Text layers ready. Cleaning the background with LaMa...");
      const result = await this.api.cleanPreview(context.page, context.cleanRecords, context.cleanBoxes, request.signal);
      if (!this.current(context)) return;
      await this.canvas.load(result.url);
      if (this.current(context)) this.status("Preview updated. Text layers and cleaned background are current.");
    } catch (error) {
      if (this.current(context) && error.name !== "AbortError") {
        this.key = null; // A later edit or view selection can retry a failed request.
        this.status(`Could not update Preview: ${error.message}`, true);
      }
    } finally {
      if (this.request === request) this.request = null;
    }
  }
}
