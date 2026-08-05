const MAX_HISTORY = 100;

function clone(value) { return structuredClone(value); }

export class EditorStore extends EventTarget {
  constructor(config) {
    super();
    this.config = config;
    this.page = 0;
    this.stage = "ocr";
    this.document = null;
    this.records = {};
    this.selection = null;
    this.dirty = false;
    this.history = [];
    this.future = [];
  }

  draftKey(page = this.page, stage = this.stage) {
    return `tetolate-editor-v2:${this.config.category}:${this.config.jobId}:${page}:${stage}`;
  }

  setDocument(document) {
    this.document = document;
    this.page = document.page;
    this.stage = document.stage;
    this.records = clone(document.recordsByArtifact || {});
    this.selection = null;
    this.dirty = false;
    this.history = [];
    this.future = [];
    const rawDraft = localStorage.getItem(this.draftKey());
    if (rawDraft) {
      try {
        const draft = JSON.parse(rawDraft);
        if (draft.baseRevision === document.revision && draft.records) {
          this.records = draft.records;
          if (draft.translationNotes) this.document.translationNotes = draft.translationNotes;
          this.dirty = true;
        } else {
          localStorage.removeItem(this.draftKey());
        }
      } catch { localStorage.removeItem(this.draftKey()); }
    }
    this.attachDisplayRegions();
    this.emit();
  }

  attachDisplayRegions() {
    const structured = new Map(
      (this.records.ocr_structured || []).map((item) => [item.boxno, item]),
    );
    for (const artifact of ["translations", "placements"]) {
      for (const record of this.records[artifact] || []) {
        const source = structured.get(record.boxno);
        if (source?.region) record._displayRegion = [...source.region];
      }
    }
  }

  snapshot() {
    return { records: clone(this.records), selection: clone(this.selection) };
  }

  checkpoint() {
    this.history.push(this.snapshot());
    if (this.history.length > MAX_HISTORY) this.history.shift();
    this.future = [];
  }

  mutate(callback, options = {}) {
    if (options.history !== false) this.checkpoint();
    callback(this.records);
    this.dirty = true;
    this.saveDraft();
    this.emit();
  }

  undo() {
    const previous = this.history.pop();
    if (!previous) return;
    this.future.push(this.snapshot());
    this.records = previous.records;
    this.selection = previous.selection;
    this.dirty = true;
    this.saveDraft();
    this.emit();
  }

  redo() {
    const next = this.future.pop();
    if (!next) return;
    this.history.push(this.snapshot());
    this.records = next.records;
    this.selection = next.selection;
    this.dirty = true;
    this.saveDraft();
    this.emit();
  }

  select(artifact, recordId) {
    this.selection = recordId ? { artifact, recordId } : null;
    this.emit();
  }

  selectedRecord() {
    if (!this.selection) return null;
    return (this.records[this.selection.artifact] || []).find(
      (item) => item.recordId === this.selection.recordId,
    ) || null;
  }

  updateSelected(values, options = {}) {
    if (!this.selection) return;
    const { artifact, recordId } = this.selection;
    this.updateRecord(artifact, recordId, values, options);
  }

  updateRecord(artifact, recordId, values, options = {}) {
    this.mutate(() => {
      const record = (this.records[artifact] || []).find((item) => item.recordId === recordId);
      if (record) Object.assign(record, values);
    }, options);
  }

  deleteSelected() {
    if (!this.selection) return;
    const { artifact, recordId } = this.selection;
    this.mutate(() => {
      this.records[artifact] = (this.records[artifact] || []).filter(
        (item) => item.recordId !== recordId,
      );
      this.selection = null;
    });
  }

  saveDraft() {
    if (!this.document || !this.dirty) return;
    localStorage.setItem(this.draftKey(), JSON.stringify({
      baseRevision: this.document.revision,
      records: this.records,
      translationNotes: this.document.translationNotes,
      savedAt: Date.now(),
    }));
  }

  markSaved(document) {
    localStorage.removeItem(this.draftKey());
    this.setDocument(document);
  }

  emit() { this.dispatchEvent(new Event("change")); }
}
