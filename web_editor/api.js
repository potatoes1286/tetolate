const jsonHeaders = { "Content-Type": "application/json" };

async function checkedResponse(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = response.statusText;
    const body = await response.text();
    try {
      const data = JSON.parse(body);
      message = data.detail || data.error?.message || message;
    } catch {
      message = body.trim().startsWith("<") ? message : body.trim() || message;
    }
    const error = new Error(message || `Request failed (HTTP ${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return response;
}

async function request(url, options = {}) {
  return (await checkedResponse(url, options)).json();
}

export class EditorApi {
  constructor(category, jobId) {
    this.root = `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/editor/v2`;
    this.cleaning = false;
    this.pendingCleaning = null;
  }

  load(stage, page, signal) {
    return request(`${this.root}/pages/${page}/stages/${stage}`, {signal});
  }

  save(stage, page, body) {
    return request(`${this.root}/pages/${page}/stages/${stage}`, {
      method: "POST", headers: jsonHeaders, body: JSON.stringify(body),
    });
  }

  reset(stage, page, baseRevision) {
    return request(`${this.root}/pages/${page}/stages/${stage}/reset`, {
      method: "POST", headers: jsonHeaders, body: JSON.stringify({baseRevision}),
    });
  }

  protect(stage, page, body) {
    return request(`${this.root}/pages/${page}/stages/${stage}/protection`, {
      method: "POST", headers: jsonHeaders, body: JSON.stringify(body),
    });
  }

  regenerate(stage, page) {
    return request(`${this.root}/pages/${page}/stages/${stage}/regenerate`, { method: "POST" });
  }

  regeneratePage(page) {
    return request(`${this.root}/pages/${page}/regenerate`, {method: "POST"});
  }

  regenerateAllPages() {
    return request(`${this.root}/regenerate-all-pages`, {method: "POST"});
  }

  finishEdits() {
    return request(`${this.root}/finish-edits`, {method: "POST"});
  }

  continueProcessing() {
    return request(`${this.root}/continue`, { method: "POST" });
  }

  retranslate(page, record = null) {
    return request(`${this.root}/pages/${page}/retranslate`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(record || {}),
    });
  }

  preview(stage, page, records, cleanRecords, cleanBoxes = [], signal) {
    return this.queueCleaning(`${this.root}/pages/${page}/stages/${stage}/preview`,
      {records, cleanRecords, cleanBoxes}, signal);
  }

  cleanPreview(page, cleanRecords, cleanBoxes, signal) {
    return this.queueCleaning(`${this.root}/pages/${page}/clean-preview`,
      {cleanRecords, cleanBoxes}, signal);
  }

  queueCleaning(url, body, signal) {
    return new Promise((resolve, reject) => {
      const job = {url, body: JSON.stringify(body), signal, resolve, reject};
      job.cancel = () => {
        if (this.pendingCleaning === job) this.pendingCleaning = null;
        signal?.removeEventListener("abort", job.cancel);
        const error = new Error("Preview superseded.");
        error.name = "AbortError";
        reject(error);
      };
      if (signal?.aborted) { job.cancel(); return; }
      this.pendingCleaning?.cancel();
      this.pendingCleaning = job;
      signal?.addEventListener("abort", job.cancel, {once: true});
      void this.runCleaning();
    });
  }

  async runCleaning() {
    if (this.cleaning || !this.pendingCleaning) return;
    const job = this.pendingCleaning;
    this.pendingCleaning = null;
    this.cleaning = true;
    try {
      // Keep the network request alive after caller cancellation. Aborting
      // fetch does not stop LaMa, and would let obsolete server work pile up.
      job.resolve(await request(job.url, {method: "POST", headers: jsonHeaders, body: job.body}));
    } catch (error) { job.reject(error); }
    finally {
      job.signal?.removeEventListener("abort", job.cancel);
      this.cleaning = false;
      void this.runCleaning();
    }
  }

  async textLayout(page, record, signal) {
    const data = await request(`${this.root}/pages/${page}/text-layout`, {
      method: "POST", headers: jsonHeaders, body: JSON.stringify({record}), signal,
    });
    if (typeof data.text !== "string") throw new Error("The text layout returned invalid text.");
    return data.text;
  }

  async textLayers(page, records, signal) {
    const data = await request(`${this.root}/pages/${page}/text-layers`, {
      method: "POST", headers: jsonHeaders, body: JSON.stringify({records}), signal,
    });
    if (!Array.isArray(data.layers) || data.layers.length !== records.length)
      throw new Error("The preview batch returned an incorrect layer count.");
    const decoded = await Promise.allSettled(data.layers.map(async (png) => {
      const bytes = Uint8Array.from(atob(png), (character) => character.charCodeAt(0));
      return createImageBitmap(new Blob([bytes], {type: "image/png"}));
    }));
    const failed = decoded.find((item) => item.status === "rejected");
    if (failed) {
      for (const item of decoded) if (item.status === "fulfilled") item.value.close();
      throw failed.reason;
    }
    return decoded.map((item) => item.value);
  }

  cropOcr(page, region, rawBoxnoStart) {
    return request(`${this.root}/ocr-crop`, {
      method: "POST", headers: jsonHeaders,
      body: JSON.stringify({ page, region, rawBoxnoStart }),
    });
  }

  revisions() { return request(`${this.root}/revisions`); }
  restoreRevision(revision) {
    return request(`${this.root}/revisions/${revision}/restore`, { method: "POST" });
  }
}
