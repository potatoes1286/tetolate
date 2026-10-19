const jsonHeaders = { "Content-Type": "application/json" };

async function checkedResponse(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      message = (await response.text()) || message;
    }
    const error = new Error(message);
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

  regenerateChanges() {
    return request(`${this.root}/regenerate-changes`, { method: "POST" });
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
    return request(`${this.root}/pages/${page}/stages/${stage}/preview`, {
      method: "POST", headers: jsonHeaders, signal,
      body: JSON.stringify({ records, cleanRecords, cleanBoxes }),
    });
  }

  cleanPreview(page, cleanRecords, cleanBoxes, signal) {
    return request(`${this.root}/pages/${page}/clean-preview`, {
      method: "POST", headers: jsonHeaders, signal,
      body: JSON.stringify({cleanRecords, cleanBoxes}),
    });
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
