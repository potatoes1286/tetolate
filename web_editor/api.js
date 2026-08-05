const jsonHeaders = { "Content-Type": "application/json" };

async function request(url, options = {}) {
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
  return response.json();
}

export class EditorApi {
  constructor(category, jobId) {
    this.root = `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/editor/v2`;
  }

  load(stage, page) {
    return request(`${this.root}/pages/${page}/stages/${stage}`);
  }

  save(stage, page, body) {
    return request(`${this.root}/pages/${page}/stages/${stage}`, {
      method: "POST", headers: jsonHeaders, body: JSON.stringify(body),
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

  preview(stage, page, records, cleanRecords) {
    return request(`${this.root}/pages/${page}/stages/${stage}/preview`, {
      method: "POST", headers: jsonHeaders,
      body: JSON.stringify({ records, cleanRecords }),
    });
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
