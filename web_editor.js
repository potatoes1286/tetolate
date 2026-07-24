(() => {
  const config = window.TETOLATE_EDITOR || {};
  const category = config.category || "";
  const jobId = config.jobId || "";
  const pageCount = Number(config.pageCount || 0);
  const OCR_MERGE_STAGE = "ocr_merge";
  const STAGE_LABELS = {
    ocr_merge: "OCR merge",
    ocr_structured: "Structured",
    translations: "Translations",
    placements: "Placements",
  };
  const STAGE_UPSTREAM = {
    ocr_merge: [],
    ocr_structured: ["ocr_merge"],
    translations: ["ocr_merge", "ocr_structured"],
    placements: ["ocr_merge", "ocr_structured", "translations"],
  };

  const canvas = document.getElementById("page-canvas");
  const ctx = canvas.getContext("2d");
  let image = new Image();
  const HIT_CYCLE_RADIUS = 8;
  const UNSAVED_WARNING = "You have unsaved editor changes. Moving away will discard them.";

  let currentPage = 0;
  let currentStage = OCR_MERGE_STAGE;
  let records = [];
  let rawRecords = [];
  let mergedRecords = [];
  let references = [];
  let selected = null;
  let selectedLayer = "records";
  let selectedSet = new Set();
  let drawMode = false;
  let ocrCropMode = false;
  let drag = null;
  let hitCycle = null;
  let dirty = false;
  let changeInfo = {};
  let regenerationInFlight = false;
  let mutationGeneration = 0;
  let contextGeneration = 0;
  let loadedContextGeneration = -1;
  let selectedFormDirty = false;
  let loadController = null;
  let savePromise = null;
  let cropPromise = null;
  let cropController = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function status(text, warn = false) {
    const el = byId("editor-status");
    el.textContent = text || "";
    el.className = warn ? "warn" : "muted";
  }

  function markDirty() {
    mutationGeneration += 1;
    dirty = true;
  }

  function clearDirty(expectedGeneration = null) {
    if (expectedGeneration !== null && mutationGeneration !== expectedGeneration) return false;
    dirty = false;
    return true;
  }

  function hasUnsavedChanges() {
    return dirty;
  }

  function confirmDiscardUnsaved(action) {
    if (!hasUnsavedChanges()) return true;
    const lockNote = byId("lock-stage").checked
      ? "This page/stage is locked, but the current in-browser edits have not been saved."
      : "This page/stage is not locked.";
    return window.confirm(`${UNSAVED_WARNING}\n${lockNote}\n\nContinue to ${action}?`);
  }

  function isTypingTarget(target) {
    if (!target || !(target instanceof HTMLElement)) return false;
    const tag = target.tagName.toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
  }

  function setShortcutTitle(id, shortcut) {
    const el = byId(id);
    if (!el) return;
    el.title = `Shortcut: ${shortcut}`;
    el.setAttribute("aria-keyshortcuts", shortcut.replace(/\s+or\s+/gi, " "));
  }

  function isOcrMerge() {
    return currentStage === OCR_MERGE_STAGE;
  }

  function captureEditorContext() {
    return { page: currentPage, stage: currentStage, generation: contextGeneration };
  }

  function isCurrentEditorContext(context) {
    return context.generation === contextGeneration
      && context.page === currentPage
      && context.stage === currentStage;
  }

  function layerRecords(layer = selectedLayer) {
    if (isOcrMerge()) {
      if (layer === "raw") return rawRecords;
      if (layer === "merged") return mergedRecords;
    }
    return records;
  }

  function replaceLayerRecords(layer, nextRecords) {
    if (isOcrMerge() && layer === "raw") {
      rawRecords = nextRecords;
    } else if (isOcrMerge() && layer === "merged") {
      mergedRecords = nextRecords;
    } else {
      records = nextRecords;
    }
  }

  function selectedRecord() {
    if (selected === null) return null;
    return layerRecords()[selected] || null;
  }

  function updatePageButtons() {
    byId("prev-page-btn").disabled = pageCount < 2 || currentPage <= 0;
    byId("next-page-btn").disabled = pageCount < 2 || currentPage >= pageCount - 1;
  }

  function updateStageControls() {
    document.body.classList.toggle("ocr-merge-stage", isOcrMerge());
    byId("merge-btn").textContent = isOcrMerge() ? "Create merged group (M)" : "Merge selected (M)";
    byId("draw-btn").textContent = drawMode
      ? "Drawing... (D)"
      : isOcrMerge()
        ? "Draw raw box (D)"
        : "Draw box (D)";
    byId("ocr-crop-btn").textContent = ocrCropMode ? "Drawing OCR area... (O)" : "OCR new area (O)";
    updateAsyncControls();
  }

  function updateAsyncControls() {
    const editorBusy = !!loadController || !!savePromise || !!cropPromise;
    byId("save-btn").disabled = editorBusy;
    byId("ocr-crop-btn").disabled = editorBusy;
  }

  function setRegenerationInFlight(active) {
    regenerationInFlight = active;
    byId("regen-btn").disabled = active;
    byId("regen-changed-btn").disabled = active || Number(changeInfo.allChangedCount || 0) < 1;
  }

  function renderChangeNotice() {
    const notice = byId("outdated-notice");
    const regenerateButton = byId("regen-changed-btn");
    const changedStages = new Set(Array.isArray(changeInfo.changedStages) ? changeInfo.changedStages : []);
    for (const option of byId("stage-select").options) {
      const stage = option.value;
      const upstreamDirty = (STAGE_UPSTREAM[stage] || []).some((item) => changedStages.has(item));
      option.textContent = `${STAGE_LABELS[stage] || stage}${changedStages.has(stage) ? " *" : ""}${upstreamDirty ? " (out of date)" : ""}`;
    }

    const changedCount = Number(changeInfo.allChangedCount || 0);
    regenerateButton.disabled = regenerationInFlight || changedCount < 1;
    regenerateButton.title = changedCount
      ? `Regenerate ${changedCount} page(s) with saved editor changes`
      : "No saved editor changes need regeneration";

    const messages = [];
    const upstreamLabels = Array.isArray(changeInfo.outdatedBecauseLabels)
      ? changeInfo.outdatedBecauseLabels
      : [];
    if (upstreamLabels.length) {
      messages.push(`This stage may be out of date because saved changes exist in: ${upstreamLabels.join(", ")}.`);
    }
    if (changeInfo.currentStageChanged) {
      messages.push("This stage has saved edits; downstream output needs regeneration.");
    }
    if (changedCount > 0) {
      const pages = Array.isArray(changeInfo.allChangedPages) ? changeInfo.allChangedPages.join(", ") : "";
      messages.push(`Saved editor changes pending on ${changedCount} page(s)${pages ? `: ${pages}` : ""}.`);
    }
    notice.textContent = messages.join(" ");
    notice.className = messages.length ? "warn" : "muted";
  }

  async function goToPage(page) {
    const select = byId("page-select");
    const next = Math.max(0, Math.min(pageCount - 1, Number(page)));
    if (next === currentPage) {
      select.value = String(currentPage);
      updatePageButtons();
      return;
    }
    if (!confirmDiscardUnsaved(`page ${next}`)) {
      select.value = String(currentPage);
      updatePageButtons();
      return;
    }
    select.value = String(next);
    await loadData();
  }

  function changePage(delta) {
    goToPage(currentPage + delta);
  }

  async function goToStage(stage) {
    const select = byId("stage-select");
    if (stage === currentStage) {
      select.value = currentStage;
      return;
    }
    if (!confirmDiscardUnsaved(`stage ${stage}`)) {
      select.value = currentStage;
      return;
    }
    select.value = stage;
    await loadData();
  }

  function changeStage(delta) {
    const select = byId("stage-select");
    const values = [...select.options].map((option) => option.value);
    const currentIndex = Math.max(0, values.indexOf(currentStage));
    const nextIndex = Math.max(0, Math.min(values.length - 1, currentIndex + delta));
    goToStage(values[nextIndex]);
  }

  async function reloadData() {
    if (!confirmDiscardUnsaved("reload this page/stage")) return;
    await loadData();
  }

  function nextBoxno(list) {
    return list.reduce((max, rec) => {
      return Math.max(max, Number.isInteger(rec.boxno) ? rec.boxno + 1 : max);
    }, 0);
  }

  function referenceFor(record) {
    if (!record || !Number.isInteger(record.boxno)) return null;
    return references.find((item) => item && item.boxno === record.boxno) || null;
  }

  function boolValue(value) {
    if (value === true || value === false) return value;
    if (typeof value === "string") {
      const lower = value.trim().toLowerCase();
      if (lower === "true") return true;
      if (lower === "false") return false;
    }
    return null;
  }

  function ownBoolField(record, field) {
    if (!record || !Object.prototype.hasOwnProperty.call(record, field)) return null;
    return boolValue(record[field]);
  }

  function recordFlag(record, field) {
    const own = ownBoolField(record, field);
    if (own !== null) return own;
    const ref = referenceFor(record);
    const referenced = ownBoolField(ref, field);
    return referenced === true;
  }

  function recordRegion(record) {
    if (!record) return null;
    const ref = referenceFor(record);
    const region = record.placementRegion || record.region || (ref && (ref.placementRegion || ref.region));
    return Array.isArray(region) && region.length === 4 ? region.map(Number) : null;
  }

  function setRecordRegion(record, region) {
    const clean = region.map((value) => Math.round(Number(value)));
    if (currentStage === "placements") {
      record.placementRegion = clean;
    } else {
      record.region = clean;
    }
  }

  function unionRegions(regions) {
    return [
      Math.min(...regions.map((region) => region[0])),
      Math.min(...regions.map((region) => region[1])),
      Math.max(...regions.map((region) => region[2])),
      Math.max(...regions.map((region) => region[3])),
    ];
  }

  function rawByBoxno() {
    return new Map(
      rawRecords
        .filter((record) => record && Number.isInteger(record.boxno))
        .map((record) => [record.boxno, record]),
    );
  }

  function sourceBoxnosFor(record) {
    if (!record || !Array.isArray(record.sourceBoxnos)) return [];
    const seen = new Set();
    const rawMap = rawByBoxno();
    const result = [];
    for (const boxno of record.sourceBoxnos) {
      if (!Number.isInteger(boxno) || seen.has(boxno) || !rawMap.has(boxno)) continue;
      seen.add(boxno);
      result.push(boxno);
    }
    return result;
  }

  function syncMergedSources() {
    const rawMap = rawByBoxno();
    const nextMerged = [];
    for (const record of mergedRecords) {
      if (!record) continue;
      const sourceBoxnos = sourceBoxnosFor(record);
      if (!sourceBoxnos.length) continue;
      const sourceTexts = sourceBoxnos.map((boxno) => String(rawMap.get(boxno).text || ""));
      const merged = {
        ...record,
        page: currentPage,
        boxno: nextMerged.length,
        sourceBoxnos,
        sourceTexts,
        text: sourceTexts.join(""),
      };
      nextMerged.push(merged);
    }
    mergedRecords = nextMerged;
  }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height,
    };
  }

  function boxContains(region, point) {
    return point.x >= region[0] && point.x <= region[2] && point.y >= region[1] && point.y <= region[3];
  }

  function hitKey(hit) {
    return `${hit.layer}:${hit.index}`;
  }

  function sameHitList(a, b) {
    if (!a || a.length !== b.length) return false;
    return a.every((value, index) => value === b[index]);
  }

  function sameCyclePoint(point) {
    if (!hitCycle) return false;
    return Math.hypot(point.x - hitCycle.x, point.y - hitCycle.y) <= HIT_CYCLE_RADIUS;
  }

  function hitLayerAll(point, layer, list) {
    const hits = [];
    for (let index = list.length - 1; index >= 0; index--) {
      const region = recordRegion(list[index]);
      if (region && boxContains(region, point)) hits.push({ layer, index });
    }
    return hits;
  }

  function hitRecords(point) {
    if (isOcrMerge()) {
      const hits = [];
      if (byId("show-merged").checked) {
        hits.push(...hitLayerAll(point, "merged", mergedRecords));
      }
      if (byId("show-raw").checked) {
        hits.push(...hitLayerAll(point, "raw", rawRecords));
      }
      return hits;
    }
    return hitLayerAll(point, "records", records);
  }

  function hitRecord(point, additive = false) {
    const hits = hitRecords(point);
    if (!hits.length) {
      hitCycle = null;
      return null;
    }
    if (additive) {
      hitCycle = null;
      return hits[0];
    }

    const keys = hits.map(hitKey);
    if (!sameCyclePoint(point) || !sameHitList(hitCycle.keys, keys)) {
      hitCycle = { x: point.x, y: point.y, keys, nextIndex: 0 };
    }
    const hit = hits[hitCycle.nextIndex % hits.length];
    hitCycle.nextIndex = (hitCycle.nextIndex + 1) % hits.length;
    return hit;
  }

  function drawLayer(list, layer, color, labelPrefix, lineWidth) {
    list.forEach((record, index) => {
      const region = recordRegion(record);
      if (!region) return;
      const isSelected = selectedLayer === layer && selectedSet.has(index);
      const drawColor = isSelected ? "#00a86b" : color;
      ctx.strokeStyle = drawColor;
      ctx.lineWidth = isSelected ? lineWidth + 2 : lineWidth;
      ctx.strokeRect(region[0], region[1], region[2] - region[0], region[3] - region[1]);
      const flags = [];
      if (recordFlag(record, "sfx")) flags.push("SFX");
      if (recordFlag(record, "openLettering")) flags.push("OPEN");
      const label = `${labelPrefix}${record.boxno ?? index}${flags.length ? ` ${flags.join(" ")}` : ""}`;
      ctx.font = "22px system-ui";
      const labelWidth = Math.ceil(ctx.measureText(label).width) + 12;
      ctx.fillStyle = drawColor;
      ctx.fillRect(region[0], Math.max(0, region[1] - 28), labelWidth, 28);
      ctx.fillStyle = "white";
      ctx.fillText(label, region[0] + 6, Math.max(22, region[1] - 7));
      if (isSelected) {
        ctx.fillStyle = "#00a86b";
        ctx.fillRect(region[2] - 8, region[3] - 8, 16, 16);
      }
    });
  }

  function draw() {
    if (!image.complete || !image.naturalWidth) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0);
    if (isOcrMerge()) {
      if (byId("show-raw").checked) drawLayer(rawRecords, "raw", "#ff2d55", "R", 2);
      if (byId("show-merged").checked) drawLayer(mergedRecords, "merged", "#ff9500", "M", 4);
    } else {
      drawLayer(records, "records", "#2586ff", "", 3);
    }
  }

  function syncSelectedList() {
    const items = [...selectedSet].sort((a, b) => a - b).map((index) => {
      const rec = layerRecords()[index] || {};
      return `${selectedLayer} #${index} boxno ${rec.boxno ?? ""}`;
    });
    byId("selected-list").textContent = items.join("\n");
  }

  function loadSelectedForm() {
    syncSelectedList();
    const rec = selectedRecord();
    const region = recordRegion(rec) || ["", "", "", ""];
    byId("field-left").value = region[0] ?? "";
    byId("field-top").value = region[1] ?? "";
    byId("field-right").value = region[2] ?? "";
    byId("field-bottom").value = region[3] ?? "";
    byId("field-text").value = rec ? (rec.text || "") : "";
    byId("field-english").value = rec ? (rec.englishText || "") : "";
    byId("field-sfx").checked = recordFlag(rec, "sfx");
    byId("field-open").checked = recordFlag(rec, "openLettering");
    byId("field-fill").value = rec ? (rec.fill || rec.color || rec.colour || "") : "";
    byId("field-font").value = rec ? (rec.font || "") : "";
    byId("record-json").value = rec ? JSON.stringify(rec, null, 2) : "";
    selectedFormDirty = false;
  }

  function applySelectedForm() {
    if (selected === null || !selectedRecord()) return true;
    let rec;
    try {
      rec = JSON.parse(byId("record-json").value || "{}");
    } catch (error) {
      status(`Record JSON error: ${error.message}. Fix it before changing records; edits were kept.`, true);
      return false;
    }
    if (!rec || typeof rec !== "object" || Array.isArray(rec)) {
      status("Record JSON must be an object. Fix it before changing records; edits were kept.", true);
      return false;
    }
    const regionValues = ["field-left", "field-top", "field-right", "field-bottom"]
      .map((id) => byId(id).value.trim());
    const hasRegion = regionValues.some(Boolean) || !!recordRegion(selectedRecord());
    const region = regionValues.map(Number);
    if (hasRegion && (!regionValues.every(Boolean)
      || !region.every(Number.isFinite)
      || region[2] <= region[0]
      || region[3] <= region[1])) {
      status("Record region needs all four finite coordinates with right > left and bottom > top; edits were kept.", true);
      return false;
    }
    if (hasRegion) {
      setRecordRegion(rec, region);
    }
    const text = byId("field-text").value;
    const english = byId("field-english").value;
    if (text || "text" in rec || (isOcrMerge() && selectedLayer === "raw")) rec.text = text;
    if (english || "englishText" in rec || currentStage === "translations") rec.englishText = english;
    rec.sfx = byId("field-sfx").checked;
    rec.openLettering = byId("field-open").checked;
    const fill = byId("field-fill").value.trim();
    const font = byId("field-font").value.trim();
    if (fill) rec.fill = fill; else delete rec.fill;
    if (font) rec.font = font; else delete rec.font;
    layerRecords()[selected] = rec;
    if (isOcrMerge() && selectedLayer === "raw") syncMergedSources();
    loadSelectedForm();
    draw();
    markDirty();
    status("Applied selected record form.");
    return true;
  }

  function applyPendingSelectedForm() {
    return !selectedFormDirty || applySelectedForm();
  }

  function selectItem(layer, index, additive = false) {
    if (!applyPendingSelectedForm()) return false;
    if (index === null || index === undefined) {
      selected = null;
      if (!additive) selectedSet.clear();
    } else {
      if (isOcrMerge() && layer !== selectedLayer) selectedSet.clear();
      selectedLayer = isOcrMerge() ? layer : "records";
      selected = index;
      if (additive) {
        if (selectedSet.has(index)) selectedSet.delete(index);
        else selectedSet.add(index);
      } else {
        selectedSet = new Set([index]);
      }
    }
    loadSelectedForm();
    draw();
    return true;
  }

  async function loadData() {
    const page = Number(byId("page-select").value);
    const stage = byId("stage-select").value;
    contextGeneration += 1;
    currentPage = page;
    currentStage = stage;
    const requestContext = captureEditorContext();
    loadedContextGeneration = -1;
    if (loadController) loadController.abort();
    if (cropController) cropController.abort();
    const controller = new AbortController();
    loadController = controller;
    drawMode = false;
    ocrCropMode = false;
    drag = null;
    records = [];
    rawRecords = [];
    mergedRecords = [];
    references = [];
    changeInfo = {};
    selected = null;
    selectedLayer = isOcrMerge() ? "merged" : "records";
    selectedSet.clear();
    hitCycle = null;
    byId("lock-stage").checked = false;
    byId("job-notes").value = "";
    byId("page-notes").value = "";
    image = new Image();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    loadSelectedForm();
    clearDirty();
    updatePageButtons();
    updateStageControls();
    renderChangeNotice();
    status(`Loading ${stage} page ${page}...`);

    try {
      const response = await fetch(
        `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/edit/${stage}/${page}`,
        { cache: "no-store", signal: controller.signal },
      );
      if (!isCurrentEditorContext(requestContext)) return false;
      if (!response.ok) {
        const message = await response.text();
        if (isCurrentEditorContext(requestContext)) status(message, true);
        return false;
      }
      const data = await response.json();
      if (!isCurrentEditorContext(requestContext)) return false;

      records = data.records || [];
      rawRecords = data.rawRecords || [];
      mergedRecords = data.mergedRecords || [];
      references = data.referenceRecords || [];
      changeInfo = data.changeInfo || {};
      byId("lock-stage").checked = !!data.locked;
      byId("job-notes").value = data.translationNotes?.job || "";
      byId("page-notes").value = data.translationNotes?.page || "";
      const loadedImage = new Image();
      loadedImage.onload = () => {
        if (!isCurrentEditorContext(requestContext)) return;
        image = loadedImage;
        draw();
      };
      loadedImage.onerror = () => {
        if (isCurrentEditorContext(requestContext)) status("Page data loaded, but the page image failed to load.", true);
      };
      loadedImage.src = `${data.imageUrl}?t=${Date.now()}`;
      const countText = stage === OCR_MERGE_STAGE
        ? `${rawRecords.length} raw records and ${mergedRecords.length} merged groups`
        : `${records.length} records`;
      loadSelectedForm();
      clearDirty();
      renderChangeNotice();
      loadedContextGeneration = requestContext.generation;
      status(`Loaded ${stage} page ${page} with ${countText}.`);
      return true;
    } catch (error) {
      if (error.name !== "AbortError" && isCurrentEditorContext(requestContext)) {
        status(`Unable to load ${stage} page ${page}: ${error.message}`, true);
      }
      return false;
    } finally {
      if (loadController === controller) {
        loadController = null;
        updateAsyncControls();
      }
    }
  }

  async function saveData() {
    if (savePromise) {
      status("A save is already in progress.", true);
      return savePromise;
    }
    if (cropPromise) {
      status("Wait for the OCR crop to finish before saving.", true);
      return false;
    }
    const pendingSave = performSaveData();
    savePromise = pendingSave;
    updateAsyncControls();
    try {
      return await pendingSave;
    } finally {
      if (savePromise === pendingSave) savePromise = null;
      updateAsyncControls();
    }
  }

  async function performSaveData() {
    if (loadedContextGeneration !== contextGeneration) {
      status("Load this page/stage successfully before saving.", true);
      return false;
    }
    if (!applyPendingSelectedForm()) return false;
    const saveContext = captureEditorContext();
    if (saveContext.stage === OCR_MERGE_STAGE) syncMergedSources();
    const payload = {
      translationNotes: {
        job: byId("job-notes").value,
        page: byId("page-notes").value,
      },
    };
    if (saveContext.stage === OCR_MERGE_STAGE) {
      payload.rawRecords = rawRecords;
      payload.mergedRecords = mergedRecords;
    } else {
      payload.records = records;
    }
    const body = JSON.stringify(payload);
    const savedMutationGeneration = mutationGeneration;
    status("Saving...");

    try {
      const response = await fetch(
        `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/edit/${saveContext.stage}/${saveContext.page}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        },
      );
      if (!isCurrentEditorContext(saveContext)) return false;
      if (!response.ok) {
        const message = await response.text();
        if (isCurrentEditorContext(saveContext)) status(message, true);
        return false;
      }
      const data = await response.json();
      if (!isCurrentEditorContext(saveContext)) return false;
      changeInfo = data.changeInfo || changeInfo;
      if (saveContext.stage === OCR_MERGE_STAGE) byId("lock-stage").checked = true;
      const savedAllCurrentChanges = clearDirty(savedMutationGeneration);
      renderChangeNotice();
      if (savedAllCurrentChanges) {
        status(saveContext.stage === OCR_MERGE_STAGE ? "Saved raw and merged OCR edits, and locked both stages." : "Saved.");
      } else {
        status("Saved, but newer edits remain unsaved.", true);
      }
      return savedAllCurrentChanges;
    } catch (error) {
      if (isCurrentEditorContext(saveContext)) status(`Unable to save: ${error.message}`, true);
      return false;
    }
  }

  async function setLock() {
    const locked = byId("lock-stage").checked;
    const response = await fetch(
      `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/edit/${currentStage}/${currentPage}/lock`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locked }),
      },
    );
    if (!response.ok) status(await response.text(), true);
    else status(locked ? "Locked." : "Unlocked.");
  }

  function deleteSelected() {
    if (!selectedSet.size) return;
    if (!applyPendingSelectedForm()) return;
    const remove = new Set(selectedSet);
    if (isOcrMerge() && selectedLayer === "raw") {
      const removedBoxnos = new Set(
        rawRecords
          .filter((_, index) => remove.has(index))
          .map((record) => record.boxno)
          .filter(Number.isInteger),
      );
      rawRecords = rawRecords.filter((_, index) => !remove.has(index));
      mergedRecords = mergedRecords.map((record) => {
        if (!record || !Array.isArray(record.sourceBoxnos)) return record;
        return {
          ...record,
          sourceBoxnos: record.sourceBoxnos.filter((boxno) => !removedBoxnos.has(boxno)),
        };
      });
      syncMergedSources();
    } else if (isOcrMerge() && selectedLayer === "merged") {
      mergedRecords = mergedRecords.filter((_, index) => !remove.has(index));
      syncMergedSources();
    } else {
      records = records.filter((_, index) => !remove.has(index));
    }
    selectItem(selectedLayer, null);
    draw();
    markDirty();
  }

  function legacyMergeSelected() {
    if (!applyPendingSelectedForm()) return;
    const indexes = [...selectedSet].sort((a, b) => a - b);
    if (indexes.length < 2) {
      status("Select at least two boxes to merge.", true);
      return;
    }
    const selectedRecords = indexes.map((index) => records[index]).filter(Boolean);
    const regions = selectedRecords.map(recordRegion).filter(Boolean);
    if (!regions.length) return;
    const merged = {
      page: currentPage,
      boxno: selectedRecords[0].boxno ?? nextBoxno(records),
      sourceBoxnos: selectedRecords
        .flatMap((rec) => Array.isArray(rec.sourceBoxnos) ? rec.sourceBoxnos : [rec.boxno])
        .filter(Number.isInteger),
      sourceTexts: selectedRecords.flatMap((rec) => Array.isArray(rec.sourceTexts) ? rec.sourceTexts : [rec.text || ""]),
      text: selectedRecords.map((rec) => rec.text || "").join(""),
      sfx: selectedRecords.some((rec) => recordFlag(rec, "sfx")),
      openLettering: selectedRecords.some((rec) => recordFlag(rec, "openLettering")),
    };
    setRecordRegion(merged, unionRegions(regions));
    records = records.filter((_, index) => !selectedSet.has(index));
    records.push(merged);
    selectItem("records", records.length - 1);
    markDirty();
    status("Merged selected records.");
  }

  function createMergedFromRawSelection() {
    if (!applyPendingSelectedForm()) return;
    if (selectedLayer !== "raw") {
      status("Select raw OCR boxes before creating a merged group.", true);
      return;
    }
    const indexes = [...selectedSet].sort((a, b) => a - b);
    if (!indexes.length) {
      status("Select at least one raw OCR box.", true);
      return;
    }
    const selectedRaw = indexes.map((index) => rawRecords[index]).filter(Boolean);
    const sourceBoxnos = selectedRaw.map((record) => record.boxno).filter(Number.isInteger);
    const regions = selectedRaw.map(recordRegion).filter(Boolean);
    if (!sourceBoxnos.length || !regions.length) return;
    const sourceSet = new Set(sourceBoxnos);
    mergedRecords = mergedRecords.map((record) => {
      if (!record || !Array.isArray(record.sourceBoxnos)) return record;
      return {
        ...record,
        sourceBoxnos: record.sourceBoxnos.filter((boxno) => !sourceSet.has(boxno)),
      };
    });
    syncMergedSources();
    const sourceTexts = selectedRaw.map((record) => String(record.text || ""));
    mergedRecords.push({
      page: currentPage,
      boxno: nextBoxno(mergedRecords),
      sourceBoxnos,
      sourceTexts,
      text: sourceTexts.join(""),
      region: unionRegions(regions),
    });
    syncMergedSources();
    selectItem("merged", mergedRecords.length - 1);
    markDirty();
    status("Created merged group from selected raw boxes.");
  }

  function mergeSelected() {
    if (isOcrMerge()) createMergedFromRawSelection();
    else legacyMergeSelected();
  }

  function unmergeSelected() {
    if (!applyPendingSelectedForm()) return;
    if (!isOcrMerge() || selectedLayer !== "merged" || !selectedSet.size) {
      status("Select one or more merged groups to unmerge.", true);
      return;
    }
    const rawMap = rawByBoxno();
    const remove = new Set(selectedSet);
    const newGroups = [];
    for (const index of [...selectedSet].sort((a, b) => a - b)) {
      const group = mergedRecords[index];
      for (const boxno of sourceBoxnosFor(group)) {
        const raw = rawMap.get(boxno);
        const region = recordRegion(raw);
        if (!raw || !region) continue;
        newGroups.push({
          page: currentPage,
          boxno: 0,
          sourceBoxnos: [boxno],
          sourceTexts: [String(raw.text || "")],
          text: String(raw.text || ""),
          region: [...region],
        });
      }
    }
    mergedRecords = mergedRecords.filter((_, index) => !remove.has(index));
    mergedRecords.push(...newGroups);
    syncMergedSources();
    selectItem("merged", newGroups.length ? mergedRecords.length - newGroups.length : null);
    markDirty();
    status(`Unmerged into ${newGroups.length} group(s).`);
  }

  function remapCropOcrRecords(data, page) {
    const rawBoxnoMap = new Map();
    let rawBoxno = nextBoxno(rawRecords);
    const addedRaw = (data.rawRecords || data.records || []).map((record) => {
      const oldBoxno = Number(record && record.boxno);
      const next = {
        ...record,
        page,
        boxno: rawBoxno,
      };
      if (Number.isInteger(oldBoxno)) rawBoxnoMap.set(oldBoxno, rawBoxno);
      rawBoxno += 1;
      return next;
    });

    const addedMerged = (data.mergedRecords || []).map((record) => {
      const sourceBoxnos = Array.isArray(record.sourceBoxnos)
        ? record.sourceBoxnos
            .map((boxno) => rawBoxnoMap.get(Number(boxno)))
            .filter(Number.isInteger)
        : [];
      const sourceTexts = sourceBoxnos.map((boxno) => {
        const raw = addedRaw.find((item) => item.boxno === boxno);
        return raw ? String(raw.text || "") : "";
      });
      return {
        ...record,
        page,
        boxno: nextBoxno(mergedRecords),
        sourceBoxnos,
        sourceTexts,
        text: sourceTexts.join(""),
      };
    }).filter((record) => record.sourceBoxnos.length);

    return { addedRaw, addedMerged };
  }

  async function ocrCropRegion(region) {
    if (cropPromise) {
      status("OCR is already running on a crop.", true);
      return false;
    }
    if (savePromise) {
      status("Wait for the save to finish before running OCR.", true);
      return false;
    }
    if (!applyPendingSelectedForm()) return false;
    if (loadedContextGeneration !== contextGeneration) {
      status("Load this page/stage successfully before running OCR.", true);
      return false;
    }
    if (!isOcrMerge()) {
      status("Switch to OCR merge stage before adding OCR crop boxes.", true);
      return false;
    }
    const pendingCrop = performOcrCropRegion(region);
    cropPromise = pendingCrop;
    updateAsyncControls();
    try {
      return await pendingCrop;
    } finally {
      if (cropPromise === pendingCrop) cropPromise = null;
      updateAsyncControls();
    }
  }

  async function performOcrCropRegion(region) {
    const cropContext = captureEditorContext();
    const rawBoxnoStart = nextBoxno(rawRecords);
    const controller = new AbortController();
    cropController = controller;
    status("Running OCR on selected crop...");

    try {
      const response = await fetch(
        `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/edit/ocr-crop`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ page: cropContext.page, region, rawBoxnoStart }),
          signal: controller.signal,
        },
      );
      if (!isCurrentEditorContext(cropContext)) return false;
      if (!response.ok) {
        const message = await response.text();
        if (isCurrentEditorContext(cropContext)) status(message, true);
        return false;
      }
      const data = await response.json();
      if (!isCurrentEditorContext(cropContext)) return false;
      const { addedRaw, addedMerged } = remapCropOcrRecords(data, cropContext.page);
      if (!addedRaw.length) {
        status("OCR found no records in that crop.", true);
        return false;
      }
      const firstRawIndex = rawRecords.length;
      const firstMergedIndex = mergedRecords.length;
      rawRecords.push(...addedRaw);
      mergedRecords.push(...addedMerged);
      syncMergedSources();
      const selectionChanged = addedMerged.length
        ? selectItem("merged", Math.min(firstMergedIndex, mergedRecords.length - 1))
        : selectItem("raw", firstRawIndex);
      markDirty();
      if (selectionChanged) {
        status(`Added ${addedRaw.length} raw OCR records and ${addedMerged.length} merged groups from crop.`);
      }
      return true;
    } catch (error) {
      if (error.name !== "AbortError" && isCurrentEditorContext(cropContext)) {
        status(`Unable to OCR crop: ${error.message}`, true);
      }
      return false;
    } finally {
      if (cropController === controller) cropController = null;
    }
  }

  function toggleOcrCropMode() {
    if (cropPromise) {
      status("OCR is already running on a crop.", true);
      return;
    }
    if (savePromise) {
      status("Wait for the save to finish before running OCR.", true);
      return;
    }
    if (!isOcrMerge()) {
      status("Switch to OCR merge stage before drawing an OCR crop.", true);
      return;
    }
    ocrCropMode = !ocrCropMode;
    if (ocrCropMode) drawMode = false;
    updateStageControls();
    status(ocrCropMode ? "Drag a box around the missed text to OCR that area." : "");
  }

  function toggleDrawMode() {
    drawMode = !drawMode;
    if (drawMode) ocrCropMode = false;
    updateStageControls();
    status(drawMode ? "Drag a box to create a new record." : "");
  }

  function cancelTransientMode() {
    if (!drawMode && !ocrCropMode && !drag) return;
    drawMode = false;
    ocrCropMode = false;
    drag = null;
    updateStageControls();
    draw();
    status("");
  }

  async function regeneratePage() {
    if (regenerationInFlight) return;
    setRegenerationInFlight(true);
    let navigating = false;
    try {
      const saved = await saveData();
      if (!saved) return;
      const response = await fetch(
        `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/edit/regenerate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ page: currentPage, stage: currentStage }),
        },
      );
      if (!response.ok) {
        status(await response.text(), true);
        return;
      }
      navigating = true;
      window.location.href = `/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}`;
    } finally {
      if (!navigating) setRegenerationInFlight(false);
    }
  }

  async function regenerateChangedPages() {
    if (regenerationInFlight) return;
    setRegenerationInFlight(true);
    let navigating = false;
    try {
      if (hasUnsavedChanges()) {
        const saved = await saveData();
        if (!saved) return;
      }
      const changedCount = Number(changeInfo.allChangedCount || 0);
      if (changedCount < 1) {
        status("No saved editor changes need regeneration.", true);
        return;
      }
      const response = await fetch(
        `/api/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}/edit/regenerate-changes`,
        { method: "POST" },
      );
      if (!response.ok) {
        status(await response.text(), true);
        return;
      }
      const data = await response.json();
      navigating = true;
      window.location.href = data.url || `/job/${encodeURIComponent(category)}/${encodeURIComponent(jobId)}`;
    } finally {
      if (!navigating) setRegenerationInFlight(false);
    }
  }

  function toggleLock() {
    const checkbox = byId("lock-stage");
    checkbox.checked = !checkbox.checked;
    setLock();
  }

  function toggleCheckbox(id) {
    const checkbox = byId(id);
    checkbox.checked = !checkbox.checked;
    draw();
  }

  function initializeShortcuts() {
    setShortcutTitle("prev-page-btn", "[ or Alt+Left");
    setShortcutTitle("next-page-btn", "] or Alt+Right");
    setShortcutTitle("load-btn", "L");
    setShortcutTitle("save-btn", "Ctrl+S");
    setShortcutTitle("draw-btn", "D");
    setShortcutTitle("delete-btn", "Delete");
    setShortcutTitle("merge-btn", "M");
    setShortcutTitle("unmerge-btn", "U");
    setShortcutTitle("ocr-crop-btn", "O");
    setShortcutTitle("regen-btn", "G");
    setShortcutTitle("regen-changed-btn", "R");
    setShortcutTitle("apply-record-btn", "A");

    byId("page-select").title = "Page ([ / ])";
    byId("page-select").setAttribute("aria-keyshortcuts", "[ ] Alt+ArrowLeft Alt+ArrowRight");
    byId("stage-select").title = "Stage (, / .)";
    byId("stage-select").setAttribute("aria-keyshortcuts", ", . Alt+ArrowUp Alt+ArrowDown");
    byId("lock-stage").title = "Lock Stage & Page (K)";
    byId("lock-stage").setAttribute("aria-keyshortcuts", "K");
    byId("show-raw").title = "Show raw OCR boxes (1)";
    byId("show-raw").parentElement.title = "Show raw OCR boxes (1)";
    byId("show-raw").setAttribute("aria-keyshortcuts", "1");
    byId("show-merged").title = "Show merged OCR boxes (2)";
    byId("show-merged").parentElement.title = "Show merged OCR boxes (2)";
    byId("show-merged").setAttribute("aria-keyshortcuts", "2");
  }

  async function changePage(delta) {
    await goToPage(currentPage + delta);
  }

  async function changeStage(delta) {
    const select = byId("stage-select");
    const options = Array.from(select.options);
    const index = options.findIndex((option) => option.value === currentStage);
    const next = Math.max(0, Math.min(options.length - 1, index + delta));
    await goToStage(options[next].value);
  }

  function handleKeyboardShortcut(event) {
    const key = event.key;
    const lower = key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && lower === "s") {
      event.preventDefault();
      saveData();
      return;
    }
    if (isTypingTarget(event.target)) return;
    if (event.ctrlKey || event.metaKey) return;

    if (event.altKey) {
      if (key === "ArrowLeft") {
        event.preventDefault();
        changePage(-1);
      } else if (key === "ArrowRight") {
        event.preventDefault();
        changePage(1);
      } else if (key === "ArrowUp") {
        event.preventDefault();
        changeStage(-1);
      } else if (key === "ArrowDown") {
        event.preventDefault();
        changeStage(1);
      }
      return;
    }

    switch (key) {
      case "[":
        event.preventDefault();
        changePage(-1);
        break;
      case "]":
        event.preventDefault();
        changePage(1);
        break;
      case ",":
        event.preventDefault();
        changeStage(-1);
        break;
      case ".":
        event.preventDefault();
        changeStage(1);
        break;
      case "Delete":
      case "Backspace":
        event.preventDefault();
        deleteSelected();
        break;
      case "Escape":
        event.preventDefault();
        cancelTransientMode();
        break;
      case "1":
        event.preventDefault();
        if (isOcrMerge()) toggleCheckbox("show-raw");
        break;
      case "2":
        event.preventDefault();
        if (isOcrMerge()) toggleCheckbox("show-merged");
        break;
      default:
        if (lower === "l") {
          event.preventDefault();
          reloadData();
        } else if (lower === "d") {
          event.preventDefault();
          toggleDrawMode();
        } else if (lower === "m") {
          event.preventDefault();
          mergeSelected();
        } else if (lower === "u") {
          event.preventDefault();
          unmergeSelected();
        } else if (lower === "o") {
          event.preventDefault();
          toggleOcrCropMode();
        } else if (lower === "g") {
          event.preventDefault();
          regeneratePage();
        } else if (lower === "r") {
          event.preventDefault();
          regenerateChangedPages();
        } else if (lower === "a") {
          event.preventDefault();
          applySelectedForm();
        } else if (lower === "k") {
          event.preventDefault();
          toggleLock();
        }
    }
  }

  canvas.addEventListener("pointerdown", (event) => {
    const point = pointFromEvent(event);
    if (drawMode || ocrCropMode) {
      drag = { mode: "draw", start: point, current: point, ocrCrop: ocrCropMode };
      return;
    }
    const hit = hitRecord(point, event.shiftKey);
    if (!hit) {
      if (!selectItem(selectedLayer, null)) return;
      return;
    }
    if (!selectItem(hit.layer, hit.index, event.shiftKey)) return;
    const rec = layerRecords(hit.layer)[hit.index];
    const region = recordRegion(rec);
    if (region) {
      const nearHandle = Math.abs(point.x - region[2]) < 18 && Math.abs(point.y - region[3]) < 18;
      drag = { mode: nearHandle ? "resize" : "move", layer: hit.layer, index: hit.index, start: point, region: [...region] };
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const point = pointFromEvent(event);
    if (drag.mode === "draw") {
      drag.current = point;
      draw();
      ctx.strokeStyle = drag.ocrCrop ? "#8e44ad" : "#ff9500";
      ctx.lineWidth = 4;
      ctx.strokeRect(drag.start.x, drag.start.y, point.x - drag.start.x, point.y - drag.start.y);
      return;
    }
    const list = layerRecords(drag.layer);
    const rec = list[drag.index];
    if (!rec) return;
    const dx = point.x - drag.start.x;
    const dy = point.y - drag.start.y;
    let region = [...drag.region];
    if (drag.mode === "move") {
      region = [region[0] + dx, region[1] + dy, region[2] + dx, region[3] + dy];
    } else if (drag.mode === "resize") {
      region = [region[0], region[1], Math.max(region[0] + 2, region[2] + dx), Math.max(region[1] + 2, region[3] + dy)];
    }
    setRecordRegion(rec, region);
    if (isOcrMerge() && drag.layer === "raw") syncMergedSources();
    markDirty();
    loadSelectedForm();
    draw();
  });

  canvas.addEventListener("pointerup", async () => {
    if (drag && drag.mode === "draw") {
      const left = Math.min(drag.start.x, drag.current.x);
      const top = Math.min(drag.start.y, drag.current.y);
      const right = Math.max(drag.start.x, drag.current.x);
      const bottom = Math.max(drag.start.y, drag.current.y);
      const wasOcrCrop = !!drag.ocrCrop;
      drag = null;
      drawMode = false;
      ocrCropMode = false;
      updateStageControls();
      if (right - left > 4 && bottom - top > 4) {
        if (wasOcrCrop) {
          draw();
          await ocrCropRegion([left, top, right, bottom]);
          return;
        } else if (isOcrMerge()) {
          const rec = { page: currentPage, boxno: nextBoxno(rawRecords), text: "", region: [left, top, right, bottom] };
          rawRecords.push(rec);
          selectItem("raw", rawRecords.length - 1);
          markDirty();
        } else {
          const rec = { page: currentPage, boxno: nextBoxno(records), text: "", sourceBoxnos: [], sourceTexts: [], sfx: false, openLettering: false };
          setRecordRegion(rec, [left, top, right, bottom]);
          records.push(rec);
          selectItem("records", records.length - 1);
          markDirty();
        }
      }
      draw();
      return;
    }
    drag = null;
  });

  byId("load-btn").addEventListener("click", reloadData);
  byId("prev-page-btn").addEventListener("click", () => changePage(-1));
  byId("next-page-btn").addEventListener("click", () => changePage(1));
  byId("save-btn").addEventListener("click", saveData);
  byId("draw-btn").addEventListener("click", toggleDrawMode);
  byId("delete-btn").addEventListener("click", deleteSelected);
  byId("merge-btn").addEventListener("click", mergeSelected);
  byId("unmerge-btn").addEventListener("click", unmergeSelected);
  byId("ocr-crop-btn").addEventListener("click", toggleOcrCropMode);
  byId("regen-btn").addEventListener("click", regeneratePage);
  byId("regen-changed-btn").addEventListener("click", regenerateChangedPages);
  byId("apply-record-btn").addEventListener("click", applySelectedForm);
  byId("lock-stage").addEventListener("change", setLock);
  byId("page-select").addEventListener("change", (event) => goToPage(Number(event.target.value)));
  byId("stage-select").addEventListener("change", (event) => goToStage(event.target.value));
  byId("show-raw").addEventListener("change", draw);
  byId("show-merged").addEventListener("change", draw);

  for (const id of [
    "field-left",
    "field-top",
    "field-right",
    "field-bottom",
    "field-text",
    "field-english",
    "field-fill",
    "field-font",
    "record-json",
  ]) {
    byId(id).addEventListener("input", () => {
      selectedFormDirty = true;
      markDirty();
    });
  }
  for (const id of ["field-sfx", "field-open"]) {
    byId(id).addEventListener("change", () => {
      selectedFormDirty = true;
      markDirty();
    });
  }
  for (const id of ["job-notes", "page-notes"]) {
    byId(id).addEventListener("input", markDirty);
  }
  window.addEventListener("beforeunload", (event) => {
    if (!hasUnsavedChanges()) return;
    event.preventDefault();
    event.returnValue = "";
  });
  document.addEventListener("keydown", handleKeyboardShortcut);
  initializeShortcuts();

  if (pageCount < 1) status("No extracted page images found.", true);
  else loadData();
})();
