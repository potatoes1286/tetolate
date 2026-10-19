export const STAGES = ["ocr", "text", "placement"];

export const ALIGNMENT_LABELS = {
  north: "Up", center: "Center", south: "Down", west: "Left", east: "Right",
  northwest: "Up left", northeast: "Up right", southwest: "Down left", southeast: "Down right",
};

export const SCREEN = {
  ocr: { artifacts: ["ocr_merged", "ocr_raw"], primary: "ocr_merged", region: "region", movable: true },
  text: { artifacts: ["ocr_structured"], primary: "ocr_structured", region: "region", movable: false },
  placement: { artifacts: ["placements"], primary: "placements", region: "placementRegion", movable: true },
};

export function recordLabel(stage, record) {
  const number = Number.isInteger(record.boxno) ? record.boxno : "?";
  const text = stage === "text" && record.englishText
    ? `${record.text || ""} → ${record.englishText}` : (record.text || record.englishText || "");
  return `${number}  ${String(text || "(empty)").replace(/\s+/g, " ").slice(0, 80)}`;
}

export function fieldsFor(stage, record, fonts = []) {
  const regionKey = stage === "placement" ? "placementRegion" : "region";
  const region = record[regionKey] || record.region || [0, 0, 0, 0];
  const regionFields = [
    ["left", "Left", "number", region[0]], ["top", "Top", "number", region[1]],
    ["right", "Right", "number", region[2]], ["bottom", "Bottom", "number", region[3]],
  ];
  if (stage === "ocr") return [...regionFields, ["text", "OCR text", "textarea", record.text || ""]];
  if (stage === "text") return [
    ["text", "Corrected source text", "textarea", record.text || ""],
    ["sfx", "Sound effect", "checkbox", !!record.sfx],
    ["openLettering", "Open lettering", "checkbox", !!record.openLettering],
    ["safeToEraseOriginal", "Safe to erase original", "checkbox", !!record.safeToEraseOriginal],
    ["altPlacementReason", "Reason", "select", record.altPlacementReason || "unclear",
      ["bubble", "caption_box", "bordered_box", "blank_text_area", "sign_label", "over_art", "over_face_body", "integrated_sfx", "unclear"]],
    ["englishText", "English text", "textarea", record.englishText || ""],
  ];
  if (stage === "placement") return [
    ...regionFields,
    ["font", "Font", "select", record.font || "", ["", ...fonts]],
    ["fill", "Text colour", "select", record.fill || "black", ["black", "white"]],
    ["stroke", "Outline colour", "select", record.stroke || (record.fill === "white" ? "black" : "white"), ["black", "white"]],
    ["strokeWidth", "Outline width", "number", record.strokeWidth ?? 2],
    ["gravity", "Alignment", "select", record.gravity || "center", ["north", "center", "south", "west", "east", "northwest", "northeast", "southwest", "southeast"]],
    ["fontSizeWidthPercent", `Font size (wv; blank = auto; auto: ${Number(record._autoFontSizeWidthPercent || 0).toFixed(2)} wv)`, "number", record.fontSizeWidthPercent ?? ""],
    ["manualLineBreaks", "Manual line breaks", "textarea", record.manualLineBreaks || ""],
  ];
  return [];
}

export function textFieldTarget(records, source, field) {
  if (field === "englishText") {
    const record = (records.translations || []).find((item) => item.boxno === source.boxno);
    return {pipelineStage: "translation", artifact: "translations", recordId: record?.recordId};
  }
  return {pipelineStage: ["openLettering", "safeToEraseOriginal", "altPlacementReason"].includes(field) ? "erase" : "structure",
    artifact: "ocr_structured", recordId: source.recordId};
}

export function updateTextField(records, source, field, value, page, newId) {
  if (field === "englishText") {
    let translation = (records.translations || []).find((item) => item.boxno === source.boxno);
    if (!translation) {
      translation = {page, boxno: source.boxno, recordId: newId(), englishText: ""};
      (records.translations ||= []).push(translation);
    }
    translation.englishText = value;
  } else {
    source[field] = value;
    if (field === "openLettering") source.safeToEraseOriginal = !value;
    if (field === "safeToEraseOriginal") source.openLettering = !value;
  }
}

export function applyFields(stage, record, values) {
  const result = { ...record };
  const regionKey = stage === "placement" ? "placementRegion" : "region";
  if (["left", "top", "right", "bottom"].some((key) => key in values)) {
    const old = result[regionKey] || result.region || [0, 0, 0, 0];
    result[regionKey] = [
      Number(values.left ?? old[0]), Number(values.top ?? old[1]),
      Number(values.right ?? old[2]), Number(values.bottom ?? old[3]),
    ];
  }
  for (const [key, value] of Object.entries(values)) {
    if (!["left", "top", "right", "bottom"].includes(key)) {
      if (key === "fontSizeWidthPercent" && value === "") delete result[key];
      else result[key] = value;
    }
  }
  return result;
}
