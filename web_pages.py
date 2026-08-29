"""Server-rendered pages for the Tetolate web interface."""

from __future__ import annotations

import html
import json
from typing import Any

from fastapi.responses import HTMLResponse

import paddle_ocr_image
import translate_cbz


DEFAULT_WEBP_QUALITY = translate_cbz.TRANSLATED_WEBP_QUALITY
DEFAULT_JXL_QUALITY = translate_cbz.TRANSLATED_JXL_QUALITY
DEFAULT_THINKING_BUDGET_TOKENS = translate_cbz.DEFAULT_VLM_THINKING_BUDGET_TOKENS
DEFAULT_PROOFREAD_TRANSLATIONS = True
DEFAULT_WRITE_TRANSLATION_NOTES = True
DEFAULT_ALT_PLACEMENT_ENABLED = translate_cbz.DEFAULT_ALT_PLACEMENT_ENABLED
DEFAULT_OCR_ENGINE = paddle_ocr_image.DEFAULT_OCR_ENGINE
DEFAULT_PADDLEOCR_VL_SERVER_URL = paddle_ocr_image.DEFAULT_PADDLEOCR_VL_SERVER_URL
DEFAULT_PADDLEOCR_VL_MODEL = paddle_ocr_image.DEFAULT_PADDLEOCR_VL_MODEL
DEFAULT_SOURCE_LANGUAGE = translate_cbz.DEFAULT_SOURCE_LANGUAGE
DEFAULT_OCR_PAGE_WORKERS = translate_cbz.DEFAULT_OCR_PAGE_WORKERS
DEFAULT_LAMA_WORKERS = translate_cbz.DEFAULT_LAMA_WORKERS
DEFAULT_IMAGEMAGICK_WORKERS = translate_cbz.DEFAULT_IMAGEMAGICK_WORKERS
UPLOAD_COMIC_ARCHIVE_ACCEPT = (
    ".cbz,.zip,application/vnd.comicbook+zip,application/zip"
)
OCR_MERGE_EDITOR_STAGE = "ocr_merge"
RERUN_JOB_PACKAGE_STAGE = "package"
UPLOAD_PAGE_IMAGE_ACCEPT = (
    ".png,.jpg,.jpeg,.webp,.jxl,.bmp,.tif,.tiff,.avif,.gif,"
    "image/png,image/jpeg,image/webp,image/jxl,image/bmp,image/tiff,image/avif,image/gif"
)
DELETE_JOB_CONFIRM = "Delete this job and its generated files?"
CATEGORY_DELETE_CONFIRM = (
    "Delete this category and every job, input, output, log, and download inside it?"
)
TERMINATE_JOB_CONFIRM = (
    "Terminate this running job? The local VLM request stream will be closed, "
    "but external providers may handle cancellation differently."
)


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _thinking_budget(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("thinking budget must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    raise ValueError("thinking budget must be an integer")


def base_page(title: str, body: str, *, wide: bool = False) -> HTMLResponse:
    body_class = ' class="wide-page"' if wide else ""
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 56rem; line-height: 1.45; }}
    body.wide-page {{ max-width: none; }}
    label {{ display: block; margin: 0 0 0.75rem; }}
    input, button, select, textarea {{ font: inherit; padding: 0.45rem; }}
    input[type="text"], input[type="url"], input[type="password"] {{ width: min(32rem, 100%); box-sizing: border-box; }}
    textarea {{ width: min(40rem, 100%); }}
    button {{ cursor: pointer; }}
    details > fieldset {{ margin: 1rem 0; padding: 1rem; }}
    details > fieldset > legend {{ font-weight: 700; }}
    .row {{ margin: 1rem 0; }}
    .muted {{ color: #555; }}
    .status {{ font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.45rem; text-align: left; vertical-align: top; }}
    pre {{ background: #f5f5f5; padding: 1rem; overflow: auto; max-height: 24rem; }}
    pre.job-log {{ box-sizing: border-box; width: 100%; height: clamp(32rem, 68vh, 64rem); max-height: none; white-space: pre; }}
    a.button {{ display: inline-block; padding: 0.5rem 0.7rem; border: 1px solid #222; text-decoration: none; color: #111; }}
  </style>
</head>
<body{body_class}>
{body}
</body>
</html>"""
    )


def admin_login_page(message: str = "") -> HTMLResponse:
    message_html = f"<p>{escape(message)}</p>" if message else ""
    return base_page(
        "Admin Login",
        f"""
<h1>Admin</h1>
{message_html}
<form action="/admin/login" method="post">
  <label>Password<br><input name="password" type="password" required autocomplete="current-password"></label>
  <button type="submit">Log in</button>
</form>
""",
    )


def admin_state_label(status: dict[str, Any]) -> str:
    if status.get("paused"):
        return "Paused"
    if status.get("active"):
        return "Processing"
    if not status.get("workerRunning"):
        return "Stopped"
    if status.get("queuedCount"):
        return "Queued"
    return "Idle"


def admin_dashboard_page(manager: JobManager, message: str = "") -> HTMLResponse:
    status = manager.admin_status()
    state = admin_state_label(status)
    message_html = f'<p class="status">{escape(message)}</p>' if message else ""
    signal_note = (
        ""
        if status.get("posixSignals")
        else '<p class="muted">This platform can pause the queue, but may not suspend a running process.</p>'
    )
    active_rows: list[str] = []
    for item in status.get("active", []):
        page = "" if item.get("page") is None else str(item.get("page"))
        active_rows.append(
            "<tr>"
            f'<td><a href="{escape(item.get("url"))}">{escape(item.get("inputFilename") or item.get("jobId"))}</a></td>'
            f'<td>{escape(item.get("category"))}</td>'
            f'<td>{escape(item.get("status"))}</td>'
            f'<td>{escape(item.get("phase"))}</td>'
            f"<td>{escape(page)}</td>"
            "</tr>"
        )
    active_html = (
        """
<table>
  <thead><tr><th>Job</th><th>Category</th><th>Status</th><th>Phase</th><th>Page</th></tr></thead>
  <tbody>
"""
        + "\n".join(active_rows)
        + """
  </tbody>
</table>
"""
        if active_rows
        else '<p class="muted">No active process.</p>'
    )
    category_rows: list[str] = []
    for item in status.get("categories", []):
        category = str(item.get("category") or "")
        url = str(item.get("url") or f"/category/{category}")
        category_rows.append(
            "<tr>"
            f'<td><a href="{escape(url)}">{escape(category)}</a></td>'
            f'<td>{escape(item.get("jobCount", 0))}</td>'
            f'<td><a class="button" href="/admin/categories/{escape(category)}/delete">Delete...</a></td>'
            "</tr>"
        )
    categories_html = (
        """
<table>
  <thead><tr><th>Category</th><th>Jobs</th><th>Actions</th></tr></thead>
  <tbody>
"""
        + "\n".join(category_rows)
        + """
  </tbody>
</table>
"""
        if category_rows
        else '<p class="muted">No job categories. Create one to submit a job.</p>'
    )
    return base_page(
        "Admin",
        f"""
<h1>Admin</h1>
{message_html}
<dl>
  <dt>Worker State</dt><dd class="status">{escape(state)}</dd>
  <dt>Queued Jobs</dt><dd>{escape(status.get("queuedCount"))}</dd>
</dl>
<form action="/admin/pause" method="post">
  <button type="submit">Pause all jobs</button>
</form>
<form action="/admin/resume" method="post">
  <button type="submit">Resume all jobs</button>
</form>
<form action="/admin/logout" method="post">
  <button type="submit">Log out</button>
</form>
{signal_note}
<h2>Job Categories</h2>
<form action="/admin/categories" method="post">
  <label>New category<br><input name="category" type="text" required pattern="[A-Za-z0-9_-]{{1,64}}" maxlength="64" autocomplete="off"></label>
  <button type="submit">Create category</button>
</form>
{categories_html}
<h2>Active Job</h2>
{active_html}
<h2>Change Admin Password</h2>
<form action="/admin/password" method="post">
  <label>Current password<br><input name="current_password" type="password" required autocomplete="current-password"></label>
  <label>New password<br><input name="new_password" type="password" required autocomplete="new-password"></label>
  <label>Confirm new password<br><input name="confirm_password" type="password" required autocomplete="new-password"></label>
  <button type="submit">Change password</button>
</form>
""",
    )


def category_delete_page(category: str, counts: dict[str, int], message: str = "") -> HTMLResponse:
    total = sum(counts.values())
    count_rows = "".join(
        f"<li>{escape(status)}: {escape(count)}</li>"
        for status, count in sorted(counts.items())
    )
    message_html = f'<p class="status">{escape(message)}</p>' if message else ""
    return base_page(
        f"Delete {category}",
        f"""
<h1>Delete Category: {escape(category)}</h1>
{message_html}
<p><strong>{escape(CATEGORY_DELETE_CONFIRM)}</strong></p>
<p>This permanently removes {total} job(s), including every original upload, translated image, log, editor change, and download.</p>
<ul>{count_rows}</ul>
<form action="/admin/categories/{escape(category)}/delete" method="post">
  <label>Type <code>{escape(category)}</code> to confirm<br><input name="confirmation" type="text" required autocomplete="off"></label>
  <button type="submit">Delete category and all jobs</button>
</form>
<p><a href="/admin">Cancel</a></p>
""",
    )


def download_links_html(code: str, job_id: str, status: dict[str, Any]) -> str:
    downloads = status.get("downloads")
    if not isinstance(downloads, dict):
        downloads = {"png": status.get("hasDownload")}
    labels = {
        "png": "Download PNG CBZ",
        "webp": "Download WebP CBZ",
        "jxl": "Download JXL CBZ",
    }
    summary_parts: list[str] = []
    input_size = str(status.get("inputSize") or "")
    if input_size:
        summary_parts.append(f"Original: {input_size}")
    links: list[str] = []
    original_links: list[str] = []
    if status.get("canViewOriginal"):
        view_url = str(
            status.get("originalViewUrl") or f"/job/{code}/{job_id}/view-original"
        )
        page_count = status.get("originalPageCount")
        page_text = f" ({page_count} pages)" if page_count else ""
        original_links.append(
            f'<a class="button" href="{escape(view_url)}">View original{escape(page_text)}</a>'
        )
    if status.get("hasOriginalDownload"):
        href = str(
            status.get("originalDownloadUrl") or f"/job/{code}/{job_id}/download-original"
        )
        token = str(status.get("inputDownloadToken") or "")
        if token:
            href += f"?v={escape(token)}"
        suffix = f" ({input_size})" if input_size else ""
        archive_type = (
            "ZIP"
            if str(status.get("inputFilename") or "").lower().endswith(".zip")
            else "CBZ"
        )
        original_links.append(
            f'<a class="button" href="{escape(href)}">Download original {archive_type}{escape(suffix)}</a>'
        )
    for variant, label in labels.items():
        item = downloads.get(variant)
        if isinstance(item, dict):
            available = bool(item.get("available"))
            size = str(item.get("size") or "")
            token = str(item.get("downloadToken") or "")
        else:
            available = bool(item)
            size = ""
            token = ""
        if not available:
            continue
        summary_label = label.removeprefix("Download ")
        summary_parts.append(f"{summary_label}: {size}" if size else summary_label)
        href = f"/job/{escape(code)}/{escape(job_id)}/download/{escape(variant)}"
        if token:
            href += f"?v={escape(token)}"
        suffix = f" ({size})" if size else ""
        links.append(f'<a class="button" href="{href}">{escape(label + suffix)}</a>')
    if status.get("canView"):
        view_url = str(status.get("viewUrl") or f"/job/{code}/{job_id}/view")
        page_count = status.get("finalPageCount")
        page_text = f" ({page_count} pages)" if page_count else ""
        links.insert(
            0,
            f'<a class="button" href="{escape(view_url)}">View in browser{escape(page_text)}</a>',
        )
    links = original_links + links
    summary_html = (
        f'<p class="muted">{escape("; ".join(summary_parts))}</p>'
        if summary_parts
        else ""
    )
    if not links:
        return summary_html
    return summary_html + "<p>" + " ".join(links) + "</p>"


def input_display_html(status: dict[str, Any]) -> str:
    filename = escape(status.get("inputFilename"))
    size = str(status.get("inputSize") or "")
    if not size:
        return filename
    return f"{filename} <span class=\"muted\">({escape(size)})</span>"


def generated_translation_notes_html(status: dict[str, Any]) -> str:
    notes = str(status.get("generatedTranslationNotes") or "")
    if not notes:
        return ""
    return f"""
<details>
  <summary>Translation notes</summary>
  <pre>{escape(notes)}</pre>
</details>
"""


def job_viewer_page(
    code: str,
    job_id: str,
    status: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    original: bool = False,
) -> HTMLResponse:
    view_route = "view-original" if original else "view"
    page_kind = "Original" if original else "Translated"
    page_items: list[str] = []
    for page in pages:
        index = int(page["index"])
        token = str(page.get("token") or "")
        src = f"/job/{escape(code)}/{escape(job_id)}/{view_route}/image/{index}"
        if token:
            src += f"?v={escape(token)}"
        page_items.append(
            f"""
<figure id="page-{index}">
  <figcaption>Page {index}</figcaption>
  <img src="{src}" alt="{page_kind} page {index}" loading="lazy" decoding="async">
</figure>
"""
        )
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(status.get("inputFilename") or job_id)} - {page_kind} Browser View</title>
  <style>
    body {{ margin: 0; background: #222; color: #eee; font-family: system-ui, sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 1; background: #111; border-bottom: 1px solid #444; padding: 0.7rem 1rem; }}
    header a {{ color: #fff; }}
    .meta {{ color: #bbb; margin-left: 0.75rem; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 1rem; }}
    figure {{ margin: 0 0 1rem; }}
    figcaption {{ color: #bbb; font-size: 0.9rem; margin: 0 0 0.35rem; }}
    img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; background: #fff; }}
  </style>
</head>
<body>
  <header>
    <a href="/job/{escape(code)}/{escape(job_id)}">Back to job</a>
    <span class="meta">{page_kind} - {escape(status.get("inputFilename") or job_id)} - {len(pages)} pages</span>
  </header>
  <main>
    {"".join(page_items)}
  </main>
</body>
</html>""",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def source_language_options(selected: str) -> str:
    try:
        selected = translate_cbz.normalize_source_language(selected)
    except translate_cbz.PipelineError:
        selected = DEFAULT_SOURCE_LANGUAGE
    options: list[str] = []
    for code in ("jp", "kr", "cn"):
        profile = translate_cbz.SOURCE_LANGUAGE_PROFILES[code]
        selected_attr = " selected" if code == selected else ""
        options.append(
            f'<option value="{escape(code)}"{selected_attr}>{escape(profile.name)}</option>'
        )
    return "\n".join(options)


def advanced_options_fields(
    thinking_budget_tokens: Any,
    vlm_base_url: str,
    summary: str = "Advanced options",
    open_details: bool = False,
    include_translation_notes: bool = False,
    translation_notes: str = "",
    pause_after_ocr: bool = False,
    proofread_translations: bool = DEFAULT_PROOFREAD_TRANSLATIONS,
    write_translation_notes: bool = DEFAULT_WRITE_TRANSLATION_NOTES,
    alt_placement_enabled: bool = DEFAULT_ALT_PLACEMENT_ENABLED,
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    paddleocr_vl_server_url: str = DEFAULT_PADDLEOCR_VL_SERVER_URL,
    paddleocr_vl_model: str = DEFAULT_PADDLEOCR_VL_MODEL,
    ocr_page_workers: int = DEFAULT_OCR_PAGE_WORKERS,
    lama_workers: int = DEFAULT_LAMA_WORKERS,
    imagemagick_workers: int = DEFAULT_IMAGEMAGICK_WORKERS,
    has_vlm_auth_token: bool = False,
    has_paddleocr_vl_auth_token: bool = False,
) -> str:
    try:
        budget = _thinking_budget(thinking_budget_tokens)
    except ValueError:
        budget = DEFAULT_THINKING_BUDGET_TOKENS
    try:
        selected_ocr_engine = paddle_ocr_image.normalize_ocr_engine(ocr_engine)
    except paddle_ocr_image.InputError:
        selected_ocr_engine = DEFAULT_OCR_ENGINE
    paddle_selected = " selected" if selected_ocr_engine == paddle_ocr_image.OCR_ENGINE_PADDLE else ""
    vl_selected = " selected" if selected_ocr_engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL else ""
    open_attr = " open" if open_details else ""
    pause_checked = " checked" if pause_after_ocr else ""
    proofread_checked = " checked" if proofread_translations else ""
    write_notes_checked = " checked" if write_translation_notes else ""
    alt_placement_checked = " checked" if alt_placement_enabled else ""
    translation_notes_html = (
        '<label>Translation notes<br><textarea name="translation_notes" rows="4" '
        f'placeholder="Names, terms, tone, style guidance">{escape(translation_notes)}</textarea></label>'
        if include_translation_notes
        else ""
    )
    vlm_token_hint = (
        "A saved token is set. Leave this blank to keep it."
        if has_vlm_auth_token
        else "Leave blank to use the config or environment default."
    )
    paddle_token_hint = (
        "A saved token is set. Leave this blank to keep it."
        if has_paddleocr_vl_auth_token
        else "Leave blank when the OCR endpoint does not require authentication."
    )
    clear_vlm_html = (
        '<label><input name="clear_vlm_auth_token" type="checkbox" value="1"> Remove saved translation VLM token</label>'
        if has_vlm_auth_token
        else ""
    )
    clear_paddle_html = (
        '<label><input name="clear_paddleocr_vl_auth_token" type="checkbox" value="1"> Remove saved PaddleOCR-VL token</label>'
        if has_paddleocr_vl_auth_token
        else ""
    )
    return f"""
<details{open_attr}>
  <summary>{escape(summary)}</summary>
  <fieldset>
    <legend>Workflow</legend>
    {translation_notes_html}
    <label><input name="enable_alt_placement" type="checkbox" value="1"{alt_placement_checked}> Enable alt-placement</label>
    <label><input name="enable_proofreading" type="checkbox" value="1"{proofread_checked}> Enable proofreading</label>
    <label><input name="enable_translation_notes" type="checkbox" value="1"{write_notes_checked}> Enable translation notes</label>
  </fieldset>
  <fieldset>
    <legend>OCR</legend>
    <label>Source language<br><select name="source_language">
      {source_language_options(source_language)}
    </select></label>
    <label>OCR engine<br><select name="ocr_engine">
      <option value="{paddle_ocr_image.OCR_ENGINE_PADDLE}"{paddle_selected}>PaddleOCR</option>
      <option value="{paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL}"{vl_selected}>PaddleOCR-VL 1.6</option>
    </select></label>
    <label><input name="pause_after_ocr" type="checkbox" value="1"{pause_checked}> Pause after OCR for review</label>
    <label>PaddleOCR-VL endpoint<br><input name="paddleocr_vl_server_url" type="text" value="{escape(paddleocr_vl_server_url)}"></label>
    <label>PaddleOCR-VL model<br><input name="paddleocr_vl_model" type="text" value="{escape(paddleocr_vl_model)}"></label>
    <label>PaddleOCR-VL auth token<br><input name="paddleocr_vl_auth_token" type="password" value="" autocomplete="new-password"></label>
    <p class="muted">{escape(paddle_token_hint)}</p>
    {clear_paddle_html}
  </fieldset>
  <fieldset>
    <legend>Parallel work</legend>
    <label>PaddleOCR-VL page workers<br><input name="ocr_page_workers" type="number" min="1" max="32" step="1" value="{escape(ocr_page_workers)}"></label>
    <p class="muted">This setting applies only to PaddleOCR-VL. Each worker can process one page.</p>
    <label>LaMa cleaning workers<br><input name="lama_workers" type="number" min="1" max="32" step="1" value="{escape(lama_workers)}"></label>
    <label>ImageMagick workers<br><input name="imagemagick_workers" type="number" min="1" max="32" step="1" value="{escape(imagemagick_workers)}"></label>
    <p class="muted">More workers can use more CPU and memory. LaMa workers share one loaded model.</p>
  </fieldset>
  <fieldset>
    <legend>Translation VLM</legend>
    <label>Endpoint<br><input name="vlm_base_url" type="url" value="{escape(vlm_base_url)}" required></label>
    <label>Auth token<br><input name="vlm_auth_token" type="password" value="" autocomplete="new-password"></label>
    <p class="muted">{escape(vlm_token_hint)}</p>
    {clear_vlm_html}
    <label>No. of thinking tokens<br><input name="thinking_budget_tokens" type="number" step="1" value="{escape(budget)}"></label>
    <p class="muted">Use 0 to disable thinking where supported. Use a negative value for unlimited/server-defined thinking.</p>
  </fieldset>
</details>
"""


def category_jobs_page(code: str, data: dict[str, Any]) -> HTMLResponse:
    jobs = data.get("jobs", [])
    translation_notes = str(data.get("translationNotes") or "")
    default_thinking_budget_tokens = data.get(
        "defaultThinkingBudgetTokens",
        DEFAULT_THINKING_BUDGET_TOKENS,
    )
    default_vlm_base_url = str(data.get("defaultVlmBaseUrl") or "")
    pause_after_ocr = bool(data.get("pauseAfterOcr", False))
    proofread_translations = bool(
        data.get("proofreadTranslations", DEFAULT_PROOFREAD_TRANSLATIONS)
    )
    write_translation_notes = bool(
        data.get("writeTranslationNotes", DEFAULT_WRITE_TRANSLATION_NOTES)
    )
    default_alt_placement_enabled = bool(
        data.get("defaultAltPlacementEnabled", DEFAULT_ALT_PLACEMENT_ENABLED)
    )
    default_source_language = str(data.get("defaultSourceLanguage") or DEFAULT_SOURCE_LANGUAGE)
    default_ocr_engine = str(data.get("defaultOcrEngine") or DEFAULT_OCR_ENGINE)
    default_paddleocr_vl_server_url = str(
        data.get("defaultPaddleocrVlServerUrl") or DEFAULT_PADDLEOCR_VL_SERVER_URL
    )
    default_paddleocr_vl_model = str(
        data.get("defaultPaddleocrVlModel") or DEFAULT_PADDLEOCR_VL_MODEL
    )
    default_ocr_page_workers = int(data.get("defaultOcrPageWorkers") or 1)
    default_lama_workers = int(data.get("defaultLamaWorkers") or 1)
    default_imagemagick_workers = int(data.get("defaultImagemagickWorkers") or 1)
    rows: list[str] = []
    for job in jobs:
        job_id = str(job.get("jobId", ""))
        filename = str(job.get("inputFilename") or job_id)
        page = "" if job.get("page") is None else str(job.get("page"))
        view_html = (
            f'<a class="button" href="/job/{escape(code)}/{escape(job_id)}/view">View</a>'
            if job.get("canView")
            else ""
        )
        delete_html = (
            f'<form action="/job/{escape(code)}/{escape(job_id)}/delete" method="post" '
            f"onsubmit=\"return confirm({escape(json.dumps(DELETE_JOB_CONFIRM))});\">"
            '<button type="submit">Delete</button></form>'
            if job.get("canDelete")
            else ""
        )
        actions_html = " ".join(item for item in (view_html, delete_html) if item)
        rows.append(
            "<tr>"
            f'<td><a href="/job/{escape(code)}/{escape(job_id)}">{escape(filename)}</a><br>'
            f'<span class="muted">{escape(job_id)}</span></td>'
            f'<td>{escape(job.get("status"))}</td>'
            f'<td>{escape(job.get("phase"))}</td>'
            f"<td>{escape(page)}</td>"
            f'<td>{escape(job.get("age"))}</td>'
            f'<td>{escape(job.get("elapsed"))}</td>'
            f"<td>{actions_html}</td>"
            "</tr>"
        )
    jobs_html = (
        """
<table>
  <thead><tr><th>Job</th><th>Status</th><th>Phase</th><th>Page</th><th>Age</th><th>Elapsed</th><th>Actions</th></tr></thead>
  <tbody>
"""
        + "\n".join(rows)
        + """
  </tbody>
</table>
"""
        if rows
        else '<p class="muted">No jobs have been submitted in this category.</p>'
    )
    return base_page(
        f"Category {code}",
        f"""
<h1>Category: {escape(code)}</h1>
<p><a href="/admin">Admin</a></p>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input name="category" type="hidden" value="{escape(code)}">
  <label>CBZ or ZIP archive<br><input name="cbz" type="file" accept="{UPLOAD_COMIC_ARCHIVE_ACCEPT}"></label>
  <label>Page image files<br><input name="page_images" type="file" accept="{UPLOAD_PAGE_IMAGE_ACCEPT}" multiple></label>
  <p class="muted">Upload one CBZ or ZIP archive, or upload multiple image pages. A ZIP archive must contain image pages and can contain metadata. Separate images use the picker order and are converted to PNG.</p>
  {advanced_options_fields(
      default_thinking_budget_tokens,
      default_vlm_base_url,
      include_translation_notes=True,
      translation_notes=translation_notes,
      pause_after_ocr=pause_after_ocr,
      proofread_translations=proofread_translations,
      write_translation_notes=write_translation_notes,
      alt_placement_enabled=default_alt_placement_enabled,
      source_language=default_source_language,
      ocr_engine=default_ocr_engine,
      paddleocr_vl_server_url=default_paddleocr_vl_server_url,
      paddleocr_vl_model=default_paddleocr_vl_model,
      ocr_page_workers=default_ocr_page_workers,
      lama_workers=default_lama_workers,
      imagemagick_workers=default_imagemagick_workers,
  )}
  <button type="submit">Queue new job</button>
</form>
<h2>Jobs</h2>
{jobs_html}
""",
    )


def job_page(code: str, job_id: str, status: dict[str, Any]) -> HTMLResponse:
    log_text = "\n".join(status.get("recentLog", []))
    page_value = status.get("page")
    page_text = "" if page_value is None else str(page_value)
    restart_from = status.get("restartResumeFrom")
    restart_page = status.get("restartResumePage")
    ocr_review_checkpoint = bool(status.get("ocrReviewCheckpoint"))
    if restart_from is None:
        restart_target_text = "from the beginning"
    elif restart_from == "package":
        restart_target_text = "from package"
    else:
        restart_target_text = f"from {restart_from} page {restart_page}"
    download_html = download_links_html(code, job_id, status)
    webp_quality = status.get("webpQuality") or status.get("defaultWebpQuality") or DEFAULT_WEBP_QUALITY
    jxl_quality = status.get("jxlQuality") or status.get("defaultJxlQuality") or DEFAULT_JXL_QUALITY
    thinking_budget = status.get("thinkingBudgetTokens")
    if thinking_budget is None:
        thinking_budget = status.get(
            "defaultThinkingBudgetTokens",
            DEFAULT_THINKING_BUDGET_TOKENS,
        )
    vlm_base_url = str(
        status.get("vlmBaseUrl")
        or status.get("defaultVlmBaseUrl")
        or ""
    )
    pause_after_ocr = bool(status.get("pauseAfterOcr"))
    proofread_translations = bool(
        status.get("proofreadTranslations", DEFAULT_PROOFREAD_TRANSLATIONS)
    )
    write_translation_notes = bool(
        status.get("writeTranslationNotes", DEFAULT_WRITE_TRANSLATION_NOTES)
    )
    alt_placement_enabled = bool(
        status.get("altPlacementEnabled", status.get("defaultAltPlacementEnabled", DEFAULT_ALT_PLACEMENT_ENABLED))
    )
    source_language = str(
        status.get("sourceLanguage")
        or status.get("defaultSourceLanguage")
        or DEFAULT_SOURCE_LANGUAGE
    )
    ocr_engine = str(status.get("ocrEngine") or status.get("defaultOcrEngine") or DEFAULT_OCR_ENGINE)
    paddleocr_vl_server_url = str(
        status.get("paddleocrVlServerUrl")
        or status.get("defaultPaddleocrVlServerUrl")
        or DEFAULT_PADDLEOCR_VL_SERVER_URL
    )
    paddleocr_vl_model = str(
        status.get("paddleocrVlModel")
        or status.get("defaultPaddleocrVlModel")
        or DEFAULT_PADDLEOCR_VL_MODEL
    )
    ocr_page_workers = int(status.get("ocrPageWorkers") or 1)
    lama_workers = int(status.get("lamaWorkers") or 1)
    imagemagick_workers = int(status.get("imagemagickWorkers") or 1)
    advanced_options_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/advanced-options" method="post">
  {advanced_options_fields(
      thinking_budget,
      vlm_base_url,
      open_details=True,
      pause_after_ocr=pause_after_ocr,
      proofread_translations=proofread_translations,
      write_translation_notes=write_translation_notes,
      alt_placement_enabled=alt_placement_enabled,
      source_language=source_language,
      ocr_engine=ocr_engine,
      paddleocr_vl_server_url=paddleocr_vl_server_url,
      paddleocr_vl_model=paddleocr_vl_model,
      ocr_page_workers=ocr_page_workers,
      lama_workers=lama_workers,
      imagemagick_workers=imagemagick_workers,
      has_vlm_auth_token=bool(status.get("hasVlmAuthToken")),
      has_paddleocr_vl_auth_token=bool(status.get("hasPaddleocrVlAuthToken")),
  )}
  <button type="submit">Save advanced options</button>
</form>
"""
        if status.get("canUpdateAdvancedOptions")
        else ""
    )
    restart_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/restart" method="post">
  <button type="submit">{"Continue processing" if ocr_review_checkpoint else "Restart run"}</button>
  <span class="muted">{"Use the reviewed OCR and start Structure." if ocr_review_checkpoint else f"Resume {escape(restart_target_text)}"}</span>
</form>
"""
        if status.get("canRestart")
        else ""
    )
    terminate_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/terminate" method="post" onsubmit="return confirm({escape(json.dumps(TERMINATE_JOB_CONFIRM))});">
  <button type="submit">Terminate job</button>
</form>
"""
        if status.get("canTerminate")
        else ""
    )
    delete_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/delete" method="post" onsubmit="return confirm({escape(json.dumps(DELETE_JOB_CONFIRM))});">
  <button type="submit">Delete job</button>
</form>
"""
        if status.get("canDelete")
        else ""
    )
    rerun_job_html = (
        f"""
<details>
  <summary>Rerun job</summary>
  <form action="/job/{escape(code)}/{escape(job_id)}/rerun-job" method="post">
    <fieldset>
      <legend>Passes</legend>
      <label><input name="rerun_stage" type="checkbox" value="{OCR_MERGE_EDITOR_STAGE}"> OCR &amp; Merge</label>
      <label><input name="rerun_stage" type="checkbox" value="ocr_structured"> Structure</label>
      <label><input name="rerun_stage" type="checkbox" value="alt_placement"> Erase &amp; Alternate Placement</label>
      <label><input name="rerun_stage" type="checkbox" value="translations"> Translation</label>
      <label><input name="rerun_stage" type="checkbox" value="placements"> Typesetting</label>
      <label><input name="rerun_stage" type="checkbox" value="render"> Render</label>
      <label><input name="rerun_stage" type="checkbox" value="{RERUN_JOB_PACKAGE_STAGE}"> Package</label>
    </fieldset>
    <label>Pages<br><input name="page_spec" type="text" placeholder="0-3,6,8"></label>
    <p class="muted">Leave pages empty to rerun every page. Package alone rebuilds download archives.</p>
    <label>WebP quality<br><input name="webp_quality" type="number" min="1" max="100" value="{escape(webp_quality)}"></label>
    <label>JXL quality<br><input name="jxl_quality" type="number" min="1" max="100" value="{escape(jxl_quality)}"></label>
    <button type="submit">Queue rerun</button>
  </form>
</details>
"""
        if status.get("canRerunPages") or status.get("canRegenerateDownloads")
        else ""
    )
    edit_html = (
        f'<p><a class="button" href="/job/{escape(code)}/{escape(job_id)}/edit">{"Review OCR" if ocr_review_checkpoint else "Edit job"}</a></p>'
        if status.get("canEdit")
        else ""
    )
    generated_notes_html = generated_translation_notes_html(status)
    return base_page(
        f"Job {code} {job_id}",
        f"""
<h1>Job {escape(job_id)}</h1>
<p><a href="/category/{escape(code)}">Back to category {escape(code)}</a></p>
<dl>
  <dt>Category</dt><dd>{escape(code)}</dd>
  <dt>Input</dt><dd>{input_display_html(status)}</dd>
  <dt>Status</dt><dd id="status" class="status">{escape(status.get("status"))}</dd>
  <dt>Age</dt><dd id="age">{escape(status.get("age"))}</dd>
  <dt>Phase</dt><dd id="phase">{escape(status.get("phase"))}</dd>
  <dt>Page</dt><dd id="page">{escape(page_text)}</dd>
  <dt>Elapsed</dt><dd id="elapsed">{escape(status.get("elapsed"))}</dd>
  <dt>Thinking Tokens</dt><dd id="thinking-budget">{escape(status.get("thinkingBudget"))}</dd>
  <dt>Message</dt><dd id="message">{escape(status.get("message"))}</dd>
</dl>
<div id="advanced-options">{advanced_options_html}</div>
<div id="restart">{restart_html}</div>
<div id="terminate">{terminate_html}</div>
<div id="edit">{edit_html}</div>
<div id="download">{download_html}</div>
<div id="generated-translation-notes">{generated_notes_html}</div>
<div id="rerun-job">{rerun_job_html}</div>
<div id="delete">{delete_html}</div>
<h2>Recent Log</h2>
<pre id="log" class="job-log">{escape(log_text)}</pre>
<script>
const code = {json.dumps(code)};
const jobId = {json.dumps(job_id)};
const deleteJobConfirm = {json.dumps(DELETE_JOB_CONFIRM)};
const terminateJobConfirm = {json.dumps(TERMINATE_JOB_CONFIRM)};
function htmlEscape(value) {{
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }}[char]));
}}
function restartMarkup(data) {{
  if (!data.canRestart) return "";
  let target = "from the beginning";
  if (data.restartResumeFrom === "package") {{
    target = "from package";
  }} else if (data.restartResumeFrom) {{
    target = `from ${{data.restartResumeFrom}} page ${{data.restartResumePage ?? 0}}`;
  }}
  const label = data.ocrReviewCheckpoint ? "Continue processing" : "Restart run";
  const detail = data.ocrReviewCheckpoint ? "Use the reviewed OCR and start Structure." : `Resume ${{target}}`;
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/restart" method="post"><button type="submit">${{label}}</button> <span class="muted">${{detail}}</span></form>`;
}}
function downloadMarkup(data) {{
  const labels = {{png: "Download PNG CBZ", webp: "Download WebP CBZ", jxl: "Download JXL CBZ"}};
  const downloads = data.downloads || {{}};
  const summary = [];
  if (data.inputSize) {{
    summary.push(`Original: ${{data.inputSize}}`);
  }}
  const links = Object.keys(labels)
    .filter((variant) => {{
      const item = downloads[variant];
      return item && (typeof item !== "object" || item.available);
    }})
    .map((variant) => {{
      const item = downloads[variant];
      const size = item && typeof item === "object" && item.size ? ` (${{item.size}})` : "";
      const summaryLabel = labels[variant].replace("Download ", "");
      summary.push(size ? `${{summaryLabel}}: ${{item.size}}` : summaryLabel);
      const token = item && typeof item === "object" && item.downloadToken ? `?v=${{encodeURIComponent(item.downloadToken)}}` : "";
      const href = `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/download/${{variant}}${{token}}`;
      return `<a class="button" href="${{href}}">${{labels[variant]}}${{size}}</a>`;
    }});
  if (data.canView) {{
    const viewUrl = data.viewUrl || `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/view`;
    const pageCount = data.finalPageCount ? ` (${{data.finalPageCount}} pages)` : "";
    links.unshift(`<a class="button" href="${{viewUrl}}">View in browser${{pageCount}}</a>`);
  }}
  if (data.hasOriginalDownload) {{
    const inputSize = data.inputSize ? ` (${{data.inputSize}})` : "";
    const token = data.inputDownloadToken ? `?v=${{encodeURIComponent(data.inputDownloadToken)}}` : "";
    const url = data.originalDownloadUrl || `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/download-original`;
    const archiveType = String(data.inputFilename || "").toLowerCase().endsWith(".zip") ? "ZIP" : "CBZ";
    links.unshift(`<a class="button" href="${{url}}${{token}}">Download original ${{archiveType}}${{inputSize}}</a>`);
  }}
  if (data.canViewOriginal) {{
    const viewUrl = data.originalViewUrl || `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/view-original`;
    const pageCount = data.originalPageCount ? ` (${{data.originalPageCount}} pages)` : "";
    links.unshift(`<a class="button" href="${{viewUrl}}">View original${{pageCount}}</a>`);
  }}
  const summaryHtml = summary.length ? `<p class="muted">${{summary.join("; ")}}</p>` : "";
  const linksHtml = links.length ? `<p>${{links.join(" ")}}</p>` : "";
  return summaryHtml + linksHtml;
}}
function deleteMarkup(data) {{
  if (!data.canDelete) return "";
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/delete" method="post" onsubmit="return confirm(${{JSON.stringify(deleteJobConfirm)}});"><button type="submit">Delete job</button></form>`;
}}
function terminateMarkup(data) {{
  if (!data.canTerminate) return "";
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/terminate" method="post" onsubmit="return confirm(${{JSON.stringify(terminateJobConfirm)}});"><button type="submit">Terminate job</button></form>`;
}}
function advancedOptionsMarkup(data) {{
  if (!data.canUpdateAdvancedOptions) return "";
  const thinkingBudget = data.thinkingBudgetTokens ?? data.defaultThinkingBudgetTokens ?? 2048;
  const vlmBaseUrl = htmlEscape(data.vlmBaseUrl || data.defaultVlmBaseUrl || "");
  const pauseChecked = data.pauseAfterOcr ? " checked" : "";
  const proofreadChecked = data.proofreadTranslations ? " checked" : "";
  const notesChecked = data.writeTranslationNotes ? " checked" : "";
  const altPlacementChecked = (data.altPlacementEnabled ?? data.defaultAltPlacementEnabled ?? true) ? " checked" : "";
  const sourceLanguage = data.sourceLanguage || data.defaultSourceLanguage || "jp";
  const jpSelected = sourceLanguage === "jp" ? " selected" : "";
  const krSelected = sourceLanguage === "kr" ? " selected" : "";
  const cnSelected = sourceLanguage === "cn" ? " selected" : "";
  const ocrEngine = data.ocrEngine || data.defaultOcrEngine || "paddle";
  const paddleSelected = ocrEngine === "paddle" ? " selected" : "";
  const vlSelected = ocrEngine === "paddleocr_vl" ? " selected" : "";
  const vlServerUrl = htmlEscape(data.paddleocrVlServerUrl || data.defaultPaddleocrVlServerUrl || "http://127.0.0.1:8081/v1");
  const vlModel = htmlEscape(data.paddleocrVlModel || data.defaultPaddleocrVlModel || "PaddlePaddle/PaddleOCR-VL-1.6");
  const ocrPageWorkers = data.ocrPageWorkers ?? data.defaultOcrPageWorkers ?? 1;
  const lamaWorkers = data.lamaWorkers ?? data.defaultLamaWorkers ?? 1;
  const imagemagickWorkers = data.imagemagickWorkers ?? data.defaultImagemagickWorkers ?? 1;
  const paddleTokenHint = data.hasPaddleocrVlAuthToken
    ? "A saved token is set. Leave this blank to keep it."
    : "Leave blank when the OCR endpoint does not require authentication.";
  const vlmTokenHint = data.hasVlmAuthToken
    ? "A saved token is set. Leave this blank to keep it."
    : "Leave blank to use the config or environment default.";
  const clearPaddle = data.hasPaddleocrVlAuthToken
    ? `<label><input name="clear_paddleocr_vl_auth_token" type="checkbox" value="1"> Remove saved PaddleOCR-VL token</label>`
    : "";
  const clearVlm = data.hasVlmAuthToken
    ? `<label><input name="clear_vlm_auth_token" type="checkbox" value="1"> Remove saved translation VLM token</label>`
    : "";
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/advanced-options" method="post">
    <details open><summary>Advanced options</summary>
      <fieldset><legend>Workflow</legend>
        <label><input name="enable_alt_placement" type="checkbox" value="1"${{altPlacementChecked}}> Enable alt-placement</label>
        <label><input name="enable_proofreading" type="checkbox" value="1"${{proofreadChecked}}> Enable proofreading</label>
        <label><input name="enable_translation_notes" type="checkbox" value="1"${{notesChecked}}> Enable translation notes</label>
      </fieldset>
      <fieldset><legend>OCR</legend>
        <label>Source language<br><select name="source_language"><option value="jp"${{jpSelected}}>Japanese</option><option value="kr"${{krSelected}}>Korean</option><option value="cn"${{cnSelected}}>Chinese</option></select></label>
        <label>OCR engine<br><select name="ocr_engine"><option value="paddle"${{paddleSelected}}>PaddleOCR</option><option value="paddleocr_vl"${{vlSelected}}>PaddleOCR-VL 1.6</option></select></label>
        <label><input name="pause_after_ocr" type="checkbox" value="1"${{pauseChecked}}> Pause after OCR for review</label>
        <label>PaddleOCR-VL endpoint<br><input name="paddleocr_vl_server_url" type="text" value="${{vlServerUrl}}"></label>
        <label>PaddleOCR-VL model<br><input name="paddleocr_vl_model" type="text" value="${{vlModel}}"></label>
        <label>PaddleOCR-VL auth token<br><input name="paddleocr_vl_auth_token" type="password" value="" autocomplete="new-password"></label>
        <p class="muted">${{paddleTokenHint}}</p>${{clearPaddle}}
      </fieldset>
      <fieldset><legend>Parallel work</legend>
        <label>PaddleOCR-VL page workers<br><input name="ocr_page_workers" type="number" min="1" max="32" step="1" value="${{ocrPageWorkers}}"></label>
        <p class="muted">This setting applies only to PaddleOCR-VL. Each worker can process one page.</p>
        <label>LaMa cleaning workers<br><input name="lama_workers" type="number" min="1" max="32" step="1" value="${{lamaWorkers}}"></label>
        <label>ImageMagick workers<br><input name="imagemagick_workers" type="number" min="1" max="32" step="1" value="${{imagemagickWorkers}}"></label>
        <p class="muted">More workers can use more CPU and memory. LaMa workers share one loaded model.</p>
      </fieldset>
      <fieldset><legend>Translation VLM</legend>
        <label>Endpoint<br><input name="vlm_base_url" type="url" value="${{vlmBaseUrl}}" required></label>
        <label>Auth token<br><input name="vlm_auth_token" type="password" value="" autocomplete="new-password"></label>
        <p class="muted">${{vlmTokenHint}}</p>${{clearVlm}}
        <label>No. of thinking tokens<br><input name="thinking_budget_tokens" type="number" step="1" value="${{thinkingBudget}}"></label>
        <p class="muted">Use 0 to disable thinking where supported. Use a negative value for unlimited/server-defined thinking.</p>
      </fieldset>
    </details><button type="submit">Save advanced options</button></form>`;
}}
function rerunJobMarkup(data) {{
  if (!data.canRerunPages && !data.canRegenerateDownloads) return "";
  const webpQuality = data.webpQuality ?? data.defaultWebpQuality ?? 90;
  const jxlQuality = data.jxlQuality ?? data.defaultJxlQuality ?? 90;
  return `<details><summary>Rerun job</summary><form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/rerun-job" method="post"><fieldset><legend>Passes</legend><label><input name="rerun_stage" type="checkbox" value="ocr_merge"> OCR &amp; Merge</label><label><input name="rerun_stage" type="checkbox" value="ocr_structured"> Structure</label><label><input name="rerun_stage" type="checkbox" value="alt_placement"> Erase &amp; Alternate Placement</label><label><input name="rerun_stage" type="checkbox" value="translations"> Translation</label><label><input name="rerun_stage" type="checkbox" value="placements"> Typesetting</label><label><input name="rerun_stage" type="checkbox" value="render"> Render</label><label><input name="rerun_stage" type="checkbox" value="package"> Package</label></fieldset><label>Pages<br><input name="page_spec" type="text" placeholder="0-3,6,8"></label><p class="muted">Leave pages empty to rerun every page. Package alone rebuilds download archives.</p><label>WebP quality<br><input name="webp_quality" type="number" min="1" max="100" value="${{webpQuality}}"></label><label>JXL quality<br><input name="jxl_quality" type="number" min="1" max="100" value="${{jxlQuality}}"></label><button type="submit">Queue rerun</button></form></details>`;
}}
function editMarkup(data) {{
  if (!data.canEdit) return "";
  const label = data.ocrReviewCheckpoint ? "Review OCR" : "Edit job";
  return `<p><a class="button" href="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/edit">${{label}}</a></p>`;
}}
function generatedTranslationNotesMarkup(data) {{
  if (!data.generatedTranslationNotes) return "";
  return `<details><summary>Translation notes</summary><pre>${{htmlEscape(data.generatedTranslationNotes)}}</pre></details>`;
}}
async function refreshStatus() {{
  const response = await fetch(`/api/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}`, {{cache: "no-store"}});
  if (!response.ok) return;
  const data = await response.json();
  const previousStatus = document.getElementById("status").textContent;
  document.getElementById("status").textContent = data.status || "";
  document.getElementById("age").textContent = data.age || "";
  document.getElementById("phase").textContent = data.phase || "";
  document.getElementById("page").textContent = data.page ?? "";
  document.getElementById("elapsed").textContent = data.elapsed || "";
  document.getElementById("thinking-budget").textContent = data.thinkingBudget || "";
  document.getElementById("message").textContent = data.message || "";
  const log = document.getElementById("log");
  const keepAtBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  log.textContent = (data.recentLog || []).join("\\n");
  if (keepAtBottom) log.scrollTop = log.scrollHeight;
  document.getElementById("download").innerHTML = downloadMarkup(data);
  document.getElementById("generated-translation-notes").innerHTML = generatedTranslationNotesMarkup(data);
  if (!(previousStatus === "complete" && data.status === "complete")) {{
    document.getElementById("advanced-options").innerHTML = advancedOptionsMarkup(data);
    document.getElementById("restart").innerHTML = restartMarkup(data);
    document.getElementById("terminate").innerHTML = terminateMarkup(data);
    document.getElementById("edit").innerHTML = editMarkup(data);
    document.getElementById("rerun-job").innerHTML = rerunJobMarkup(data);
    document.getElementById("delete").innerHTML = deleteMarkup(data);
  }}
  if (data.status === "complete" || data.status === "failed" || data.status === "cancelled" || (data.status === "paused" && data.canRestart)) {{
    window.clearInterval(statusTimer);
  }}
}}
const statusTimer = window.setInterval(refreshStatus, 5000);
</script>
""",
        wide=True,
    )


def editor_page(manager: JobManager, code: str, job_id: str) -> HTMLResponse:
    status = manager.require_editable_job(code, job_id)
    manager.load_editor_v2_manifest(code, job_id)
    available_stages = manager.editor_available_stages(status)
    ocr_review_checkpoint = status.get("reviewCheckpoint") == "ocr"
    page_count = len(manager.original_page_files(code, job_id))
    page_options = "\n".join(
        f'<option value="{index}">Page {index}</option>' for index in range(page_count)
    )
    stage_labels = {
        "ocr": "OCR & Merge",
        "structure": "Structure",
        "erase": "Erase & Alternate Placement",
        "translation": "Translation",
        "placement": "Typesetting",
    }
    stage_buttons = "\n".join(
        f'<button class="stage-tab" type="button" data-stage="{stage}">{index}. {escape(label)}<span class="stage-state" aria-hidden="true"></span></button>'
        for index, stage in enumerate(available_stages, start=1)
        for label in (stage_labels[stage],)
    )
    review_notice = (
        '<p class="status-line"><strong>OCR review checkpoint.</strong> '
        'Save the OCR changes, then select Continue processing.</p>'
        if ocr_review_checkpoint
        else ""
    )
    complete_only = " hidden" if ocr_review_checkpoint else ""
    continue_only = "" if ocr_review_checkpoint else " hidden"
    return base_page(
        f"Edit {code} {job_id}",
        f"""
<link rel="stylesheet" href="/assets/editor-v2/editor.css?v=7">
<div class="editor-app" id="editor-app">
  <header class="editor-header">
    <div>
      <a href="/job/{escape(code)}/{escape(job_id)}">Back to job</a>
      <h1>Edit {escape(job_id)}</h1>
      <p class="muted editor-input">{input_display_html(status)}</p>
    </div>
    <div class="page-nav">
      <button id="prev-page" type="button" title="Previous page (A)">Previous [A]</button>
      <label>Page<select id="page-select">{page_options}</select></label>
      <button id="next-page" type="button" title="Next page (D)">Next [D]</button>
    </div>
  </header>
  {review_notice}
  <nav class="stage-tabs" aria-label="Editor stages">{stage_buttons}</nav>
  <div class="action-bar">
    <button id="undo" type="button" title="Undo (Ctrl+Z)" disabled>Undo</button>
    <button id="redo" type="button" title="Redo (Ctrl+Shift+Z)" disabled>Redo</button>
    <button id="save" type="button" title="Save changes (Ctrl+S)">Save</button>
    <button id="freeze-stage" type="button" data-tooltip="Protect all current values in this stage on this page. Later reruns keep them until you unprotect the stage.">Protect stage</button>
    <button id="freeze-page" type="button" data-tooltip="Protect all editor stages on this page. Later reruns keep their current values until you unprotect the page."{complete_only}>Protect page</button>
    <button id="regenerate" type="button" data-tooltip="Save this page, then rerun only the stages after the current stage for this page. The current stage is used as input and is not regenerated."{complete_only}>Regenerate downstream</button>
    <button id="regenerate-all" type="button" data-tooltip="Rerun every page with saved editor changes from the earliest required downstream pass. Then update all later stages and rebuild the downloads."{complete_only}>Regenerate all changes</button>
    <button id="continue-processing" type="button" data-tooltip="Save this page and continue the pipeline from Structure. The pipeline uses the reviewed OCR and merge data."{continue_only}>Continue processing</button>
    <span id="save-state" class="muted" role="status">Loading...</span>
  </div>
  <div id="editor-tooltip" class="editor-tooltip" role="tooltip" tabindex="0" hidden></div>
  <main class="editor-workspace">
    <aside class="record-pane">
      <div class="pane-heading">
        <h2 id="record-heading">Records</h2>
        <input id="record-filter" type="search" placeholder="Filter" aria-label="Filter records">
      </div>
      <div id="layer-controls" class="layer-controls"></div>
      <div id="record-list" class="record-list"></div>
    </aside>
    <div id="record-resizer" class="pane-resizer" role="separator" aria-label="Resize records pane" aria-orientation="vertical" tabindex="0"></div>
    <section class="canvas-pane">
      <div id="stage-tools" class="stage-tools"></div>
      <div class="canvas-wrap" id="canvas-wrap"><canvas id="page-canvas"></canvas></div>
      <p id="editor-status" class="status-line muted" role="status"></p>
    </section>
    <div id="inspector-resizer" class="pane-resizer" role="separator" aria-label="Resize inspector pane" aria-orientation="vertical" tabindex="0"></div>
    <aside class="inspector-pane">
      <div class="pane-heading"><h2>Inspector</h2><span id="selection-label" class="muted">No selection</span></div>
      <form id="record-form"></form>
      <div id="notes-panel" hidden>
        <h3>Translation Notes</h3>
        <label>Job notes<textarea id="job-notes" rows="4"></textarea></label>
        <label>Page notes<textarea id="page-notes" rows="4"></textarea></label>
      </div>
      <details class="history-panel">
        <summary>Saved revisions</summary>
        <div id="revision-list"></div>
      </details>
    </aside>
  </main>
</div>
<script>
window.TETOLATE_EDITOR_V2 = {{
  category: {json.dumps(code)},
  jobId: {json.dumps(job_id)},
  pageCount: {page_count},
  availableStages: {json.dumps(available_stages)},
  reviewCheckpoint: {json.dumps(status.get("reviewCheckpoint"))},
}};
</script>
<script type="module" src="/assets/editor-v2/app.js?v=8"></script>
""",
    )
