"""JSON control API. All mutations run through the live job manager."""

from __future__ import annotations

import codecs
import logging
import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler

from version import VERSION


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Input):
    password: str
    name: str = "terminal"
    read_only: StrictBool = False


class TokenName(Input):
    name: str
    read_only: StrictBool = False


class Name(Input):
    name: str


class Delete(Input):
    confirmation: str


class Password(Input):
    current_password: str
    new_password: str
    confirm_password: str


class VLMTest(Input):
    vlm_base_url: str | None = None
    vlm_auth_token: str = ""
    category: str | None = None
    job_id: str | None = None


class TitleRetry(Input):
    manual_title: str = ""


class Settings(Input):
    thinking_budget_tokens: StrictInt | None = None
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    vlm_auth_token: str | None = None
    paddleocr_vl_auth_token: str | None = None
    pause_after_ocr: StrictBool | None = None
    proofread_translations: StrictBool | None = None
    write_translation_notes: StrictBool | None = None
    auto_translate_comicinfo_title: StrictBool | None = None
    alt_placement_enabled: StrictBool | None = None
    source_language: str | None = None
    ocr_engine: str | None = None
    paddleocr_vl_server_url: str | None = None
    paddleocr_vl_model: str | None = None
    ocr_page_workers: StrictInt | None = None
    lama_workers: StrictInt | None = None
    imagemagick_workers: StrictInt | None = None
    clear_vlm_auth_token: StrictBool = False
    clear_paddleocr_vl_auth_token: StrictBool = False

    def changes(self) -> dict[str, Any]:
        data = self.model_dump(exclude_unset=True)
        if any(value is None for value in data.values()):
            raise HTTPException(422, "Settings cannot be null.")
        return data


class CategorySettings(Settings):
    translation_notes: str | None = None


class SeriesUpdate(Input):
    name: str | None = None
    instructions: str | None = None
    notes: str | None = None


class Membership(Input):
    series_id: str = ""
    chapter: str = ""


class Publish(Input):
    use_current: StrictBool = False


class Export(Input):
    format: Literal["png", "webp", "jxl"] = "png"
    title_mode: Literal["original", "translated"] = "original"
    alt_title_in_notes: StrictBool = False
    regenerate: StrictBool = False


class Rerun(Input):
    stage: Literal["ocr_raw", "ocr_merge", "ocr_structured", "alt_placement", "translations", "placements", "render"] = "ocr_raw"
    pages: list[StrictInt] = []


def error_response(status: int, message: str) -> JSONResponse:
    code = {400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
            409: "conflict", 413: "too_large", 422: "invalid_request", 429: "rate_limited",
            502: "endpoint_error"}.get(status, "server_error")
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status,
                        headers={"Cache-Control": "no-store"})


def install_control_api(app, get_manager, resolve_settings, save_upload, validate_archive,
                        save_images, submit_series, probe_models) -> None:
    router = APIRouter(prefix="/api/v1", tags=["control"])

    @app.exception_handler(Exception)
    async def control_server_error(request: Request, error: Exception):
        if not request.url.path.startswith("/api/v1/"):
            raise error
        logging.getLogger(__name__).error("Control API request failed", exc_info=error)
        return error_response(500, "The server could not complete the request. Check the server logs.")

    @app.exception_handler(StarletteHTTPException)
    async def control_http_error(request: Request, error: StarletteHTTPException):
        if request.url.path.startswith("/api/v1/"):
            response = error_response(error.status_code, str(error.detail))
            response.headers.update(error.headers or {})
            return response
        return await http_exception_handler(request, error)

    @app.exception_handler(RequestValidationError)
    async def control_validation_error(request: Request, error: RequestValidationError):
        if request.url.path.startswith("/api/v1/"):
            # Do not echo rejected values: settings and login bodies can contain keys.
            fields = [".".join(str(part) for part in item["loc"]) for item in error.errors()]
            return error_response(422, "Invalid fields: " + ", ".join(fields))
        return await request_validation_exception_handler(request, error)

    def find_job(manager, job_id: str, category: str | None):
        job_id = manager.validate_job_id(job_id)
        codes = (manager.validate_category(category),) if category else manager.categories()
        found = [code for code in codes if manager.load_status(code, job_id) is not None]
        if not found:
            raise HTTPException(404, "Unknown job.")
        if len(found) > 1:
            raise HTTPException(409, "Job ID is ambiguous. Specify its category.")
        return found[0]

    def public_job(manager, job_id: str, category: str | None):
        code = find_job(manager, job_id, category)
        return manager.public_status(code, job_id)

    def upload_options(manager, code, settings):
        try:
            changes = Settings.model_validate_json(settings).changes()
        except ValidationError:
            raise HTTPException(422, "Invalid job settings JSON. See the control API schema.") from None
        if changes.get("clear_vlm_auth_token") or changes.get("clear_paddleocr_vl_auth_token"):
            raise HTTPException(400, "Clear saved credentials through the settings action.")
        return resolve_settings(manager, code, changes)

    @router.post("/auth/login", status_code=201)
    async def login(request: Request, data: Login):
        manager = get_manager(request)
        client = request.client.host if request.client else "unknown"
        delay = manager.admin_login_retry_after(client)
        if delay:
            raise HTTPException(429, "Too many login attempts.", headers={"Retry-After": str(delay)})
        if not await run_in_threadpool(manager.admin_password_matches, data.password):
            manager.record_admin_login_failure(client)
            raise HTTPException(401, "Invalid admin password.")
        manager.clear_admin_login_failures(client)
        return manager.create_control_token(data.name, data.read_only)

    @router.post("/auth/password")
    async def password(request: Request, data: Password):
        manager = get_manager(request)
        client = request.client.host if request.client else "unknown"
        delay = manager.admin_login_retry_after(client)
        if delay:
            raise HTTPException(429, "Too many login attempts.", headers={"Retry-After": str(delay)})
        if not await run_in_threadpool(manager.admin_password_matches, data.current_password):
            manager.record_admin_login_failure(client)
            raise HTTPException(401, "Current password is incorrect.")
        try:
            session = await run_in_threadpool(manager.change_admin_password, data.current_password,
                                               data.new_password, data.confirm_password)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        manager.revoke_admin_session(session)
        manager.clear_admin_login_failures(client)
        return {"changed": True, "message": "Password changed. Browser sessions signed out. Control tokens remain valid."}

    @router.post("/vlm/test")
    async def test_vlm(request: Request, data: VLMTest):
        manager = get_manager(request)
        code = find_job(manager, data.job_id, data.category) if data.job_id else data.category
        if data.job_id:
            saved_endpoint = manager.status_vlm_base_url(manager.load_status(code, data.job_id))
        elif code is not None:
            manager.validate_category(code)
            saved_endpoint = manager.load_category_advanced_options(code)["vlmBaseUrl"]
        else:
            saved_endpoint = manager.config.default_vlm_base_url
        return await probe_models(manager, data.vlm_base_url if data.vlm_base_url is not None else saved_endpoint,
                                   data.vlm_auth_token, code, data.job_id)

    @router.get("/tokens")
    async def tokens(request: Request):
        return {"tokens": get_manager(request).list_control_tokens()}

    @router.post("/tokens", status_code=201)
    async def create_token(request: Request, data: TokenName):
        # Filesystem access grants the same authority as the browser administrator.
        auth = getattr(request.state, "control_auth", None)
        if auth and not auth.get("local"):
            raise HTTPException(403, "Use admin login to create a control token.")
        return get_manager(request).create_control_token(data.name, data.read_only)

    @router.delete("/tokens/{token_id}")
    async def revoke_token(request: Request, token_id: str):
        get_manager(request).revoke_control_token(token_id)
        return {"revoked": token_id}

    @router.get("/openapi.json")
    async def schema():
        return app.openapi()

    @router.get("/status")
    async def status(request: Request):
        manager = get_manager(request)
        jobs = [manager.load_status(code, job_id) or {} for code in manager.categories()
                for job_id in manager.iter_job_ids(code)]
        return {"apiVersion": 1, "version": VERSION, **manager.admin_status(),
                "waitingCount": sum(item.get("status") == "queued" and item.get("phase") == "Waiting for series" for item in jobs),
                "jobCount": len(jobs)}

    @router.post("/pause")
    async def pause(request: Request):
        get_manager(request).pause_jobs()
        return {"paused": True}

    @router.post("/resume")
    async def resume(request: Request):
        get_manager(request).resume_jobs()
        return {"paused": False}

    @router.get("/categories")
    async def categories(request: Request):
        return {"categories": get_manager(request).admin_status()["categories"]}

    @router.post("/categories", status_code=201)
    async def create_category(request: Request, data: Name):
        return {"category": get_manager(request).create_category(data.name)}

    @router.patch("/categories/{code}")
    async def rename_category(request: Request, code: str, data: Name):
        return {"category": get_manager(request).rename_category(code, data.name)}

    @router.delete("/categories/{code}")
    async def delete_category(request: Request, code: str, data: Delete):
        get_manager(request).delete_category(code, data.confirmation)
        return {"deleted": code}

    @router.get("/categories/{code}/settings")
    async def category_settings(request: Request, code: str):
        manager = get_manager(request)
        manager.validate_category(code)
        return manager.load_category_advanced_options(code)

    @router.post("/categories/{code}/settings/apply-to-queued")
    async def apply_category_settings(request: Request, code: str):
        updated = get_manager(request).apply_category_options_to_queue(code)
        return {"updated": updated, "count": len(updated)}

    @router.patch("/categories/{code}/settings")
    async def update_category_settings(request: Request, code: str, data: CategorySettings):
        manager = get_manager(request)
        manager.validate_category(code)
        changes = data.changes()
        if "paddleocr_vl_auth_token" in changes or changes.get("clear_paddleocr_vl_auth_token"):
            raise HTTPException(400, "Set OCR authentication on individual jobs.")
        options = resolve_settings(manager, code, changes, require_model=False)
        if "translation_notes" in changes:
            options["translation_notes"] = changes["translation_notes"]
        token = options.pop("vlm_auth_token")
        options.pop("paddleocr_vl_auth_token")
        manager.remember_category_advanced_options(code, **options)
        if changes.get("clear_vlm_auth_token") or token:
            manager.remember_vlm_token(options["vlm_base_url"], None if changes.get("clear_vlm_auth_token") else token)
        return manager.load_category_advanced_options(code)

    @router.get("/jobs")
    async def jobs(request: Request, category: str | None = None, state: str | None = None):
        manager = get_manager(request)
        codes = (manager.validate_category(category),) if category else manager.categories()
        values = [job for code in codes for job in manager.public_category_jobs(code)["jobs"]
                  if state is None or job["status"] == state]
        fields = {"category", "jobId", "jobName", "translatedJobName", "status", "phase", "page", "message",
                  "createdAt", "age", "elapsed", "seriesId", "chapterOrder", "hasDownload", "ocrReviewCheckpoint", "exportsRunning"}
        return {"jobs": [{key: value for key, value in job.items() if key in fields} for job in values]}

    @router.post("/jobs", status_code=201)
    async def submit(request: Request, file: UploadFile | None = File(None),
                     page_images: list[UploadFile] | None = File(None), category: str = Form("default"),
                     settings: str = Form("{}"), translation_notes: str = Form(""),
                     series_id: str = Form(""), chapter: str = Form("")):
        manager = get_manager(request)
        code = manager.validate_category(category)
        options = upload_options(manager, code, settings)
        manager.validate_series_chapter(code, series_id, chapter)
        images = [item for item in page_images or [] if item.filename]
        if bool(file and file.filename) == bool(images):
            raise HTTPException(400, "Upload one CBZ or ZIP archive, or separate page images, not both.")
        filename = file.filename if file else ""
        if file and Path(filename).suffix.lower() not in {".cbz", ".zip"}:
            raise HTTPException(400, "Upload a CBZ or ZIP archive.")
        job_id = manager.create_job_id(code)
        path = manager.job_dir(code, job_id)
        path.mkdir(parents=True)
        try:
            if file:
                await save_upload(file, manager.input_path(code, job_id), manager.config.max_upload_bytes, "Uploaded archive")
                await run_in_threadpool(validate_archive, manager.input_path(code, job_id), filename)
            else:
                _, filename = await save_images(images, manager.input_path(code, job_id), manager.config.max_upload_bytes)
            manager.submit_job(code, job_id, filename, **options, translation_notes=translation_notes,
                               series_id=series_id, chapter=chapter)
        except Exception:
            shutil.rmtree(path)
            raise
        finally:
            if file:
                await file.close()
            for item in page_images or []:
                await item.close()
        return manager.public_status(code, job_id, include_log=False)

    @router.get("/jobs/{job_id}")
    async def job(request: Request, job_id: str, category: str | None = None):
        return public_job(get_manager(request), job_id, category)

    @router.patch("/jobs/{job_id}/settings")
    async def settings(request: Request, job_id: str, data: Settings, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        changes = data.changes()
        with manager._lock:
            options = resolve_settings(manager, code, changes, job_id, require_model=False)
            manager.update_job_advanced_options(code, job_id, **options,
                clear_vlm_auth_token=data.clear_vlm_auth_token, clear_paddleocr_vl_auth_token=data.clear_paddleocr_vl_auth_token)
        return manager.public_status(code, job_id, include_log=False)

    @router.post("/jobs/{job_id}/settings/use-category")
    async def use_category_settings(request: Request, job_id: str, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        manager.use_category_options_for_job(code, job_id)
        return manager.public_status(code, job_id, include_log=False)

    @router.post("/jobs/{job_id}/restart")
    async def restart(request: Request, job_id: str, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        manager.restart_failed_job(code, job_id)
        return manager.public_status(code, job_id, include_log=False)

    @router.post("/jobs/{job_id}/terminate")
    async def terminate(request: Request, job_id: str, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        manager.terminate_active_job(code, job_id)
        return manager.public_status(code, job_id, include_log=False)

    @router.patch("/jobs/{job_id}/name")
    async def rename_job(request: Request, job_id: str, data: Name, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        try:
            manager.rename_job(code, job_id, data.name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        return manager.public_status(code, job_id, include_log=False)

    @router.delete("/jobs/{job_id}")
    async def delete_job(request: Request, job_id: str, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        manager.delete_job(code, job_id)
        return {"deleted": job_id, "category": code}

    @router.patch("/jobs/{job_id}/translated-title")
    async def set_title(request: Request, job_id: str, data: Name, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        try:
            manager.set_translated_title(code, job_id, data.name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        return manager.public_status(code, job_id, include_log=False)

    async def run_title(request, job_id, category, retry, manual_title=""):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        try:
            if retry:
                await run_in_threadpool(manager.retry_title_translation, code, job_id, manual_title)
            else:
                await run_in_threadpool(manager.translate_title, code, job_id)
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(400, "The translated title is not valid. Set a manual title or retry.") from None
        except Exception:
            raise HTTPException(502, "Title translation failed. Check the job warning and VLM settings, then retry.") from None
        return manager.public_status(code, job_id, include_log=False)

    @router.post("/jobs/{job_id}/translate-title")
    async def translate_title(request: Request, job_id: str, category: str | None = None):
        return await run_title(request, job_id, category, False)

    @router.post("/jobs/{job_id}/retry-title-translation")
    async def retry_title(request: Request, job_id: str, data: TitleRetry, category: str | None = None):
        return await run_title(request, job_id, category, True, data.manual_title)

    def page_files(manager, code, job_id, kind):
        if kind == "original":
            return manager.original_page_files(code, job_id)
        if (manager.load_status(code, job_id) or {}).get("status") != "complete":
            raise HTTPException(409, "Wait for the job to complete before viewing translated pages.")
        return manager.final_page_files(code, job_id)

    @router.get("/jobs/{job_id}/pages")
    async def pages(request: Request, job_id: str, kind: Literal["original", "final"] = "final", category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        return {"pages": [{"page": index, "filename": path.name, "bytes": path.stat().st_size}
                          for index, path in enumerate(page_files(manager, code, job_id, kind))]}

    @router.get("/jobs/{job_id}/pages/{page}")
    async def page(request: Request, job_id: str, page: int, kind: Literal["original", "final"] = "final", category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        paths = page_files(manager, code, job_id, kind)
        if page < 0 or page >= len(paths):
            raise HTTPException(404, "Page is not available.")
        return FileResponse(paths[page], filename=paths[page].name)

    @router.post("/jobs/{job_id}/rerun")
    async def rerun(request: Request, job_id: str, data: Rerun, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        if any(page < 0 for page in data.pages):
            raise HTTPException(400, "Page numbers cannot be negative.")
        manager.rerun_completed_job_pages(code, job_id, data.pages, resume_from=data.stage)
        return manager.public_status(code, job_id, include_log=False)

    @router.post("/jobs/{job_id}/export")
    async def export(request: Request, job_id: str, data: Export, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        existing = manager.translated_cbz_variant_path(code, job_id, data.format)
        status = manager.load_status(code, job_id) or {}
        # An existing archive can be downloaded immediately unless replacement was requested.
        if data.regenerate or not existing.is_file() or status.get("status") != "complete":
            manager.generate_download(code, job_id, data.format, title_mode=data.title_mode,
                alt_title_in_notes=data.alt_title_in_notes, regenerate=data.regenerate)
        return manager.public_status(code, job_id, include_log=False)

    @router.get("/jobs/{job_id}/download/{variant}")
    async def download(request: Request, job_id: str, variant: Literal["png", "webp", "jxl", "original"], category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        if variant == "original":
            path = manager.input_path(code, job_id)
            filename = Path((manager.load_status(code, job_id) or {}).get("inputFilename") or "original.cbz").name
        else:
            if (manager.load_status(code, job_id) or {}).get("status") != "complete":
                raise HTTPException(409, "Wait for the job to complete before downloading.")
            path = manager.translated_cbz_variant_path(code, job_id, variant)
            filename = manager.translated_cbz_download_name(code, job_id)
        if not path.is_file():
            raise HTTPException(404, "Archive is not available.")
        return FileResponse(path, filename=filename, media_type="application/zip")

    @router.get("/jobs/{job_id}/logs")
    async def logs(request: Request, job_id: str, category: str | None = None,
                   cursor: str = "", limit: int = Query(65536, ge=1024, le=1048576)):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        path = manager.log_path(code, job_id)
        state = (manager.load_status(code, job_id) or {}).get("status")
        if not path.exists():
            return {"text": "", "cursor": "", "reset": bool(cursor), "hasMore": False, "status": state}
        with path.open("rb") as source:
            import os
            stat = os.fstat(source.fileno())
            identity = str(stat.st_ino)
            offset = 0
            reset = False
            if cursor:
                try:
                    previous_identity, value = cursor.split(":")
                    offset = int(value)
                    if offset < 0:
                        raise ValueError
                except ValueError:
                    raise HTTPException(400, "Invalid log cursor.") from None
                reset = previous_identity != identity or offset > stat.st_size
                if reset:
                    offset = 0
            source.seek(offset)
            data = source.read(limit)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            text = decoder.decode(data, final=source.tell() >= stat.st_size)
            offset = source.tell() - len(decoder.getstate()[0])
        return {"text": text, "cursor": f"{identity}:{offset}", "reset": reset,
                "hasMore": offset < stat.st_size, "status": state}

    @router.get("/series")
    async def series(request: Request, category: str | None = None):
        manager = get_manager(request)
        codes = (manager.validate_category(category),) if category else manager.categories()
        return {"series": [{"category": code, **{key: value for key, value in item.items()
                            if key not in {"notes", "instructions"}}}
                           for code in codes for item in manager.public_series(code)]}

    @router.post("/categories/{code}/series", status_code=201)
    async def create_series(request: Request, code: str, data: Name):
        manager = get_manager(request)
        series_id = manager.create_series(code, data.name)
        return {"id": series_id, "category": code, **manager.get_series(code, series_id)}

    @router.get("/categories/{code}/series/{series_id}")
    async def get_series(request: Request, code: str, series_id: str):
        manager = get_manager(request)
        return {"id": series_id, "category": code, **manager.get_series(code, series_id),
                "chapters": [manager.public_status(code, item["jobId"], include_log=False)
                             for item in manager.series_chapters(code, series_id)]}

    @router.post("/categories/{code}/series/{series_id}/upload", status_code=201)
    async def upload_chapters(request: Request, code: str, series_id: str, chapters: list[UploadFile] = File(...),
                              first_chapter: str = Form("1"), settings: str = Form("{}"), translation_notes: str = Form("")):
        manager = get_manager(request)
        manager.validate_category(code)
        options = upload_options(manager, code, settings)
        options.update(translation_notes=translation_notes, write_translation_notes=True)
        ids = await submit_series(manager, code, series_id, chapters, first_chapter, options)
        return {"jobs": [manager.public_status(code, job_id, include_log=False) for job_id in ids]}

    @router.patch("/categories/{code}/series/{series_id}")
    async def update_series(request: Request, code: str, series_id: str, data: SeriesUpdate):
        manager = get_manager(request)
        series = manager.get_series(code, series_id)
        manager.update_series(code, series_id, data.name if data.name is not None else series["name"],
            data.instructions if data.instructions is not None else series["instructions"],
            data.notes if data.notes is not None else series["notes"])
        return {"id": series_id, "category": code, **manager.get_series(code, series_id)}

    @router.patch("/jobs/{job_id}/series")
    async def assign_series(request: Request, job_id: str, data: Membership, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        manager.assign_job_series(code, job_id, data.series_id, data.chapter)
        return manager.public_status(code, job_id, include_log=False)

    @router.post("/jobs/{job_id}/series-notes")
    async def publish_notes(request: Request, job_id: str, data: Publish, category: str | None = None):
        manager = get_manager(request)
        code = find_job(manager, job_id, category)
        manager.publish_series_notes(code, job_id, explicit=True, use_current=data.use_current)
        return {"published": True, "jobId": job_id}

    app.include_router(router)
    original_openapi = app.openapi

    def control_openapi():
        document = original_openapi()
        document.setdefault("components", {}).setdefault("securitySchemes", {})["ControlToken"] = {
            "type": "http", "scheme": "bearer", "description": "The local CLI reads its private credential from the jobs directory. Remote clients obtain a control token with /api/v1/auth/login.",
        }
        for path, operations in document["paths"].items():
            if path.startswith("/api/v1/"):
                for method, operation in operations.items():
                    if method in {"get", "post", "patch", "put", "delete"}:
                        operation["security"] = [] if path == "/api/v1/auth/login" else [{"ControlToken": []}]
        return document

    app.openapi = control_openapi
