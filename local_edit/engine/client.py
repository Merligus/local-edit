"""The `sd-server` native API, over urllib.

Only the `/sdcpp/v1` family is used. The server also speaks an A1111-compatible
`/sdapi/v1` and an OpenAI-compatible `/v1`, but both are shaped for text-to-image
and neither carries a *list* of reference images, which is the one thing this
app exists to send.

The flow is submit / poll / cancel:

    POST /sdcpp/v1/img_gen        -> 202 {"id": ..., "poll_url": ...}
    GET  /sdcpp/v1/jobs/{id}      -> queued | generating | completed | failed
                                     | cancelled
    POST /sdcpp/v1/jobs/{id}/cancel

`status` carries no step counter — only a `queue_position` — so nothing here
reports progress. That comes from the server process's own output, which the app
can read because it owns the process; see `progress` and `server`.

Images cross as base64 in JSON, which costs a third in size over the wire. On
loopback against a model that takes minutes per image, that is not a cost worth
engineering around, and it keeps the request one self-contained object.

Field names were taken from `examples/common/common.cpp` and
`examples/server/routes_sdcpp.cpp` rather than from `api.md`, which is close but
not exact — `increase_ref_index` appears in neither the documented request table
nor the CLI-flag list in the same spelling.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Recipe

API = "/sdcpp/v1"

STATUS_QUEUED = "queued"
STATUS_GENERATING = "generating"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED})


class ApiError(Exception):
    """The server refused a request or could not be reached."""


def encode_image(path: Path) -> str:
    """A file as base64, the form every image field takes."""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as e:
        raise ApiError(f"could not read {path.name}: {e}") from e


@dataclass
class ImgGenRequest:
    """One edit, as the server wants it.

    Deliberately a flat dataclass rather than a dict built at the call site, so
    `runner` cannot misspell a field into silence: the server ignores keys it
    does not recognise, and a typo in `ref_images` would produce a plausible
    image that simply ignored every reference.
    """

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = -1
    batch_count: int = 1
    #: Base64, no data-URL prefix needed.
    init_image: str = ""
    mask_image: str = ""
    ip_adapter_image: str = ""
    ref_images: list[str] = field(default_factory=list)
    strength: float = 0.75
    ip_adapter_strength: float = 1.0
    #: Numbers the references 1..N so a prompt can say "the jacket in image 2".
    increase_ref_index: bool = True
    output_format: str = "png"

    # sample_params
    sample_steps: int = 20
    sample_method: str = "euler"
    scheduler: str = ""
    flow_shift: float | None = None
    txt_cfg: float = 7.0
    img_cfg: float | None = None
    distilled_guidance: float | None = None

    def to_json(self) -> dict:
        guidance: dict[str, float] = {"txt_cfg": self.txt_cfg}
        if self.img_cfg is not None:
            guidance["img_cfg"] = self.img_cfg
        if self.distilled_guidance is not None:
            guidance["distilled_guidance"] = self.distilled_guidance

        sample: dict = {
            "sample_steps": self.sample_steps,
            "sample_method": self.sample_method,
            "guidance": guidance,
        }
        if self.scheduler:
            sample["scheduler"] = self.scheduler
        if self.flow_shift is not None:
            sample["flow_shift"] = self.flow_shift

        body: dict = {
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "batch_count": self.batch_count,
            "output_format": self.output_format,
            "sample_params": sample,
        }
        if self.negative_prompt:
            body["negative_prompt"] = self.negative_prompt
        if self.init_image:
            body["init_image"] = self.init_image
            body["strength"] = self.strength
        if self.mask_image:
            body["mask_image"] = self.mask_image
        if self.ip_adapter_image:
            body["ip_adapter_image"] = self.ip_adapter_image
            body["ip_adapter_strength"] = self.ip_adapter_strength
        if self.ref_images:
            body["ref_images"] = self.ref_images
            body["increase_ref_index"] = self.increase_ref_index
        return body

    def redacted(self) -> dict:
        """`to_json` with the base64 blobs replaced by their sizes, for logs."""
        body = self.to_json()
        for key in ("init_image", "mask_image", "ip_adapter_image"):
            if body.get(key):
                body[key] = f"<{len(body[key])} b64 chars>"
        if body.get("ref_images"):
            body["ref_images"] = [f"<{len(r)} b64 chars>" for r in body["ref_images"]]
        return body


def request_from_recipe(recipe: Recipe, prompt: str, *,
                        width: int, height: int, seed: int = -1,
                        steps: int | None = None,
                        cfg_scale: float | None = None,
                        strength: float | None = None,
                        negative_prompt: str = "",
                        init_image: str = "", ref_images: list[str] | None = None,
                        ip_adapter_image: str = "",
                        ip_adapter_strength: float = 1.0) -> ImgGenRequest:
    """Build a request with the recipe's defaults, overridden where asked."""
    return ImgGenRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width, height=height, seed=seed,
        init_image=init_image,
        ref_images=list(ref_images or []),
        ip_adapter_image=ip_adapter_image,
        ip_adapter_strength=ip_adapter_strength,
        strength=strength if strength is not None else recipe.strength,
        sample_steps=steps if steps is not None else recipe.steps,
        sample_method=recipe.sampling_method,
        scheduler=recipe.scheduler,
        flow_shift=recipe.flow_shift,
        txt_cfg=cfg_scale if cfg_scale is not None else recipe.cfg_scale,
        distilled_guidance=recipe.guidance,
    )


class Client:
    """Talks to one `sd-server`. Cheap to make; holds no connection."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1234,
                 timeout: float = 30.0) -> None:
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------
    def _call(self, method: str, path: str, body: dict | None = None,
              timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read() or b"{}").get("error") or ""
            except (ValueError, OSError):
                pass
            raise ApiError(f"HTTP {e.code} from {path}"
                           + (f": {detail}" if detail else "")) from e
        except urllib.error.URLError as e:
            raise ApiError(f"could not reach the engine: {e.reason}") from e
        except OSError as e:
            raise ApiError(f"could not reach the engine: {e}") from e
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError as e:
            raise ApiError(f"{path} returned something that is not JSON") from e

    # -- endpoints --------------------------------------------------------
    def capabilities(self, timeout: float = 5.0) -> dict:
        """Also the readiness probe: it answers only once weights are loaded."""
        return self._call("GET", f"{API}/capabilities", timeout=timeout)

    def alive(self, timeout: float = 3.0) -> bool:
        try:
            self.capabilities(timeout=timeout)
            return True
        except ApiError:
            return False

    def submit(self, request: ImgGenRequest) -> str:
        """Queue an edit. Returns the job id."""
        # The body carries base64 images and can be tens of megabytes, so this
        # single call gets a longer timeout than the polls that follow.
        reply = self._call("POST", f"{API}/img_gen", request.to_json(),
                           timeout=max(self.timeout, 120.0))
        job_id = reply.get("id")
        if not job_id:
            raise ApiError("the engine accepted the request but returned no job id")
        return str(job_id)

    def job(self, job_id: str) -> dict:
        return self._call("GET", f"{API}/jobs/{job_id}")

    def cancel(self, job_id: str) -> bool:
        """Ask the engine to stop. False when it had already finished."""
        try:
            self._call("POST", f"{API}/jobs/{job_id}/cancel", {})
            return True
        except ApiError:
            # 404 (unknown) and 409 (already complete) are both "nothing to do",
            # and the caller's next poll will see the real state anyway.
            return False


def images_from_job(job: dict) -> list[bytes]:
    """Decode a completed job's images."""
    result = job.get("result") or {}
    out = []
    for entry in result.get("images") or []:
        b64 = entry.get("b64_json") or ""
        if b64:
            try:
                out.append(base64.b64decode(b64))
            except (ValueError, TypeError) as e:
                raise ApiError("the engine returned an image that is not "
                               "valid base64") from e
    return out


def job_error(job: dict) -> str:
    return str(job.get("error") or job.get("error_message")
               or "the engine reported a failure with no message")
