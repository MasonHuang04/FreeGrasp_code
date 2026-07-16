from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import (
    INPUT_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    point_to_dataset_id,
    parse_freegrasp_response,
    read_json,
    write_json,
)
from .metrics import (
    DEFAULT_IOU_THRESHOLD,
    GroundTruthMaskError,
    score_prediction_against_gt_files,
)
from .segmentation import SegmentationInfrastructureError, predict_object_mask


MOLMO_PROMPT = "Point out all objects in the green tray"
LOCALIZATION_MODES = ("gt", "molmo")
RELAY_BASE_URL = "https://www.highland-api.top/v1"
MAX_API_REQUEST_BYTES = 8_000_000
FREEGRASP_REQUEST_PROFILE = "original_freegrasp_request_v1"

FREEGRASP_SYSTEM_PROMPT = (
    "You are a robotic system for bin picking, using a parallel gripper. I labeled all objects id in the image."
    "You have two possible actions:"
    "1. remove obstacle, object_id: This action moves the specified object out of the way so it does not interfere with grasping the desired target object. This action can only be performed if the specified object is free of obstacles (not occluded by any other object)."
    "2. pick object, object_id: This action picks up the specified object. It can only be performed if the object is free of obstacles."
    "An object is considered an obstacle if it occludes another object."
    "Task:"
    "Given a target object description as input, determine the first object that needs to be grasped to enable picking the target object. If the target object is free of obstacles, return the target object ID itself. Otherwise, identify an object that is occluding the target and is itself free of obstacles. If multiple objects could be removed, return any one valid option."
    "Output Format:"
    "The output should only be the object ID of the first object to grasp, "
    "must formatted as: [object_id, color class_name]\n"
)


CSV_FIELDS = (
    "testcase",
    "scene_id",
    "query_obj_id",
    "difficulty",
    "ambiguous",
    "split",
    "annotation",
    "model",
    "api_response_model",
    "api_transport",
    "api_base_url",
    "localization_mode",
    "status",
    "predicted_localization_id",
    "predicted_molmo_id",
    "predicted_npz_label",
    "predicted_object_id",
    "predicted_class_name",
    "point_x",
    "point_y",
    "predicted_mask",
    "ground_truth_mask_manifest",
    "iou",
    "ssr",
    "rsr",
    "iou_threshold",
    "best_ground_truth_object_id",
    "metric_status",
    "raw_response",
    "elapsed_seconds",
    "result_json",
)


class APIRequestError(RuntimeError):
    """A transport/API failure that produced no usable model result."""


class LocalizationInfrastructureError(RuntimeError):
    """Required localization output/model infrastructure is unavailable."""


def _scene_dirs(input_root: Path, args: argparse.Namespace) -> list[Path]:
    manifest = read_json(input_root / "manifest.json")
    requested_testcases = set(args.testcase or [])
    requested_scenes = set(args.scene_id or [])
    result = []
    for testcase in manifest["testcases"]:
        name = testcase["name"]
        slug = name.split("_", 1)[-1]
        if requested_testcases and name not in requested_testcases and slug not in requested_testcases:
            continue
        for scene_id in testcase["scene_ids"]:
            if requested_scenes and int(scene_id) not in requested_scenes:
                continue
            result.append(input_root / name / f"scene_{scene_id}")
    if args.limit_scenes is not None:
        result = result[: args.limit_scenes]
    return result


def _localization_paths(
    output_root: Path,
    localization_mode: str,
    scene_id: int,
) -> tuple[Path, Path, Path]:
    folder = output_root / "localization" / localization_mode / f"scene{scene_id}"
    return (
        folder / "localization_result.json",
        folder / f"{scene_id}.png",
        folder / f"{scene_id}_id.txt",
    )


def run_molmo_scene(
    scene_dir: Path,
    output_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    metadata = read_json(scene_dir / "metadata.json")
    scene_id = int(metadata["scene_id"])
    result_path, labeled_image_path, id_text_path = _localization_paths(
        output_root, "molmo", scene_id
    )
    if result_path.exists() and labeled_image_path.exists() and id_text_path.exists() and not force:
        print(f"[molmo cached] scene={scene_id}", flush=True)
        payload = read_json(result_path)
        if payload.get("labeled_image_style") != "freegrasp_matplotlib":
            # Older rsr outputs replaced FreeGrasp's Matplotlib artifact with
            # a smaller opaque-label PIL image. Restore the original visual
            # prompt from cached points without running Molmo again.
            image = Image.open(scene_dir / "image.png").convert("RGB")
            _save_freegrasp_matplotlib_png(
                image,
                payload.get("points", []),
                labeled_image_path,
            )
            payload["labeled_image_style"] = "freegrasp_matplotlib"
            payload["labeled_image_restored_without_molmo_rerun"] = True
            write_json(result_path, payload)
            print(
                f"[molmo label restored] scene={scene_id} style=freegrasp_matplotlib",
                flush=True,
            )
        return payload
    if force:
        # A failed fresh localization must not leave an older localization
        # available for the reasoning stage.
        for path in (result_path, labeled_image_path, id_text_path):
            path.unlink(missing_ok=True)

    print(f"[molmo start] scene={scene_id}", flush=True)
    started = time.time()
    # Importing the official module loads the official Molmo model. No source file is changed.
    import molmo_eval

    image = Image.open(scene_dir / "image.png").convert("RGB")
    # Molmo's custom processor mutates the PIL image in place in this runtime.
    # Keep the original intact for the numbered visual prompt.
    points = molmo_eval.run_molmo_inference(image.copy(), MOLMO_PROMPT)
    instances = np.load(scene_dir / "instances_objects.npy")
    mapping = {}
    point_records = []
    for molmo_id, x, y in points:
        mapped = point_to_dataset_id(instances, int(x), int(y))
        mapping[int(molmo_id)] = mapped["npz_label"] if mapped["npz_label"] is not None else -1
        point_records.append({
            "localization_id": int(molmo_id),
            "molmo_id": int(molmo_id),
            "x": int(x),
            "y": int(y),
            **mapped,
        })

    molmo_eval.OUTPUT_DIR = str(output_root / "localization" / "molmo")
    molmo_eval.save_results(scene_id, image, points, mapping)
    payload = {
        "schema_version": 1,
        "scene_id": scene_id,
        "localization_mode": "molmo",
        "prompt": MOLMO_PROMPT,
        "points": point_records,
        "labeled_image": str(labeled_image_path.resolve()),
        "labeled_image_style": "freegrasp_matplotlib",
        "id_text": str(id_text_path.resolve()),
        "elapsed_seconds": round(time.time() - started, 3),
        "mapping_definition": "predicted_object_id = instances_objects[y, x] - 1",
    }
    write_json(result_path, payload)
    print(f"[molmo done] scene={scene_id} points={len(point_records)}", flush=True)
    return payload


def _visible_instance_point(instances: np.ndarray, npz_label: int) -> tuple[int, int]:
    """Choose a visible point inside a GT instance, near its visible centroid."""
    ys, xs = np.nonzero(instances == npz_label)
    if not len(xs):
        raise ValueError(f"NPZ label {npz_label} has no visible pixels")
    center_x = int(round(float(xs.mean())))
    center_y = int(round(float(ys.mean())))
    if int(instances[center_y, center_x]) == npz_label:
        return center_x, center_y
    closest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    return int(xs[closest]), int(ys[closest])


def _save_numbered_png(image: Image.Image, points: list[dict[str, Any]], path: Path) -> None:
    """Draw IDs without resizing or changing the lossless PNG format."""
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    font_size = max(14, min(labeled.size) // 24)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    padding = max(2, font_size // 6)
    for point in points:
        text = str(point["localization_id"])
        x, y = int(point["x"]), int(point["y"])
        left, top, right, bottom = draw.textbbox((x, y), text, font=font, anchor="mm")
        draw.rounded_rectangle(
            (left - padding, top - padding, right + padding, bottom + padding),
            radius=padding,
            fill=(0, 0, 0),
        )
        draw.text((x, y), text, fill=(255, 255, 0), font=font, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.save(path, format="PNG", optimize=True)


def _save_freegrasp_matplotlib_png(
    image: Image.Image,
    points: list[dict[str, Any]],
    path: Path,
) -> None:
    """Reproduce FreeGrasp's original Matplotlib numbered PNG."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    for point in points:
        plt.text(
            int(point["x"]),
            int(point["y"]),
            int(point["localization_id"]),
            color="yellow",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
        )
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


def _encode_original_png(source: Path) -> tuple[str, dict[str, Any]]:
    """Upload the exact FreeGrasp Matplotlib PNG bytes without conversion."""
    encoded = source.read_bytes()
    with Image.open(source) as image:
        source_format = image.format
        size = list(image.size)
        mode = image.mode
    if source_format != "PNG":
        raise ValueError(f"Expected a PNG visual prompt, got {source_format}: {source}")
    return base64.b64encode(encoded).decode("utf-8"), {
        "source": str(source.resolve()),
        "source_format": source_format,
        "transport_format": "PNG",
        "size": size,
        "mode": mode,
        "encoded_image_bytes": len(encoded),
        "transport_uses_source_bytes": True,
        "resized": False,
        "recompressed": False,
    }


def _freegrasp_chat_payload(
    model: str,
    instruction: str,
    base64_image: str,
) -> dict[str, Any]:
    """Build the original FreeGrasp Chat Completions payload exactly."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": FREEGRASP_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Grasp {instruction}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 713,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "seed": 0,
    }


def run_gt_scene(
    scene_dir: Path,
    output_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    metadata = read_json(scene_dir / "metadata.json")
    scene_id = int(metadata["scene_id"])
    result_path, labeled_image_path, id_text_path = _localization_paths(
        output_root, "gt", scene_id
    )
    if result_path.exists() and labeled_image_path.exists() and id_text_path.exists() and not force:
        print(f"[gt cached] scene={scene_id}", flush=True)
        return read_json(result_path)
    if force:
        for path in (result_path, labeled_image_path, id_text_path):
            path.unlink(missing_ok=True)

    print(f"[gt start] scene={scene_id}", flush=True)
    started = time.time()
    image = Image.open(scene_dir / "image.png").convert("RGB")
    instances = np.load(scene_dir / "instances_objects.npy")
    npz_labels = sorted(int(value) for value in np.unique(instances) if int(value) > 0)
    points = []
    for npz_label in npz_labels:
        x, y = _visible_instance_point(instances, npz_label)
        points.append({
            "localization_id": npz_label,
            "molmo_id": None,
            "x": x,
            "y": y,
            "npz_label": npz_label,
            "dataset_object_id": npz_label - 1,
        })

    _save_numbered_png(image, points, labeled_image_path)
    id_text_path.write_text(
        "Localization_ID X Y NPZ_Label Dataset_Object_ID\n"
        + "".join(
            f"{item['localization_id']} {item['x']} {item['y']} "
            f"{item['npz_label']} {item['dataset_object_id']}\n"
            for item in points
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "scene_id": scene_id,
        "localization_mode": "gt",
        "points": points,
        "labeled_image": str(labeled_image_path.resolve()),
        "id_text": str(id_text_path.resolve()),
        "elapsed_seconds": round(time.time() - started, 3),
        "mapping_definition": "localization_id = npz_label; predicted_object_id = npz_label - 1",
    }
    write_json(result_path, payload)
    print(f"[gt done] scene={scene_id} points={len(points)}", flush=True)
    return payload


def run_localization_scene(
    localization_mode: str,
    scene_dir: Path,
    output_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    if localization_mode == "gt":
        return run_gt_scene(scene_dir, output_root, force=force)
    if localization_mode == "molmo":
        return run_molmo_scene(scene_dir, output_root, force=force)
    raise ValueError(f"Unknown localization mode: {localization_mode}")


class SDKChatClient:
    transport_name = "openai_sdk"

    def __init__(self, api_key: str, base_url: str | None, timeout_seconds: float):
        import httpx
        from openai import OpenAI

        self.base_url = base_url
        # Apply the requested limit to every network phase. Previously the
        # connect/TLS phase was capped at 60 seconds even when --api-timeout
        # was 420, so the SDK could report a timeout well before seven minutes.
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=timeout_seconds,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        ) if base_url else OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    def create(self, **payload: Any) -> dict[str, Any]:
        return self.client.chat.completions.create(**payload).model_dump()


class CurlChatClient:
    """OpenAI-compatible transport for relays whose TLS fails in httpx."""

    transport_name = "curl"

    def __init__(self, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def create(self, **payload: Any) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["FREEGRASP_RELAY_BASE_URL"] = self.base_url
        # The key remains in OPENAI_API_KEY, and the image JSON is streamed on
        # stdin. Neither credential nor request payload is written to a file.
        request_body = json.dumps(payload, ensure_ascii=False)
        request_bytes = len(request_body.encode("utf-8"))
        if request_bytes > MAX_API_REQUEST_BYTES:
            raise RuntimeError(
                f"API request is {request_bytes} bytes; relay-safe limit is "
                f"{MAX_API_REQUEST_BYTES} bytes"
            )
        environment["FREEGRASP_CONTENT_LENGTH"] = str(request_bytes)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                """
                exec curl --silent --show-error --fail-with-body \
                  --http1.1 \
                  --connect-timeout "$FREEGRASP_CONNECT_TIMEOUT" \
                  --max-time "$FREEGRASP_API_TIMEOUT" \
                  --request POST \
                  --header "Authorization: Bearer $OPENAI_API_KEY" \
                  --header "Content-Type: application/json" \
                  --header "Content-Length: $FREEGRASP_CONTENT_LENGTH" \
                  --header "Expect:" \
                  --data-binary @- \
                  --url "$FREEGRASP_RELAY_BASE_URL/chat/completions"
                """,
            ],
            input=request_body,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds + 15.0,
            env={
                **environment,
                "FREEGRASP_CONNECT_TIMEOUT": str(self.timeout_seconds),
                "FREEGRASP_API_TIMEOUT": str(self.timeout_seconds),
            },
        )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = (completed.stderr or completed.stdout or "empty response").strip()[-1000:]
            raise RuntimeError(
                f"Compatible API returned non-JSON output (curl code {completed.returncode}): {detail}"
            ) from exc
        if completed.returncode != 0 or response.get("error"):
            error = response.get("error") or {}
            detail = error.get("message") or completed.stderr.strip() or str(response)
            code = error.get("code")
            raise RuntimeError(
                f"Compatible API request failed (curl code {completed.returncode}, api code {code}): {detail}"
            )
        return response


def _chat_client(api_transport: str, timeout_seconds: float):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    selected = api_transport
    if selected == "auto":
        selected = "openai"
    if selected == "curl":
        return CurlChatClient(RELAY_BASE_URL, timeout_seconds)
    return SDKChatClient(api_key, RELAY_BASE_URL, timeout_seconds)


def _request_chat_completion(
    client: Any,
    payload: dict[str, Any],
    *,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], int]:
    """Retry transport failures while preserving one logical evaluation run."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")

    errors: list[str] = []
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.time()
        print(f"[reason api attempt] {attempt}/{max_attempts}", flush=True)
        try:
            response = client.create(**payload)
        except Exception as exc:
            last_error = exc
            elapsed = time.time() - attempt_started
            detail = f"{type(exc).__name__}: {exc}"
            errors.append(f"attempt {attempt} after {elapsed:.3f}s: {detail}")
            if attempt == max_attempts:
                break
            delay = retry_backoff_seconds * attempt
            print(
                f"[reason api retry] attempt={attempt} elapsed={elapsed:.3f}s "
                f"backoff={delay:.3f}s error={detail}",
                file=sys.stderr,
                flush=True,
            )
            if delay:
                time.sleep(delay)
        else:
            elapsed = time.time() - attempt_started
            print(
                f"[reason api response] attempt={attempt} elapsed={elapsed:.3f}s",
                flush=True,
            )
            return response, attempt

    message = f"API failed after {max_attempts} attempts; " + " | ".join(errors)
    raise APIRequestError(message) from last_error


def _api_cache_matches(
    cached: dict[str, Any],
    *,
    metadata: dict[str, Any],
    scene_id: int,
    split: int,
    instruction: str,
    localization_mode: str,
    model: str,
    points_by_id: dict[int, dict[str, Any]],
) -> bool:
    """Return whether a successful API result is safe to reuse."""
    if (
        cached.get("status") != "ok"
        or cached.get("testcase") != metadata.get("testcase")
        or cached.get("scene_id") != scene_id
        or cached.get("split") != split
        or cached.get("annotation") != instruction
        or cached.get("localization_mode") != localization_mode
        or cached.get("model") != model
        or cached.get("request_profile") != FREEGRASP_REQUEST_PROFILE
        or not isinstance(cached.get("upload_image"), dict)
        or cached["upload_image"].get("transport_format") != "PNG"
        or cached["upload_image"].get("transport_uses_source_bytes") is not True
    ):
        return False

    parsed = parse_freegrasp_response(str(cached.get("raw_response", "")), instruction)
    localization_id = parsed["selected_object_id"]
    point = points_by_id.get(localization_id) if localization_id is not None else None
    if not parsed["parsed"] or point is None or point.get("dataset_object_id") is None:
        return False
    return (
        cached.get("predicted_localization_id") == localization_id
        and cached.get("predicted_npz_label") == point.get("npz_label")
        and cached.get("predicted_object_id") == point.get("dataset_object_id")
        and cached.get("predicted_class_name") == parsed["class_name"]
        and cached.get("point_x") == point.get("x")
        and cached.get("point_y") == point.get("y")
    )


def run_reason_case(
    client: Any,
    scene_dir: Path,
    output_root: Path,
    annotation: dict[str, Any],
    *,
    localization_mode: str,
    model: str,
    iou_threshold: float,
    compute_metrics: bool,
    force: bool,
    fresh: bool,
    api_max_attempts: int,
    api_retry_backoff: float,
) -> dict[str, Any]:
    metadata = read_json(scene_dir / "metadata.json")
    scene_id = int(metadata["scene_id"])
    split = int(annotation["split"])
    result_path = (
        output_root
        / "reason"
        / localization_mode
        / metadata["testcase"]
        / f"scene_{scene_id}"
        / f"split_{split}.json"
    )
    api_result_path = result_path.parent / "api_cache" / f"split_{split}.json"
    predicted_mask_path = result_path.with_name(f"split_{split}_predicted_mask.png")
    if fresh:
        # Fresh mode is intentionally non-resumable for the selected run: no
        # previous final result, mask, or intermediate API response may leak in.
        for path in (result_path, predicted_mask_path, api_result_path):
            path.unlink(missing_ok=True)
        for path in result_path.parent.glob(
            f"split_{split}_ground_truth_mask_object_*.png"
        ):
            path.unlink(missing_ok=True)
    if result_path.exists() and not force:
        cached = read_json(result_path)
        cached_computed = (
            cached.get("ground_truth_compared") is True
            and cached.get("iou_threshold") == float(iou_threshold)
            and cached.get("ssr") is not None
            and cached.get("rsr") is not None
        )
        cached_manual = (
            cached.get("ground_truth_compared") is False
            and cached.get("metric_status") == "manual_review_required"
            and cached.get("ssr") is None
            and cached.get("rsr") is None
        )
        if (
            cached.get("model") == model
            and cached.get("request_profile") == FREEGRASP_REQUEST_PROFILE
            and (
                (compute_metrics and cached_computed)
                or (not compute_metrics and cached_manual)
            )
        ):
            print(
                f"[reason cached] mode={localization_mode} scene={scene_id} split={split}",
                flush=True,
            )
            return cached

    localization_result_path, labeled_image_path, _ = _localization_paths(
        output_root, localization_mode, scene_id
    )
    if not localization_result_path.exists() or not labeled_image_path.exists():
        raise LocalizationInfrastructureError(
            f"Missing {localization_mode} localization output for scene {scene_id}"
        )
    localization = read_json(localization_result_path)
    points_by_id = {
        int(item["localization_id"]): item for item in localization["points"]
    }
    instruction = str(annotation["annotation"])

    print(
        f"[reason start] mode={localization_mode} scene={scene_id} split={split}",
        flush=True,
    )
    started = time.time()
    api_cache_reused = False
    cached_api = None
    if api_result_path.exists() and not force and not fresh:
        try:
            candidate = read_json(api_result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            candidate = None
        if isinstance(candidate, dict) and _api_cache_matches(
            candidate,
            metadata=metadata,
            scene_id=scene_id,
            split=split,
            instruction=instruction,
            localization_mode=localization_mode,
            model=model,
            points_by_id=points_by_id,
        ):
            cached_api = candidate

    if cached_api is not None:
        api_cache_reused = True
        raw_response = str(cached_api["raw_response"])
        parsed = parse_freegrasp_response(raw_response, instruction)
        localization_id = parsed["selected_object_id"]
        point = points_by_id[localization_id]
        status = "ok"
        upload_image = cached_api["upload_image"]
        usage = cached_api.get("usage")
        api_attempts = int(cached_api.get("api_attempts") or 0)
        api_transport = cached_api.get("api_transport", client.transport_name)
        api_base_url = cached_api.get("api_base_url", client.base_url)
        api_response_model = cached_api.get("api_response_model")
        print(
            f"[reason api cached] mode={localization_mode} scene={scene_id} "
            f"split={split} localization_id={localization_id}",
            flush=True,
        )
    else:
        base64_image, upload_image = _encode_original_png(labeled_image_path)
        print(
            f"[reason upload] png={upload_image['encoded_image_bytes']} bytes "
            f"resolution={upload_image['size'][0]}x{upload_image['size'][1]}",
            flush=True,
        )
        response, api_attempts = _request_chat_completion(
            client,
            _freegrasp_chat_payload(model, instruction, base64_image),
            max_attempts=api_max_attempts,
            retry_backoff_seconds=api_retry_backoff,
        )
        raw_response = response["choices"][0]["message"].get("content") or ""
        parsed = parse_freegrasp_response(raw_response, instruction)
        localization_id = parsed["selected_object_id"]
        point = points_by_id.get(localization_id) if localization_id is not None else None
        if not parsed["parsed"]:
            status = "unparsed_response"
        elif point is None:
            status = "unknown_localization_id"
        elif point["dataset_object_id"] is None:
            status = "unmapped_background"
        else:
            status = "ok"
        usage = response.get("usage")
        api_response_model = response.get("model")
        api_transport = client.transport_name
        api_base_url = client.base_url

        if not fresh:
            # Normal runs preserve API output for resuming. Fresh runs
            # deliberately do not create this intermediate cache.
            write_json(api_result_path, {
                "schema_version": 1,
                "testcase": metadata["testcase"],
                "scene_id": scene_id,
                "split": split,
                "annotation": instruction,
                "model": model,
                "request_profile": FREEGRASP_REQUEST_PROFILE,
                "api_transport": api_transport,
                "api_base_url": api_base_url,
                "api_response_model": api_response_model,
                "localization_mode": localization_mode,
                "upload_image": upload_image,
                "raw_response": raw_response,
                "predicted_localization_id": localization_id,
                "predicted_npz_label": point["npz_label"] if point else None,
                "predicted_object_id": point["dataset_object_id"] if point else None,
                "predicted_class_name": parsed["class_name"],
                "point_x": point["x"] if point else None,
                "point_y": point["y"] if point else None,
                "status": status,
                "usage": usage,
                "api_attempts": api_attempts,
                "metrics": None,
                "excluded_from_statistics": True,
            })

    segmentation_details = None
    segmentation_error = None
    predicted_mask = None
    # A syntactically valid response that does not select a usable numbered
    # object is an algorithm failure, not an infrastructure failure.
    if status == "ok":
        try:
            predicted_mask, segmentation_details = predict_object_mask(
                str(scene_dir / "image.png"),
                parsed["class_name"],
                (int(point["x"]), int(point["y"])),
            )
            predicted_mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(predicted_mask.astype(np.uint8) * 255).save(predicted_mask_path)
        except SegmentationInfrastructureError:
            # Missing packages/weights are evaluation infrastructure failures,
            # not evidence that the model selected the wrong object.
            raise
        except Exception as exc:
            segmentation_error = repr(exc)
            status = "segmentation_failure"

    ground_truth_root = (
        scene_dir.parents[2]
        / "ground_truth_masks"
        / metadata["testcase"]
        / f"scene_{scene_id}"
        / "gt"
    )
    ground_truth_mask_manifest = ground_truth_root / "summary.json"
    ground_truth_masks = []
    if ground_truth_mask_manifest.exists():
        ground_truth_masks = read_json(ground_truth_mask_manifest).get("masks", [])

    has_selected_object_and_mask = (
        status == "ok"
        and localization_id is not None
        and point is not None
        and point.get("dataset_object_id") is not None
        and predicted_mask is not None
    )
    if compute_metrics and has_selected_object_and_mask:
        metrics = score_prediction_against_gt_files(
            predicted_mask,
            metadata.get("ground_truth_object_ids"),
            ground_truth_root / "mask",
            threshold=iou_threshold,
        )
        metric_status = "ok"
        ground_truth_compared = True
        excluded_from_statistics = False
    elif compute_metrics:
        metrics = {
            "iou": None,
            "ssr": None,
            "rsr": None,
            "iou_threshold": float(iou_threshold),
            "threshold_operator": ">",
            "best_ground_truth_object_id": None,
            "per_ground_truth_iou": {},
            "compared_ground_truth_masks": [],
            "metric_definition": (
                "missing selected object ID or corresponding predicted mask; excluded"
            ),
        }
        metric_status = "missing_selected_object_or_mask_excluded"
        ground_truth_compared = False
        excluded_from_statistics = True
    else:
        metrics = {
            "iou": None,
            "ssr": None,
            "rsr": None,
            "iou_threshold": None,
            "threshold_operator": None,
            "best_ground_truth_object_id": None,
            "per_ground_truth_iou": {},
            "metric_definition": "manual mask comparison; not computed",
        }
        metric_status = "manual_review_required"
        ground_truth_compared = False
        excluded_from_statistics = True
    payload = {
        "schema_version": 1,
        "testcase": metadata["testcase"],
        "scene_id": scene_id,
        "query_obj_id": int(metadata["query_obj_id"]),
        "difficulty": metadata["difficulty"],
        "ambiguous": bool(metadata["ambiguous"]),
        "split": split,
        "annotation": instruction,
        "model": model,
        "request_profile": FREEGRASP_REQUEST_PROFILE,
        "api_transport": api_transport,
        "api_base_url": api_base_url,
        "api_response_model": api_response_model,
        "api_attempts": api_attempts,
        "api_cache_reused": api_cache_reused,
        "fresh_run": fresh,
        "localization_mode": localization_mode,
        "status": status,
        "raw_response": raw_response,
        "predicted_localization_id": localization_id,
        "predicted_molmo_id": localization_id if localization_mode == "molmo" else None,
        "predicted_npz_label": point["npz_label"] if point else None,
        "predicted_object_id": point["dataset_object_id"] if point else None,
        "predicted_class_name": parsed["class_name"],
        "point_x": point["x"] if point else None,
        "point_y": point["y"] if point else None,
        "predicted_mask": str(predicted_mask_path.resolve()) if predicted_mask is not None else None,
        "ground_truth_object_ids": metadata.get("ground_truth_object_ids"),
        "ground_truth_mask_manifest": (
            str(ground_truth_mask_manifest.resolve())
            if ground_truth_mask_manifest.exists() else None
        ),
        "ground_truth_masks": ground_truth_masks,
        "segmentation": segmentation_details,
        "segmentation_error": segmentation_error,
        "localization_result": str(localization_result_path.resolve()),
        "upload_image": upload_image,
        "api_result": str(api_result_path.resolve()) if api_result_path.exists() else None,
        "elapsed_seconds": round(time.time() - started, 3),
        "usage": usage,
        **metrics,
        "metric_status": metric_status,
        "rsr_success": metrics["rsr"],
        "ground_truth_compared": ground_truth_compared,
        "excluded_from_statistics": excluded_from_statistics,
    }
    write_json(result_path, payload)
    metric_text = (
        f"ssr={payload['ssr']:.6f} rsr={payload['rsr']}"
        if payload["ssr"] is not None else f"metrics={metric_status}"
    )
    print(
        f"[reason done] mode={localization_mode} scene={scene_id} split={split} "
        f"localization_id={localization_id} "
        f"predicted_object_id={payload['predicted_object_id']} status={status} "
        f"{metric_text}",
        flush=True,
    )
    return payload


def write_reports(
    output_root: Path,
    localization_modes: set[str] | None = None,
) -> dict[str, Any]:
    records = [
        read_json(path)
        for path in sorted((output_root / "reason").glob("*/*/scene_*/split_*.json"))
    ]
    if localization_modes is not None:
        records = [
            item for item in records
            if item.get("localization_mode") in localization_modes
        ]
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "predicted_object_ids.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["result_json"] = str(
                output_root
                / "reason"
                / record["localization_mode"]
                / record["testcase"]
                / f"scene_{record['scene_id']}"
                / f"split_{record['split']}.json"
            )
            writer.writerow(row)
    jsonl_path = report_root / "predicted_object_ids.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    evaluated = [
        item for item in records
        if item.get("ssr") is not None and item.get("rsr") is not None
    ]
    failures_path = output_root / "run_failures.json"
    failures = read_json(failures_path).get("failures", []) if failures_path.exists() else []
    excluded_reason_failures = [
        item for item in failures
        if item.get("stage") == "reason" and item.get("excluded_from_statistics") is True
    ]
    excluded_api_failures = [
        item for item in excluded_reason_failures
        if item.get("failure_type") == "api_or_transport_failure"
    ]
    excluded_infrastructure_failures = [
        item for item in excluded_reason_failures
        if item.get("failure_type") in {
            "localization_infrastructure_failure",
            "segmentation_infrastructure_failure",
            "ground_truth_mask_infrastructure_failure",
        }
    ]

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "mean_ssr": None, "mean_rsr": None}
        return {
            "count": len(items),
            "mean_ssr": float(sum(float(item["ssr"]) for item in items) / len(items)),
            "mean_rsr": float(sum(float(item["rsr"]) for item in items) / len(items)),
        }

    by_localization_mode = {
        mode: aggregate([item for item in evaluated if item["localization_mode"] == mode])
        for mode in LOCALIZATION_MODES
    }
    summary = {
        "num_predictions": len(records),
        "num_evaluated": len(evaluated),
        "num_api_failures_excluded_current_run": len(excluded_api_failures),
        "num_infrastructure_failures_excluded_current_run": len(excluded_infrastructure_failures),
        "overall": aggregate(evaluated),
        "by_localization_mode": by_localization_mode,
        "status_counts": dict(sorted(Counter(item["status"] for item in records).items())),
        "localization_mode_counts": dict(
            sorted(Counter(item["localization_mode"] for item in records).items())
        ),
        "predicted_object_ids_csv": str(csv_path.resolve()),
        "predicted_object_ids_jsonl": str(jsonl_path.resolve()),
        "metric_definition": (
            "SSR=max GT IoU; RSR=1 if SSR>0.5 else 0"
            if evaluated else "manual mask comparison; SSR/RSR not computed"
        ),
        "threshold_operator": ">" if evaluated else None,
        "api_failure_policy": "timeout/connection/API failure => SSR=null, RSR=null, excluded",
        "infrastructure_failure_policy": "missing localization/segmentation/GT mask infrastructure => SSR=null, RSR=null, excluded",
        "selection_or_mask_failure_policy": "missing selected object ID or corresponding predicted mask => SSR=null, RSR=null, excluded",
        "rsr_is_computed": bool(evaluated),
        "ground_truth_compared": any(
            item.get("ground_truth_compared") is True for item in records
        ),
    }
    write_json(report_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FreeGrasp and save masks for manual review by default."
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--testcase", action="append", default=None)
    parser.add_argument("--scene-id", type=int, action="append", default=None)
    parser.add_argument("--split", type=int, action="append", choices=[0, 1, 2], default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--api-transport",
        choices=("auto", "openai", "curl"),
        default="auto",
        help="auto matches SmartGrasp and uses the OpenAI SDK with the fixed relay.",
    )
    parser.add_argument("--api-timeout", type=float, default=420.0)
    parser.add_argument(
        "--api-max-attempts",
        type=int,
        default=3,
        help="Maximum transport attempts for one GPT-4o evaluation (default: 3).",
    )
    parser.add_argument(
        "--api-retry-backoff",
        type=float,
        default=5.0,
        help="Base retry backoff in seconds, multiplied by the failed attempt number.",
    )
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument(
        "--manual-mask-review",
        action="store_true",
        help="Disable automatic IoU/SSR/RSR and save masks for manual review only.",
    )
    parser.add_argument(
        "--localization-mode",
        action="append",
        choices=LOCALIZATION_MODES,
        default=None,
        help="Repeatable; defaults to both gt and molmo.",
    )
    parser.add_argument("--localization-only", action="store_true")
    parser.add_argument("--molmo-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reason-only", action="store_true")
    parser.add_argument("--force-localization", action="store_true")
    parser.add_argument("--force-molmo", action="store_true")
    parser.add_argument("--force-reason", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Recompute localization, GPT-4o reasoning, and segmentation "
            "from input without reading or writing intermediate API cache."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if (args.localization_only or args.molmo_only) and args.reason_only:
        parser.error("--localization-only and --reason-only are mutually exclusive")
    if args.api_max_attempts < 1:
        parser.error("--api-max-attempts must be at least 1")
    if args.api_retry_backoff < 0:
        parser.error("--api-retry-backoff cannot be negative")

    os.chdir(PROJECT_ROOT)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    scene_dirs = _scene_dirs(input_root, args)
    selected_splits = set(args.split or [0, 1, 2])
    localization_modes = list(args.localization_mode or LOCALIZATION_MODES)
    if args.molmo_only:
        localization_modes = ["molmo"]
        args.localization_only = True
    failures = []

    if not args.reason_only:
        for localization_mode in localization_modes:
            for scene_dir in scene_dirs:
                try:
                    run_localization_scene(
                        localization_mode,
                        scene_dir,
                        output_root,
                        force=args.force_localization
                        or args.fresh
                        or (localization_mode == "molmo" and args.force_molmo),
                    )
                except Exception as exc:
                    failure = {
                        "stage": "localization",
                        "localization_mode": localization_mode,
                        "scene_dir": str(scene_dir),
                        "error": repr(exc),
                        "failure_type": "localization_infrastructure_failure",
                        "excluded_from_statistics": True,
                    }
                    failures.append(failure)
                    print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
                    if args.fail_fast:
                        raise

    if not args.localization_only:
        client = _chat_client(args.api_transport, args.api_timeout)
        for localization_mode in localization_modes:
            for scene_dir in scene_dirs:
                metadata = read_json(scene_dir / "metadata.json")
                for annotation in metadata["annotations"]:
                    if int(annotation["split"]) not in selected_splits:
                        continue
                    try:
                        run_reason_case(
                            client,
                            scene_dir,
                            output_root,
                            annotation,
                            localization_mode=localization_mode,
                            model=args.model,
                            iou_threshold=args.iou_threshold,
                            compute_metrics=not args.manual_mask_review,
                            force=args.force_reason or args.fresh,
                            fresh=args.fresh,
                            api_max_attempts=args.api_max_attempts,
                            api_retry_backoff=args.api_retry_backoff,
                        )
                    except Exception as exc:
                        is_api_failure = isinstance(exc, APIRequestError)
                        is_segmentation_infrastructure_failure = isinstance(
                            exc, SegmentationInfrastructureError
                        )
                        is_localization_infrastructure_failure = isinstance(
                            exc, LocalizationInfrastructureError
                        )
                        is_gt_mask_infrastructure_failure = isinstance(
                            exc, GroundTruthMaskError
                        )
                        is_excluded = (
                            is_api_failure
                            or is_segmentation_infrastructure_failure
                            or is_localization_infrastructure_failure
                            or is_gt_mask_infrastructure_failure
                        )
                        if is_api_failure:
                            failure_type = "api_or_transport_failure"
                        elif is_localization_infrastructure_failure:
                            failure_type = "localization_infrastructure_failure"
                        elif is_segmentation_infrastructure_failure:
                            failure_type = "segmentation_infrastructure_failure"
                        elif is_gt_mask_infrastructure_failure:
                            failure_type = "ground_truth_mask_infrastructure_failure"
                        else:
                            failure_type = "pipeline_failure"
                        failure = {
                            "stage": "reason",
                            "localization_mode": localization_mode,
                            "scene_id": metadata["scene_id"],
                            "split": annotation["split"],
                            "error": repr(exc),
                            "failure_type": failure_type,
                            "excluded_from_statistics": is_excluded,
                            "ssr": None if is_excluded else 0.0,
                            "rsr": None if is_excluded else 0,
                        }
                        failures.append(failure)
                        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
                        if args.fail_fast:
                            raise

    write_json(output_root / "run_failures.json", {"failures": failures})
    print(json.dumps(
        write_reports(output_root, localization_modes=set(localization_modes)),
        ensure_ascii=False,
        indent=2,
    ))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
