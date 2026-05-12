"""
Packy GPT-Image-2 Images API engine

Calls /v1/images/generations for text-to-image and /v1/images/edits
for image-to-image using gpt-image-2.
Docs: https://docs.packyapi.com/docs/paint/GPTImage.html
"""
import asyncio
import base64
import uuid
from typing import Any

import httpx

from zhenxun.services.log import logger

from ..config import base_config
from . import DrawEngine

MAX_RETRIES = 3
RETRY_DELAY_BASE = 5.0
REQUEST_TIMEOUT = 1200  # 20 minutes
PACKY_VALID_QUALITIES = {"auto", "low", "medium", "high"}
PACKY_VALID_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
PACKY_VALID_RESPONSE_FORMATS = {"b64_json", "url"}
PACKY_VALID_BACKGROUNDS = {"auto", "opaque"}
PACKY_VALID_MODERATIONS = {"auto", "low"}
PACKY_VALID_INPUT_FIDELITIES = {"high"}
PACKY_EDIT_IMAGE_FIELD_NAME = "image"
PACKY_DIMENSION_MULTIPLE = 16
PACKY_MIN_N = 1
PACKY_MAX_N = 1
PACKY_MAX_OUTPUT_COMPRESSION = 100
PACKY_MIN_PIXELS = 655360
PACKY_MAX_SIDE = 3840
PACKY_MAX_ASPECT_RATIO = 3
PACKY_MAX_PIXELS = 8294400


def _detect_image_mime(image_data: bytes) -> str:
    """Best-effort MIME detection for images received from chat adapters."""
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    if image_data.startswith(b"BM"):
        return "image/bmp"
    return "image/png"


def _mime_to_ext(mime_type: str) -> str:
    return {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
    }.get(mime_type, "png")


def _is_retryable_status(status_code: int) -> bool:
    return status_code in (429, 500, 502, 503, 504, 524)


def _build_packy_edit_multipart(
    payload: dict[str, Any], images_data: list[bytes]
) -> tuple[bytes, str]:
    boundary = f"packy-edit-{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in payload.items():
        if value is None:
            continue
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for index, image_data in enumerate(images_data, start=1):
        mime_type = _detect_image_mime(image_data)
        ext = _mime_to_ext(mime_type)
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            (
                "Content-Disposition: form-data; "
                f'name="{PACKY_EDIT_IMAGE_FIELD_NAME}"; '
                f'filename="input-{index}.{ext}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("ascii"))
        body.extend(image_data)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body), boundary


def _parse_choice(
    value: Any,
    valid_values: set[str],
    param_name: str,
    default: str | None = None,
) -> str | None:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized not in valid_values:
        allowed = ", ".join(sorted(valid_values))
        raise ValueError(f"Packy 参数 {param_name} 仅支持: {allowed}")
    return normalized


def _parse_int_range(
    value: Any,
    param_name: str,
    min_value: int,
    max_value: int,
    default: int | None = None,
) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Packy 参数 {param_name} 必须是整数") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"Packy 参数 {param_name} 必须在 {min_value} 到 {max_value} 之间")
    return parsed


def _get_packy_option(
    api_options: dict[str, Any], option_key: str, config_key: str, default: Any = None
) -> Any:
    if option_key in api_options:
        return api_options[option_key]
    return base_config.get(config_key, default)


def normalize_packy_api_options(api_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize Packy Images API options from config and command overrides."""
    options = dict(api_options or {})

    resolved: dict[str, Any] = {
        "quality": _parse_choice(
            _get_packy_option(options, "quality", "packy_image_quality", "high"),
            PACKY_VALID_QUALITIES,
            "quality",
            "high",
        ),
        "output_format": _parse_choice(
            _get_packy_option(options, "output_format", "packy_output_format", "png"),
            PACKY_VALID_OUTPUT_FORMATS,
            "output_format",
            "png",
        ),
        "response_format": _parse_choice(
            _get_packy_option(options, "response_format", "packy_response_format", "url"),
            PACKY_VALID_RESPONSE_FORMATS,
            "response_format",
            "url",
        ),
        "n": _parse_int_range(
            _get_packy_option(options, "n", "packy_n", 1),
            "n",
            PACKY_MIN_N,
            PACKY_MAX_N,
            1,
        ),
    }

    background = _parse_choice(
        _get_packy_option(options, "background", "packy_background", ""),
        PACKY_VALID_BACKGROUNDS,
        "background",
    )
    if background:
        resolved["background"] = background

    moderation = _parse_choice(
        _get_packy_option(options, "moderation", "packy_moderation", ""),
        PACKY_VALID_MODERATIONS,
        "moderation",
    )
    if moderation:
        resolved["moderation"] = moderation

    output_compression = _parse_int_range(
        _get_packy_option(options, "output_compression", "packy_output_compression", None),
        "output_compression",
        0,
        PACKY_MAX_OUTPUT_COMPRESSION,
    )
    if output_compression is not None:
        resolved["output_compression"] = output_compression

    input_fidelity = _parse_choice(
        _get_packy_option(options, "input_fidelity", "packy_input_fidelity", ""),
        PACKY_VALID_INPUT_FIDELITIES,
        "input_fidelity",
    )
    if input_fidelity:
        resolved["input_fidelity"] = input_fidelity

    user = str(_get_packy_option(options, "user", "packy_user", "")).strip()
    if user:
        resolved["user"] = user

    return resolved


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _packy_size_error(size: str) -> str | None:
    """Return a concrete Packy size validation error, or None when valid."""
    if size == "auto":
        return None

    generic_rule = (
        "Packy 分辨率需为 auto 或 WxH 格式，例如 2560:1440。"
        "宽高必须为 16 的倍数，单边不超过 3840，宽高比不超过 3:1，"
        "总像素需在 655360 到 8294400 之间。"
    )

    parts = size.lower().split("x")
    if len(parts) != 2:
        return generic_rule

    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return generic_rule

    if w <= 0 or h <= 0:
        return f"Packy 分辨率宽高必须为正整数。当前为 {w}x{h}。{generic_rule}"

    if w > PACKY_MAX_SIDE or h > PACKY_MAX_SIDE:
        return (
            f"Packy 分辨率单边不能超过 {PACKY_MAX_SIDE}。"
            f"当前为 {w}x{h}。"
        )

    if w % PACKY_DIMENSION_MULTIPLE != 0 or h % PACKY_DIMENSION_MULTIPLE != 0:
        return (
            f"Packy 分辨率宽高必须都是 {PACKY_DIMENSION_MULTIPLE} 的倍数。"
            f"当前为 {w}x{h}。"
        )

    long_side = max(w, h)
    short_side = min(w, h)
    if long_side / short_side > PACKY_MAX_ASPECT_RATIO:
        max_allowed_long_side = short_side * PACKY_MAX_ASPECT_RATIO
        return (
            f"Packy 分辨率宽高比不能超过 {PACKY_MAX_ASPECT_RATIO}:1。"
            f"当前为 {w}x{h}，较短边 {short_side} 时另一边最大为 "
            f"{max_allowed_long_side}。"
        )

    total_pixels = w * h
    if total_pixels < PACKY_MIN_PIXELS:
        required_other_side = _ceil_div(PACKY_MIN_PIXELS, short_side)
        required_valid_other_side = _ceil_to_multiple(
            required_other_side, PACKY_DIMENSION_MULTIPLE
        )
        short_side_name = "宽" if w <= h else "高"
        other_side_name = "高" if w <= h else "宽"
        detail_hints: list[str] = []
        if required_valid_other_side != required_other_side:
            detail_hints.append(
                f"由于边长还需为 {PACKY_DIMENSION_MULTIPLE} 的倍数，"
                f"实际建议至少 {required_valid_other_side}"
            )
        if required_valid_other_side > PACKY_MAX_SIDE:
            detail_hints.append(
                f"该值超过单边上限 {PACKY_MAX_SIDE}，请增大较短边"
            )
        if required_valid_other_side / short_side > PACKY_MAX_ASPECT_RATIO:
            detail_hints.append(
                f"该值会超过宽高比 {PACKY_MAX_ASPECT_RATIO}:1，请增大较短边"
            )
        detail = f"；{'；'.join(detail_hints)}" if detail_hints else ""
        return (
            f"Packy 分辨率总像素不能低于 {PACKY_MIN_PIXELS}。"
            f"当前为 {w}x{h}，总像素 {total_pixels}。"
            f"按较短边{short_side_name} {short_side} 计算，"
            f"另一个边{other_side_name}至少需要 {required_other_side}"
            f"{detail}。"
        )

    if total_pixels > PACKY_MAX_PIXELS:
        max_other_side = PACKY_MAX_PIXELS // short_side
        short_side_name = "宽" if w <= h else "高"
        other_side_name = "高" if w <= h else "宽"
        return (
            f"Packy 分辨率总像素不能超过 {PACKY_MAX_PIXELS}。"
            f"当前为 {w}x{h}，总像素 {total_pixels}。"
            f"按较短边{short_side_name} {short_side} 计算，"
            f"另一个边{other_side_name}最多为 {max_other_side}。"
        )

    return None


def _validate_packy_size(size: str) -> bool:
    """按分辨率规则计算校验尺寸：auto 或 WxH 格式。"""
    return _packy_size_error(size) is None


def normalize_packy_size(size: str) -> str:
    """Normalize CLI/config size values to Packy's WxH format."""
    normalized = str(size).strip().lower().replace(":", "x")
    normalized = normalized.replace("*", "x").replace("×", "x")
    if error := _packy_size_error(normalized):
        raise ValueError(error)
    return normalized


class PackyImageEngine(DrawEngine):
    """
    Packy GPT-Image-2 engine using the Images API.

    Returns generated image bytes, downloading URL responses when needed.
    """

    async def draw(
        self,
        prompt: str,
        image_bytes: list[bytes] | None = None,
        size_override: str | None = None,
        api_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        api_key = base_config.get("packy_api_key", "")
        if not api_key:
            raise ValueError("未配置 packy_api_key (Packy sora 分组令牌)")

        base_url = base_config.get("packy_api_base_url", "https://www.packyapi.com")
        base_url = base_url.rstrip("/")

        size = size_override or base_config.get("packy_image_size", "auto")
        try:
            size = normalize_packy_size(size)
        except ValueError:
            if size_override:
                raise
            logger.warning(
                f"[Packy] 不支持的尺寸 '{size}'，回退到 auto"
            )
            size = "auto"

        packy_options = normalize_packy_api_options(api_options)

        has_input_images = bool(image_bytes)
        if not has_input_images:
            packy_options.pop("input_fidelity", None)
        if packy_options.get("output_format") == "webp":
            logger.warning("[Packy] 官方文档不建议使用 webp，推荐 png 或 jpeg")
        endpoint = "/v1/images/edits" if has_input_images else "/v1/images/generations"
        url = f"{base_url}{endpoint}"

        payload = {
            "model": "gpt-image-2",
            "prompt": prompt,
            "size": size,
            **packy_options,
        }

        input_images: list[bytes] = []
        if has_input_images:
            input_images = [img for img in image_bytes or [] if img]
            if not input_images:
                raise ValueError("Packy 图生图模式未收到有效图片数据")
            if len(input_images) > 1:
                logger.warning(
                    f"[Packy] 官方建议 edits 一次只上传 1 张图片，"
                    f"当前将测试上传 {len(input_images)} 张图片"
                )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }

        logger.info(
            f"[Packy] 请求 GPT-Image-2: endpoint={endpoint}, size={size}, "
            f"input_images={len(image_bytes) if image_bytes else 0}, "
            f"prompt={prompt[:80]}{'...' if len(prompt) > 80 else ''}"
        )

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=REQUEST_TIMEOUT, proxy=None
                ) as client:
                    if has_input_images:
                        if not input_images:
                            raise ValueError("Packy 图生图模式未收到有效图片数据")
                        multipart_body, boundary = _build_packy_edit_multipart(
                            payload, input_images
                        )
                        resp = await client.post(
                            url,
                            headers={
                                **headers,
                                "Content-Type": f"multipart/form-data; boundary={boundary}",
                                "Content-Length": str(len(multipart_body)),
                            },
                            content=multipart_body,
                        )
                    else:
                        resp = await client.post(
                            url,
                            headers={**headers, "Content-Type": "application/json"},
                            json=payload,
                        )

                if resp.status_code != 200:
                    error_text = resp.text
                    if _is_retryable_status(resp.status_code) and attempt < MAX_RETRIES:
                        delay = RETRY_DELAY_BASE * (2 ** attempt)
                        logger.warning(
                            f"[Packy] HTTP {resp.status_code}，{delay:.0f}s 后重试 "
                            f"({attempt + 1}/{MAX_RETRIES + 1})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise Exception(
                        f"Packy API 返回 {resp.status_code}: {error_text[:300]}"
                    )

                data = resp.json()
                images: list[bytes] = []
                image_urls: list[str] = []

                for item in data.get("data", []):
                    if item.get("b64_json"):
                        images.append(base64.b64decode(item["b64_json"]))
                    if item.get("url"):
                        image_urls.append(item["url"])

                if not images and not image_urls:
                    raise Exception("Packy API 未返回任何图片数据")

                # Download URL results into bytes when Packy returns URL responses.
                if image_urls:
                    async with httpx.AsyncClient(
                        timeout=60, proxy=None
                    ) as dl_client:
                        for i, img_url in enumerate(image_urls):
                            dl_resp = await dl_client.get(img_url)
                            if dl_resp.status_code == 200:
                                images.append(dl_resp.content)
                                logger.info(
                                    f"[Packy] 图片 {i + 1} 下载成功: "
                                    f"{len(dl_resp.content) / 1024:.1f} KB"
                                )
                            else:
                                logger.warning(
                                    f"[Packy] 图片 {i + 1} 下载失败: "
                                    f"HTTP {dl_resp.status_code}"
                                )

                if not images:
                    raise Exception("Packy 图片全部获取失败")

                return {"images": images, "text": ""}

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str
                        for kw in ("timeout", "connection", "network")
                    ):
                        delay = RETRY_DELAY_BASE * (2 ** attempt)
                        logger.warning(
                            f"[Packy] 网络错误，{delay:.0f}s 后重试: {e}"
                        )
                        await asyncio.sleep(delay)
                        continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("[Packy] 重试逻辑异常")
