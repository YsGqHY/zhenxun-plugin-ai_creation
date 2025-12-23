"""
OpenAI 兼容 API / Gemini REST API 图片生成引擎

支持两种模式：
1. OpenAI SDK 兼容模式 - 用于 OpenAI 兼容的 API 服务
2. Gemini REST 模式 - 直接调用 Gemini REST API，支持中转服务和 1K/2K/4K 分辨率
"""
import asyncio
import base64
import re
from io import BytesIO
from typing import Any, Callable, TypeVar

import httpx
from openai import AsyncOpenAI

from zhenxun.services.log import logger

from ..config import base_config
from . import DrawEngine

# 重试配置
MAX_RETRIES = 3              # 最大重试次数
RETRY_DELAY_BASE = 5.0       # 重试基础延迟（秒）
GEMINI_TIMEOUT = 1800        # Gemini 超时时间（30分钟 = 1800秒）

T = TypeVar("T")


def _is_retryable_error(error: Exception) -> bool:
    """判断错误是否可重试"""
    error_str = str(error).lower()

    # 可重试的错误类型
    retryable_patterns = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily",
        "overloaded",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "resource exhausted",
        "deadline exceeded",
    ]

    for pattern in retryable_patterns:
        if pattern in error_str:
            return True

    # 检查特定异常类型
    error_type = type(error).__name__.lower()
    if any(t in error_type for t in ["timeout", "connection", "network"]):
        return True

    return False


def _is_permanent_error(error: Exception) -> bool:
    """判断是否为永久性错误（不应重试）"""
    error_str = str(error).lower()

    # 永久性错误
    permanent_patterns = [
        "401",
        "403",
        "404",
        "invalid api key",
        "unauthorized",
        "forbidden",
        "not found",
        "model not found",
        "invalid model",
        "permission denied",
        "authentication",
    ]

    for pattern in permanent_patterns:
        if pattern in error_str:
            return True

    return False


async def _retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_DELAY_BASE,
    tag: str = "API",
) -> T:
    """
    带指数退避的重试逻辑

    Args:
        func: 要执行的异步函数（无参数）
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        tag: 日志标签

    Returns:
        函数执行结果

    Raises:
        最后一次尝试的异常
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_error = e

            # 永久性错误不重试
            if _is_permanent_error(e):
                logger.error(f"[{tag}] 遇到永久性错误，不再重试: {e}")
                raise

            # 已达到最大重试次数
            if attempt >= max_retries:
                logger.error(f"[{tag}] 已达最大重试次数 ({max_retries})，放弃重试")
                raise

            # 判断是否可重试
            if not _is_retryable_error(e):
                logger.warning(f"[{tag}] 错误不可重试: {e}")
                raise

            # 计算延迟时间（指数退避）
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"[{tag}] 第 {attempt + 1}/{max_retries + 1} 次尝试失败: {e}，"
                f"{delay:.1f}秒后重试..."
            )
            await asyncio.sleep(delay)

    # 理论上不会到达这里，但为了类型安全
    if last_error:
        raise last_error
    raise RuntimeError(f"[{tag}] 重试逻辑异常")


def _get_image_info(image_data: bytes) -> dict[str, Any]:
    """获取图片的基本信息（大小、格式、分辨率）"""
    info: dict[str, Any] = {
        "size_bytes": len(image_data),
        "size_kb": round(len(image_data) / 1024, 2),
        "size_mb": round(len(image_data) / 1024 / 1024, 2),
    }

    try:
        from PIL import Image

        img = Image.open(BytesIO(image_data))
        info["format"] = img.format
        info["width"] = img.width
        info["height"] = img.height
        info["mode"] = img.mode
        info["resolution"] = f"{img.width}x{img.height}"
    except Exception as e:
        info["format_error"] = str(e)

    return info


def _extract_image_from_content(content: str) -> bytes | None:
    """从响应内容中提取图片数据"""
    # 尝试提取 Base64 图片
    # 格式: data:image/png;base64,... 或 ![image](data:image/png;base64,...)
    match = re.search(r"data:image/(\w+);base64,([a-zA-Z0-9+/=]+)", content)

    if match:
        try:
            img_b64 = match.group(2)
            return base64.b64decode(img_b64)
        except Exception as e:
            logger.warning(f"[DirectAPI] Base64 解码失败: {e}")
            return None

    return None


class LlmImageApiEngine(DrawEngine):
    """
    使用 OpenAI 兼容 API 或 Gemini REST API 生成图片的引擎

    支持两种模式（根据模型名自动判断）：
    1. OpenAI SDK 兼容模式 - 通过 Chat Completions API 生成图片
    2. Gemini REST 模式 - 直接调用 Gemini REST API，支持中转服务和 1K/2K/4K 分辨率

    配置项：
    - api_image_size: 图片尺寸，如 "1K", "2K", "4K"（仅 Gemini 模式有效）
    - api_draw_aspect_ratio: 宽高比，如 "1:1", "16:9"
    - api_draw_model: 模型名称（包含 "gemini" 则自动使用 Gemini REST API）
    - api_base_url: API 基础 URL（Gemini 模式时作为中转服务地址）
    """

    # 宽高比到模型后缀的映射（用于 OpenAI 兼容模式）
    ASPECT_RATIO_SUFFIX_MAP = {
        "1:1": "1-1",
        "16:9": "16-9",
        "9:16": "9-16",
        "3:4": "3-4",
        "4:3": "4-3",
    }

    # Gemini 支持的图片尺寸
    VALID_IMAGE_SIZES = {"1K", "2K", "4K"}

    # Gemini 支持的宽高比
    VALID_ASPECT_RATIOS = {
        "1:1", "2:3", "3:2", "3:4", "4:3",
        "4:5", "5:4", "9:16", "16:9", "21:9"
    }

    # Gemini 官方 API 地址
    GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """获取或创建 AsyncOpenAI 客户端"""
        if self._client is None:
            base_url = base_config.get("api_base_url", "")
            api_key = base_config.get("api_key", "")

            if not base_url:
                raise ValueError("未配置 api_base_url")
            if not api_key:
                raise ValueError("未配置 api_key")

            # 确保 URL 以 /v1 结尾（OpenAI SDK 要求）
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"

            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            logger.debug(f"[DirectAPI] 客户端已创建, base_url={base_url}")

        return self._client

    def _resolve_model_name(self) -> str:
        """
        解析最终使用的模型名称

        如果配置了宽高比且模型支持，会自动替换模型名称中的比例后缀
        """
        model_name = base_config.get("api_draw_model", "")
        if not model_name:
            raise ValueError("未配置 api_draw_model")

        aspect_ratio = base_config.get("api_draw_aspect_ratio", "")

        # 如果配置了宽高比，尝试替换模型名中的比例部分
        if aspect_ratio and aspect_ratio in self.ASPECT_RATIO_SUFFIX_MAP:
            target_suffix = self.ASPECT_RATIO_SUFFIX_MAP[aspect_ratio]

            # 检查模型名是否包含比例标识（如 1-1, 16-9 等）
            for ratio_key, ratio_suffix in self.ASPECT_RATIO_SUFFIX_MAP.items():
                if ratio_suffix in model_name:
                    # 替换为目标比例
                    new_model = model_name.replace(ratio_suffix, target_suffix)
                    if new_model != model_name:
                        logger.info(
                            f"[DirectAPI] 根据宽高比 {aspect_ratio} "
                            f"调整模型: {model_name} -> {new_model}"
                        )
                        return new_model
                    break

        return model_name

    @staticmethod
    def _is_gemini_model(model_name: str) -> bool:
        """判断是否为 Gemini 模型"""
        return "gemini" in model_name.lower()

    def _log_gemini_request(
        self,
        api_url: str,
        model_name: str,
        image_config: dict[str, str],
        prompt: str,
        image_count: int,
        request_body: dict[str, Any],
    ) -> None:
        """
        打印 Gemini REST API 请求的详细日志（排版优化）

        Args:
            api_url: 请求 URL
            model_name: 模型名称
            image_config: 图片配置
            prompt: 提示词
            image_count: 输入图片数量
            request_body: 完整请求体（用于 debug 级别）
        """
        # 分隔线
        separator = "=" * 60

        # 构建日志内容
        log_lines = [
            "",
            separator,
            "[GeminiREST] Request Details",
            separator,
            f"  URL          : {api_url}",
            f"  Model        : {model_name}",
            f"  Image Size   : {image_config.get('imageSize', 'N/A')}",
            f"  Aspect Ratio : {image_config.get('aspectRatio', 'default')}",
            f"  Input Images : {image_count}",
            separator,
            "  Prompt:",
        ]

        # 处理提示词（多行缩进）
        prompt_lines = prompt.split("\n")
        max_prompt_lines = 10  # 限制显示的行数
        for i, line in enumerate(prompt_lines[:max_prompt_lines]):
            # 每行最多显示 80 字符
            if len(line) > 80:
                log_lines.append(f"    {line[:80]}...")
            else:
                log_lines.append(f"    {line}")
        if len(prompt_lines) > max_prompt_lines:
            log_lines.append(f"    ... (共 {len(prompt_lines)} 行，已省略)")

        log_lines.append(separator)
        log_lines.append("")

        # 使用 info 级别打印主要信息
        logger.info("\n".join(log_lines))

        # 使用 debug 级别打印完整请求体（不含 base64 数据）
        debug_body = self._sanitize_request_body_for_log(request_body)
        logger.debug(
            f"[GeminiREST] Full Request Body (sanitized):\n"
            f"{self._format_json_for_log(debug_body)}"
        )

    @staticmethod
    def _sanitize_request_body_for_log(request_body: dict[str, Any]) -> dict[str, Any]:
        """
        清理请求体，移除 base64 图片数据（避免日志过大）

        Args:
            request_body: 原始请求体

        Returns:
            清理后的请求体副本
        """
        import copy

        sanitized = copy.deepcopy(request_body)

        # 遍历 contents -> parts，替换 inlineData.data
        contents = sanitized.get("contents", [])
        for content in contents:
            parts = content.get("parts", [])
            for part in parts:
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    if "data" in inline_data:
                        # 计算原始数据大小
                        data_len = len(inline_data["data"])
                        inline_data["data"] = f"<base64 data, {data_len} chars>"

        return sanitized

    @staticmethod
    def _format_json_for_log(data: dict[str, Any], indent: int = 2) -> str:
        """
        格式化 JSON 数据用于日志输出

        Args:
            data: 要格式化的字典
            indent: 缩进空格数

        Returns:
            格式化后的 JSON 字符串
        """
        import json

        try:
            return json.dumps(data, ensure_ascii=False, indent=indent)
        except (TypeError, ValueError) as e:
            return f"<JSON 格式化失败: {e}>"

    def _log_http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> None:
        """
        打印原生 HTTP 请求内容（模拟 curl 格式）

        Args:
            method: HTTP 方法
            url: 请求 URL
            headers: 请求头
            body: 请求体
        """
        import json
        from urllib.parse import urlparse

        # 解析 URL
        parsed = urlparse(url)

        # 构建分隔线
        separator = "-" * 60

        # 构建请求行
        log_lines = [
            "",
            separator,
            "[GeminiREST] Raw HTTP Request",
            separator,
            f"{method} {parsed.path} HTTP/1.1",
            f"Host: {parsed.netloc}",
        ]

        # 添加请求头（隐藏敏感信息）
        for key, value in headers.items():
            if key.lower() in ("x-goog-api-key", "authorization", "api-key"):
                # 隐藏 API Key，只显示前后几位
                if len(value) > 12:
                    masked = f"{value[:4]}...{value[-4:]}"
                else:
                    masked = "***"
                log_lines.append(f"{key}: {masked}")
            else:
                log_lines.append(f"{key}: {value}")

        log_lines.append("")  # 空行分隔 header 和 body

        # 处理请求体
        log_lines.append("[Request Body (JSON)]")

        # 清理请求体中的 base64 数据
        sanitized_body = self._sanitize_request_body_for_log(body)

        try:
            body_json = json.dumps(sanitized_body, ensure_ascii=False, indent=2)
            # 限制 body 的行数
            body_lines = body_json.split("\n")
            max_body_lines = 50
            for line in body_lines[:max_body_lines]:
                log_lines.append(f"  {line}")
            if len(body_lines) > max_body_lines:
                log_lines.append(f"  ... (共 {len(body_lines)} 行，已省略)")
        except (TypeError, ValueError) as e:
            log_lines.append(f"  <JSON 序列化失败: {e}>")

        log_lines.append(separator)
        log_lines.append("")

        # 使用 info 级别打印
        logger.info("\n".join(log_lines))

    async def _call_gemini_rest(
        self, prompt: str, image_bytes: list[bytes] | None = None
    ) -> dict[str, Any]:
        """
        使用 REST API 直接调用 Gemini API（支持中转服务）

        Args:
            prompt: 绘图提示词
            image_bytes: 输入图片列表

        Returns:
            包含 images 和 text 的字典
        """
        model_name = base_config.get(
            "api_draw_model", "gemini-2.0-flash-preview-image-generation"
        )
        aspect_ratio = base_config.get("api_draw_aspect_ratio", "")
        image_size = base_config.get("api_image_size", "1K").upper()
        api_key = base_config.get("api_key", "")
        base_url = base_config.get("api_base_url", "")

        if not api_key:
            raise ValueError("未配置 api_key")

        # 校验分辨率参数
        if image_size not in self.VALID_IMAGE_SIZES:
            logger.warning(f"[GeminiREST] 不支持的分辨率 '{image_size}'，使用默认 1K")
            image_size = "1K"

        # 构建 imageConfig
        image_config: dict[str, str] = {"imageSize": image_size}
        if aspect_ratio and aspect_ratio in self.VALID_ASPECT_RATIOS:
            image_config["aspectRatio"] = aspect_ratio

        logger.info(
            f"[GeminiREST] 模型: {model_name}, "
            f"宽高比: {aspect_ratio or '默认'}, "
            f"分辨率: {image_size}"
        )
        logger.debug(f"[GeminiREST] 提示词: {prompt[:100]}...")

        # 构建请求 URL
        # 中转服务或官方 API
        if base_url:
            # 清理 URL：移除末尾的 / 和可能的 /v1 后缀
            clean_base_url = base_url.rstrip("/")
            if clean_base_url.endswith("/v1"):
                clean_base_url = clean_base_url[:-3]
            api_url = f"{clean_base_url}/v1beta/models/{model_name}:generateContent"
            logger.debug(f"[GeminiREST] 使用中转服务: {clean_base_url}")
        else:
            api_url = (
                f"{self.GEMINI_DEFAULT_BASE_URL}/v1beta/models/"
                f"{model_name}:generateContent"
            )
            logger.debug("[GeminiREST] 使用官方 API")

        # 构建请求内容的 parts
        parts: list[dict[str, Any]] = []

        # 添加输入图片（如果有）
        if image_bytes:
            for i, img_data in enumerate(image_bytes):
                # 检测图片格式
                mime_type = "image/png"
                if img_data[:3] == b"\xff\xd8\xff":
                    mime_type = "image/jpeg"
                elif img_data[:4] == b"\x89PNG":
                    mime_type = "image/png"
                elif img_data[:6] in (b"GIF87a", b"GIF89a"):
                    mime_type = "image/gif"
                elif img_data[:4] == b"RIFF" and img_data[8:12] == b"WEBP":
                    mime_type = "image/webp"

                img_b64 = base64.b64encode(img_data).decode("utf-8")
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": img_b64,
                    }
                })
                logger.debug(f"[GeminiREST] 添加输入图片 {i + 1} ({mime_type})")

        # 添加文本提示
        parts.append({"text": prompt})

        # 构建请求体
        request_body: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": image_config,
            },
        }

        # 打印请求日志（排版优化）
        self._log_gemini_request(
            api_url=api_url,
            model_name=model_name,
            image_config=image_config,
            prompt=prompt,
            image_count=len(image_bytes) if image_bytes else 0,
            request_body=request_body,
        )

        async def _do_request() -> dict[str, Any]:
            """执行实际的 HTTP 请求"""
            # 构建请求头
            request_headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            # 打印原生 HTTP 请求内容
            self._log_http_request(
                method="POST",
                url=api_url,
                headers=request_headers,
                body=request_body,
            )

            # 显式设置 proxy=None，禁止 httpx 读取系统代理环境变量
            # 防止代理切换导致 QQ 连接断开
            async with httpx.AsyncClient(
                timeout=GEMINI_TIMEOUT,
                proxy=None,
            ) as client:
                response = await client.post(
                    api_url,
                    headers=request_headers,
                    json=request_body,
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise Exception(
                        f"{response.status_code} {response.reason_phrase}. {error_text}"
                    )

                result = response.json()

            # 解析响应
            images: list[bytes] = []
            text_parts: list[str] = []

            candidates = result.get("candidates", [])
            if not candidates:
                logger.warning("[GeminiREST] 响应中没有 candidates")
                return {"images": [], "text": ""}

            content = candidates[0].get("content", {})
            response_parts = content.get("parts", [])

            for part in response_parts:
                # 处理文本
                if "text" in part:
                    text_content = part["text"]
                    # 检查是否是 Markdown 格式的 base64 图片
                    # 格式: ![image](data:image/png;base64,xxxx) 或 ![](data:image/jpeg;base64,xxxx)
                    md_pattern = r'!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=]+)\)'
                    md_matches = re.findall(md_pattern, text_content)
                    if md_matches:
                        # 从 Markdown 中提取 base64 图片
                        for img_b64 in md_matches:
                            try:
                                img_bytes = base64.b64decode(img_b64)
                                images.append(img_bytes)
                                img_info = _get_image_info(img_bytes)
                                logger.info(
                                    f"[GeminiREST] 从Markdown提取图片成功: "
                                    f"大小={img_info.get('size_kb', '?')}KB, "
                                    f"分辨率={img_info.get('resolution', '?')}"
                                )
                            except Exception as e:
                                logger.warning(f"[GeminiREST] 解码Markdown图片失败: {e}")
                        # 移除 Markdown 图片部分，保留其他文本
                        remaining_text = re.sub(md_pattern, '', text_content).strip()
                        if remaining_text:
                            text_parts.append(remaining_text)
                    else:
                        text_parts.append(text_content)

                # 处理图片（inlineData 格式）
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    img_b64 = inline_data.get("data", "")
                    if img_b64:
                        try:
                            img_bytes = base64.b64decode(img_b64)
                            images.append(img_bytes)

                            # 记录图片信息
                            img_info = _get_image_info(img_bytes)
                            logger.info(
                                f"[GeminiREST] 图片生成成功: "
                                f"大小={img_info.get('size_kb', '?')}KB, "
                                f"分辨率={img_info.get('resolution', '?')}"
                            )
                        except Exception as e:
                            logger.warning(f"[GeminiREST] 解码图片失败: {e}")

            text_content = "\n".join(text_parts).strip()

            if not images and not text_content:
                logger.warning("[GeminiREST] 模型未返回任何内容")

            return {"images": images, "text": text_content}

        # 使用重试逻辑包装调用
        try:
            return await _retry_with_backoff(
                _do_request,
                max_retries=MAX_RETRIES,
                base_delay=RETRY_DELAY_BASE,
                tag="GeminiREST",
            )
        except Exception as e:
            logger.error(f"[GeminiREST] 调用失败: {e}")
            raise

    async def draw(
        self, prompt: str, image_bytes: list[bytes] | None = None
    ) -> dict[str, Any]:
        """
        执行图片生成

        Args:
            prompt: 绘图提示词
            image_bytes: 输入图片列表（用于图生图）

        Returns:
            包含 images 和 text 的字典
        """
        model_name = base_config.get("api_draw_model", "")
        use_gemini_native = base_config.get("use_gemini_native_api", False)

        # 根据配置决定使用哪种 API
        if use_gemini_native:
            logger.debug(
                f"[DirectAPI] 使用 Gemini 原生 REST API (模型: {model_name})"
            )
            return await self._call_gemini_rest(prompt, image_bytes)

        # 使用 OpenAI SDK 兼容模式
        logger.debug("[DirectAPI] 使用 OpenAI SDK 兼容模式")
        return await self._call_openai_compatible_api(prompt, image_bytes)

    async def _call_openai_compatible_api(
        self, prompt: str, image_bytes: list[bytes] | None = None
    ) -> dict[str, Any]:
        """
        使用 OpenAI SDK 调用兼容 API

        Args:
            prompt: 绘图提示词
            image_bytes: 输入图片列表

        Returns:
            包含 images 和 text 的字典
        """
        logger.debug("[DirectAPI] 开始图片生成...")

        client = self._get_client()
        model_name = self._resolve_model_name()

        logger.info(f"[DirectAPI] 使用模型: {model_name}")
        logger.debug(f"[DirectAPI] 提示词: {prompt[:100]}...")

        # 构建消息
        messages: list[dict[str, Any]] = []

        # 如果有输入图片，构建多模态消息
        if image_bytes:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

            for i, img_data in enumerate(image_bytes):
                img_b64 = base64.b64encode(img_data).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
                logger.debug(f"[DirectAPI] 添加输入图片 {i + 1}")

            messages.append({"role": "user", "content": content})
        else:
            # 纯文本消息
            messages.append({"role": "user", "content": prompt})

        async def _do_generate() -> dict[str, Any]:
            """执行实际的 API 调用"""
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4096,  # 图片数据较大，需要足够的 token
            )

            # 提取响应内容
            response_content = response.choices[0].message.content or ""
            logger.debug(f"[DirectAPI] 响应长度: {len(response_content)} 字符")

            # 尝试提取图片
            images: list[bytes] = []
            text_content = ""

            img_data = _extract_image_from_content(response_content)
            if img_data:
                images.append(img_data)

                # 记录图片信息
                img_info = _get_image_info(img_data)
                logger.info(
                    f"[DirectAPI] 图片生成成功: "
                    f"大小={img_info.get('size_kb', '?')}KB, "
                    f"分辨率={img_info.get('resolution', '?')}, "
                    f"格式={img_info.get('format', '?')}"
                )
            else:
                # 没有提取到图片，可能是纯文本回复
                text_content = response_content
                logger.warning("[DirectAPI] 未能从响应中提取图片数据")
                if response_content:
                    logger.debug(
                        f"[DirectAPI] 响应内容预览: {response_content[:200]}..."
                    )

            return {"images": images, "text": text_content}

        # 使用重试逻辑包装调用
        try:
            return await _retry_with_backoff(
                _do_generate,
                max_retries=MAX_RETRIES,
                base_delay=RETRY_DELAY_BASE,
                tag="DirectAPI",
            )
        except Exception as e:
            logger.error(f"[DirectAPI] 图片生成失败: {e}")
            raise
