from typing import Any

from zhenxun.services.llm import create_image
from zhenxun.services.log import logger

from ..config import base_config
from . import DrawEngine


def _get_image_info(image_data: bytes) -> dict[str, Any]:
    """获取图片的基本信息（大小、格式、分辨率）"""
    info: dict[str, Any] = {
        "size_bytes": len(image_data),
        "size_kb": round(len(image_data) / 1024, 2),
        "size_mb": round(len(image_data) / 1024 / 1024, 2),
    }

    # 检测图片格式和尺寸
    try:
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(image_data))
        info["format"] = img.format
        info["width"] = img.width
        info["height"] = img.height
        info["mode"] = img.mode
        info["resolution"] = f"{img.width}x{img.height}"
    except Exception as e:
        info["format_error"] = str(e)

    return info


class LlmApiEngine(DrawEngine):
    """使用 zhenxun.services.llm API 的绘图引擎"""

    async def draw(
        self, prompt: str, image_bytes: list[bytes] | None = None
    ) -> dict[str, Any]:
        logger.debug("🎨 使用 LLM API 引擎进行绘图...")
        draw_model_name = base_config.get("api_draw_model")
        if not draw_model_name:
            raise ValueError("未配置API绘图模型 (api_draw_model)")

        # 构建自定义参数
        custom_params: dict[str, Any] = {}

        # 检测是否使用 4K 模型，自动设置输出分辨率
        if "4k" in draw_model_name.lower():
            custom_params["outputResolution"] = "4K"
            logger.info("[LLM API] 检测到 4K 模型，已设置 outputResolution=4K")

        # 从配置中获取图片生成参数
        aspect_ratio = base_config.get("api_draw_aspect_ratio")
        if aspect_ratio:
            custom_params["aspectRatio"] = aspect_ratio
            logger.debug(f"[LLM API] 使用宽高比: {aspect_ratio}")

        # 传递自定义参数到 create_image
        kwargs: dict[str, Any] = {}
        if custom_params:
            kwargs["custom_params"] = custom_params

        response = await create_image(
            prompt=prompt,
            images=image_bytes,  # type: ignore
            model=draw_model_name,
            **kwargs,
        )
        images = response.images or []

        # 记录图片信息用于诊断
        if images:
            logger.info(f"[LLM API] 返回 {len(images)} 张图片")
            for i, img_data in enumerate(images):
                if isinstance(img_data, bytes):
                    img_info = _get_image_info(img_data)
                    logger.info(
                        f"[LLM API] 图片 {i + 1}: "
                        f"大小={img_info.get('size_kb', '?')}KB, "
                        f"分辨率={img_info.get('resolution', '?')}, "
                        f"格式={img_info.get('format', '?')}"
                    )
                else:
                    logger.warning(
                        f"[LLM API] 图片 {i + 1}: 非 bytes 类型, type={type(img_data)}"
                    )

        return {"images": images, "text": response.text}
