"""
OpenList/Alist 文件上传服务

用于将生成的图片上传到 OpenList 服务器并返回可访问的 URL

URL 路径说明:
- /d/path  - 直接下载 (Direct download)
- /p/path  - 预览 (Preview，用于图片在网页中显示)
"""
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from zhenxun.services.log import logger

# 常见文件扩展名对应的 MIME 类型
MIME_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "ico": "image/x-icon",
    "pdf": "application/pdf",
    "zip": "application/zip",
    "txt": "text/plain",
    "json": "application/json",
    "xml": "application/xml",
    "mp4": "video/mp4",
    "mp3": "audio/mpeg",
}


@dataclass
class UploadResult:
    """
    上传结果

    Attributes:
        preview_url: 预览URL（用于网页中显示图片）
        download_url: 下载URL（用于直接下载文件）
        raw_url: 直链URL（从API获取的原始链接）
        path: 远程路径
    """

    preview_url: str
    download_url: str
    path: str
    raw_url: str | None = None


@dataclass
class FileInfo:
    """文件信息"""

    name: str
    size: int
    is_dir: bool
    raw_url: str | None = None


def get_mime_type(filename: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(filename).suffix.lstrip(".").lower()
    return MIME_TYPES.get(ext, "application/octet-stream")


class OpenListUploader:
    """
    OpenList 文件上传工具类

    用于将本地文件上传到 OpenList 服务器并获取访问 URL
    """

    def __init__(
        self,
        host: str,
        token: str,
        upload_path: str = "/ai_images",
        timeout: aiohttp.ClientTimeout | None = None,
    ):
        """
        初始化上传器

        Args:
            host: OpenList 服务器地址，如 http://110.42.4.3:49447
            token: API Token
            upload_path: 默认上传目录路径
            timeout: 请求超时设置
        """
        self.host = host.rstrip("/")
        self.token = token
        self.upload_path = upload_path.rstrip("/")
        self.timeout = timeout or aiohttp.ClientTimeout(
            total=180, connect=30
        )

    def _generate_filename(self, data: bytes, ext: str = "png") -> str:
        """根据内容生成唯一文件名"""
        content_hash = hashlib.md5(data).hexdigest()[:12]
        timestamp = int(time.time() * 1000)
        return f"img_{timestamp}_{content_hash}.{ext}"

    async def upload(
        self,
        data: bytes,
        remote_path: str,
        content_type: str | None = None,
    ) -> UploadResult | None:
        """
        上传文件到 OpenList

        Args:
            data: 文件二进制数据
            remote_path: 远程目标路径，如 "/images/test.jpg"
            content_type: MIME 类型，不提供则自动推断

        Returns:
            成功返回 UploadResult，失败返回 None
        """
        if not content_type:
            content_type = get_mime_type(remote_path)

        headers = {
            "Authorization": self.token,
            "File-Path": remote_path,
            "Content-Type": content_type,
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.put(
                    f"{self.host}/api/fs/put",
                    headers=headers,
                    data=data,
                ) as resp:
                    response_body = await resp.text()

                    if resp.status == 200:
                        logger.info(f"[OpenList] 上传成功: {remote_path}")
                        logger.debug(f"[OpenList] 响应: {response_body}")
                        return UploadResult(
                            preview_url=f"{self.host}{remote_path}",
                            download_url=f"{self.host}{remote_path}",
                            path=remote_path,
                        )
                    else:
                        logger.error(
                            f"[OpenList] 上传失败 [{resp.status}]: {response_body}"
                        )
                        return None

        except Exception as e:
            logger.error(f"[OpenList] 上传异常: {e}")
            return None

    async def upload_file(
        self, local_path: Path | str, remote_path: str
    ) -> UploadResult | None:
        """
        上传本地文件到 OpenList

        Args:
            local_path: 本地文件路径
            remote_path: 远程目标路径

        Returns:
            成功返回 UploadResult，失败返回 None
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"[OpenList] 文件不存在: {local_path}")
            return None

        data = local_path.read_bytes()
        content_type = get_mime_type(local_path.name)
        return await self.upload(data, remote_path, content_type)

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        ext: str = "png",
    ) -> UploadResult | None:
        """
        上传图片到默认目录

        Args:
            image_bytes: 图片二进制数据
            filename: 自定义文件名，不提供则自动生成
            ext: 文件扩展名（仅在自动生成文件名时使用）

        Returns:
            成功返回 UploadResult，失败返回 None
        """
        if not filename:
            filename = self._generate_filename(image_bytes, ext)

        remote_path = f"{self.upload_path}/{filename}"
        return await self.upload(image_bytes, remote_path)

    async def upload_images(self, images_bytes: list[bytes]) -> list[UploadResult]:
        """
        批量上传图片

        Args:
            images_bytes: 图片二进制数据列表

        Returns:
            成功上传的 UploadResult 列表
        """
        results = []
        for i, img_bytes in enumerate(images_bytes):
            result = await self.upload_image(img_bytes)
            if result:
                results.append(result)
            else:
                logger.warning(f"[OpenList] 第 {i + 1} 张图片上传失败")
        return results

    async def upload_text(
        self,
        content: str,
        upload_path: str,
        filename: str | None = None,
    ) -> UploadResult | None:
        """
        上传文本文件到指定目录

        Args:
            content: 文本内容
            upload_path: 上传目录路径
            filename: 自定义文件名，不提供则自动生成

        Returns:
            成功返回 UploadResult，失败返回 None
        """
        text_bytes = content.encode("utf-8")
        if not filename:
            content_hash = hashlib.md5(text_bytes).hexdigest()[:12]
            timestamp = int(time.time() * 1000)
            filename = f"log_{timestamp}_{content_hash}.txt"

        remote_path = f"{upload_path.rstrip('/')}/{filename}"
        return await self.upload(text_bytes, remote_path, "text/plain; charset=utf-8")

    async def get_raw_url(self, remote_path: str) -> str | None:
        """
        获取文件直链

        Args:
            remote_path: 远程文件路径

        Returns:
            直链 URL，失败返回 None
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.host}/api/fs/get",
                    headers={"Authorization": self.token},
                    json={"path": remote_path},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == 200:
                            return data.get("data", {}).get("raw_url")
                    return None

        except Exception as e:
            logger.error(f"[OpenList] 获取直链异常: {e}")
            return None

    async def upload_and_get_raw_url(
        self, data: bytes, remote_path: str
    ) -> str | None:
        """
        上传文件并获取直链

        Args:
            data: 文件二进制数据
            remote_path: 远程目标路径

        Returns:
            直链 URL，失败返回 None
        """
        result = await self.upload(data, remote_path)
        if not result:
            return None
        return await self.get_raw_url(remote_path)

    async def get_file_info(self, remote_path: str) -> FileInfo | None:
        """
        获取文件信息

        Args:
            remote_path: 远程文件路径

        Returns:
            FileInfo 对象，失败返回 None
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.host}/api/fs/get",
                    headers={"Authorization": self.token},
                    json={"path": remote_path},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == 200:
                            file_data = data.get("data", {})
                            return FileInfo(
                                name=file_data.get("name", ""),
                                size=file_data.get("size", 0),
                                is_dir=file_data.get("is_dir", False),
                                raw_url=file_data.get("raw_url"),
                            )
                    return None

        except Exception as e:
            logger.error(f"[OpenList] 获取文件信息异常: {e}")
            return None


# 全局上传器实例
_uploader: OpenListUploader | None = None


def get_uploader() -> OpenListUploader | None:
    """获取全局上传器实例"""
    return _uploader


def init_uploader(
    host: str,
    token: str,
    upload_path: str = "/ai_images",
) -> OpenListUploader:
    """
    初始化全局上传器

    Args:
        host: OpenList 服务器地址
        token: API Token
        upload_path: 默认上传目录路径

    Returns:
        OpenListUploader 实例
    """
    global _uploader
    _uploader = OpenListUploader(
        host=host,
        token=token,
        upload_path=upload_path,
    )
    logger.info(f"[OpenList] 上传器已初始化: {host}")
    return _uploader


__all__ = [
    "FileInfo",
    "OpenListUploader",
    "UploadResult",
    "get_mime_type",
    "get_uploader",
    "init_uploader",
]