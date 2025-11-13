from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from maim_message import Seg
from PIL import Image, UnidentifiedImageError

from src.plugin_system import (
    ActionActivationType,
    BaseAction,
    BasePlugin,
    ComponentInfo,
    ConfigField,
    get_logger,
    register_plugin,
)

from .sketchbook_core import draw_text_auto, paste_image_auto

logger = get_logger("anan_sketchbook_plugin")

PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_ASSETS_ROOT = (PLUGIN_DIR / "assets").resolve()
DEFAULT_ASSETS_CONFIG_VALUE = "assets"
DEFAULT_MOOD_MAPPING = {
    "#普通#": "BaseImages/base.png",
    "#开心#": "BaseImages/开心.png",
    "#生气#": "BaseImages/生气.png",
    "#无语#": "BaseImages/无语.png",
    "#脸红#": "BaseImages/脸红.png",
    "#病娇#": "BaseImages/病娇.png",
}
DEFAULT_MOOD_ALIASES = {
    "normal": "#普通#",
    "happy": "#开心#",
    "anger": "#生气#",
    "speechless": "#无语#",
    "blush": "#脸红#",
    "yandere": "#病娇#",
}
DEFAULT_MOOD_TAG = "#普通#"
DEFAULT_AUTO_PICK_MESSAGE_IMAGE = True
MIN_BASE64_LENGTH = 80


class SketchbookRendererError(RuntimeError):
    """Sketchbook 渲染过程中的可预期错误。"""


@dataclass(frozen=True)
class SketchbookConfigData:
    assets_root: Path
    baseimage_mapping: Dict[str, Path]
    default_baseimage: Path
    default_mood: str
    font_file: Optional[Path]
    overlay_file: Optional[Path]
    use_overlay: bool
    text_box_topleft: Tuple[int, int]
    image_box_bottomright: Tuple[int, int]
    padding: int
    allow_upscale: bool
    max_font_height: int
    line_spacing: float
    mood_aliases: Dict[str, str]


@dataclass(frozen=True)
class RenderResult:
    png_bytes: bytes
    mood_tag: str
    text: str


class SketchbookRenderer:
    """负责把文本/图片渲染到素描本底图的核心类。"""

    def __init__(self, config: SketchbookConfigData):
        self.config = config
        self.assets_root = config.assets_root
        bounds = config.text_box_topleft, config.image_box_bottomright
        region_width = max(1, bounds[1][0] - bounds[0][0])
        region_height = max(1, bounds[1][1] - bounds[0][1])
        self._region_ratio = region_width / region_height

    @classmethod
    def from_plugin_config(cls, plugin_config: Optional[dict]) -> "SketchbookRenderer":
        section = (plugin_config or {}).get("sketchbook", {})
        assets_root = cls._resolve_base_dir(section.get("assets_root"))
        raw_mapping = section.get("mood_mapping") or DEFAULT_MOOD_MAPPING
        if not isinstance(raw_mapping, dict):
            raw_mapping = DEFAULT_MOOD_MAPPING
        baseimage_mapping = cls._load_mapping(raw_mapping, assets_root)
        if not baseimage_mapping:
            raise SketchbookRendererError("未在配置中找到任何底图映射（sketchbook.mood_mapping）。")

        default_mood = str(section.get("default_mood") or DEFAULT_MOOD_TAG)
        default_baseimage = baseimage_mapping.get(default_mood) or next(iter(baseimage_mapping.values()))

        font_file = cls._resolve_optional_path(section.get("font_file"), assets_root)
        overlay_file = cls._resolve_optional_path(section.get("base_overlay_file"), assets_root)
        use_overlay = bool(section.get("use_overlay", True))

        text_box = cls._parse_xy(section.get("text_box_topleft"), fallback=(119, 450))
        image_box = cls._parse_xy(section.get("image_box_bottomright"), fallback=(398, 625))

        padding = int(section.get("padding", 12))
        allow_upscale = bool(section.get("allow_upscale", True))
        max_font_height = int(section.get("max_font_height", 64))
        line_spacing = float(section.get("line_spacing", 0.15))

        raw_aliases = section.get("mood_aliases") or DEFAULT_MOOD_ALIASES
        if not isinstance(raw_aliases, dict):
            raw_aliases = DEFAULT_MOOD_ALIASES
        mood_aliases = {
            str(alias).strip().lower(): str(tag).strip()
            for alias, tag in raw_aliases.items()
            if alias and tag
        }

        cfg = SketchbookConfigData(
            assets_root=assets_root,
            baseimage_mapping=baseimage_mapping,
            default_baseimage=default_baseimage,
            default_mood=default_mood,
            font_file=font_file,
            overlay_file=overlay_file if use_overlay else None,
            use_overlay=use_overlay,
            text_box_topleft=text_box,
            image_box_bottomright=image_box,
            padding=padding,
            allow_upscale=allow_upscale,
            max_font_height=max_font_height,
            line_spacing=line_spacing,
            mood_aliases=mood_aliases,
        )
        return cls(cfg)

    @staticmethod
    def _resolve_base_dir(path_value: Optional[str]) -> Path:
        candidate = Path(path_value) if path_value else DEFAULT_ASSETS_ROOT
        if not candidate.is_absolute():
            candidate = (PLUGIN_DIR / candidate).resolve()
        return candidate

    @staticmethod
    def _resolve_optional_path(relative: Optional[str], base_dir: Path) -> Optional[Path]:
        if not relative:
            return None
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return candidate

    @staticmethod
    def _parse_xy(value: Any, fallback: Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return fallback

    @staticmethod
    def _load_mapping(raw_mapping: Dict[str, str], base_dir: Path) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for mood, relative in raw_mapping.items():
            if not mood or not relative:
                continue
            candidate = Path(relative)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            mapping[str(mood)] = candidate
        return mapping

    def render(
        self,
        *,
        text: str,
        mood_tag: Optional[str],
        content_image: Optional[Image.Image],
    ) -> RenderResult:
        cleaned_text = text.strip()
        resolved_mood, cleaned_text, baseimage_path = self._select_baseimage(mood_tag, cleaned_text)

        if not cleaned_text and content_image is None:
            raise SketchbookRendererError("没有可渲染的文本或图片内容。")

        if content_image is not None and content_image.mode not in {"RGBA", "LA"}:
            content_image = content_image.convert("RGBA")

        overlay = self.config.overlay_file if self.config.use_overlay else None
        x1, y1 = self.config.text_box_topleft
        x2, y2 = self.config.image_box_bottomright

        try:
            if cleaned_text and content_image:
                png_bytes = self._render_text_and_image(
                    baseimage_path, cleaned_text, content_image, overlay, (x1, y1), (x2, y2)
                )
            elif content_image:
                png_bytes = paste_image_auto(
                    image_source=str(baseimage_path),
                    image_overlay=str(overlay) if overlay else None,
                    top_left=(x1, y1),
                    bottom_right=(x2, y2),
                    content_image=content_image,
                    align="center",
                    valign="middle",
                    padding=self.config.padding,
                    allow_upscale=self.config.allow_upscale,
                    keep_alpha=True,
                )
            else:
                png_bytes = draw_text_auto(
                    image_source=str(baseimage_path),
                    image_overlay=str(overlay) if overlay else None,
                    top_left=(x1, y1),
                    bottom_right=(x2, y2),
                    text=cleaned_text,
                    color=(0, 0, 0),
                    max_font_height=self.config.max_font_height,
                    font_path=str(self.config.font_file) if self.config.font_file else None,
                    line_spacing=self.config.line_spacing,
                )
        except FileNotFoundError as exc:
            raise SketchbookRendererError(f"找不到底图资源：{exc.filename}") from exc
        except Exception as exc:
            raise SketchbookRendererError(f"渲染素描本失败：{exc}") from exc

        return RenderResult(png_bytes=png_bytes, mood_tag=resolved_mood, text=cleaned_text)

    def _render_text_and_image(
        self,
        baseimage_path: Path,
        text: str,
        content_image: Image.Image,
        overlay: Optional[Path],
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
    ) -> bytes:
        x1, y1 = top_left
        x2, y2 = bottom_right
        region_width = x2 - x1
        region_height = y2 - y1

        if self._is_vertical(content_image):
            spacing = 10
            left_width = region_width // 2 - spacing // 2
            left_region = (x1, y1), (x1 + left_width, y2)
            right_region = (x1 + left_width + spacing, y1), (x2, y2)

            intermediate = paste_image_auto(
                image_source=str(baseimage_path),
                image_overlay=None,
                top_left=left_region[0],
                bottom_right=left_region[1],
                content_image=content_image,
                align="center",
                valign="middle",
                padding=self.config.padding,
                allow_upscale=self.config.allow_upscale,
                keep_alpha=True,
            )

            return draw_text_auto(
                image_source=io.BytesIO(intermediate),
                image_overlay=str(overlay) if overlay else None,
                top_left=right_region[0],
                bottom_right=right_region[1],
                text=text,
                color=(0, 0, 0),
                max_font_height=self.config.max_font_height,
                font_path=str(self.config.font_file) if self.config.font_file else None,
                line_spacing=self.config.line_spacing,
            )

        estimated_text_height = min(region_height // 2, 100)
        image_region = (x1, y1), (x2, y1 + (region_height - estimated_text_height))
        text_region = (x1, image_region[1][1]), (x2, y2)

        intermediate = paste_image_auto(
            image_source=str(baseimage_path),
            image_overlay=None,
            top_left=image_region[0],
            bottom_right=image_region[1],
            content_image=content_image,
            align="center",
            valign="middle",
            padding=self.config.padding,
            allow_upscale=self.config.allow_upscale,
            keep_alpha=True,
        )

        return draw_text_auto(
            image_source=io.BytesIO(intermediate),
            image_overlay=str(overlay) if overlay else None,
            top_left=text_region[0],
            bottom_right=text_region[1],
            text=text,
            color=(0, 0, 0),
            max_font_height=self.config.max_font_height,
            font_path=str(self.config.font_file) if self.config.font_file else None,
            line_spacing=self.config.line_spacing,
        )

    def _is_vertical(self, image: Image.Image) -> bool:
        return image.height * self._region_ratio > image.width

    def _select_baseimage(
        self,
        mood_tag: Optional[str],
        text: str,
    ) -> Tuple[str, str, Path]:
        normalized = self._normalize_mood(mood_tag)
        cleaned_text = text
        baseimage_path = self.config.baseimage_mapping.get(normalized)

        if not baseimage_path:
            normalized = self.config.default_mood
            baseimage_path = self.config.baseimage_mapping.get(normalized, self.config.default_baseimage)

        if not baseimage_path:
            raise SketchbookRendererError("未能找到可用的素描本底图，请检查配置。")

        return normalized, cleaned_text, baseimage_path

    def _normalize_mood(self, mood_tag: Optional[str]) -> Optional[str]:
        if not mood_tag:
            return None
        tag = mood_tag.strip()
        if not tag:
            return None
        alias = self.config.mood_aliases.get(tag.lower())
        return alias or tag


class SketchbookRenderAction(BaseAction):
    """供 LLM 选择的素描本渲染 Action。"""

    action_name = "anan_sketchbook_render"
    action_description = "把指定文本和可选图片渲染到安安素描本底图，并以图片形式回复。"
    activation_type = ActionActivationType.LLM_JUDGE
    base_action_parameters = {
        "text": "当你想强调、写在素描本上的核心内容。",
        "mood": "当你想表达的表情标签（例如 #开心# 或 happy），用于匹配差分底图。",
        "image_base64": "可选，直接传入要贴到素描本上的 base64 图片。",
        "image_path": "可选，本地图片路径，便于复用已经下载的临时文件。",
        "message_image_index": "当自动读取最近一条消息的图片时，选择第 N 张（从 1 开始）。",
        "use_message_image": "布尔值，是否允许从最近消息中取图，默认遵循配置。",
    }
    action_parameters = base_action_parameters.copy()
    action_require = [
        "当你想突出自己说的话、让情绪起伏更直观时使用。",
        "当你想强调当前心情/状态并以差分底图表达时使用，自行挑选 mood。",
    ]
    associated_types = ["image"]

    _renderer: Optional[SketchbookRenderer] = None
    _renderer_error: Optional[str] = None

    async def execute(self) -> Tuple[bool, str]:
        renderer = self._ensure_renderer()
        if renderer is None:
            message = (
                "素描本插件尚未准备好："
                + (self._renderer_error or "请检查资源路径配置。")
            )
            await self.send_text(message, set_reply=True, reply_message=self.action_message)
            return False, message
        text = self._extract_text()
        mood = self._extract_mood()
        try:
            content_image = self._resolve_image(renderer.assets_root)
        except SketchbookRendererError as exc:
            await self.send_text(str(exc), set_reply=True, reply_message=self.action_message)
            return False, str(exc)

        if not text and content_image is None:
            message = "没有提供文本内容，也没有可用图片，无法渲染素描本。"
            await self.send_text(message, set_reply=True, reply_message=self.action_message)
            return False, message

        try:
            result = renderer.render(
                text=text,
                mood_tag=mood,
                content_image=content_image,
            )
        except SketchbookRendererError as exc:
            await self.send_text(str(exc), set_reply=True, reply_message=self.action_message)
            return False, str(exc)

        image_base64 = base64.b64encode(result.png_bytes).decode("ascii")
        sent = await self.send_image(
            image_base64,
            set_reply=True,
            reply_message=self.action_message,
        )
        if not sent:
            message = "素描本图片生成成功，但发送失败。"
            await self.send_text(message, set_reply=True, reply_message=self.action_message)
            return False, message

        summary = f"生成素描本图片并发送，mood={result.mood_tag}"
        return True, summary

    def _ensure_renderer(self) -> Optional[SketchbookRenderer]:
        if self._renderer:
            return self._renderer
        if self._renderer_error:
            return None
        try:
            renderer = SketchbookRenderer.from_plugin_config(self.plugin_config or {})
            self._renderer = renderer
            self._update_mood_help(renderer.config.baseimage_mapping.keys())
            return renderer
        except SketchbookRendererError as exc:
            self._renderer_error = str(exc)
            logger.error("%s 构建渲染器失败：%s", self.log_prefix, exc)
            return None

    def _extract_text(self) -> str:
        for key in ("text", "content", "message"):
            value = self.action_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_mood(self) -> Optional[str]:
        mood = self.action_data.get("mood") or self.action_data.get("emotion")
        if isinstance(mood, str):
            return mood.strip()
        return None

    def _update_mood_help(self, mood_tags: Iterable[str]) -> None:
        tags = sorted(tag for tag in mood_tags if isinstance(tag, str) and tag.strip())
        if not tags:
            return
        self.action_parameters = self.base_action_parameters.copy()
        tag_list = ", ".join(tags)
        self.action_parameters["mood"] = (
            f"当你想表达的表情标签，可用：{tag_list}"
        )

    def _resolve_image(self, assets_root: Path) -> Optional[Image.Image]:
        base64_payload = self.action_data.get("image_base64") or self.action_data.get("image")
        if isinstance(base64_payload, str) and base64_payload.strip():
            return self._decode_base64_image(base64_payload.strip())

        image_path = self.action_data.get("image_path")
        if isinstance(image_path, str) and image_path.strip():
            loaded = self._load_image_from_path(image_path.strip(), assets_root)
            if loaded:
                return loaded

        use_message_image = self._should_use_message_image()
        if not use_message_image:
            return None

        candidates = self._collect_message_images()
        if not candidates:
            return None

        index = self._preferred_image_index()
        idx = min(max(index - 1, 0), len(candidates) - 1)
        return self._decode_base64_image(candidates[idx])

    def _should_use_message_image(self) -> bool:
        action_override = self.action_data.get("use_message_image")
        if action_override is not None:
            if isinstance(action_override, bool):
                return action_override
            if isinstance(action_override, str):
                return action_override.strip().lower() not in {"false", "0", "no"}
        default_value = DEFAULT_AUTO_PICK_MESSAGE_IMAGE
        return bool(self.get_config("sketchbook.auto_pick_message_image", default_value))

    def _preferred_image_index(self) -> int:
        index = self.action_data.get("message_image_index")
        try:
            idx = int(index)
            return idx if idx > 0 else 1
        except (TypeError, ValueError):
            return 1

    def _collect_message_images(self) -> List[str]:
        candidates: List[str] = []
        segments = []

        if self.action_message and hasattr(self.action_message, "message_segment"):
            segments.append(self.action_message.message_segment)

        if self.chat_stream and hasattr(self.chat_stream, "context") and self.chat_stream.context:
            try:
                last_message = self.chat_stream.context.get_last_message()
                if last_message and hasattr(last_message, "message_segment"):
                    segments.append(last_message.message_segment)
                reply_message = getattr(last_message, "reply", None)
                if reply_message and hasattr(reply_message, "message_segment"):
                    segments.append(reply_message.message_segment)
            except Exception:
                pass

        for segment in segments:
            candidates.extend(self._collect_images_from_any(segment))

        return [encoded for encoded in candidates if encoded]

    def _collect_images_from_any(self, payload: Any) -> List[str]:
        if payload is None:
            return []
        if isinstance(payload, Seg):
            seg_type = getattr(payload, "type", "")
            data = getattr(payload, "data", None)
            if seg_type in {"image", "emoji"}:
                normalized = self._normalize_base64_string(data)
                return [normalized] if normalized else []
            if seg_type == "seglist" and isinstance(data, list):
                results: List[str] = []
                for sub in data:
                    results.extend(self._collect_images_from_any(sub))
                return results
            if isinstance(data, (list, dict, str)):
                return self._collect_images_from_any(data)
            return []
        if isinstance(payload, list):
            results: List[str] = []
            for item in payload:
                results.extend(self._collect_images_from_any(item))
            return results
        if isinstance(payload, dict):
            results: List[str] = []
            if "type" in payload and "data" in payload:
                try:
                    seg = Seg.from_dict(payload)
                    results.extend(self._collect_images_from_any(seg))
                except Exception:
                    pass
            for key in ("base64", "data", "content", "message_segment", "segments"):
                if key in payload:
                    results.extend(self._collect_images_from_any(payload[key]))
            return results
        if isinstance(payload, str):
            normalized = self._normalize_base64_string(payload)
            return [normalized] if normalized else []
        return []

    def _normalize_base64_string(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.startswith("base64://"):
            candidate = candidate[len("base64://") :]
        elif candidate.startswith("data:"):
            parts = candidate.split(",", 1)
            if len(parts) == 2:
                candidate = parts[1]
        elif candidate.startswith("[CQ:image"):
            match = re.search(r"base64=([^,\\]]+)", candidate)
            if match:
                candidate = match.group(1)
            else:
                return None
        if len(candidate) < MIN_BASE64_LENGTH:
            return None
        try:
            base64.b64decode(candidate, validate=True)
        except Exception:
            return None
        return candidate

    def _decode_base64_image(self, payload: str) -> Image.Image:
        normalized = self._normalize_base64_string(payload)
        if not normalized:
            raise SketchbookRendererError("提供的 base64 图片格式不正确或内容过短。")
        try:
            binary = base64.b64decode(normalized)
            return Image.open(io.BytesIO(binary)).convert("RGBA")
        except UnidentifiedImageError as exc:
            raise SketchbookRendererError("无法解析 base64 图片内容。") from exc

    def _load_image_from_path(self, path_value: str, assets_root: Path) -> Optional[Image.Image]:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (assets_root / candidate).resolve()
        if not candidate.exists():
            logger.warning("%s 指定的图片路径不存在：%s", self.log_prefix, candidate)
            return None
        try:
            return Image.open(candidate).convert("RGBA")
        except UnidentifiedImageError:
            logger.warning("%s 无法打开图片文件：%s", self.log_prefix, candidate)
            return None


@register_plugin
class AnanSketchbookPlugin(BasePlugin):
    """安安素描本渲染插件入口。"""

    plugin_name = "anan_sketchbook_plugin"
    enable_plugin = False
    dependencies: List[str] = []
    python_dependencies: List[str] = ["Pillow>=12.0.0"]
    config_file_name = "config.toml"

    config_section_descriptions = {
        "plugin": "插件基础配置",
        "sketchbook": "素描本渲染相关参数",
    }

    config_schema: Dict[str, Dict[str, ConfigField]] = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=False, description="是否启用插件"),
            "config_version": ConfigField(type=str, default="1.0.0", description="配置文件版本"),
        },
        "sketchbook": {
            "assets_root": ConfigField(
                type=str,
                default=DEFAULT_ASSETS_CONFIG_VALUE,
                description="素材根目录，默认指向本插件目录内的 assets。",
            ),
            "font_file": ConfigField(
                type=str,
                default="font.ttf",
                description="字体文件路径，默认在素材目录下。",
            ),
            "base_overlay_file": ConfigField(
                type=str,
                default="BaseImages/base_overlay.png",
                description="覆盖层文件路径，可为空。",
            ),
            "use_overlay": ConfigField(type=bool, default=True, description="是否使用覆盖层。"),
            "text_box_topleft": ConfigField(
                type=list,
                default=[119, 450],
                description="文本区域左上角坐标。",
            ),
            "image_box_bottomright": ConfigField(
                type=list,
                default=[398, 625],
                description="文本/图片区域右下角坐标。",
            ),
            "padding": ConfigField(type=int, default=12, description="内容与边框的内边距。"),
            "allow_upscale": ConfigField(type=bool, default=True, description="图片是否允许放大以填满区域。"),
            "max_font_height": ConfigField(type=int, default=64, description="文本的最大字号像素。"),
            "line_spacing": ConfigField(type=float, default=0.15, description="文本行间距比例。"),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (SketchbookRenderAction.get_action_info(), SketchbookRenderAction),
        ]
