from __future__ import annotations

import os
from io import BytesIO
from typing import List, Literal, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

RGBColor = Tuple[int, int, int]

Align = Literal["left", "center", "right"]
VAlign = Literal["top", "middle", "bottom"]


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """
    加载指定路径的字体文件，如果失败则加载默认字体。
    """
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()  # type: ignore # 如果没有可用的 TTF 字体，则加载默认位图字体


def wrap_lines(
    draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.FreeTypeFont, max_w: int
) -> List[str]:
    """
    将文本按指定宽度拆分为多行。
    """
    lines: List[str] = []

    for para in txt.splitlines() or [""]:
        has_space = " " in para
        units = para.split(" ") if has_space else list(para)
        buf = ""

        def unit_join(a: str, b: str, *, space: bool = has_space) -> str:
            if not a:
                return b
            return (a + " " + b) if space else (a + b)

        for u in units:
            trial = unit_join(buf, u)
            w = draw.textlength(trial, font=font)

            # 如果加入当前单元后宽度未超限，则继续累积
            if w <= max_w:
                buf = trial
                continue

            # 否则先将缓冲区内容作为一行输出
            if buf:
                lines.append(buf)

            # 处理当前单元
            if has_space and len(u) > 1:
                tmp = ""
                for ch in u:
                    if draw.textlength(tmp + ch, font=font) <= max_w:
                        tmp += ch
                        continue

                    if tmp:
                        lines.append(tmp)
                    tmp = ch
                buf = tmp
                continue

            if draw.textlength(u, font=font) <= max_w:
                buf = u
            else:
                lines.append(u)
                buf = ""
        if buf != "":
            lines.append(buf)
        if para == "" and (not lines or lines[-1] != ""):
            lines.append("")
    return lines


def parse_color_segments(
    s: str, in_bracket: bool, bracket_color: RGBColor, color: RGBColor
) -> Tuple[List[Tuple[str, RGBColor]], bool]:
    """
    解析字符串为带颜色信息的片段列表。
    中括号及其内部内容使用 bracket_color。
    """
    segs: List[Tuple[str, RGBColor]] = []
    buf = ""
    for ch in s:
        if ch == "[" or ch == "【":
            if buf:
                segs.append((buf, bracket_color if in_bracket else color))
                buf = ""
            segs.append((ch, bracket_color))
            in_bracket = True
        elif ch == "]" or ch == "】":
            if buf:
                segs.append((buf, bracket_color))
                buf = ""
            segs.append((ch, bracket_color))
            in_bracket = False
        else:
            buf += ch
    if buf:
        segs.append((buf, bracket_color if in_bracket else color))
    return segs, in_bracket


def measure_block(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float,
) -> Tuple[int, int, int]:
    """
    测量文本块的宽度、高度和行高。

    :return: (最大宽度, 总高度, 行高)
    """
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * (1 + line_spacing))
    max_w = 0
    for ln in lines:
        max_w = max(max_w, int(draw.textlength(ln, font=font)))
    total_h = max(line_h * max(1, len(lines)), 1)
    return max_w, total_h, line_h


def draw_text_auto(
    image_source: Union[str, Image.Image],
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    text: str,
    color: RGBColor = (0, 0, 0),
    max_font_height: Optional[int] = None,
    font_path: Optional[str] = None,
    align: Align = "center",
    valign: VAlign = "middle",
    line_spacing: float = 0.15,
    bracket_color: RGBColor = (128, 0, 128),  # 中括号及内部内容颜色
    image_overlay: Union[str, Image.Image, None] = None,
) -> bytes:
    """
    在指定矩形内自适应字号绘制文本；
    中括号及括号内文字使用 bracket_color。
    """

    # --- 1. 打开图像 ---
    if isinstance(image_source, Image.Image):
        img = image_source.copy()
    else:
        img = Image.open(image_source).convert("RGBA")
    draw = ImageDraw.Draw(img)

    if image_overlay is not None:
        if isinstance(image_overlay, Image.Image):
            img_overlay = image_overlay.copy()
        else:
            img_overlay = (
                Image.open(image_overlay).convert("RGBA")
                if os.path.isfile(image_overlay)
                else None
            )
    else:
        img_overlay = None

    x1, y1 = top_left
    x2, y2 = bottom_right
    if not (x2 > x1 and y2 > y1):
        raise ValueError("无效的文字区域。")
    region_w, region_h = x2 - x1, y2 - y1

    # --- 2. 搜索最大字号 ---
    hi = min(region_h, max_font_height) if max_font_height else region_h
    lo, best_size, best_lines, best_line_h, best_block_h = 1, 0, [], 0, 0

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        lines = wrap_lines(draw, text, font, region_w)
        w, h, lh = measure_block(draw, lines, font, line_spacing)
        if w <= region_w and h <= region_h:
            best_size, best_lines, best_line_h, best_block_h = mid, lines, lh, h
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size == 0:
        font = _load_font(font_path, 1)
        best_lines = wrap_lines(draw, text, font, region_w)
        best_block_h, best_line_h = 1, 1
        best_size = 1
    else:
        font = _load_font(font_path, best_size)

    # --- 3. 垂直对齐 ---
    if valign == "top":
        y_start = y1
    elif valign == "middle":
        y_start = y1 + (region_h - best_block_h) // 2
    else:
        y_start = y2 - best_block_h

    # --- 4. 绘制 ---
    y = y_start
    in_bracket = False
    for ln in best_lines:
        line_w = int(draw.textlength(ln, font=font))
        if align == "left":
            x = x1
        elif align == "center":
            x = x1 + (region_w - line_w) // 2
        else:
            x = x2 - line_w
        segments, in_bracket = parse_color_segments(
            ln, in_bracket, bracket_color, color
        )
        for seg_text, seg_color in segments:
            if seg_text:
                draw.text((x, y), seg_text, font=font, fill=seg_color)
                x += int(draw.textlength(seg_text, font=font))
        y += best_line_h
        if y - y_start > region_h:
            break

    # 覆盖置顶图层（如果有）
    if image_overlay is not None and img_overlay is not None:
        img.paste(img_overlay, (0, 0), img_overlay)
    elif image_overlay is not None and img_overlay is None:
        print("Warning: overlay image is not exist.")

    # --- 5. 输出 PNG ---
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
