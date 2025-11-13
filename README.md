# 安安素描本插件

一个专注于“自我表达”的 MaiBot 插件：帮助麦麦把任意想说的话与可选图片绘制到《安安的素描本》风格的底图上，帮助你用差分表情直观展示情绪。

## 功能亮点
- **心情差分**：配置多个底图（如普通、开心、脸红等），由 LLM 选择合适的标签来切换表情。
- **文字排版**：自动根据素描本文本框尺寸调整字号和换行，保留 `[]` / `【】` 的紫色强调效果。
- **图文合成**：支持只写文字、只贴图片，或文字+图片组合（会根据横竖图动态分区）。
- **自动取图**：当 Action 没提供图片时，可按配置自动尝试提取最近一条消息中的图片（可关闭）。

## 使用方式
1. 安装后先运行麦麦，将自动生成插件文件
2. 在插件管理中启用 `anan_sketchbook_plugin`。
3. 确认 `plugins/anan_sketchbook_plugin/assets` 目录包含 `BaseImages` 差分底图和 `font.ttf` 字体。
4. 可选修改配置：
   - `assets_root`：素材根目录（默认 `assets`）。
   - `font_file`：字体路径。
   - `base_overlay_file`：可选遮罩层。
   - `text_box_topleft` / `image_box_bottomright`：文字/图片区域坐标。
   - `padding` / `max_font_height` / `line_spacing` 等排版参数。
   - 若要自定义差分，把 `sketchbook.mood_mapping` / `sketchbook.mood_aliases` 手动写入 `config.toml`。
4. LLM Planner 会在需要表达心情时选择 `anan_sketchbook_render` Action，传入 `text`、`mood` 及可选 `image_base64` 等字段。

## 依赖需求
- Python 3.11+
- Pillow >= 12.0.0（用于图片处理）

插件来源：https://github.com/MarkCup-Official/Anan-s-Sketchbook-Chat-Box。 好用请点star~

其他依赖由 MaiBot 主程序提供，无需额外安装。
