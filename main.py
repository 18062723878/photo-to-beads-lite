# -*- coding: utf-8 -*-
"""照片转拼豆图纸 —— 主程序 (全能优化版)
包含：Lab色彩匹配、防内存溢出、多格式导出(PNG/PDF/CSV)、专业界面、异常捕获。
"""
import csv
import io
import os
import tempfile
import traceback
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE_PATH = os.path.join(HERE, "palette.csv")

# --------------------------------------------------------------------------- #
# 颜色空间转换：sRGB(0-255) -> CIE Lab
# 注意：必须在 load_palette 之前定义，因为 load_palette 内部会调用它。
# --------------------------------------------------------------------------- #
def srgb_to_lab(rgb):
    """rgb: (...,3) float 0-255 -> Lab (...,3)。D65 白点。"""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    mask = c > 0.04045
    # IEC 61966-2-1 标准：偏移量 0.055（0.04045 是分段阈值，不是偏移）
    lin = np.where(mask, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    x = lin[..., 0] * 0.4124 + lin[..., 1] * 0.3576 + lin[..., 2] * 0.1805
    y = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    z = lin[..., 0] * 0.0193 + lin[..., 1] * 0.1192 + lin[..., 2] * 0.9505

    x /= 0.95047  # D65 白点
    z /= 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

# --------------------------------------------------------------------------- #
# 色卡加载与全局缓存 (修复：只在启动时加载一次，提升后续生成速度)
# --------------------------------------------------------------------------- #
def load_palette(path=PALETTE_PATH):
    """读取色卡 CSV，返回 (ids, names, rgb_array, lab_array)。"""
    ids, names, rgbs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["color_id"])
            names.append(row["color_name"])
            rgbs.append([int(row["r"]), int(row["g"]), int(row["b"])])

    rgb_array = np.array(rgbs, dtype=np.float64)
    lab_array = srgb_to_lab(rgb_array) # 提前计算好色卡的 Lab 值
    return ids, names, rgb_array, lab_array

# 全局加载，避免每次点击都读文件
try:
    PALETTE_IDS, PALETTE_NAMES, PALETTE_RGB, PALETTE_LAB = load_palette()
    if len(PALETTE_IDS) == 0:
        raise ValueError("色卡数据为空")
    print("色卡加载成功：" + str(len(PALETTE_IDS)) + " 种颜色")
except Exception as e:
    print("严重错误：无法加载色卡文件 " + PALETTE_PATH + " (" + str(e) + ")，请检查文件是否存在且格式正确！")
    PALETTE_IDS, PALETTE_NAMES, PALETTE_RGB, PALETTE_LAB = [], [], np.array([]), np.array([])

# --------------------------------------------------------------------------- #
# 核心：图片 -> 拼豆图纸
# --------------------------------------------------------------------------- #
def image_to_beads(img, board_width):
    """返回 (grid_ids 二维列表, preview_rgb (H,W,3) uint8)。"""
    img = img.convert("RGB")
    w, h = img.size
    
    # 等比缩放
    scale = board_width / w
    board_height = max(1, round(h * scale))
    small = img.resize((board_width, board_height), Image.LANCZOS)
    
    pixels = np.asarray(small, dtype=np.float64)          # (H,W,3)
    flat = pixels.reshape(-1, 3)                           # (N,3)
    
    # 在 Lab 空间做最近邻匹配 (优化：分批计算防止大图内存溢出)
    lab_pix = srgb_to_lab(flat)
    
    # 分批处理，每批 2000 个像素，极大降低内存峰值
    batch_size = 2000
    n_pixels = lab_pix.shape[0]
    idx = np.empty(n_pixels, dtype=np.int32)
    
    for i in range(0, n_pixels, batch_size):
        batch = lab_pix[i:i+batch_size]
        # 计算当前批次像素与所有色卡的距离
        dists = np.linalg.norm(batch[:, None, :] - PALETTE_LAB[None, :, :], axis=2)
        idx[i:i+batch_size] = dists.argmin(axis=1)
    
    mapped_rgb = PALETTE_RGB[idx]                          # (N,3)
    preview = mapped_rgb.reshape(board_height, board_width, 3).astype(np.uint8)
    grid_ids = np.array(PALETTE_IDS)[idx].reshape(board_height, board_width)
    
    return grid_ids.tolist(), preview

def render_preview(preview_rgb, bead_px=18):
    """把每颗豆放大成 bead_px 方块，保持像素感。"""
    h, w, _ = preview_rgb.shape
    img = Image.fromarray(preview_rgb, "RGB")
    return img.resize((w * bead_px, h * bead_px), Image.NEAREST)

# --------------------------------------------------------------------------- #
# 增加：生成带网格的 PNG 和 PDF 图纸
# --------------------------------------------------------------------------- #
def generate_export_files(grid_ids, bead_px=18):
    """根据网格数据生成 PNG 和 PDF 文件"""
    h, w = len(grid_ids), len(grid_ids[0])
    cell_size = bead_px
    img_w, img_h = w * cell_size, h * cell_size
    
    # 创建一个纯白背景
    img = Image.new('RGB', (img_w, img_h), color='white')
    draw = ImageDraw.Draw(img)
    
    # 画网格线和色块
    for r in range(h):
        for c in range(w):
            cid = grid_ids[r][c]
            # 查找颜色对应的 RGB
            try:
                color_idx = PALETTE_IDS.index(cid)
                rgb = tuple(int(v) for v in PALETTE_RGB[color_idx])
            except ValueError:
                rgb = (128, 128, 128) # 找不到颜色就用灰色
            
            # 画填充色块
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=rgb, outline='gray', width=1)

    # 导出 PNG
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)
    
    # 导出 PDF (将图片放入 A4 纸张)
    pdf_buffer = io.BytesIO()
    # 转换为 RGB 以防报错
    rgb_img = img.convert("RGB")
    rgb_img.save(pdf_buffer, format="PDF", resolution=300.0)
    pdf_buffer.seek(0)
    
    return png_buffer, pdf_buffer

# --------------------------------------------------------------------------- #
# Gradio 处理函数 (修复：增加全局异常捕获)
# --------------------------------------------------------------------------- #
def process(image, board_width, bead_px):
    if image is None:
        return None, "⚠️ 请先上传一张图片。", None, None, None
        
    if len(PALETTE_IDS) == 0:
        return None, "❌ 致命错误：未找到 palette.csv 或色卡数据为空，请检查文件！", None, None, None

    try:
        board_width = int(board_width)
        bead_px = int(bead_px)
        
        grid_ids, preview = image_to_beads(image, board_width)
        preview_img = render_preview(preview, bead_px)
        
        # 用色统计
        flat_ids = [c for row in grid_ids for c in row]
        total = len(flat_ids)
        name_of = dict(zip(PALETTE_IDS, PALETTE_NAMES))
        rgb_of = {cid: tuple(int(v) for v in PALETTE_RGB[i])
                  for i, cid in enumerate(PALETTE_IDS)}
                  
        uniq, counts = np.unique(flat_ids, return_counts=True)
        order = np.argsort(-counts)
        
        lines = [f"📊 豆板尺寸：{board_width} × {len(grid_ids)}，共 {total} 颗豆，"
                 f"使用 {len(uniq)} 种颜色\n"]
        lines.append(f"{'色号':<6}{'颜色名':<16}{'用量':>6}  hex")
        lines.append("-" * 44)
        
        for i in order:
            cid = uniq[i]
            cnt = int(counts[i])
            r, g, b = rgb_of[cid]
            lines.append(f"{cid:<6}{name_of[cid]:<16}{cnt:>6}  #{r:02X}{g:02X}{b:02X}")
            
        usage_text = "\n".join(lines)
        
        # 导出图纸网格 CSV
        tmp_csv = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        with tmp_csv as f:
            w = csv.writer(f)
            for row in grid_ids:
                w.writerow(row)
        csv_path = tmp_csv.name
        
        # 生成并保存 PNG 和 PDF
        png_buf, pdf_buf = generate_export_files(grid_ids, bead_px)
        png_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        with open(png_path, "wb") as f: f.write(png_buf.read())
        with open(pdf_path, "wb") as f: f.write(pdf_buf.read())
        
        return preview_img, usage_text, png_path, pdf_path, csv_path
        
    except Exception as e:
        # 捕获所有未知错误，防止界面崩溃
        error_msg = f"❌ 转换过程中发生错误：\n{str(e)}\n\n详细追踪:\n{traceback.format_exc()}"
        return None, error_msg, None, None, None

# --------------------------------------------------------------------------- #
# 界面 (优化版：增加了格式说明和美观排版)
# --------------------------------------------------------------------------- #
def build_ui():
    # 设置界面主题，让它看起来更现代
    theme = gr.themes.Soft(primary_hue="orange", secondary_hue="blue")

    with gr.Blocks(title="照片转拼豆图纸") as demo:
        gr.Markdown("# 🟫 照片转拼豆图纸生成器\n"
                    "上传照片，自动匹配 Artkal 色卡，一键生成专业拼豆图纸！")
                    
        with gr.Row():
            # ----------------- 左侧：上传与设置 -----------------
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ 上传与设置")
                img_in = gr.Image(label="上传照片", type="pil", height=300)
                board_width = gr.Slider(16, 160, value=48, step=1,
                                        label="豆板宽度（颗数）", 
                                        info="数值越大，图纸越精细，但拼起来越耗时")
                bead_px = gr.Slider(8, 40, value=18, step=1,
                                    label="预览每豆像素（放大倍数）",
                                    info="仅影响网页上的预览大小，不影响下载的文件")
                btn = gr.Button("✨ 生成图纸", variant="primary", size="lg")
            
            # ----------------- 右侧：预览与统计 -----------------
            with gr.Column(scale=2):
                gr.Markdown("### 2️⃣ 图纸预览与用色统计")
                img_out = gr.Image(label="图纸效果预览", height=350)
                usage_out = gr.Textbox(label="📊 用色统计清单", lines=14, 
                                       interactive=False,
                                       info="这里列出了需要用到的所有颜色、色号和具体颗数")
        
        # ----------------- 底部：下载中心 (增加了详细说明) -----------------
        gr.Markdown("---")
        gr.Markdown("### 3️⃣ 下载中心 (请根据需要选择格式)")
        
        with gr.Row():
            # PNG 下载区
            with gr.Column():
                png_out = gr.File(label="📥 下载拼豆图纸 (PNG 图片)")
                gr.Markdown(
                    "**💡 适用场景：**\n"
                    "* 适合保存在**手机或平板**上，随时放大缩小核对细节。\n"
                    "* 适合直接发到**微信群或朋友圈**，和拼友分享交流图纸。"
                )
            
            # PDF 下载区
            with gr.Column():
                pdf_out = gr.File(label="📥 下载拼豆图纸 (PDF 文档)")
                gr.Markdown(
                    "**💡 适用场景：**\n"
                    "* 适合去打印店用 **A4 纸打印**出来。\n"
                    "* 适合**看着纸质图纸**一边拼一边打勾，保护眼睛，不容易拼错行。"
                )
                
            # CSV 下载区
            with gr.Column():
                csv_out = gr.File(label="📥 下载色号网格 (CSV 表格)")
                gr.Markdown(
                    "**💡 适用场景：**\n"
                    "* 可以用 **Excel 或 WPS** 打开，方便自己手动修改某个格子的颜色。\n"
                    "* 适合做二次统计，或者导入到其他专业的拼豆软件中使用。"
                )
                
        # 绑定按钮点击事件
        btn.click(process,
                  inputs=[img_in, board_width, bead_px],
                  outputs=[img_out, usage_out, png_out, pdf_out, csv_out])
                  
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()