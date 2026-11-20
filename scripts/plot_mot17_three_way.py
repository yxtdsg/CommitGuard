import argparse
import colorsys
import csv
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results" / "paper" / "figures" / "mot17"
RUNS_ROOT = REPO / "runs"
IMAGE_ROOT = REPO / "external_inputs" / "images"
VIDEOS = ("02", "04", "05", "09", "10", "11", "13")
DEVELOPMENT = {"02", "04", "05"}
SEGMENT = 0
LOCAL_FRAME = 8
REPETITION = 1
PANEL_WIDTH = 960
FONT_REGULAR = os.environ.get("COMMITGUARD_FONT_REGULAR", font_manager.findfont("DejaVu Sans"))
FONT_BOLD = os.environ.get("COMMITGUARD_FONT_BOLD", font_manager.findfont("DejaVu Sans:bold"))


def track_color(track_id):
    hue = (track_id * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return tuple(round(channel * 255) for channel in rgb)


def load_frame_tracks(path, frame):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and int(row[0]) == frame:
                rows.append((int(row[1]), tuple(map(float, row[2:6]))))
    return sorted(rows, key=lambda item: item[0])


def centered_text(draw, center_x, y, text, font, fill=(20, 25, 32)):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def render_panel(image_path, tracks):
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        scale = PANEL_WIDTH / source.width
        panel_height = round(source.height * scale)
        panel = source.resize((PANEL_WIDTH, panel_height), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(panel)
    id_font = ImageFont.truetype(FONT_REGULAR, 16)
    line_width = 3
    for track_id, (x, y, width, height) in tracks:
        x1 = x * scale
        y1 = y * scale
        x2 = (x + width) * scale
        y2 = (y + height) * scale
        color = track_color(track_id)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = str(track_id)
        text_box = draw.textbbox((0, 0), label, font=id_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = min(max(0.0, x1), PANEL_WIDTH - text_width - 8.0)
        label_y = y1 - text_height - 9.0
        if label_y < 0:
            label_y = y1 + line_width
        label_y = min(max(0.0, label_y), panel_height - text_height - 7.0)
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 8, label_y + text_height + 7),
            fill=color,
        )
        draw.text((label_x + 4, label_y + 2), label, font=id_font, fill=(255, 255, 255))
    return panel


def source_paths(video):
    experiment = "mot17_development" if video in DEVELOPMENT else "mot17_additional"
    result_root = RUNS_ROOT / experiment
    metadata_path = result_root / f"c0_A{video}_r{REPETITION}" / "runtime.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    annotation = json.loads(Path(metadata["annotation"]).read_text(encoding="utf-8"))
    images = sorted(annotation["images"], key=lambda item: item["id"])
    image_info = images[SEGMENT * 30 + LOCAL_FRAME - 1]
    image_path = IMAGE_ROOT / image_info["file_name"]

    segment_name = f"MOT17-{video}-FRCNN-{SEGMENT:03d}.txt"
    runs = (
        ("(a) ByteTrack (clean)", f"c0_clean{video}_r{REPETITION}"),
        ("(b) ByteTrack under F&F attack", f"c0_A{video}_r{REPETITION}"),
        ("(c) CommitGuard-Review (ours)", f"e19_fi40_A{video}_r{REPETITION}"),
    )
    panels = [(label, result_root / run / segment_name) for label, run in runs]
    return image_path, image_info, panels


def make_figure(video):
    image_path, image_info, panel_specs = source_paths(video)
    tracks = [load_frame_tracks(path, LOCAL_FRAME) for _, path in panel_specs]
    panels = [render_panel(image_path, rows) for rows in tracks]

    margin = 24
    gap = 16
    header_height = 82
    caption_height = 58
    bottom_margin = 20
    panel_height = panels[0].height
    canvas_width = margin * 2 + PANEL_WIDTH * 3 + gap * 2
    canvas_height = header_height + panel_height + caption_height + bottom_margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = ImageFont.truetype(FONT_BOLD, 34)
    subtitle_font = ImageFont.truetype(FONT_REGULAR, 22)
    caption_font = ImageFont.truetype(FONT_REGULAR, 24)
    centered_text(draw, canvas_width / 2, 8, f"MOT17-{video}: normal, attacked, and defended tracking", title_font)
    source_frame = Path(image_info["file_name"]).stem
    subtitle = (
        f"Segment {SEGMENT:03d} | repetition {REPETITION} | local frame {LOCAL_FRAME} "
        f"(source frame {source_frame}) | F&F attack on local frames 6-8"
    )
    centered_text(draw, canvas_width / 2, 49, subtitle, subtitle_font, fill=(70, 78, 88))

    for index, ((label, _), panel, rows) in enumerate(zip(panel_specs, panels, tracks)):
        x = margin + index * (PANEL_WIDTH + gap)
        canvas.paste(panel, (x, header_height))
        draw.rectangle(
            (x, header_height, x + PANEL_WIDTH - 1, header_height + panel_height - 1),
            outline=(185, 190, 198),
            width=1,
        )
        centered_text(
            draw,
            x + PANEL_WIDTH / 2,
            header_height + panel_height + 13,
            f"{label} | {len(rows)} boxes",
            caption_font,
        )

    output_path = OUTPUT / f"MOT17-{video}_fnf_three_way.png"
    canvas.save(output_path, dpi=(300, 300), optimize=True)
    pdf_path = OUTPUT / f"MOT17-{video}_fnf_three_way.pdf"
    canvas.save(pdf_path, "PDF", resolution=300.0)
    print(f"{pdf_path} | clean/attack/defense boxes: {[len(rows) for rows in tracks]}")


def main():
    global OUTPUT, RUNS_ROOT, IMAGE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--videos", nargs="+", default=list(VIDEOS))
    args = parser.parse_args()
    RUNS_ROOT, IMAGE_ROOT, OUTPUT = args.runs_root, args.image_root, args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        if video not in VIDEOS:
            raise ValueError(f"Unsupported MOT17 video: {video}")
        make_figure(video)


if __name__ == "__main__":
    main()
