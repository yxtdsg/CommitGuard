import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from plot_mot17_three_way import (
    FONT_BOLD, FONT_REGULAR, PANEL_WIDTH, centered_text, load_frame_tracks, render_panel,
)

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results" / "paper" / "figures" / "dancetrack"
RESULTS = REPO / "runs" / "dancetrack_main"
ANNOTATIONS = REPO / "external_inputs" / "annotations" / "dancetrack_main"
IMAGE_ROOT = REPO / "external_inputs" / "images"
SEQUENCES = ("dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010")
SEGMENT = 0
LOCAL_FRAME = 8
REPETITION = 1


def source_paths(sequence):
    video = sequence.removeprefix("dancetrack")
    annotation = json.loads((ANNOTATIONS / f"clean4_{sequence}.json").read_text(encoding="utf-8"))
    images = sorted(annotation["images"], key=lambda item: item["id"])
    image_info = images[SEGMENT * 30 + LOCAL_FRAME - 1]
    image_path = IMAGE_ROOT / image_info["file_name"]
    segment_name = f"{sequence}-{SEGMENT:03d}.txt"
    runs = (
        ("(a) ByteTrack (clean)", f"c0_clean{video}_r{REPETITION}"),
        ("(b) ByteTrack under F&F attack", f"c0_A{video}_r{REPETITION}"),
        ("(c) CommitGuard-Review (ours)", f"e19_fi40_A{video}_r{REPETITION}"),
    )
    panels = [(label, RESULTS / run / segment_name) for label, run in runs]
    return image_path, image_info, panels


def make_figure(sequence):
    image_path, image_info, panel_specs = source_paths(sequence)
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
    centered_text(draw, canvas_width / 2, 8, f"{sequence}: normal, attacked, and defended tracking", title_font)
    source_frame = Path(image_info["file_name"]).stem
    subtitle = (
        f"E26 frozen protocol | segment {SEGMENT:03d} | repetition {REPETITION} | "
        f"local frame {LOCAL_FRAME} (source frame {source_frame}) | F&F attack on local frames 6-8"
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

    output_path = OUTPUT / f"{sequence}_fnf_three_way.pdf"
    canvas.save(output_path, "PDF", resolution=300.0)
    print(f"{output_path} | clean/attack/defense boxes: {[len(rows) for rows in tracks]}")


def main():
    global OUTPUT, RESULTS, ANNOTATIONS, IMAGE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RESULTS)
    parser.add_argument("--input-root", type=Path, default=ANNOTATIONS.parent)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    RESULTS, IMAGE_ROOT, OUTPUT = args.runs_root, args.image_root, args.output_dir
    ANNOTATIONS = args.input_root / "dancetrack_main"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for sequence in args.sequences:
        make_figure(sequence)


if __name__ == "__main__":
    main()
