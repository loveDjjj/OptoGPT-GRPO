from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _save_image(image: Image.Image, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def select_sample_plot_rows(
    rows: Sequence[dict],
    worst_count: int,
    random_count: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    valid_rows = [
        row
        for row in rows
        if row.get("generated_valid") and row.get("spectrum_rmse") is not None
    ]
    worst_count = max(0, int(worst_count))
    random_count = max(0, int(random_count))
    worst_rows = sorted(valid_rows, key=lambda row: float(row["spectrum_rmse"]), reverse=True)[:worst_count]

    worst_ids = {row.get("sample_id") for row in worst_rows}
    remaining_rows = [row for row in valid_rows if row.get("sample_id") not in worst_ids]
    rng = random.Random(seed)
    if random_count >= len(remaining_rows):
        random_rows = list(remaining_rows)
    else:
        random_rows = rng.sample(remaining_rows, k=random_count)
    return {"worst": worst_rows, "random": random_rows}


def plot_metric_histogram(
    values: Iterable[float],
    title: str,
    xlabel: str,
    output_path: str | Path,
) -> None:
    numeric_values = []
    for value in values:
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            numeric_values.append(numeric_value)
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 60, 80
    chart_w = width - left - right
    chart_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _get_font()

    draw.text((left, 18), title, fill="black", font=font)

    if numeric_values:
        min_value = min(numeric_values)
        max_value = max(numeric_values)
        if min_value == max_value:
            min_value -= 0.5
            max_value += 0.5
        bin_count = min(20, max(1, len(numeric_values)))
        bin_width = (max_value - min_value) / bin_count
        counts = [0] * bin_count
        for value in numeric_values:
            idx = int((value - min_value) / bin_width) if bin_width > 0 else 0
            if idx >= bin_count:
                idx = bin_count - 1
            counts[idx] += 1
        max_count = max(counts) if counts else 1
        bar_gap = 4
        bar_width = max(1, (chart_w - bar_gap * (bin_count - 1)) // bin_count)

        origin_x = left
        origin_y = top + chart_h
        draw.line((origin_x, top, origin_x, origin_y), fill="#1f1f1f", width=2)
        draw.line((origin_x, origin_y, origin_x + chart_w, origin_y), fill="#1f1f1f", width=2)

        for idx, count in enumerate(counts):
            bar_h = int((count / max_count) * (chart_h - 10)) if max_count else 0
            x0 = origin_x + idx * (bar_width + bar_gap)
            y0 = origin_y - bar_h
            x1 = x0 + bar_width
            y1 = origin_y
            draw.rectangle((x0, y0, x1, y1), fill="#2a6f97", outline="#1f1f1f")

        axis_label = f"{xlabel}  n={len(numeric_values)}"
        label_w, label_h = _measure_text(draw, axis_label, font)
        draw.text((left + (chart_w - label_w) / 2, height - bottom + 22), axis_label, fill="black", font=font)
        min_text = f"{min_value:.4f}"
        max_text = f"{max_value:.4f}"
        draw.text((left, origin_y + 6), min_text, fill="black", font=font)
        max_w, _ = _measure_text(draw, max_text, font)
        draw.text((left + chart_w - max_w, origin_y + 6), max_text, fill="black", font=font)
    else:
        empty_text = "no finite values to plot"
        text_w, text_h = _measure_text(draw, empty_text, font)
        draw.text(
            (left + (chart_w - text_w) / 2, top + (chart_h - text_h) / 2),
            empty_text,
            fill="black",
            font=font,
        )

    _save_image(image, output_path)


def _draw_series(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    *,
    chart_box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = chart_box
    width = right - left
    height = bottom - top
    if values.size == 0:
        return
    x_step = width / max(1, values.size - 1)
    points = []
    for idx, value in enumerate(values):
        x = left + idx * x_step
        y = bottom - float(value) * height
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    else:
        x, y = points[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    text_w, text_h = _measure_text(draw, label, font)
    draw.rectangle((left + 8, top + 8, left + 20, top + 20), fill=color)
    draw.text((left + 28, top + 6), label, fill="black", font=font)


def plot_sample_spectrum(
    row: dict,
    output_path: str | Path,
    num_points: int,
) -> None:
    if num_points < 0:
        raise ValueError("num_points must be non-negative")

    target_spectrum = np.asarray(row["target_spectrum_rt"], dtype=np.float32).reshape(-1)
    predicted_spectrum = np.asarray(row["predicted_spectrum_rt"], dtype=np.float32).reshape(-1)
    expected_length = 2 * int(num_points)
    if target_spectrum.size != expected_length:
        raise ValueError(
            f"target_spectrum_rt must contain exactly {expected_length} values; got {target_spectrum.size}"
        )
    if predicted_spectrum.size != expected_length:
        raise ValueError(
            f"predicted_spectrum_rt must contain exactly {expected_length} values; got {predicted_spectrum.size}"
        )
    target_r = target_spectrum[:num_points]
    target_t = target_spectrum[num_points : num_points * 2]
    pred_r = predicted_spectrum[:num_points]
    pred_t = predicted_spectrum[num_points : num_points * 2]

    width, height = 1320, 600
    left, right, top, bottom = 80, 40, 110, 70
    legend_width = 210
    chart_box = (left, top, width - right - legend_width, height - bottom)
    legend_left = width - right - legend_width + 20
    legend_top = top + 8

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _get_font()

    title = (
        f"sample={row.get('sample_id')} | target={row.get('target_layer_count')} | "
        f"pred={row.get('prediction_layer_count')} | exact={row.get('token_exact_match')} | "
        f"rmse={row.get('spectrum_rmse')}"
    )
    draw.text((left, 18), title, fill="black", font=font)

    origin_left, origin_top, origin_right, origin_bottom = chart_box
    draw.rectangle(chart_box, outline="#1f1f1f", width=2)
    draw.line((origin_left, origin_bottom, origin_right, origin_bottom), fill="#1f1f1f", width=1)
    draw.line((origin_left, origin_top, origin_left, origin_bottom), fill="#1f1f1f", width=1)

    for label, values, color in (
        ("target_R", target_r, (31, 31, 31)),
        ("pred_R", pred_r, (214, 40, 40)),
        ("target_T", target_t, (29, 78, 216)),
        ("pred_T", pred_t, (22, 163, 74)),
    ):
        _draw_series(draw, np.asarray(values, dtype=np.float32), chart_box=chart_box, color=color, label=label, font=font)

    legend_items = [
        ("target_R", (31, 31, 31)),
        ("pred_R", (214, 40, 40)),
        ("target_T", (29, 78, 216)),
        ("pred_T", (22, 163, 74)),
    ]
    draw.rectangle(
        (width - right - legend_width, top, width - right, top + 120),
        outline="#1f1f1f",
        width=1,
    )
    draw.text((legend_left, legend_top), "Legend", fill="black", font=font)
    for index, (legend_label, color) in enumerate(legend_items, start=1):
        row_top = legend_top + 18 * index
        draw.rectangle((legend_left, row_top + 2, legend_left + 14, row_top + 16), fill=color, outline=color)
        draw.text((legend_left + 22, row_top), legend_label, fill="black", font=font)

    x_label = "wavelength index"
    label_w, _ = _measure_text(draw, x_label, font)
    draw.text((origin_left + (origin_right - origin_left - label_w) / 2, height - bottom + 18), x_label, fill="black", font=font)
    draw.text((18, origin_top + 10), "value", fill="black", font=font)

    _save_image(image, output_path)
