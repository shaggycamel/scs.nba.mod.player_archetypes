"""Minimal inline-SVG chart helpers.

Marks are thin, grids are hairlines, fills carry 4px rounded data-ends anchored
to the baseline, and every chart ships a hover tooltip plus a table view. Colors
are referenced as CSS custom properties so light/dark swap in one place.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

MUTED = "var(--muted)"
GRID = "var(--gridline)"
AXIS = "var(--baseline)"
INK = "var(--text-primary)"
INK2 = "var(--text-secondary)"


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def fmt(v: float, dp: int = 3) -> str:
    return f"{v:,.{dp}f}"


@dataclass
class Box:
    w: int = 760
    h: int = 320
    l: int = 64
    r: int = 20
    t: int = 16
    b: int = 44

    @property
    def iw(self) -> int:
        return self.w - self.l - self.r

    @property
    def ih(self) -> int:
        return self.h - self.t - self.b


def _open(box: Box, label: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {box.w} {box.h}" role="img" '
        f'style="max-width:{box.w}px" '
        f'aria-label="{esc(label)}" preserveAspectRatio="xMidYMid meet">'
    )


def _yticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi == lo:
        hi = lo + 1
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def table_view(headers: list[str], rows: list[list[object]], caption: str) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return (
        f'<details class="tableview"><summary>Table view &mdash; {esc(caption)}</summary>'
        f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
        f"<tbody>{tr}</tbody></table></div></details>"
    )


def line_chart(
    xs: list[float],
    ys: list[float],
    *,
    label: str,
    x_label: str,
    y_label: str,
    highlight: float | None = None,
    highlight_note: str = "",
    box: Box | None = None,
    y_dp: int = 2,
) -> str:
    """Single-series line. No legend: the title names the series."""
    box = box or Box()
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.12 or 1
    lo, hi = lo - pad, hi + pad
    xlo, xhi = min(xs), max(xs)

    def px(x: float) -> float:
        return box.l + (x - xlo) / ((xhi - xlo) or 1) * box.iw

    def py(y: float) -> float:
        return box.t + box.ih - (y - lo) / ((hi - lo) or 1) * box.ih

    out = [_open(box, label)]
    for t in _yticks(lo, hi):
        y = py(t)
        out.append(
            f'<line x1="{box.l}" y1="{y:.1f}" x2="{box.l + box.iw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{box.l - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt(t, y_dp)}</text>'
        )
    out.append(
        f'<line x1="{box.l}" y1="{box.t + box.ih}" x2="{box.l + box.iw}" y2="{box.t + box.ih}" stroke="{AXIS}" stroke-width="1"/>'
    )
    for x in xs:
        out.append(
            f'<text x="{px(x):.1f}" y="{box.t + box.ih + 20}" class="tick" text-anchor="middle">{int(x)}</text>'
        )

    if highlight is not None:
        hx = px(highlight)
        out.append(
            f'<line x1="{hx:.1f}" y1="{box.t}" x2="{hx:.1f}" y2="{box.t + box.ih}" stroke="var(--series-2)" stroke-width="2"/>'
        )
        if highlight_note:
            anchor = "end" if hx > box.l + box.iw * 0.6 else "start"
            dx = -8 if anchor == "end" else 8
            out.append(
                f'<text x="{hx + dx:.1f}" y="{box.t + 14}" class="annot" text-anchor="{anchor}">{esc(highlight_note)}</text>'
            )

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    out.append(
        f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>'
    )
    for x, y in zip(xs, ys):
        out.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.5" fill="var(--series-1)" '
            f'stroke="var(--surface-1)" stroke-width="2" class="hit" '
            f'data-tip="{esc(x_label)} {int(x)} &middot; {esc(y_label)} {fmt(y, 4)}"><title>{esc(x_label)} {int(x)}: {fmt(y, 4)}</title></circle>'
        )
    out.append(
        f'<text x="{box.l + box.iw / 2:.0f}" y="{box.h - 6}" class="axislabel" text-anchor="middle">{esc(x_label)}</text>'
    )
    out.append("</svg>")
    return "".join(out)


def hbar_chart(
    labels: list[str],
    values: list[float],
    *,
    label: str,
    value_label: str,
    groups: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    dp: int = 4,
    row_h: int = 26,
    width: int = 760,
    label_w: int = 250,
) -> str:
    """Horizontal bars with direct value labels. `groups` adds a legend."""
    h = len(labels) * row_h + 34
    box = Box(w=width, h=h, l=label_w, r=86, t=8, b=26)
    vmax = max(values) * 1.02 or 1
    out = [_open(box, label)]
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = box.t + i * row_h
        colour = "var(--series-1)"
        if groups and group_colors:
            colour = group_colors.get(groups[i], colour)
        bw = max(val / vmax * box.iw, 1.5)
        out.append(
            f'<text x="{box.l - 10}" y="{y + row_h / 2 + 4:.1f}" class="catlabel" text-anchor="end">{esc(lab)}</text>'
        )
        out.append(
            f'<rect x="{box.l}" y="{y + 4:.1f}" width="{bw:.1f}" height="{row_h - 10}" rx="4" ry="4" '
            f'fill="{colour}" class="hit" data-tip="{esc(lab)} &middot; {esc(value_label)} {fmt(val, dp)}">'
            f"<title>{esc(lab)}: {fmt(val, dp)}</title></rect>"
        )
        out.append(
            f'<text x="{box.l + bw + 8:.1f}" y="{y + row_h / 2 + 4:.1f}" class="vallabel">{fmt(val, dp)}</text>'
        )
    out.append(
        f'<line x1="{box.l}" y1="{box.t}" x2="{box.l}" y2="{box.t + len(labels) * row_h}" stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append("</svg>")
    legend = ""
    if groups and group_colors:
        seen = [g for g in dict.fromkeys(groups)]
        chips = "".join(
            f'<span class="chip"><i style="background:{group_colors[g]}"></i>{esc(g)}</span>'
            for g in seen
        )
        legend = f'<div class="legend">{chips}</div>'
    return legend + "".join(out)


def heatmap(
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    *,
    label: str,
    diverging: bool = True,
    cell: int = 34,
    label_w: int = 190,
    value_dp: int = 2,
    cell_dp: int | None = None,
    vmax: float | None = None,
    unit: str = "",
) -> str:
    """Cells on a diverging (signed) or sequential (magnitude) ramp.

    Column labels sit above the grid and the scale legend below it, so neither
    can collide with the other. A 2px surface gap separates cells rather than a
    border, and a scale legend is always drawn because color carries magnitude.
    """
    # Rotated headers extend up and to the right, so both paddings scale with
    # the longest label or the last column clips off the edge.
    reach = 0.707 * 6.0 * max((len(c) for c in col_labels), default=0)
    if cell_dp is None:
        cell_dp = 1 if diverging else value_dp
    top = int(24 + reach)
    grid_h = len(row_labels) * cell
    legend_y = top + grid_h + 26
    w = int(label_w + len(col_labels) * cell + reach + 24)
    h = legend_y + 46
    out = [
        (
            f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
            f'style="max-width:{w}px" '
            f'aria-label="{esc(label)}" preserveAspectRatio="xMidYMid meet">'
        )
    ]
    flat = [v for row in matrix for v in row if v is not None]
    # A single extreme cell otherwise flattens every other cell to near-neutral,
    # so the ramp saturates at `vmax` and outliers simply clip.
    vmax = vmax or (max(abs(min(flat)), abs(max(flat))) or 1)
    smax = max(flat) or 1

    def colour(v: float) -> str:
        if diverging:
            t = max(-1.0, min(1.0, v / vmax))
            pole = "var(--pole-pos)" if t >= 0 else "var(--pole-neg)"
            return f"color-mix(in oklab, {pole} {abs(t) * 100:.0f}%, var(--mid-neutral))"
        t = max(0.0, min(1.0, v / smax))
        return f"color-mix(in oklab, var(--seq-hi) {t * 100:.0f}%, var(--seq-lo))"

    for j, cl in enumerate(col_labels):
        x = label_w + j * cell + cell / 2
        y = top - 8
        out.append(
            f'<text x="{x:.1f}" y="{y}" class="tick" text-anchor="start" '
            f'transform="rotate(-45 {x:.1f} {y})">{esc(cl)}</text>'
        )
    for i, rlab in enumerate(row_labels):
        y = top + i * cell
        out.append(
            f'<text x="{label_w - 10}" y="{y + cell / 2 + 4:.1f}" class="catlabel" '
            f'text-anchor="end">{esc(rlab)}</text>'
        )
        for j, v in enumerate(matrix[i]):
            if v is None:
                continue
            x = label_w + j * cell
            out.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" '
                f'rx="3" ry="3" fill="{colour(v)}" class="hit" '
                f'data-tip="{esc(rlab)} &middot; {esc(col_labels[j])} &middot; '
                f'{fmt(v, value_dp)}{esc(unit)}">'
                f"<title>{esc(rlab)} / {esc(col_labels[j])}: {fmt(v, value_dp)}{esc(unit)}</title></rect>"
            )
            strong = abs(v) >= 0.72 * vmax if diverging else v >= 0.62 * smax
            if strong:
                out.append(
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                    f'class="cellval" text-anchor="middle">'
                    f"{fmt(v, cell_dp)}</text>"
                )

    steps = 9
    sw = 20
    out.append(
        f'<text x="{label_w - 10}" y="{legend_y + 11}" class="tick" text-anchor="end">scale</text>'
    )
    for s in range(steps):
        v = (-vmax + 2 * vmax * s / (steps - 1)) if diverging else (smax * s / (steps - 1))
        out.append(
            f'<rect x="{label_w + s * (sw + 2)}" y="{legend_y}" width="{sw}" height="13" '
            f'rx="2" fill="{colour(v)}"/>'
        )
    lo_txt = f"{-vmax:.1f}{unit}" if diverging else f"0{unit}"
    end_x = label_w + steps * (sw + 2) - 2
    out.append(f'<text x="{label_w}" y="{legend_y + 30}" class="tick">{esc(lo_txt)}</text>')
    out.append(
        f'<text x="{end_x}" y="{legend_y + 30}" class="tick" text-anchor="end">'
        f"{vmax:.1f}{esc(unit)}</text>"
    )
    out.append("</svg>")
    return "".join(out)


def grouped_hbar(
    categories: list[str],
    series: dict[str, list[float]],
    *,
    label: str,
    value_label: str,
    colors: dict[str, str],
    dp: int = 3,
    bar_h: int = 13,
    gap: int = 2,
    group_gap: int = 12,
    width: int = 760,
    label_w: int = 190,
) -> str:
    """Grouped horizontal bars: one label per category, one bar per series.

    Bars within a group are separated by a 2px surface gap rather than a border,
    and each is direct-labelled, so the comparison reads without a lookup.
    """
    names = list(series)
    group_h = len(names) * bar_h + (len(names) - 1) * gap + group_gap
    h = len(categories) * group_h + 10
    box = Box(w=width, h=h, l=label_w, r=78, t=6, b=4)
    vmax = max((max(v) for v in series.values()), default=1) * 1.04 or 1

    out = [
        (
            f'<svg class="chart" viewBox="0 0 {box.w} {box.h}" role="img" '
            f'style="max-width:{box.w}px" aria-label="{esc(label)}" '
            f'preserveAspectRatio="xMidYMid meet">'
        )
    ]
    for i, cat in enumerate(categories):
        gy = box.t + i * group_h
        block = len(names) * bar_h + (len(names) - 1) * gap
        out.append(
            f'<text x="{box.l - 10}" y="{gy + block / 2 + 4:.1f}" class="catlabel" '
            f'text-anchor="end">{esc(cat)}</text>'
        )
        for j, nm in enumerate(names):
            v = series[nm][i]
            y = gy + j * (bar_h + gap)
            bw = max(v / vmax * box.iw, 1.0)
            out.append(
                f'<rect x="{box.l}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="4" ry="4" '
                f'fill="{colors[nm]}" class="hit" '
                f'data-tip="{esc(cat)} &middot; {esc(nm)} &middot; {esc(value_label)} {fmt(v, dp)}">'
                f"<title>{esc(cat)} / {esc(nm)}: {fmt(v, dp)}</title></rect>"
            )
            out.append(
                f'<text x="{box.l + bw + 7:.1f}" y="{y + bar_h - 2:.1f}" '
                f'class="vallabel">{fmt(v, dp)}</text>'
            )
    out.append(
        f'<line x1="{box.l}" y1="{box.t}" x2="{box.l}" '
        f'y2="{box.t + len(categories) * group_h - group_gap:.1f}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append("</svg>")
    chips = "".join(
        f'<span class="chip"><i style="background:{colors[n]}"></i>{esc(n)}</span>' for n in names
    )
    return f'<div class="legend">{chips}</div>' + "".join(out)
