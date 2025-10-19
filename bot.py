#!/usr/bin/env python3
# bot.py
import time
import itertools
import math

# Use your existing detector as a module
import scan_shape_shade as D

# ==================== BOT CONFIG ====================
# Tap cadence and rescan delays
TAP_INTERVAL_S        = 0.06   # delay between taps within a set
POST_SET_PAUSE_S      = 0.5   # wait after tapping a set before rescanning
PRE_START_PAUSE_S     = 0.6   # wait after tapping START_BTN

# Acceptance rules
MIN_CELL_SCORE        = 0.6   # ignore very uncertain cells
PENALIZE_LOWCONF      = 0.02   # penalty to set quality per lowconf cell

# Set selection strategy
# "max_min": maximize the minimum score across the 3 cells (robust)
# "max_sum": maximize the sum of scores (greedy overall confidence)
SELECT_STRATEGY       = "max_min"

# Stop condition: "accuracy drop" heuristic
DROP_WINDOW           = 3      # how many recent rounds are averaged
DROP_FACTOR           = 0.60   # stop if current avg < DROP_FACTOR * prior avg
MIN_ROUNDS_FOR_DROP   = 5      # don't enforce drop rule before this many rounds

# Safety limits
MAX_ROUNDS            = 200    # hard cap
# ====================================================


def center_of_box(box):
    x, y, w, h = box
    return int(x + w/2), int(y + h/2)


def attributes_all_same_or_all_diff(vals):
    """True if all values are the same OR all different."""
    s = set(vals)
    return len(s) == 1 or len(s) == len(vals)


def is_valid_set(cells, idx_triple):
    """
    cells: list of dict with keys: shape, shade, color, score
    idx_triple: (i, j, k)
    """
    i, j, k = idx_triple
    a = cells[i]; b = cells[j]; c = cells[k]

    # Must be reasonably confident on each cell
    if a["score"] < MIN_CELL_SCORE or b["score"] < MIN_CELL_SCORE or c["score"] < MIN_CELL_SCORE:
        return False

    shapes = [a["shape"], b["shape"], c["shape"]]
    shades = [a["shade"], b["shade"], c["shade"]]
    colors = [a["color"], b["color"], c["color"]]

    return (attributes_all_same_or_all_diff(shapes) and
            attributes_all_same_or_all_diff(shades) and
            attributes_all_same_or_all_diff(colors))


def set_quality(cells, idx_triple, strategy=SELECT_STRATEGY):
    """Score a set; higher is better."""
    i, j, k = idx_triple
    s = [cells[i]["score"], cells[j]["score"], cells[k]["score"]]
    base = min(s) if strategy == "max_min" else sum(s)

    # penalize low-confidence labels a bit (below your strict threshold)
    lowconf_pen = sum(1 for v in s if v < D.MATCH_STRICT_THRESHOLD) * PENALIZE_LOWCONF
    return base - lowconf_pen


def choose_best_set(cells, strategy=SELECT_STRATEGY):
    """Return best (i,j,k) and its quality, or (None, None) if none."""
    best = None
    best_q = -1e9
    for tri in itertools.combinations(range(len(cells)), 3):
        if not is_valid_set(cells, tri):
            continue
        q = set_quality(cells, tri, strategy)
        if q > best_q:
            best_q, best = q, tri
    return best, best_q


def classify_cell(screen, box, template_lib, color_lib):
    """
    Returns a dict: {shape, shade, color, score, color_conf}
    Uses the same logic as your scan file (including jitter & stripe bias).
    """
    cell = D.crop(screen, box)
    inner, info = D.remove_green_ring(cell)

    # Base classify
    if inner is not None:
        (shape_base, shade_base), score_base = D.classify_shape_shade(inner, template_lib)
    else:
        shape_base, shade_base, score_base = "unknown", "unknown", -1.0

    # Jitter (always or if low score / no inner)
    do_jitter = D.ALWAYS_JITTER or (score_base < D.RETRY_THRESHOLD) or (inner is None)
    if do_jitter:
        (shape_jit, shade_jit), score_jit, used_box = D.refine_with_jitter(screen, box, template_lib)
        if score_jit > score_base:
            inner_for_color, _ = D.remove_green_ring(D.crop(screen, used_box))
            shape, shade, score = shape_jit, shade_jit, score_jit
        else:
            inner_for_color = inner
            shape, shade, score = shape_base, shade_base, score_base
    else:
        inner_for_color = inner
        shape, shade, score = shape_base, shade_base, score_base

    # Color from inked pixels
    if inner_for_color is not None:
        color_name, color_conf = D.classify_color(inner_for_color, color_lib)
    else:
        color_name, color_conf = "unknown", 0.0

    return {
        "shape": shape,
        "shade": shade,
        "color": color_name,
        "score": float(score),
        "color_conf": float(color_conf),
    }


def scan_grid(template_lib, color_lib):
    """
    Screenshot and classify all 9 cells.
    Returns: (screen, cells list)
    """
    screen = D.screencap_bgr()
    cells = [classify_cell(screen, box, template_lib, color_lib) for box in D.GRID]
    return screen, cells


def tap_set(triple):
    """Tap the three cells (centers) with short gaps."""
    for idx in triple:
        cx, cy = center_of_box(D.GRID[idx])
        D.tap(cx, cy)
        time.sleep(TAP_INTERVAL_S)


def drastic_drop(history, window=DROP_WINDOW, factor=DROP_FACTOR, min_rounds=MIN_ROUNDS_FOR_DROP):
    """
    history: list of per-round quality numbers (one per set played)
    Return True if the most recent avg < factor * previous avg.
    """
    if len(history) < max(min_rounds, 2 * window):
        return False
    prev = history[-2*window:-window]
    curr = history[-window:]
    prev_avg = sum(prev) / len(prev)
    curr_avg = sum(curr) / len(curr)
    if prev_avg < 1e-6:
        return False
    return curr_avg < factor * prev_avg


def print_board(cells):
    def fmt(i):
        c = cells[i]
        return f"{c['shape']}/{c['shade']}/{c['color']} ({c['score']:.2f})"
    for r in range(3):
        row = [fmt(r*3 + k) for k in range(3)]
        print(" | ".join(row))


def main():
    print("[bot] Building libraries from scan_shape_shade.py...")
    template_lib = D.build_template_library(D.TEMPLATES)
    color_lib    = D.build_color_library(D.COLOR_TEMPLATES)

    print("[bot] Tapping START and waiting...")
    D.tap(*D.START_BTN)
    time.sleep(PRE_START_PAUSE_S)

    quality_history = []
    rounds = 0

    while rounds < MAX_ROUNDS:
        # 1) scan
        screen, cells = scan_grid(template_lib, color_lib)
        print("\n[bot] Current grid (shape/shade/color (score)):")
        print_board(cells)

        # 2) find best set
        best_set, best_q = choose_best_set(cells, SELECT_STRATEGY)
        if best_set is None:
            print("[bot] No valid set found — stopping.")
            break

        print(f"[bot] Choosing set {best_set} with quality={best_q:.3f}")
        # 3) tap the set (3 cells)
        tap_set(best_set)

        # 4) record quality & stop if accuracy drops a lot
        quality_history.append(best_q)
        rounds += 1
        if drastic_drop(quality_history):
            print("[bot] Accuracy dropped significantly — stopping.")
            break

        # 5) wait and re-scan
        time.sleep(POST_SET_PAUSE_S)

    print(f"[bot] Done. Rounds played: {rounds}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR in bot:", e)
        raise
