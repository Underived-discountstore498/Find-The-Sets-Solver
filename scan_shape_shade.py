#!/usr/bin/env python3
import os, time, subprocess, sys, math
import numpy as np
import cv2

# ==================== USER CONFIG ====================
ADB = os.environ.get("ADB", "adb")

# Start button
START_BTN = (518, 1690)

# 3x3 grid cells (x, y, w, h)
GRID = [
    (82,  1105, 290, 290), (405, 1105, 290, 290), (722, 1105, 290, 290),
    (82,  1429, 290, 290), (405, 1429, 290, 290), (722, 1429, 290, 290),
    (82,  1747, 290, 290), (405, 1747, 290, 290), (722, 1747, 290, 290),
]

# Templates: 3 shapes x 3 shades
TEMPLATES = {
    ("triangle","empty"):  "templates/triangle_empty.png",
    ("triangle","stripe"): "templates/triangle_stripe.png",
    ("triangle","full"):   "templates/triangle_full.png",

    ("square","empty"):    "templates/square_empty.png",
    ("square","stripe"):   "templates/square_stripe.png",
    ("square","full"):     "templates/square_full.png",

    ("circle","empty"):    "templates/circle_empty.png",
    ("circle","stripe"):   "templates/circle_stripe.png",
    ("circle","full"):     "templates/circle_full.png",
}

# --- Color templates (exactly 3) ---
COLOR_TEMPLATES = {
    "red":   "templates/red.png",
    "green": "templates/green.png",
    "blue":  "templates/blue.png",
}

# HSV range for the mint-green border (OpenCV HSV: H 0..179)
GREEN_LOWER = (35,  25,  80)
GREEN_UPPER = (90, 255, 255)

# Thin ring (~2%)
THIN_FRAC = 0.02
THIN_MIN_FRAC = 0.005
THIN_MAX_FRAC = 0.05
EDGE_GREEN_COVERAGE = 0.10

# Canonical size (fit without stretch to this square)
CANON_SIZE = 128

# Edge matching (color-invariant)
CANNY_LOW, CANNY_HIGH = 60, 120

# Shade heuristic
SHADE_FILL_SOLID_THR = 0.55
SHADE_FILL_EMPTY_THR = 0.20
SHADE_EDGE_STRIPED_THR = 0.14
SAT_MIN, VAL_MIN = 40, 50  # consider pixel "inked" if above both
# -------- Matching strictness & jitter policy (tuned) --------
RETRY_THRESHOLD = 0.750          # was 0.45
MATCH_STRICT_THRESHOLD = 0.8    # accept-as-confident bar (reporting only)
ALWAYS_JITTER = True             # run jitter search for every cell

# Jitter search params
JITTER_RANGE = 10
JITTER_STEP = 4
MAX_JITTER_EVALS = 25

# Stripe bias when confidence is low
STRIPE_BIAS_THRESHOLD = 0.7
STRIPE_BIAS_BOOST = 0.2
STRIPE_OVERRIDE_THRESHOLD = 0.40
# ===============================================================

# -------- Color match settings ----------
COLOR_MIN_PIXELS = 80     # need at least this many "inked" pixels to trust color
COLOR_SIGMA = 18.0        # controls how fast confidence drops with Lab distance
# =======================================

# ORB matcher (2nd signal)
ORB_FEATURES = 500
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def adb_call(*args): return subprocess.call([ADB] + list(args))
def adb_out(*args):  return subprocess.check_output([ADB] + list(args))
def tap(x, y):       adb_call("shell", "input", "tap", str(int(x)), str(int(y)))

def screencap_bgr():
    raw = adb_out("exec-out", "screencap", "-p")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("Failed to decode screenshot from adb")
    return img

def crop(img, box):
    x, y, w, h = box
    H, W = img.shape[:2]
    x0 = max(0, min(W-1, x))
    y0 = max(0, min(H-1, y))
    x1 = max(0, min(W, x + w))
    y1 = max(0, min(H, y + h))
    if x1 <= x0 or y1 <= y0: return img[0:1,0:1].copy()
    return img[y0:y1, x0:x1].copy()

def green_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(GREEN_LOWER, np.uint8), np.array(GREEN_UPPER, np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return mask

def clamp(v, lo, hi): return max(lo, min(hi, v))

def estimate_thickness_per_edge(mask):
    H, W = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0: return None, None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    roi = mask[y0:y1+1, x0:x1+1]
    h, w = roi.shape[:2]

    base = int(round(THIN_FRAC * min(h, w)))
    tmin = max(1, int(round(THIN_MIN_FRAC * min(h, w))))
    tmax = max(tmin+1, int(round(THIN_MAX_FRAC * min(h, w))))

    def scan_top(a):
        for r in range(h):
            if np.count_nonzero(a[r,:]) / float(w) < EDGE_GREEN_COVERAGE: return r
        return h
    def scan_bottom(a):
        for r in range(h-1, -1, -1):
            if np.count_nonzero(a[r,:]) / float(w) < EDGE_GREEN_COVERAGE: return (h-1)-r
        return h
    def scan_left(a):
        for c in range(w):
            if np.count_nonzero(a[:,c]) / float(h) < EDGE_GREEN_COVERAGE: return c
        return w
    def scan_right(a):
        for c in range(w-1, -1, -1):
            if np.count_nonzero(a[:,c]) / float(h) < EDGE_GREEN_COVERAGE: return (w-1)-c
        return w

    t = scan_top(roi); b = scan_bottom(roi); l = scan_left(roi); r = scan_right(roi)
    if t == 0: t = base
    if b == 0: b = base
    if l == 0: l = base
    if r == 0: r = base
    t = clamp(t, tmin, tmax); b = clamp(b, tmin, tmax); l = clamp(l, tmin, tmax); r = clamp(r, tmin, tmax)
    return (t, r, b, l), (x0, y0, x1, y1)

def remove_green_ring(cell_bgr):
    H, W = cell_bgr.shape[:2]
    m = green_mask(cell_bgr)
    if cv2.countNonZero(m) < 10:
        delta = int(round(THIN_FRAC * min(H, W)))
        y0, y1 = delta, H - delta
        x0, x1 = delta, W - delta
        if x1 <= x0 or y1 <= y0:
            return None, {"reason":"fallback-too-small"}
        return cell_bgr[y0:y1, x0:x1].copy(), {"fallback":True}

    thick, bbox = estimate_thickness_per_edge(m)
    if thick is None:
        return None, {"reason":"no-green-detected"}

    (t, r, b, l) = thick
    (x0g, y0g, x1g, y1g) = bbox
    x0i = x0g + l + 1; x1i = x1g - r - 1
    y0i = y0g + t + 1; y1i = y1g - b - 1

    x0i = clamp(x0i, 0, W-1); x1i = clamp(x1i, 0, W-1)
    y0i = clamp(y0i, 0, H-1); y1i = clamp(y1i, 0, H-1)
    if x1i <= x0i or y1i <= y0i:
        return None, {"reason":"inner-invalid"}

    return cell_bgr[y0i:y1i, x0i:x1i].copy(), {"fallback":False}

def corner_bg_color(img):
    h,w = img.shape[:2]
    pts = [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]
    cols = [img[y,x] for (y,x) in pts]
    return np.median(np.array(cols), axis=0).astype(np.uint8)

def fit_to_square(img_bgr, target=128, pad_color=None):
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0: return None
    scale = float(target) / max(h, w)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    top = (target - new_h)//2; bottom = target - new_h - top
    left = (target - new_w)//2; right = target - new_w - left
    if pad_color is None: pad_color = corner_bg_color(resized).tolist()
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=pad_color)

def preprocess_edges_no_stretch(inner_bgr):
    fitted = fit_to_square(inner_bgr, target=CANON_SIZE)
    gray = cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    vec = edges.astype(np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    if n > 1e-6: vec /= n
    return fitted, gray, edges, vec

# ---------- Color helpers ----------
def _ink_mask(bgr):
    """Pixels belonging to the shape (colored/inked), not background."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:,:,1], hsv[:,:,2]
    return (s > SAT_MIN) & (v > VAL_MIN)

def _mean_lab_over_mask(bgr, mask_bool):
    if mask_bool is None or not mask_bool.any():
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ys, xs = np.where(mask_bool)
    if ys.size < COLOR_MIN_PIXELS:
        return None
    vec = lab[ys, xs].astype(np.float32)
    return vec.mean(axis=0)  # (L,a,b)

def build_color_library(color_templates_dict):
    """Load each color template and compute its mean Lab over inked pixels."""
    lib = {}
    for name, path in color_templates_dict.items():
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Missing color template: {path}")
            continue
        m = _ink_mask(img)
        mu = _mean_lab_over_mask(img, m)
        if mu is not None:
            lib[name] = mu  # store (L,a,b)
    return lib

def classify_color(inner_bgr, color_lib):
    """Return (color_name, confidence 0..1). Confidence via Lab distance."""
    m = _ink_mask(inner_bgr)
    mu = _mean_lab_over_mask(inner_bgr, m)
    if mu is None or not color_lib:
        return "unknown", 0.0
    # choose nearest mean in Lab
    best = None
    best_d = 1e9
    for name, proto in color_lib.items():
        d = float(np.linalg.norm(mu - proto))
        if d < best_d:
            best_d, best = d, name
    # convert distance to confidence (smaller distance => higher conf)
    conf = float(np.exp(-best_d / COLOR_SIGMA))
    return best, conf
# -----------------------------------

def estimate_shade(inner_bgr):
    hsv = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:,:,1], hsv[:,:,2]
    ink = (s > SAT_MIN) & (v > VAL_MIN)
    fill_ratio = np.count_nonzero(ink) / max(1, ink.size)

    g = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, CANNY_LOW, CANNY_HIGH)
    edge_density = np.count_nonzero(edges) / edges.size

    if fill_ratio >= SHADE_FILL_SOLID_THR:
        return "full", min(1.0, 0.5 + 0.8*(fill_ratio - SHADE_FILL_SOLID_THR))
    if fill_ratio <= SHADE_FILL_EMPTY_THR:
        return "empty", min(1.0, 0.5 + 2.0*(SHADE_FILL_EMPTY_THR - fill_ratio))
    if edge_density >= SHADE_EDGE_STRIPED_THR:
        return "stripe", min(1.0, 0.5 + 3.0*(edge_density - SHADE_EDGE_STRIPED_THR))
    return "full", 0.45

def orb_similarity(gray_fit, tmpl_des, tmpl_kps, kps2=None, des2=None):
    if tmpl_des is None:
        return 0.0
    if des2 is None:
        kps2, des2 = orb.detectAndCompute(gray_fit, None)
    if des2 is None or len(kps2) == 0:
        return 0.0
    matches = bf.match(tmpl_des, des2)
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda m: m.distance)
    keep = matches[:max(10, len(matches)//4)]
    sc = sum(1.0/(1.0 + m.distance/40.0) for m in keep) / len(keep)
    cov = min(len(keep) / max(1, min(len(tmpl_kps), len(kps2))), 1.0)
    return float(sc * (0.5 + 0.5*cov))

def contour_geometry_shape(edges_fit):
    contours,_ = cv2.findContours(edges_fit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "unknown", 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 50: return "unknown", 0.0
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03*peri, True)
    verts = len(approx)
    circ = 4*math.pi*area / (peri*peri + 1e-6)
    if verts <= 3: return "triangle", 0.7
    if verts == 4: return "square", 0.65
    if circ > 0.72: return "circle", 0.75
    return "circle", 0.55

def build_template_library(templates_dict):
    """Build per-(shape,shade) entries with edge vec + ORB features."""
    lib = {}
    for label, path in templates_dict.items():
        if label not in lib:
            lib[label] = []
        entries = lib[label]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Missing template: {path}")
            continue
        t_inner, _ = remove_green_ring(img)
        if t_inner is None: t_inner = img.copy()
        t_fit, t_gray, t_edges, t_vec = preprocess_edges_no_stretch(t_inner)
        t_kps, t_des = orb.detectAndCompute(t_gray, None)
        entries.append({"vec": t_vec, "fit": t_fit, "edges": t_edges, "kps": t_kps, "des": t_des, "path": path})
    return lib

def _apply_stripe_bias_if_low_conf(combined_scores):
    if not combined_scores:
        return ("unknown","unknown"), 0.0
    best_label, best_score = max(combined_scores.items(), key=lambda kv: kv[1])
    if best_score < STRIPE_BIAS_THRESHOLD:
        boosted = dict(combined_scores)
        for (shape, shade), sc in list(boosted.items()):
            if shade == "stripe":
                boosted[(shape, shade)] = sc + STRIPE_BIAS_BOOST
        best_label, best_score = max(boosted.items(), key=lambda kv: kv[1])
        if best_score < STRIPE_OVERRIDE_THRESHOLD:
            best_shape = max(boosted.items(), key=lambda kv: kv[1])[0][0]
            return (best_shape, "stripe"), best_score
    return best_label, best_score

def classify_shape_shade(inner_bgr, template_lib):
    fit, gray, edges, vec = preprocess_edges_no_stretch(inner_bgr)

    # cosine to all templates
    cos_list = []
    for label, entries in template_lib.items():
        best = -1.0; best_t = None
        for t in entries:
            sc = float(np.dot(vec, t["vec"]))
            if sc > best:
                best, best_t = sc, t
        cos_list.append((label, best, best_t))

    cos_list = sorted(list(cos_list), key=lambda x: x[1], reverse=True)
    candidates = cos_list[:3]

    # ORB refine (compute once on the cell)
    kps2, des2 = orb.detectAndCompute(gray, None)
    orb_scores = {}
    for label, _, t in candidates:
        orb_scores[label] = orb_similarity(gray, t["des"], t["kps"], kps2, des2)

    # independent hints
    shade_guess, shade_conf = estimate_shade(fit)
    shape_guess, shape_conf = contour_geometry_shape(edges)

    # fuse scores
    w_cos, w_orb, w_shade, w_shape = 0.50, 0.30, 0.12, 0.08
    combined = {}
    for label, cos_sc, _ in candidates:
        (shape, shade) = label
        cos01 = max(0.0, (cos_sc + 1.0)/2.0)
        orb_sc = orb_scores.get(label, 0.0)
        shade_bonus = shade_conf if shade == shade_guess else 0.0
        shape_bonus = shape_conf if shape == shape_guess else 0.0
        score = w_cos*cos01 + w_orb*orb_sc + w_shade*shade_bonus + w_shape*shape_bonus
        combined[label] = score

    best_label, best_score = _apply_stripe_bias_if_low_conf(combined)
    if not combined:
        return ("unknown","unknown"), 0.0
    return best_label if best_score >= 0.0 else ("unknown","unknown"), float(best_score)

# ---------- JITTER REFINEMENT ----------
def refine_with_jitter(screen, base_box, template_lib):
    """Try small shifts around the provided box; keep the best scoring result."""
    x, y, w, h = base_box
    best = {"score": -1.0, "label": ("unknown","unknown"), "box": base_box}

    deltas = [(0,0)]
    rng = list(range(-JITTER_RANGE, JITTER_RANGE+1, JITTER_STEP))
    for dx in rng:
        for dy in rng:
            if dx == 0 and dy == 0: continue
            deltas.append((dx, dy))
            if len(deltas) >= MAX_JITTER_EVALS: break
        if len(deltas) >= MAX_JITTER_EVALS: break

    for (dx, dy) in deltas:
        box = (x+dx, y+dy, w, h)
        cell = crop(screen, box)
        inner, _ = remove_green_ring(cell)
        if inner is None:
            continue
        fit, gray, edges, vec = preprocess_edges_no_stretch(inner)

        # quick cosine prefilter
        cos_list = []
        for label, entries in template_lib.items():
            best_cos = -1.0
            for t in entries:
                sc = float(np.dot(vec, t["vec"]))
                if sc > best_cos:
                    best_cos = sc
            cos_list.append((label, best_cos))
        cos_list = sorted(cos_list, key=lambda x: x[1], reverse=True)
        top_label = cos_list[0][0]
        cos01 = max(0.0, (cos_list[0][1] + 1.0)/2.0)

        # estimate shade + geometry bonus
        shade_guess, shade_conf = estimate_shade(fit)
        shape_guess, shape_conf = contour_geometry_shape(edges)
        shade_bonus = 0.10 if top_label[1] == shade_guess else 0.0
        shape_bonus = 0.08 if top_label[0] == shape_guess else 0.0

        score = 0.70*cos01 + shade_bonus + shape_bonus
        if score > best["score"]:
            best.update({"score": score, "label": top_label, "box": box})

    jl, js = best["label"], best["score"]
    biased_label, biased_score = _apply_stripe_bias_if_low_conf({jl: js, (jl[0], "stripe"): js})
    return biased_label, float(biased_score), best["box"]
# --------------------------------------

def main():
    print("[i] Starting classification (with color).")

    # Tap start & wait
    tap(*START_BTN)
    time.sleep(1.0)

    # Screenshot
    screen = screencap_bgr()

    # Build libs
    template_lib = build_template_library(TEMPLATES)
    color_lib = build_color_library(COLOR_TEMPLATES)

    labels = []
    for i, box in enumerate(GRID):
        cell = crop(screen, box)

        inner, info = remove_green_ring(cell)
        if inner is None:
            (shape, shade), score, used_box = refine_with_jitter(screen, box, template_lib)
            color_name, color_conf = classify_color(crop(screen, used_box), color_lib)  # use the jitter crop
            labels.append(((shape, shade, color_name), score))
            print(f"[{i}] (no ring) -> jitter best: {shape}/{shade}/{color_name}  score={score:.3f}  color_conf={color_conf:.2f}  box={used_box}")
            continue

        # Base classify (shape/shade)
        (shape_base, shade_base), score_base = classify_shape_shade(inner, template_lib)

        # Jitter if needed/always
        do_jitter = ALWAYS_JITTER or (score_base < RETRY_THRESHOLD)
        if do_jitter:
            (shape_jit, shade_jit), score_jit, used_box = refine_with_jitter(screen, box, template_lib)
            if score_jit > score_base:
                shape, shade, score = shape_jit, shade_jit, score_jit
                # recompute inner for color from the jittered crop
                inner_for_color, _ = remove_green_ring(crop(screen, used_box))
            else:
                shape, shade, score = shape_base, shade_base, score_base
                inner_for_color = inner
        else:
            shape, shade, score = shape_base, shade_base, score_base
            inner_for_color = inner

        # Color classification (uses inked pixels only)
        color_name, color_conf = classify_color(inner_for_color, color_lib)

        conf_tag = "CONFIDENT" if score >= MATCH_STRICT_THRESHOLD else "LOWCONF"
        labels.append(((shape, shade, color_name), score))
        print(f"[{i}] shape={shape:8s} shade={shade:6s} color={color_name:6s} score={score:.3f} ({conf_tag})  color_conf={color_conf:.2f}")

    print("\n3x3 grid (shape/shade/color):")
    for r in range(3):
        row = labels[r*3:(r+1)*3]
        print(" | ".join(f"{sh}/{sd}/{col}" for ((sh,sd,col), _) in row))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
