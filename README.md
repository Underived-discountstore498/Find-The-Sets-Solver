# Find-The-Sets-Solver

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-vision-green)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automates finding and tapping valid **sets** on a 3×3 Android board using ADB + OpenCV.
Automates finding and tapping valid **sets** on a 3×3 board captured from an Android device via ADB.

- **Detection:** shape (triangle / square / circle), shade (empty / stripe / full), color (red / green / blue via templates)
- **Engine:** OpenCV (edges + ORB), HSV heuristics, jitter alignment, low-confidence stripe bias
- **Bot:** picks valid sets, taps them, rescans, repeats until no set or accuracy drops

---

## 📱 Connecting your Android device

Before running the scripts, you need to enable **Developer Mode** and **USB Debugging** on your phone.

### Step-by-step
1. **Enable Developer Mode**
   - Open **Settings → About phone**
   - Tap **Build number** 7 times until you see “You are now a developer!”

2. **Enable USB Debugging**
   - Open **Settings → System → Developer options**
   - Enable **USB debugging**

3. **Connect your phone via USB**
   - Plug the phone into your computer.
   - On your phone, allow the prompt *“Allow USB debugging?”* and check “Always allow”.

4. **Verify the connection**
   ```bash
   adb devices
    ```

    You should see your device listed as **device** (not unauthorized).

> ⚠️ Tip: Make sure the phone screen stays on and the game is visible before running the scripts.

---

## 📂 Folder layout
```

.
├── bot.py                 # autoplay loop (uses scan_shape_shade as library)
├── scan_shape_shade.py    # detector (shape/shade/color + jitter)
├── templates/             # required template PNGs (see below)
├── requirements.txt
├── LICENSE
└── README.md

```

---

## ⚙️ Requirements
- Python 3.9+
- Android Platform Tools (`adb` in PATH)
- Android device with **Developer options → USB debugging** enabled
- USB cable (recommended)

---

## 🧩 Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🎨 Templates (Shapes & Colors)

Place these files in `templates/`:

**Shapes:**

```
triangle_empty.png
triangle_stripe.png
triangle_full.png
square_empty.png
square_stripe.png
square_full.png
circle_empty.png
circle_stripe.png
circle_full.png
```

**Colors:**

```
red.png
green.png
blue.png
```

> ⚠️ **Note on assets:**
> If your game’s shapes or colors are copyrighted, please supply your own templates or draw look-alike graphics.
> This project is for **personal and educational use only.**

---

## 📸 Screenshots & Game Assets

This repository may include example screenshots to demonstrate measurement and detection.
All trademarks and graphics remain property of their respective owners.

* Screenshots are provided **for personal/educational use only**.
* They are **not licensed for redistribution or commercial use**.
* If you are a rights holder and wish for an image to be removed, please open an issue.

---

## 🧭 Device calibration (important!)

Every phone has a different resolution and layout — so you **must** update the grid coordinates and start button position before running the bot.

Open **`scan_shape_shade.py`** and edit:

```python
# Start button (tap coordinates)
START_BTN = (518, 1690)

# 3×3 grid cells (x, y, w, h)
GRID = [
    (82, 1105, 290, 290), (405, 1105, 290, 290), (722, 1105, 290, 290),
    (82, 1429, 290, 290), (405, 1429, 290, 290), (722, 1429, 290, 290),
    (82, 1747, 290, 290), (405, 1747, 290, 290), (722, 1747, 290, 290),
]
```

### 📍 How to get positions (recommended method)

1. On your Android device, open **Settings → Developer options → Pointer location**.
2. Turn it **ON** — you’ll now see live `(x, y)` coordinates at the top of the screen.
3. Open the game and touch:

   * The **top-left corner of each cell** → note the top-left corner (`x`, `y`) and measure width/height.
   * The **Start / Retry button** → use that center position for `START_BTN`.
4. Turn **Pointer location OFF** when done.


---

## 🧪 Test detection

```bash
python3 scan_shape_shade.py
```

Expected output: 9 lines of `shape/shade/color (score)` and a 3×3 grid summary.

---

## 🤖 Run the bot

```bash
python3 bot.py
```

The bot:

1. Finds a valid set (each attribute all-same or all-different)
2. Taps the three cells
3. Waits for the board to update
4. Rescans and repeats
   Stops when no set is found or accuracy drops significantly.

---

## 🔧 Tweakable settings

### Detection (`scan_shape_shade.py`)

| Variable                                       | Description                               |
| ---------------------------------------------- | ----------------------------------------- |
| `RETRY_THRESHOLD`, `ALWAYS_JITTER`, `JITTER_*` | Jitter alignment aggressiveness           |
| `MATCH_STRICT_THRESHOLD`                       | “Confident” score cutoff                  |
| `STRIPE_BIAS_*`                                | Slight bias to “stripe” on low confidence |
| `SAT_MIN`, `VAL_MIN`                           | Adjust for dim or pale colors             |
| `COLOR_MIN_PIXELS`, `COLOR_SIGMA`              | Color detection strictness                |

### Bot pacing & strategy (`bot.py`)

| Variable                                                  | Description                                  |
| --------------------------------------------------------- | -------------------------------------------- |
| `TAP_INTERVAL_S`, `POST_SET_PAUSE_S`, `PRE_START_PAUSE_S` | Tap timing                                   |
| `MIN_CELL_SCORE`                                          | Filter uncertain cells                       |
| `SELECT_STRATEGY`                                         | `"max_min"` (robust) or `"max_sum"` (greedy) |
| `DROP_WINDOW`, `DROP_FACTOR`                              | Stop if quality degrades                     |
| `MAX_ROUNDS`                                              | Safety cap                                   |

---

## 🛡️ Safety & TOS

* Prefer **USB debugging**; avoid ADB over Wi-Fi on untrusted networks.
* Respect the game’s **terms of service** and fair play guidelines.
* This project is for **personal and educational use only.**

---

## ⚖️ License

Released under the [MIT License](LICENSE).
