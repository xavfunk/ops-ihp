#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PsychoPy Experiment — Experimenter-facing display
===================================================
Screen layout (1280 × 720):
  ┌──────────────────────────┬────────────────────┐
  │                          │  Block · Trial      │
  │   Camera stream  /       │  COMMAND TEXT       │
  │   Captured photo         │  ─────────────────  │
  │   (780 × 439 px)         │  ● phase status     │
  │                          │  Experimenter notes │
  └──────────────────────────┴────────────────────┘
  (participant does NOT see this screen)

Trial flow
----------
  1. Audio command plays automatically.  Stream shown.
  2. [R]      → repeat audio command
  3. [SPACE]  → capture photo, replace stream with photo
  4. [SPACE]  → accept  → save accepted image, move to next trial
     [R]      → discard → save discarded image, replay audio, retry

Calibration flow  (before each block + at end)
---------------
  1. Stream shown.
  2. [SPACE]  → capture
  3. [SPACE]  → accept     [R] → discard & retake

Images saved to
---------------
  data/pictures/trials/accepted/
  data/pictures/trials/discarded/
  data/pictures/calibration/accepted/
  data/pictures/calibration/discarded/

Requirements: psychopy   opencv-python   numpy
  pip install psychopy opencv-python
  
TODO 
- improve instructions etc
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
import random

import cv2
import numpy as np
from psychopy import core, data, gui, logging, sound, visual, event
from psychopy.hardware import keyboard
from psychopy import prefs
import pygame
pygame.mixer.init()


# prefs.hardware['audioLib'] = [ 'sounddevice']
# prefs.hardware['audioLatencyMode'] = 2


# # Specify your preferred audio library order (e.g., PTB, sounddevice, pyo, pygame)
# prefs.hardware['audioLib'] = ['pygame']  # or ['sounddevice'], ['pyo'], etc.
# from psychopy import sound



# ══════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION   (edit this block to suit your study)
# ══════════════════════════════════════════════════════════════

EXP_NAME = "implicit-hand-perception"
EXP_VER  = "0.1.0"

# ── Trial structure ──────────────────────────────────────────
# Option A: point CONDITIONS at a CSV file.
#   Required columns: block (int), trial (int), command (str), audio_file (str)
# Option B: leave CONDITIONS = None → auto-generate from COMMANDS below.
CONDITIONS = None


# (command_text, path_to_audio_file) — cycles if count < N_TRIALS_PER_BLOCK
LANGUAGE = 'english'
if LANGUAGE == 'english':
    COMMANDS = [
    ("thumb knuckle",   "audio/thumb_knuckle.mp3"),
    ("index knuckle",  "audio/index_knuckle.mp3"),
    ("middle knuckle",     "audio/middle_knuckle.mp3"),
    ("ring knuckle",   "audio/ring_knuckle.mp3"),
    ("pinky knuckle", "audio/pinky_knuckle.mp3"),
    ("thumb tip",   "audio/thumb_tip.mp3"),
    ("index tip",  "audio/index_tip.mp3"),
    ("middle tip",     "audio/middle_tip.mp3"),
    ("ring tip",   "audio/ring_tip.mp3"),
    ("pinky tip", "audio/pinky_tip.mp3"),
    ]
elif LANGUAGE == 'dutch':
#     COMMANDS = [
#     ("duimknokkel",         "audio/duim_knokkel.mp3"),
#     ("wijsvingerknokkel",   "audio/wijsvinger_knokkel.mp3"),
#     ("middelvingerknokkel", "audio/middelvinger_knokkel.mp3"),
#     ("ringvingerknokkel",   "audio/ringvinger_knokkel.mp3"),
#     ("pinkknokkel",         "audio/pink_knokkel.mp3"),
#     ("duimtop",             "audio/duim_top.mp3"),
#     ("wijsvingertop",       "audio/wijsvinger_top.mp3"),
#     ("middelvingertop",     "audio/middelvinger_top.mp3"),
#     ("ringvingertop",       "audio/ringvinger_top.mp3"),
#     ("pinktop",             "audio/pink_top.mp3"),
# ]
    # if we want english in the backend
    COMMANDS = [
    ("thumb knuckle",         "audio/duim_knokkel.mp3"),
    ("index knuckle",   "audio/wijsvinger_knokkel.mp3"),
    ("middle knuckle", "audio/middelvinger_knokkel.mp3"),
    ("ring knuckle",   "audio/ringvinger_knokkel.mp3"),
    ("pinky knuckle",         "audio/pink_knokkel.mp3"),
    ("thumb tip",             "audio/duim_top.mp3"),
    ("index tip",       "audio/wijsvinger_top.mp3"),
    ("middle tip",     "audio/middelvinger_top.mp3"),
    ("ring tip",       "audio/ringvinger_top.mp3"),
    ("pinky tip",             "audio/pink_top.mp3"),
]
else:
    raise NotImplementedError("No other languages are implemented.")

N_BLOCKS           = 3
N_REPEATS_PER_BLOCK = 3
N_TRIALS_PER_BLOCK = N_REPEATS_PER_BLOCK * len(COMMANDS)

# ── Paths ────────────────────────────────────────────────────
DATA_DIR = "data"

# ── Camera ───────────────────────────────────────────────────
# webcam: Camera: 640×480
CAMERA_INDEX  = 4        # 0 = default webcam
CAMERA_WIDTH  = 640#1280
CAMERA_HEIGHT = 480#720
BRIGHTENER = .5

# ── Window ───────────────────────────────────────────────────
WIN_SIZE   = (1280, 720)
FULLSCREEN = False

# ── Layout (pixels, origin = window centre) ──────────────────
#
#   x: −640 ──────────────── 0 ─────────── +640
#           ←── camera ──→ │ ←── info ──────→
#   camera: pos=(−250, 0), size=(780, 439)
#   info:   centre x = 390, width = 500
#   divider at x = 140
#
CAM_W, CAM_H = 640,480#780, 439    # camera/photo display size (≈ 16:9)
CAM_POS      = (-250, 0)   # centre of camera area
DIV_X        = 140         # x-coordinate of vertical divider
INFO_X       = 390         # x-centre of info panel

# ── Timing ───────────────────────────────────────────────────
ITI_DURATION = 0.3   # seconds between trial acceptance and next trial start

# ── Experimenter instructions per phase ──────────────────────
# Fill in whatever guidance is useful for your protocol.
EXP_INSTR = {
    "stream": (
        "[SPACE]  →  capture photo\n"
        "[R]       →  repeat audio"
    ),
    "photo": (
        "[SPACE]  →  accept & continue\n"
        "[R]       →  discard & retake"
    ),
    "cal_stream": (
        "[SPACE]  →  capture reference photo"
    ),
    "cal_photo": (
        "[SPACE]  →  accept\n"
        "[R]       →  discard & retake"
    ),
}

task_instructions="placeholder"

# ══════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
# SECTION 2 — PARTICIPANT DIALOG
# ──────────────────────────────────────────────────────────────
exp_info = {"Participant ID": "test", "Session": "1"}
dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title=EXP_NAME)
if not dlg.OK:
    core.quit()

PID = exp_info["Participant ID"].zfill(3)
SES = exp_info["Session"].zfill(2)


# ──────────────────────────────────────────────────────────────
# SECTION 3 — PATHS & IMAGE DIRECTORIES
# ──────────────────────────────────────────────────────────────
run_ts   = data.getDateStr(format="%Y-%m-%d_%Hh%M")
print(PID, SES)
run_stem = f"sub-{PID}_ses-{SES}_dt-{run_ts}"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
data_stem = Path(DATA_DIR) / f"sub-{PID}" / f"ses-{SES}"/ run_stem

# All four image output directories, keyed by (category, outcome)
# TODO maybe add hand location to path
PIC_DIRS = {
    ("trial",       "accepted"):  Path(DATA_DIR) / f"sub-{PID}" / f"ses-{SES}"/ "pictures" / "trials"      / "accepted",
    ("trial",       "discarded"): Path(DATA_DIR) / f"sub-{PID}"/ f"ses-{SES}"/ "pictures" / "trials"      / "discarded",
    ("calibration", "accepted"):  Path(DATA_DIR) / f"sub-{PID}"/ f"ses-{SES}"/ "pictures" / "calibration" / "accepted",
    ("calibration", "discarded"): Path(DATA_DIR) / f"sub-{PID}"/ f"ses-{SES}"/ "pictures" / "calibration" / "discarded",
}
for _p in PIC_DIRS.values():
    _p.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# SECTION 4 — LOGGING
# ──────────────────────────────────────────────────────────────
logging.LogFile(str(data_stem) + ".log", level=logging.EXP)
logging.console.setLevel(logging.WARNING)
logging.exp(f"Experiment : {EXP_NAME}  v{EXP_VER}")
logging.exp(f"Participant: {PID}   Session: {SES}")
logging.exp(f"Data stem  : {data_stem}")


# ──────────────────────────────────────────────────────────────
# SECTION 5 — CSV DATA FILE
# Manual CSV gives full control and handles variable retry counts cleanly.
# ──────────────────────────────────────────────────────────────
_CSV_COLS = [
    "participant", "session",
    "phase", "block", "trial",
    "command", "audio_file",
    "attempt", "outcome",
    "photo_filename", "photo_ts",
]
_csv_fh = open(str(data_stem) + "_trials.csv", "w", newline="", encoding="utf-8")
_csv_wr = csv.DictWriter(_csv_fh, fieldnames=_CSV_COLS)
_csv_wr.writeheader()


def write_data_row(**kw: object) -> None:
    """Write one row to the data CSV, supplying participant/session automatically."""
    row = {c: "" for c in _CSV_COLS}
    row.update(participant=PID, session=SES)
    row.update(kw)          # type: ignore[arg-type]
    _csv_wr.writerow(row)
    _csv_fh.flush()


# ──────────────────────────────────────────────────────────────
# SECTION 6 — TRIAL LIST
# ──────────────────────────────────────────────────────────────
# TODO check if working
if CONDITIONS and os.path.isfile(CONDITIONS):
    _raw = data.importConditions(CONDITIONS)
    _block_nums = sorted({int(r["block"]) for r in _raw})
    blocks: list[list[dict]] = []
    for _b in _block_nums:
        _bt = sorted(
            [r for r in _raw if int(r["block"]) == _b],
            key=lambda r: int(r["trial"]),
        )
        blocks.append(_bt)
else:
    blocks = []
    for _b in range(1, N_BLOCKS + 1):
        _bt = []
        # for _t in range(1, N_TRIALS_PER_BLOCK + 1):
        #     _cmd, _audio = COMMANDS[(_t - 1) % len(COMMANDS)]
        #     _bt.append({"block": _b, "trial": _t,
        #                 "command": _cmd, "audio_file": _audio})
        for _rep in range(N_REPEATS_PER_BLOCK):
            _replist = []
            random.shuffle(COMMANDS)
            for _t, command in enumerate(COMMANDS):
                _cmd, _audio = command
                _replist.append({"block": _b, "command": _cmd, "audio_file": _audio})
            
            _bt += _replist
        
        # add trial number
        for i, trial in enumerate(_bt):
            trial["trial"] = i+1
        blocks.append(_bt)



print(blocks)

# ──────────────────────────────────────────────────────────────
# SECTION 7 — WINDOW
# ──────────────────────────────────────────────────────────────
win = visual.Window(
    size=WIN_SIZE,
    fullscr=FULLSCREEN,
    screen=0,
    winType="pyglet",
    allowGUI=True,
    color=(-0.8, -0.8, -0.8),   # dark-grey background
    colorSpace="rgb",
    units="pix",
    checkTiming=False,
)
win.mouseVisible = False


# ──────────────────────────────────────────────────────────────
# SECTION 8 — CAMERA
# ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)

# fixes that dont work:
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # 1 = manual on most drivers
# cap.set(cv2.CAP_PROP_EXPOSURE, -4)          # negative = shorter shutter; try -3 to -6
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)       # 0–255 typically
# cap.set(cv2.CAP_PROP_GAIN, 100)
if not cap.isOpened():
    logging.error("Cannot open camera — aborting.")
    win.close()
    core.quit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Warm-up: discard frames so auto-exposure/white-balance settle
for _ in range(10):
    cap.read()

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logging.exp(f"Camera: {actual_w}×{actual_h}")

# _blank_frame = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)
_blank_frame = np.full((actual_h, actual_w, 3), -1.0, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# SECTION 9 — VISUAL STIMULI
# ──────────────────────────────────────────────────────────────

# ── Left side: camera stream / captured photo ────────────────
cam_stim = visual.ImageStim(
    win, image=_blank_frame,
    pos=CAM_POS, size=(CAM_W, CAM_H),
)
photo_stim = visual.ImageStim(
    win, image=_blank_frame,
    pos=CAM_POS, size=(CAM_W, CAM_H),
)

# ── Right side: info panel ───────────────────────────────────

# Panel background  (slightly lighter than window bg)
ui_bg = visual.Rect(
    win, pos=(INFO_X, 0),
    width=WIN_SIZE[0] - DIV_X - (WIN_SIZE[0] // 2 - INFO_X),
    height=WIN_SIZE[1],
    fillColor=(-0.72, -0.72, -0.72), lineColor=None,
)
# Recompute width cleanly: panel spans from DIV_X to +640
_panel_w = WIN_SIZE[0] // 2 - DIV_X   # = 500
ui_bg.width = _panel_w

# Vertical divider
ui_divider = visual.Line(
    win,
    start=(DIV_X, -WIN_SIZE[1] // 2),
    end  =(DIV_X,  WIN_SIZE[1] // 2),
    lineColor=(-0.35, -0.35, -0.35), lineWidth=2,
)

# Block · Trial counter  (top)
ui_header = visual.TextStim(
    win, text="",
    pos=(INFO_X, 295), height=22,
    color=(0.8, 0.8, 0.8), bold=True,
    wrapWidth=460, alignText="center",
)

# "COMMAND" / "CALIBRATION" section label
ui_section_label = visual.TextStim(
    win, text="COMMAND",
    pos=(INFO_X, 213), height=17,
    color=(0.55, 0.55, 0.55),
    wrapWidth=460, alignText="center",
)

# The command text itself  (large, prominent)
ui_command = visual.TextStim(
    win, text="",
    pos=(INFO_X, 148), height=40,
    color=(1.0, 0.85, 0.1), bold=True,   # yellow
    wrapWidth=460, alignText="center",
)

# Phase indicator  (● LIVE STREAM  /  ◼ PHOTO REVIEW)
ui_phase = visual.TextStim(
    win, text="",
    pos=(INFO_X, 48), height=20,
    bold=True, wrapWidth=460, alignText="center",
)

# Thin horizontal rule  (separates status from instructions)
ui_hrule = visual.Line(
    win,
    start=(DIV_X + 15, 2),
    end  =(WIN_SIZE[0] // 2 - 15, 2),
    lineColor=(-0.4, -0.4, -0.4), lineWidth=1,
)

# "EXPERIMENTER" heading above instructions
ui_instr_hdr = visual.TextStim(
    win, text="EXPERIMENTER",
    pos=(INFO_X, -38), height=14,
    color=(0.4, 0.4, 0.4),
    wrapWidth=460, alignText="center",
)

# Instruction body
ui_instr = visual.TextStim(
    win, text="",
    pos=(INFO_X, -125), height=19,
    color=(0.75, 0.75, 0.75),
    wrapWidth=440, alignText="center",
)

# Attempt counter  (shown from attempt 2 onwards)
ui_attempt = visual.TextStim(
    win, text="",
    pos=(INFO_X, -295), height=16,
    color=(0.45, 0.45, 0.45),
    wrapWidth=460, alignText="center",
)


# ──────────────────────────────────────────────────────────────
# SECTION 10 — AUDIO
# ──────────────────────────────────────────────────────────────
_all_audio_files = {t["audio_file"] for blk in blocks for t in blk}

# audio_bank: dict[str, sound.Sound | None] = {}
# for _af in _all_audio_files:
#     if os.path.isfile(_af):
#         audio_bank[_af] = sound.Sound(_af)
#         logging.exp(f"Loaded audio: {_af!r}")
#     else:
#         audio_bank[_af] = None
#         logging.warning(
#             f"Audio file not found — trial will run silently: {_af!r}"
#         )


# ──────────────────────────────────────────────────────────────
# SECTION 11 — KEYBOARD & CLOCK
# ──────────────────────────────────────────────────────────────
global_clock = core.Clock()


# ══════════════════════════════════════════════════════════════
# SECTION 12 — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def brighten(rgb, brightener):
    rgb_ = rgb.copy() + brightener
    rgb_[rgb_>1] = 1.0
    # return (rgb.astype(np.float32) / 127.5) - 1.0
    return rgb_

def read_frame():  # returns np.ndarray or None
    """Return the latest camera frame as a uint8 RGB array, or None on error.
 
    Transforms applied:
      1. BGR → RGB             (OpenCV default → PsychoPy default)
      2. Horizontal mirror     (participant-facing camera feels natural)
      3. Vertical flip         (numpy arrays appear upside-down in PsychoPy
                                without this; no need to set flipVert on stim)
    """
    ret, bgr = cap.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 1)   # mirror left ↔ right
    rgb = cv2.flip(rgb, 0)   # correct y-axis for PsychoPy
    # Normalize to float32 in [-1, 1] — PsychoPy's ImageStim expects this
    rgb_norm = (rgb.astype(np.float32) / 127.5) - 1.0

    # Brighten
    # rgb_norm = brighten(rgb_norm, brightener)
    # print(brightener == rgb_norm)
    # rgb_norm[rgb_norm<0] = 0
    # return (rgb.astype(np.float32) / 127.5) - 1.0
    return rgb_norm

 
def save_image(
    frame,          # np.ndarray — frame in PsychoPy display orientation
    category,       # str: "trial" | "calibration"
    outcome,        # str: "accepted" | "discarded"
    block,          # int | str
    trial,          # int | str | None
    attempt,        # int
):
    # type: (...) -> Tuple[str, str]
    """Save *frame* (in PsychoPy display orientation) to the correct directory.
 
    Undoes the two flips from read_frame() so the saved JPEG has natural
    camera orientation (not mirrored, y-axis as expected in a photograph).
 
    Returns (filename, iso_timestamp_string).
    """
    # Convert float32 [-1,1] back to uint8 [0,255], then undo display transforms
    uint8 = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    natural = cv2.flip(cv2.flip(uint8, 0), 1)
    bgr = cv2.cvtColor(natural, cv2.COLOR_RGB2BGR)
 
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    bstr   = f"{int(block):02d}" if str(block).lstrip("-").isdigit() else str(block)
 
    if category == "trial":
        fname = f"block-{bstr}_trial-{int(trial):03d}_attempt-{attempt:02d}_dt-{ts_str}.jpg"
    else:
        fname = f"cal_block-{bstr}_attempt-{attempt:02d}_dt-{ts_str}.jpg"
 
    fpath = PIC_DIRS[(category, outcome)] / fname
    cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logging.exp(f"Saved [{outcome}] {category} image → {fpath}")
    return fname, ts_str


# def play_command_audio(audio_file: str) -> None:
#     """Stop any current playback of *audio_file*, then replay it from the start."""
#     snd = audio_bank.get(audio_file)
#     if snd is None:
#         logging.warning(f"Skipping missing audio: {audio_file!r}")
#         return
#     snd.stop()
#     snd.play()


def play_command_audio(audio_file):
    if not os.path.isfile(audio_file):
        logging.warning("Audio not found: {}".format(audio_file))
        return
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

def draw_ui(
    phase: str,
    block_num: int | str,
    trial_num: int | str | None,
    command: str | None,
    attempt: int,
) -> None:
    """Draw one complete experimenter UI frame and call win.flip().

    The caller is responsible for updating cam_stim / photo_stim *before*
    calling this function:
      • stream phases  → update cam_stim.setImage(frame) each loop iteration
      • photo phases   → set photo_stim.setImage(captured) once before the loop
    """
    # ── Left: camera stream or captured photo ────────────────
    if "stream" in phase:
        cam_stim.draw()
    else:
        photo_stim.draw()

    # ── Right: info panel ────────────────────────────────────
    ui_bg.draw()
    ui_divider.draw()

    # Block · trial header
    if trial_num is not None:
        ui_header.text = f"Block {block_num}   ·   Trial {trial_num}"
    else:
        ui_header.text = f"Block {block_num}"
    ui_header.draw()

    # Command or calibration label
    if command:
        ui_section_label.color = (0.55, 0.55, 0.55)
        ui_section_label.text  = "COMMAND"
        ui_section_label.draw()
        ui_command.text = command.upper()
        ui_command.draw()
    else:
        ui_section_label.color = (0.45, 0.65, 0.95)   # blue tint for calibration
        ui_section_label.text  = "CALIBRATION"
        ui_section_label.draw()
        # Reset for next call
        ui_section_label.color = (0.55, 0.55, 0.55)

    # Phase status indicator
    if "stream" in phase:
        ui_phase.color = (0.3, 0.88, 0.3)          # green
        ui_phase.text  = "●  LIVE  STREAM"
    else:
        ui_phase.color = (0.95, 0.68, 0.18)         # amber
        ui_phase.text  = "◼  PHOTO  REVIEW"
    ui_phase.draw()

    ui_hrule.draw()
    ui_instr_hdr.draw()

    # Experimenter instructions for this phase
    ui_instr.text = EXP_INSTR.get(phase, "")
    ui_instr.draw()

    # Attempt counter (only shown from attempt 2 onwards)
    if attempt > 1:
        ui_attempt.text = f"Attempt  {attempt}"
        ui_attempt.draw()

    win.flip()


def quit_exp() -> None:
    """Stop all audio, release camera, flush data to disk, and exit cleanly."""
    # for snd in audio_bank.values():
    #     if snd:
    #         snd.stop()
    cap.release()
    _csv_fh.close()
    logging.flush()
    win.close()
    core.quit()


# ══════════════════════════════════════════════════════════════
# SECTION 13 — PHASE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def run_calibration(block_num: int | str) -> None:
    """
    Calibration phase — repeated until the experimenter accepts.

    Step 1 (stream): experimenter presses [SPACE] to capture.
    Step 2 (review): [SPACE] accepts, [R] discards and returns to step 1.
    Image saved after every capture (to accepted/ or discarded/).
    """
    attempt = 0

    while True:
        attempt += 1
        logging.exp(f"Calibration  block={block_num}  attempt={attempt}  [stream]")
        event.clearEvents()
        captured: np.ndarray | None = None

        # ── Step 1: live stream, wait for capture ─────────────
        while captured is None:
            frame = read_frame()
            if frame is not None:
                cam_stim.setImage(brighten(frame, brightener=BRIGHTENER))
            draw_ui("cal_stream", block_num, None, None, attempt)

            for k in event.getKeys(keyList=["space", "escape"]):
                if k == "escape":
                    quit_exp()
                if k == "space":

                    captured = frame.copy() if frame is not None else _blank_frame.copy()

        # ── Step 2: photo review ──────────────────────────────
        logging.exp(f"Calibration  block={block_num}  attempt={attempt}  [review]")
        photo_stim.setImage(brighten(captured, brightener=BRIGHTENER))
        # photo_stim.setImage(captured)
        event.clearEvents()
        decision: str | None = None

        while decision is None:
            draw_ui("cal_photo", block_num, None, None, attempt)
            for k in event.getKeys(keyList = ["space", "r", "escape"]):
                if k == "escape":
                    quit_exp()
                if k == "space":
                    decision = "accepted"
                if k == "r":
                    decision = "discarded"

        fname, ts = save_image(captured, "calibration", decision,
                                block_num, None, attempt)
        write_data_row(
            phase="calibration", block=block_num, trial="",
            command="", audio_file="",
            attempt=attempt, outcome=decision,
            photo_filename=fname, photo_ts=ts,
        )
        logging.exp(f"Calibration  block={block_num}  attempt={attempt}  → {decision}")

        if decision == "accepted":
            return
        # else: loop — attempt += 1, retake


def run_trial(
    block_num: int,
    trial_num: int,
    command: str,
    audio_file: str,
) -> None:
    """
    Single trial — repeated until the experimenter accepts.

    Step 1 (stream + audio): audio plays automatically.
      [R]     → replay audio
      [SPACE] → capture photo, go to step 2
    Step 2 (photo review):
      [SPACE] → accept  → save accepted image, brief ITI, return
      [R]     → discard → save discarded image, replay audio from step 1
    """
    attempt = 0

    while True:
        attempt += 1
        logging.exp(
            f"Trial  b={block_num}  t={trial_num}  attempt={attempt}  [stream]"
        )

        # ── Step 1: play audio + live stream ──────────────────
        play_command_audio(audio_file)
        event.clearEvents()
        captured: np.ndarray | None = None

        while captured is None:
            frame = read_frame()
            if frame is not None:
                cam_stim.setImage(brighten(frame, brightener=BRIGHTENER))
                
            draw_ui("stream", block_num, trial_num, command, attempt)

            for k in event.getKeys(keyList =["space", "r", "escape"]):
                if k == "escape":
                    quit_exp()
                if k == "r":
                    play_command_audio(audio_file)      # repeat instruction
                if k == "space":
                    captured = frame.copy() if frame is not None else _blank_frame.copy()

        # ── Step 2: photo review ──────────────────────────────
        logging.exp(
            f"Trial  b={block_num}  t={trial_num}  attempt={attempt}  [review]"
        )
        photo_stim.setImage(brighten(captured, brightener=BRIGHTENER))
        event.clearEvents()
        decision: str | None = None

        while decision is None:
            draw_ui("photo", block_num, trial_num, command, attempt)
            for k in event.getKeys(keyList=["space", "r", "escape"]):
                if k == "escape":
                    quit_exp()
                if k == "space":
                    decision = "accepted"
                if k == "r":
                    decision = "discarded"

        fname, ts = save_image(captured, "trial", decision,
                                block_num, trial_num, attempt)
        write_data_row(
            phase="trial", block=block_num, trial=trial_num,
            command=command, audio_file=audio_file,
            attempt=attempt, outcome=decision,
            photo_filename=fname, photo_ts=ts,
        )
        logging.exp(
            f"Trial  b={block_num}  t={trial_num}  attempt={attempt}  → {decision}"
        )

        if decision == "accepted":
            core.wait(ITI_DURATION)
            return
        # else: loop — attempt += 1, replay audio, retake


# ══════════════════════════════════════════════════════════════
# SECTION 14 — READY SCREEN
# ══════════════════════════════════════════════════════════════
_n_trials_total = sum(len(b) for b in blocks)

visual.TextStim(
    win,
    text=(
        f"Experiment :  {EXP_NAME}  (v{EXP_VER})\n"
        f"Participant:  {PID}   —   Session {SES}\n\n"
        f"{len(blocks)} block(s)   ·   {_n_trials_total} trial(s) total\n\n"
        "Calibration runs before each block and once at the end.\n\n"
        "Press  [ESCAPE]  at any time to quit and save data.\n\n"
        "Press  [SPACE]  to begin."
    ),
    height=24, color=(0.8, 0.8, 0.8),
    wrapWidth=WIN_SIZE[0] - 100,
).draw()
win.flip()
event.clearEvents()
event.waitKeys(keyList=["space"])

visual.TextStim(
    win,
    text=(
        "Please remind the participant about the task instructions and make sure they understand:\n\n"
        f"{task_instructions}.\n\n"
        "Press  [ESCAPE]  at any time to quit and save data.\n\n"
        "Press  [SPACE]  to move to the first calibration block."
    ),
    height=24, color=(0.8, 0.8, 0.8),
    wrapWidth=WIN_SIZE[0] - 100,
).draw()
win.flip()
event.clearEvents()
event.waitKeys(keyList=["space"])


# ══════════════════════════════════════════════════════════════
# SECTION 15 — MAIN EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════
for _block_idx, _block_trials in enumerate(blocks):
    _block_num = _block_idx + 1

    # ── Calibration before each block ────────────────────────
    run_calibration(_block_num)

    # ── Start block ────────────────────────

    visual.TextStim(
    win,
    text=(
        f"Experiment :  {EXP_NAME}  (v{EXP_VER})\n"
        f"Participant:  {PID}   —   Session {SES}\n\n"
        f"{len(blocks)} block(s)   ·   {_n_trials_total} trial(s) total\n\n"
        "Calibration run successfully.\n\n"
        "Press  [ESCAPE]  at any time to quit and save data.\n\n"
        "Press  [SPACE]  to start this block."
    ),
    height=24, color=(0.8, 0.8, 0.8),
    wrapWidth=WIN_SIZE[0] - 100,
    ).draw()
    win.flip()
    event.clearEvents()
    event.waitKeys(keyList=["space"])

    # ── All trials in this block ──────────────────────────────
    for _t in _block_trials:
        run_trial(
            block_num=int(_t["block"]),
            trial_num=int(_t["trial"]),
            command=_t["command"],
            audio_file=_t["audio_file"],
        )

# ── Final calibration at end of experiment ────────────────────
run_calibration("END")


# ══════════════════════════════════════════════════════════════
# SECTION 16 — END SCREEN
# ══════════════════════════════════════════════════════════════
visual.TextStim(
    win,
    text=(
        "Experiment complete.\n\n"
        f"Trial data  →  {data_stem}_trials.csv\n"
        f"Images      →  {DATA_DIR}/pictures/\n\n"
        "Press  [SPACE]  to exit."
    ),
    height=24, color=(0.8, 0.8, 0.8),
    wrapWidth=WIN_SIZE[0] - 100,
).draw()
win.flip()
event.clearEvents()
event.waitKeys(keyList=["space"])

quit_exp()