import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import font as tkfont
from collections import Counter
import pyttsx3
from PIL import Image, ImageTk
import time
import threading

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("asl_model.keras")
print("Model loaded:", model.input_shape)

# ================= LABELS =================
classes = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z'
]

# ================= SPEECH (non-blocking) =================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    if text.strip():
        def _speak():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

ROI_SIZE = 300
background = None

# ================= PREPROCESS =================
def preprocess(roi):
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
    img = cv2.resize(thresh, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# ================= COLOR PALETTE =================
BG       = "#0d0f1a"
PANEL    = "#131629"
ACCENT   = "#00e5ff"
ACCENT2  = "#7c4dff"
SUCCESS  = "#00e676"
WARN     = "#ff6d00"
TEXT     = "#e0e6f0"
SUBTEXT  = "#8891aa"
BTN_BG   = "#1c2040"
BTN_HOV  = "#252b52"

# ================= ROOT WINDOW =================
root = tk.Tk()
root.title("ASL Sign Language Recognition")
root.geometry("1100x760")
root.configure(bg=BG)
root.resizable(False, False)

# Fonts
try:
    FONT_TITLE   = tkfont.Font(family="Courier New", size=17, weight="bold")
    FONT_PRED    = tkfont.Font(family="Courier New", size=28, weight="bold")
    FONT_CONF    = tkfont.Font(family="Courier New", size=13)
    FONT_WORD    = tkfont.Font(family="Courier New", size=30, weight="bold")
    FONT_LABEL   = tkfont.Font(family="Courier New", size=10, weight="bold")
    FONT_BTN     = tkfont.Font(family="Courier New", size=11, weight="bold")
    FONT_HIST    = tkfont.Font(family="Courier New", size=11)
    FONT_STATUS  = tkfont.Font(family="Courier New", size=10)
except:
    FONT_TITLE  = ("Courier New", 17, "bold")
    FONT_PRED   = ("Courier New", 28, "bold")
    FONT_CONF   = ("Courier New", 13)
    FONT_WORD   = ("Courier New", 30, "bold")
    FONT_LABEL  = ("Courier New", 10, "bold")
    FONT_BTN    = ("Courier New", 11, "bold")
    FONT_HIST   = ("Courier New", 11)
    FONT_STATUS = ("Courier New", 10)

# ================= STATE =================
word         = ""
char_history = []
last_char    = ""
stable_count = 0
cooldown     = 0
fps_list     = []
last_time    = time.time()
sentence_list = []

# ================= TKINTER VARS =================
pred_var      = tk.StringVar(value="—")
conf_var      = tk.StringVar(value="Confidence: —")
word_var      = tk.StringVar(value="")
status_var    = tk.StringVar(value="● Initializing camera...")
conf_pct_var  = tk.IntVar(value=0)
fps_var       = tk.StringVar(value="FPS: —")

# ================= LAYOUT: left (video) | right (panels) =================
left_frame  = tk.Frame(root, bg=BG)
left_frame.pack(side="left", fill="both", padx=(16, 8), pady=16)

right_frame = tk.Frame(root, bg=BG)
right_frame.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

# ─── Title ───────────────────────────────────────────────────────
tk.Label(left_frame,
         text="◈  ASL SIGN LANGUAGE RECOGNITION  ◈",
         font=FONT_TITLE, fg=ACCENT, bg=BG).pack(anchor="w", pady=(0, 8))

# ─── Video feed ──────────────────────────────────────────────────
video_frame = tk.Frame(left_frame, bg=ACCENT, padx=2, pady=2)
video_frame.pack()

video_label = tk.Label(video_frame, bg="black")
video_label.pack()

# FPS + status bar below video
info_bar = tk.Frame(left_frame, bg=BG)
info_bar.pack(fill="x", pady=(4, 0))

tk.Label(info_bar, textvariable=status_var,
         font=FONT_STATUS, fg=SUBTEXT, bg=BG).pack(side="left")
tk.Label(info_bar, textvariable=fps_var,
         font=FONT_STATUS, fg=SUBTEXT, bg=BG).pack(side="right")

# ─── Right panel: Prediction ─────────────────────────────────────
def section(parent, title):
    f = tk.Frame(parent, bg=PANEL, padx=14, pady=10)
    f.pack(fill="x", pady=(0, 10))
    tk.Label(f, text=title, font=FONT_LABEL, fg=ACCENT2, bg=PANEL).pack(anchor="w")
    sep = tk.Frame(f, bg=ACCENT2, height=1)
    sep.pack(fill="x", pady=(2, 8))
    return f

# Prediction panel
pred_panel = section(right_frame, "CURRENT PREDICTION")

pred_row = tk.Frame(pred_panel, bg=PANEL)
pred_row.pack(fill="x")

tk.Label(pred_row, textvariable=pred_var,
         font=FONT_PRED, fg=SUCCESS, bg=PANEL, width=4,
         anchor="w").pack(side="left")

# Confidence bar (canvas)
conf_canvas_frame = tk.Frame(pred_row, bg=PANEL)
conf_canvas_frame.pack(side="left", fill="x", expand=True, padx=(10, 0))

tk.Label(conf_canvas_frame, textvariable=conf_var,
         font=FONT_CONF, fg=TEXT, bg=PANEL).pack(anchor="w")

conf_bar_bg = tk.Canvas(conf_canvas_frame, height=14, bg="#0a0c18",
                        highlightthickness=0)
conf_bar_bg.pack(fill="x", pady=(4, 0))
conf_bar_fill = conf_bar_bg.create_rectangle(0, 0, 0, 14, fill=SUCCESS, outline="")

def update_conf_bar(pct):
    conf_bar_bg.update_idletasks()
    w = conf_bar_bg.winfo_width()
    fill_w = int(w * pct / 100)
    color = SUCCESS if pct >= 75 else WARN if pct >= 50 else "#ff1744"
    conf_bar_bg.coords(conf_bar_fill, 0, 0, fill_w, 14)
    conf_bar_bg.itemconfig(conf_bar_fill, fill=color)

# Top-3 predictions
top3_frame = tk.Frame(pred_panel, bg=PANEL)
top3_frame.pack(fill="x", pady=(8, 0))

tk.Label(top3_frame, text="TOP 3", font=FONT_LABEL,
         fg=SUBTEXT, bg=PANEL).pack(anchor="w")

top3_labels = []
for _ in range(3):
    lbl = tk.Label(top3_frame, text="", font=FONT_HIST, fg=SUBTEXT, bg=PANEL, anchor="w")
    lbl.pack(fill="x")
    top3_labels.append(lbl)

# ─── Word builder panel ───────────────────────────────────────────
word_panel = section(right_frame, "WORD BUILDER")

tk.Label(word_panel, textvariable=word_var,
         font=FONT_WORD, fg=ACCENT, bg=PANEL,
         anchor="w", wraplength=380).pack(fill="x")

# Sentence history
tk.Label(word_panel, text="SENTENCE HISTORY",
         font=FONT_LABEL, fg=SUBTEXT, bg=PANEL).pack(anchor="w", pady=(10, 2))

history_box = tk.Text(word_panel, height=3, font=FONT_HIST,
                       bg="#0a0c18", fg=TEXT, bd=0,
                       insertbackground=ACCENT, state="disabled",
                       wrap="word", relief="flat")
history_box.pack(fill="x")

# ─── Buttons panel ───────────────────────────────────────────────
btn_panel = section(right_frame, "CONTROLS")

buttons_frame = tk.Frame(btn_panel, bg=PANEL)
buttons_frame.pack(fill="x")

def styled_btn(parent, text, cmd, color):
    b = tk.Button(parent, text=text, command=cmd,
                  font=FONT_BTN, fg="white", bg=color,
                  activebackground=BTN_HOV, activeforeground=ACCENT,
                  bd=0, padx=10, pady=8, cursor="hand2",
                  relief="flat")
    return b

def add_space():
    global word
    word += " "
    word_var.set(word)

def clear_word():
    global word, char_history, stable_count, last_char, cooldown
    word = ""
    char_history = []
    stable_count = 0
    last_char = ""
    cooldown = 0
    word_var.set("")

def speak_word():
    speak(word.strip())

def save_sentence():
    global word
    s = word.strip()
    if s:
        sentence_list.append(s)
        history_box.config(state="normal")
        history_box.insert("end", s + "\n")
        history_box.see("end")
        history_box.config(state="disabled")
        clear_word()
        status_var.set(f"● Saved: \"{s}\"")

def undo_char():
    global word
    if word:
        word = word[:-1]
        word_var.set(word)

btns = [
    ("⎵  SPACE",    add_space,    "#1a6b3c"),
    ("◀  UNDO",     undo_char,    "#5c3a1e"),
    ("✔  SAVE",     save_sentence, "#1a3a6b"),
    ("🔊  SPEAK",   speak_word,   "#2d1a6b"),
    ("✕  CLEAR",    clear_word,   "#6b1a1a"),
]

for i, (txt, cmd, col) in enumerate(btns):
    b = styled_btn(buttons_frame, txt, cmd, col)
    b.grid(row=0, column=i, padx=4, sticky="ew")
    buttons_frame.columnconfigure(i, weight=1)

# ─── Tip label ───────────────────────────────────────────────────
tk.Label(right_frame,
         text="Hold a sign steady for ~1 sec to add a letter",
         font=FONT_STATUS, fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(0, 4))

# ================= MAIN LOOP =================
def update():
    global background, word, char_history, last_char, stable_count, cooldown, last_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    # FPS
    now = time.time()
    fps_list.append(1.0 / max(now - last_time, 1e-6))
    last_time = now
    if len(fps_list) > 20:
        fps_list.pop(0)
    fps_var.set(f"FPS: {int(np.mean(fps_list))}")

    frame = cv2.flip(frame, 1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (21, 21), 0)

    if background is None:
        background = gray
        status_var.set("● Camera ready — show a hand sign")
        root.after(10, update)
        return

    h, w = gray.shape
    x1 = w // 2 - ROI_SIZE // 2
    y1 = h // 2 - ROI_SIZE // 2
    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE

    roi_gray = gray[y1:y2, x1:x2]
    diff     = cv2.absdiff(background[y1:y2, x1:x2], roi_gray)
    thresh_m = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    motion   = np.sum(thresh_m > 0)

    if motion < 1500:
        pred_var.set("—")
        conf_var.set("Confidence: —")
        update_conf_bar(0)
        for lbl in top3_labels:
            lbl.config(text="")
        char_history = []
        stable_count = 0
        cooldown = 0
        status_var.set("● No hand detected — show a sign in the green box")

    else:
        roi = frame[y1:y2, x1:x2]
        img = preprocess(roi)

        if img is None:
            pred_var.set("?")
        else:
            pred  = model.predict(img, verbose=0)[0]
            idx   = int(np.argmax(pred))
            conf  = float(pred[idx])
            char  = classes[idx] if conf >= 0.75 else "?"

            # Top-3
            top3_idx = np.argsort(pred)[::-1][:3]
            for i, lbl in enumerate(top3_labels):
                c = classes[top3_idx[i]]
                p = pred[top3_idx[i]] * 100
                bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
                lbl.config(text=f"  {c.upper()}  {bar}  {p:.1f}%",
                           fg=SUCCESS if i == 0 else SUBTEXT)

            pred_var.set(char.upper() if char != "?" else "?")
            conf_var.set(f"Confidence: {conf*100:.1f}%")
            update_conf_bar(int(conf * 100))

            if char != "?":
                char_history.append(char)
                if len(char_history) > 10:
                    char_history.pop(0)

                if cooldown == 0:
                    common = Counter(char_history).most_common(1)[0]
                    if common[1] >= 3:
                        if common[0] == last_char:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_char = common[0]

                        if stable_count >= 5:
                            word += common[0]
                            word_var.set(word)
                            char_history = []
                            stable_count = 0
                            cooldown = 10
                            status_var.set(f"● Added: '{common[0].upper()}'  →  \"{word}\"")

            # Cooldown progress (repurpose accent color)
            if cooldown > 0:
                cooldown -= 1

    # ── Draw ROI rectangle with corner accents ──
    col_roi = (0, 229, 255)   # ACCENT in BGR
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), col_roi, thickness)
    # Corner marks
    cs = 20
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx*cs, py), col_roi, 3)
        cv2.line(frame, (px, py), (px, py + dy*cs), col_roi, 3)

    # ── Convert & display ──
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgb).resize((640, 420))
    imgtk  = ImageTk.PhotoImage(pil_im)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(15, update)

# ================= START =================
update()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
