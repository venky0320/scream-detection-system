"""Step 3: real-time scream detection with GUI and emergency alerts.

Design:
  * audio callback only copies audio into a queue (fast, never blocks)
  * a worker thread keeps a rolling 3 s buffer and runs the model
  * the GUI is only touched from the main thread (polls a UI queue)
  * alerts are sent from a separate thread, with a cooldown
"""
import os
import time
import queue
import threading
import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import joblib
from dotenv import load_dotenv

from utils import SR, N_SAMPLES, extract_features

load_dotenv()
MODEL_PATH = "scream_detector_model.joblib"
SCALER_PATH = "scaler.joblib"
THRESHOLD = float(os.getenv("THRESHOLD", "0.8"))          # probability needed
REQUIRED_HITS = int(os.getenv("REQUIRED_HITS", "2"))      # consecutive windows
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "60"))       # min gap between alerts
MIN_RMS = float(os.getenv("MIN_RMS", "0.01"))             # ignore near-silence
HOP_SECONDS = 0.5                                         # prediction interval
ALERT_CHANNEL = os.getenv("ALERT_CHANNEL", "whatsapp").lower()  # whatsapp|telegram|both
PHONES = [p.strip() for p in os.getenv("ALERT_PHONE", "").split(",") if p.strip()]
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


# ------------------------------------------------------------------ alerts
def get_location_link():
    try:
        import geocoder
        g = geocoder.ip("me")
        if g.latlng:
            lat, lon = g.latlng
            return f"https://www.google.com/maps?q={lat},{lon}"
    except Exception:
        pass
    return "Location unavailable"


def send_whatsapp_web(phone, message, wait=25):
    """Open WhatsApp Web with the text pre-filled, then press Enter to send."""
    import webbrowser
    import urllib.parse
    import pyautogui
    number = phone.replace("+", "").replace(" ", "")
    url = f"https://web.whatsapp.com/send?phone={number}&text={urllib.parse.quote(message)}"
    webbrowser.open(url)
    time.sleep(wait)                 # wait for WhatsApp Web to load the chat
    pyautogui.press("enter")         # message box already has focus, so Enter sends
    time.sleep(3)
    pyautogui.hotkey("ctrl", "w")    # close the tab


def send_alert(ui_q, reason="A scream was detected!"):
    """Runs in its own thread. Reports progress to the GUI through ui_q."""
    link = get_location_link()
    message = (f"EMERGENCY ALERT: {reason}\n"
               f"Approximate location (IP based): {link}\n"
               f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    results = []

    if ALERT_CHANNEL in ("telegram", "both"):
        try:
            import requests
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT, "text": message}, timeout=10)
            r.raise_for_status()
            results.append("Telegram sent")
        except Exception as e:
            results.append(f"Telegram failed: {e}")

    if ALERT_CHANNEL in ("whatsapp", "both"):
        for i, phone in enumerate(PHONES):
            try:
                send_whatsapp_web(phone, message)
                results.append(f"WhatsApp sent to {phone[-4:]}")
            except Exception as e:
                results.append(f"WhatsApp failed ({phone[-4:]}): {e}")
            if i < len(PHONES) - 1:
                time.sleep(5)

    ui_q.put(("alert_result", " | ".join(results)))


# ------------------------------------------------------------------ app
class App:
    def __init__(self, root):
        self.root = root
        self.audio_q = queue.Queue(maxsize=50)
        self.ui_q = queue.Queue()
        self.stop_event = threading.Event()
        self.stream = None
        self.worker = None
        self.last_alert = 0.0

        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            messagebox.showerror("Model not found",
                                 f"Run extract_features.py and train_model.py first.\n\n{e}")
            raise SystemExit(1)
        self._predict(np.zeros(N_SAMPLES, dtype=np.float32))   # warm-up

        root.title("Scream Detection System")
        root.geometry("420x380")
        root.config(bg="white")
        tk.Label(root, text="Scream Detection System", font=("Arial", 16, "bold"),
                 bg="white").pack(pady=10)
        self.status = tk.Label(root, text="Click Start to begin", font=("Arial", 12),
                               fg="gray", bg="white", wraplength=380)
        self.status.pack(pady=5)
        self.prob = tk.Label(root, text="Scream probability: --", font=("Arial", 11), bg="white")
        self.prob.pack(pady=5)
        self.start_btn = tk.Button(root, text="Start Detection", font=("Arial", 12),
                                   bg="green", fg="white", padx=10, pady=5,
                                   command=self.start)
        self.start_btn.pack(pady=8)
        self.stop_btn = tk.Button(root, text="Stop Detection", font=("Arial", 12),
                                  bg="red", fg="white", padx=10, pady=5,
                                  state=tk.DISABLED, command=self.stop)
        self.stop_btn.pack(pady=8)
        tk.Button(root, text="Send Test Alert", font=("Arial", 10),
                  command=self.test_alert).pack(pady=8)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll_ui)

    # ---- model ----
    def _predict(self, window):
        feats = self.scaler.transform([extract_features(window, SR)])
        return float(self.model.predict_proba(feats)[0][1])

    # ---- audio ----
    def audio_callback(self, indata, frames, time_info, status):
        try:
            self.audio_q.put_nowait(indata[:, 0].copy())   # only enqueue
        except queue.Full:
            pass

    def worker_loop(self):
        buffer = np.zeros(N_SAMPLES, dtype=np.float32)
        hits = 0
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            buffer = np.concatenate([buffer[len(chunk):], chunk])
            if np.sqrt(np.mean(buffer[-SR:] ** 2)) < MIN_RMS:     # silence gate
                hits = 0
                self.ui_q.put(("status", "Listening... (quiet)", "blue", 0.0))
                continue
            p = self._predict(buffer)
            hits = hits + 1 if p > THRESHOLD else 0
            if hits >= REQUIRED_HITS:
                self.ui_q.put(("status", "SCREAM DETECTED!", "red", p))
                if time.time() - self.last_alert > COOLDOWN:
                    self.last_alert = time.time()
                    threading.Thread(target=send_alert, args=(self.ui_q,), daemon=True).start()
                    self.ui_q.put(("status", "SCREAM DETECTED! Sending alert...", "red", p))
            else:
                self.ui_q.put(("status", "No scream detected", "green", p))

    # ---- controls (main thread) ----
    def start(self):
        try:
            self.stop_event.clear()
            while not self.audio_q.empty():
                self.audio_q.get_nowait()
            self.stream = sd.InputStream(callback=self.audio_callback, samplerate=SR,
                                         channels=1, blocksize=int(SR * HOP_SECONDS))
            self.worker = threading.Thread(target=self.worker_loop, daemon=True)
            self.worker.start()
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Microphone error", str(e))
            self.stop_event.set()
            return
        self.set_status("Listening for screams...", "blue")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        self.stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.set_status("Detection stopped", "black")
        self.prob.config(text="Scream probability: --")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def test_alert(self):
        self.set_status("Sending test alert...", "orange")
        threading.Thread(target=send_alert, args=(self.ui_q, "TEST ALERT (not a real emergency)"),
                         daemon=True).start()

    def on_close(self):
        self.stop()
        self.root.destroy()

    # ---- UI updates ----
    def set_status(self, text, color):
        self.status.config(text=text, fg=color)

    def poll_ui(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                if item[0] == "status" and not self.stop_event.is_set():
                    _, text, color, p = item
                    self.set_status(text, color)
                    self.prob.config(text=f"Scream probability: {p:.2f}")
                elif item[0] == "alert_result":
                    self.set_status(f"Alert: {item[1]}", "purple")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_ui)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()