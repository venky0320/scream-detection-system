# Scream Detection System

Real-time scream detection from a microphone, with automatic emergency alerts
(WhatsApp and/or Telegram) that include an approximate location link.

A neural network (multi-layer perceptron) classifies 3-second audio windows using
MFCC, spectral and chroma features. When a scream is detected, the system alerts
one or more contacts.

> Final year project. Version **1.0.0**

---

## Features
- Live microphone detection with a simple desktop GUI (Tkinter)
- Rolling 3-second analysis window, evaluated every 0.5 s
- False-alarm protection: probability threshold, consecutive-hit rule, silence gate, alert cooldown
- Alerts to **multiple WhatsApp numbers** and/or **Telegram**
- Reproducible training with a leak-free split and a full evaluation report
- Audio augmentation (noise, pitch shift, gain) to enlarge the training data

## Results (test set, 700 unseen segments)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| NotScreaming | 0.97 | 0.96 | 0.97 |
| Screaming | 0.96 | 0.98 | 0.97 |

Accuracy **97%**, ROC-AUC **0.991**. Confusion matrix: 320 / 14 / 9 / 357.
Plots are in the [`results/`](results/) folder (`confusion_matrix.png`, `training_curves.png`,
`evaluation_report.txt`).

Note: live accuracy depends on your microphone and room, and can be lower than
the dataset results. See [Limitations](#limitations).

---

## Project structure

```
scream-detection-system/
├── README.md                  <- this file
├── LICENSE
├── requirements.txt           <- pinned Python dependencies
├── .env.example               <- settings template (copy to .env)
├── .gitignore
├── utils.py                   <- shared feature extraction (training + live)
├── extract_features.py        <- step 1: dataset -> features.csv
├── train_model.py             <- step 2: train, evaluate, save model
├── real_time_detection.py     <- step 3: live GUI + alerts
├── dataset/
│   ├── README.md
│   ├── Screaming/             <- put scream audio here
│   └── NotScreaming/          <- put non-scream audio here
└── results/                   <- evaluation report and plots
```

Files created when you run the project (not all are committed):
`features.csv`, `scream_detector_model.joblib`, `scaler.joblib`,
`evaluation_report.txt`, `confusion_matrix.png`, `training_curves.png`.

---

## Requirements

| Item | Version |
|---|---|
| Python | 3.14.6 (tested). Python 3.10 or newer should work |
| OS | Windows 10/11 (tested). Linux/macOS: see notes below |
| Microphone | Any input device |
| Browser | Chrome or Edge, logged in to WhatsApp Web (for WhatsApp alerts) |

Python packages (see `requirements.txt` for exact pinned versions):

| Package | Version | Purpose |
|---|---|---|
| numpy | 2.5.4 | arrays |
| pandas | 3.0.6 | feature table |
| scipy | 1.18.1 | signal processing |
| scikit-learn | 1.9.1 | neural network, scaling, metrics |
| joblib | 1.6.0 | saving the model |
| librosa | 1.0.0 | audio features |
| numba | 0.68.0 | librosa dependency |
| soundfile | 0.14.0 | reading audio files |
| sounddevice | 0.5.6 | live microphone input |
| matplotlib | 3.11.2 | result plots |
| geocoder | 1.38.1 | approximate location |
| requests | 2.34.2 | Telegram API |
| python-dotenv | 1.2.4 | reading `.env` |
| pyautogui | >=0.9.54 | sending WhatsApp Web messages |

`tkinter` ships with the standard Python installer on Windows.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/venky0320/scream-detection-system.git
cd scream-detection-system
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```
If PowerShell says "running scripts is disabled", run this once, then activate again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the installation
```bash
python -c "import librosa, sklearn, sounddevice, joblib, dotenv, pyautogui; print('All OK')"
```

### 5. Add the dataset
Put your audio files in `dataset/Screaming/` and `dataset/NotScreaming/`
(see [dataset/README.md](dataset/README.md)).

### 6. Create your settings file
```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```
Then edit `.env` (see [Alert setup](#alert-setup)).

---

## How to run

Run the three steps in order, inside the activated virtual environment.

```bash
python extract_features.py      # 1. builds features.csv (can take a long time)
python train_model.py           # 2. trains and evaluates the model
python real_time_detection.py   # 3. starts the live detector
```

**Step 1 - `extract_features.py`** cuts every file into 3-second segments, extracts
features and creates augmented copies. Expect about 5 to 10 minutes per 100 files.
Librosa warnings are harmless. Set `AUGMENT = False` in the file for a 4x faster run.

**Step 2 - `train_model.py`** splits the data by source file (no leakage), balances
the classes, trains the model and writes the evaluation report, confusion matrix and
training curves.

**Step 3 - `real_time_detection.py`** opens the GUI:
1. Click **Send Test Alert** to confirm alerts work.
2. Click **Start Detection** and allow microphone access.
3. Make a loud scream sound near the microphone. The status turns red and an alert is sent.
4. Click **Stop Detection** when finished.

If you downloaded the repository with the trained `scream_detector_model.joblib` and
`scaler.joblib`, you can skip steps 1 and 2. If you get a version error when loading
them, retrain with your installed scikit-learn.

---

## Alert setup

All settings live in the `.env` file (copied from `.env.example`).

| Setting | Meaning | Default |
|---|---|---|
| `ALERT_CHANNEL` | `whatsapp`, `telegram` or `both` | `whatsapp` |
| `ALERT_PHONE` | WhatsApp numbers, comma separated | - |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | - |
| `TELEGRAM_CHAT_ID` | Telegram chat id | - |
| `THRESHOLD` | Scream probability needed (0 to 1) | `0.8` |
| `REQUIRED_HITS` | Consecutive positive windows before alerting | `2` |
| `COOLDOWN_SECONDS` | Minimum time between alerts | `60` |
| `MIN_RMS` | Ignore audio quieter than this | `0.01` |

### How to set up WhatsApp alerts

The app opens WhatsApp Web in your browser, pre-fills the message, and presses Enter.

1. Open **https://web.whatsapp.com** in your **default browser** (Chrome or Edge).
2. On your phone: WhatsApp -> **Linked devices** -> **Link a device**, and scan the QR code.
3. Stay logged in. Tick "Keep me signed in" if offered.
4. Edit `.env`:
   ```
   ALERT_CHANNEL=whatsapp
   ALERT_PHONE=+911234567890,+919876543210
   ```
   Use the full international format: `+`, country code, number. No spaces. Separate
   several friends with commas.
5. Run `python real_time_detection.py` and click **Send Test Alert**.
6. **Do not touch the mouse or keyboard** until it finishes. For each number, a browser
   tab opens, waits about 25 seconds for WhatsApp Web to load, sends the message and
   closes the tab. Several numbers are sent one after another.

WhatsApp tips:
- Keep the screen on and unlocked while an alert is sent.
- If the message is typed but not sent, increase `wait=25` to `35` in
  `send_whatsapp_web()` in `real_time_detection.py` (slow internet).
- Sending to your own number works and appears in "Message yourself".
- Tell friends beforehand that test alerts are not real emergencies.
- On macOS, change `pyautogui.hotkey("ctrl", "w")` to `pyautogui.hotkey("command", "w")`.
  On Linux, `pyautogui` may need `sudo apt install python3-tk python3-dev scrot`.

### How to set up Telegram alerts (optional, more reliable)

Telegram sends through an API, with no browser needed.

1. In Telegram, message **@BotFather** and send `/newbot`. Follow the steps and copy the **token**.
2. Open your new bot and send it any message, such as "hi".
3. In a browser, open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` and find
   `"chat":{"id": ...}`. That number is your chat id.
4. Edit `.env`:
   ```
   ALERT_CHANNEL=telegram
   TELEGRAM_BOT_TOKEN=123456789:ABC...
   TELEGRAM_CHAT_ID=123456789
   ```
   Use `ALERT_CHANNEL=both` to send to WhatsApp and Telegram together.

**Security:** never commit `.env` or share your bot token or phone numbers.
`.env` is already listed in `.gitignore`.

---

## How it works

```
Microphone -> audio callback (copy only) -> queue -> worker thread
   -> rolling 3 s buffer -> feature extraction -> scaler -> neural network
   -> probability -> threshold + consecutive hits + cooldown -> alert thread
   -> WhatsApp / Telegram
```

- **Features (utils.py):** MFCC mean/std, delta MFCC, zero-crossing rate, spectral
  centroid / rolloff / contrast, RMS and chroma. The same function is used for
  training and live detection, so features always match.
- **Model:** MLPClassifier with hidden layers (128, 64), ReLU, L2 regularisation and
  early stopping.
- **Split:** 80/20 by source file, so augmented copies of one clip never appear in
  both training and test sets.
- **GUI safety:** the audio callback only queues audio, a worker thread runs the model,
  and only the main thread touches the GUI.

## Tuning

| Problem | Change in `.env` (no retraining needed) |
|---|---|
| Screams missed | Lower `THRESHOLD` to `0.6` or `0.7`, or `REQUIRED_HITS=1` |
| Too many false alarms | Raise `THRESHOLD` to `0.9` or `REQUIRED_HITS=3` |
| Reacts to quiet room noise | Raise `MIN_RMS` to `0.02` |
| Quiet screams ignored | Lower `MIN_RMS` to `0.005` |

For better live accuracy, record extra clips with your own microphone, add them to
the dataset folders and retrain.

## Troubleshooting

| Problem | Fix |
|---|---|
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt` |
| `Folder not found: dataset\Screaming` | Check the dataset layout and folder names |
| "Model not found" popup | Run `extract_features.py` and `train_model.py` first |
| Microphone error | Windows Settings -> Privacy -> Microphone; set a default input device |
| WhatsApp text typed but not sent | Increase `wait` in `send_whatsapp_web()`; stay logged in; do not touch the keyboard |
| No alert received | Check the status line in the app; check `.env` format (no comments after values) |

## Limitations
- Location comes from the IP address, so it is only approximate.
- Loud non-scream sounds (shouting, laughter) can cause false alarms.
- WhatsApp sending uses browser automation: it needs a logged-in browser, an unlocked
  screen, and takes about 30 seconds per number.
- Accuracy depends on the dataset and on the microphone and room used.

## Future work
- Convolutional network on mel-spectrograms
- Mobile app with real GPS location
- WhatsApp Business Cloud API or Twilio for fully reliable delivery
- Larger and more varied dataset, including real-world recordings

## License
MIT. See [LICENSE](LICENSE).
