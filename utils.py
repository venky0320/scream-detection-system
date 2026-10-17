"""Shared audio helpers. Used by BOTH training and real-time detection so the
features are computed identically in both places."""
import os
import numpy as np
import librosa

SR = 22050                      # sample rate used everywhere
DURATION = 3                    # seconds per analysis window
N_SAMPLES = SR * DURATION
N_MFCC = 20
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def is_audio_file(name):
    return name.lower().endswith(AUDIO_EXTS) and not name.startswith(".")


def fix_audio(audio):
    """Pad with zeros or trim so the clip is exactly DURATION seconds."""
    audio = np.asarray(audio, dtype=np.float32)
    return librosa.util.fix_length(audio, size=N_SAMPLES)


def load_segments(path, min_seconds=1.0):
    """Load a file and cut it into non-overlapping 3 s windows.
    A trailing piece shorter than 3 s is kept (zero padded) if >= min_seconds."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    segments = []
    for start in range(0, len(audio), N_SAMPLES):
        piece = audio[start:start + N_SAMPLES]
        if len(piece) >= int(min_seconds * SR):
            segments.append(fix_audio(piece))
    return segments


def extract_features(audio, sr=SR):
    """Return a fixed-length feature vector for a (up to) 3 s audio window."""
    audio = fix_audio(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak            # volume invariant (mic gain differs)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        [zcr.mean(), zcr.std()],
        [centroid.mean(), centroid.std()],
        [rolloff.mean()],
        [rms.mean(), rms.std()],
        contrast.mean(axis=1),
        chroma.mean(axis=1),
    ])
    return feats.astype(np.float32)


FEATURE_DIM = len(extract_features(np.zeros(N_SAMPLES, dtype=np.float32)))