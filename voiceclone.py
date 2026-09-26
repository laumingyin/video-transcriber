"""
voiceclone.py - "Sound like me" voice conversion using the OpenVoice V2 tone-color converter.

It changes the *timbre* of speech (who it sounds like) while keeping the pronunciation and
rhythm of the source speech. The app feeds it US-English TTS, so the result is your voice
with a US accent. Everything runs locally on the CPU.
"""
import os
import sys
import threading
import types

import numpy as np
import torch

# OpenVoice's text package pulls in Chinese/English G2P libraries that the converter never uses.
sys.modules.setdefault("openvoice.text", types.SimpleNamespace(text_to_sequence=None))
from openvoice import utils as ov_utils                    # noqa: E402
from openvoice.mel_processing import spectrogram_torch     # noqa: E402
from openvoice.models import SynthesizerTrn                # noqa: E402

SR = 22050
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "openvoice_v2")

_model = None
_hps = None
_lock = threading.Lock()
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


def available():
    return os.path.exists(os.path.join(MODEL_DIR, "checkpoint.pth"))


def _load():
    global _model, _hps
    if _model is None:
        hps = ov_utils.get_hparams_from_file(os.path.join(MODEL_DIR, "config.json"))
        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        ckpt = torch.load(os.path.join(MODEL_DIR, "checkpoint.pth"), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        _model, _hps = model, hps
    return _model, _hps


def _spec(audio, hps):
    y = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    return spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                             hps.data.hop_length, hps.data.win_length, center=False)


def speaker_embedding(audio, segment_s=10.0, max_s=240.0):
    """Voice 'fingerprint' from mono float audio at 22.05 kHz. Uses up to max_s seconds,
    split into segments; near-silent segments are skipped. Returns (embedding, seconds_used)."""
    model, hps = _load()
    audio = audio[: int(max_s * SR)]
    seg = int(segment_s * SR)
    pieces = [audio[i:i + seg] for i in range(0, len(audio), seg)]
    pieces = [p for p in pieces if len(p) >= SR * 2]
    if not pieces:
        raise ValueError("Need at least a few seconds of speech.")
    rms = np.array([np.sqrt(np.mean(p ** 2)) for p in pieces])
    keep = [p for p, r in zip(pieces, rms) if r > max(0.01, 0.25 * np.median(rms))] or pieces
    embs = []
    with _lock, torch.no_grad():
        for p in keep:
            embs.append(model.ref_enc(_spec(p, hps).transpose(1, 2)).unsqueeze(-1))
    return torch.stack(embs).mean(0), sum(len(p) for p in keep) / SR


def convert(audio, src_emb, tgt_emb, tau=0.3):
    """Re-voice mono float audio (22.05 kHz) from the src speaker to the tgt speaker."""
    model, hps = _load()
    with _lock, torch.no_grad():
        spec = _spec(audio, hps)
        lengths = torch.LongTensor([spec.size(-1)])
        out = model.voice_conversion(spec, lengths, sid_src=src_emb, sid_tgt=tgt_emb, tau=tau)[0][0, 0]
    return out.cpu().float().numpy()
