"""
Local helper for transcriber.html: serves the page and provides the voice-over API.

Easiest start: double-click "Start Transcriber.bat" (sets everything up on first run).
Manual:        .venv\\Scripts\\python server.py   (opens http://localhost:8765 in your browser)

Speech comes from Microsoft's online edge-tts service, so subtitle text is sent to Microsoft.
Videos and voice samples only go to this local server (127.0.0.1) and never leave your machine.
The "sound like me" voice conversion (OpenVoice) runs locally too.
"""
import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import edge_tts
import imageio_ffmpeg
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TMP = tempfile.mkdtemp(prefix="voiceover_")
PORT = int(os.environ.get("PORT", 8765))
# Web pages allowed to use this helper besides its own page. Add more with the ALLOWED_ORIGINS
# environment variable (comma-separated). Every other website is refused, so a random page you
# visit can't drive the helper or touch your voice profile.
ALLOWED_ORIGINS = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}", "https://laumingyin.github.io"} | {
    o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()}
RATE_RE = re.compile(r"^[+-]\d{1,3}%$")
VOICE_RE = re.compile(r"^[A-Za-z]{2,3}-[A-Za-z]{2,4}-[A-Za-z]+$")
PROFILE_DIR = os.path.join(HERE, "voice_profiles")
MODEL_DIR = os.path.join(HERE, "models", "openvoice_v2")
MODEL_URL = "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/"
VC_SR = 22050
BASE_SAMPLE = ("Hello there. I'm reading a short passage so the system can learn how this voice sounds. "
               "The weather has been pleasant this week, and the team finished the project ahead of schedule. "
               "Next, we'll review the numbers, answer a few questions, and plan what comes after that. "
               "Thanks for listening, and have a wonderful day.")
_voices = None

# ---------------------------------------------------------------- voice conversion (optional)
vc = None
vc_error = None
_base_embs = {}
_base_lock = threading.Lock()


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in ("config.json", "checkpoint.pth"):
        dest = os.path.join(MODEL_DIR, name)
        if os.path.exists(dest):
            continue
        print(f"Downloading voice model: {name} ...")
        tmp = dest + ".part"
        with urllib.request.urlopen(MODEL_URL + name) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(tmp, dest)


def init_voice_clone():
    global vc, vc_error
    try:
        import voiceclone
        if not voiceclone.available():
            download_model()
        vc = voiceclone
    except ImportError as e:
        vc_error = (f"Voice cloning isn't installed ({e.name} missing). "
                    "Start the app with 'Start Transcriber.bat' to set it up.")
    except Exception as e:
        vc_error = f"Voice cloning unavailable: {e}"


def decode_to_float(src, sr=VC_SR):
    """Decode any audio/video (path or bytes) to mono float32 at sr."""
    if isinstance(src, (bytes, bytearray)):
        r = subprocess.run([FFMPEG, "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                           input=src, capture_output=True)
    else:
        r = subprocess.run([FFMPEG, "-v", "error", "-i", src, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                           capture_output=True)
    if r.returncode:
        raise RuntimeError("Couldn't read the audio: " + r.stderr.decode(errors="ignore").strip()[-300:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def wav_bytes(audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    import wave
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def profile_meta():
    p = os.path.join(PROFILE_DIR, "me.json")
    if os.path.exists(p) and os.path.exists(os.path.join(PROFILE_DIR, "me.pt")):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def my_embedding():
    import torch
    return torch.load(os.path.join(PROFILE_DIR, "me.pt"), weights_only=True)


def base_embedding(voice):
    """Voice fingerprint of a TTS base voice (cached on disk)."""
    import torch
    with _base_lock:
        if voice in _base_embs:
            return _base_embs[voice]
        path = os.path.join(PROFILE_DIR, f"base_{voice}.pt")
        if os.path.exists(path):
            emb = torch.load(path, weights_only=True)
        else:
            emb, _ = vc.speaker_embedding(decode_to_float(tts(BASE_SAMPLE, voice, "+0%")))
            os.makedirs(PROFILE_DIR, exist_ok=True)
            torch.save(emb, path)
        _base_embs[voice] = emb
        return emb


# ---------------------------------------------------------------- TTS
async def _tts(text, voice, rate):
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)


def tts(text, voice, rate):
    last = None
    for attempt in range(4):
        try:
            data = asyncio.run(_tts(text, voice, rate))
            if data:
                return data
            last = "empty audio"
        except Exception as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"TTS failed: {last}")


def us_voices():
    global _voices
    if _voices is None:
        vs = asyncio.run(edge_tts.list_voices())
        _voices = sorted(
            ({"name": v["ShortName"], "gender": v["Gender"]} for v in vs if v["Locale"] == "en-US"),
            key=lambda v: v["name"],
        )
    return _voices


# ---------------------------------------------------------------- video merge
def run_ffmpeg(args):
    return subprocess.run([FFMPEG, "-y", "-v", "error", *args], capture_output=True, text=True)


def mux(video, wav, out, keep):
    """Put the voice track on the video; optionally keep original audio at volume `keep`."""
    def attempt(keep, vcodec):
        args = ["-i", video, "-i", wav]
        if keep > 0:
            args += ["-filter_complex",
                     f"[0:a]volume={keep}[bg];[bg][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
                     "-map", "0:v:0", "-map", "[a]"]
        else:
            args += ["-map", "0:v:0", "-map", "1:a"]
        args += vcodec + ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
        return run_ffmpeg(args)

    r = attempt(keep, ["-c:v", "copy"])
    if r.returncode and keep > 0 and "matches no streams" in r.stderr:
        keep = 0                                   # video has no audio track
        r = attempt(keep, ["-c:v", "copy"])
    if r.returncode:                               # codec can't go into MP4 as-is: re-encode
        r = attempt(keep, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-800:])


# ---------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")     # always load the latest page after an update
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def origin_allowed(self):
        """Requests without an Origin come from the helper's own page or a local tool."""
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS

    def do_OPTIONS(self):
        # CORS / Private Network Access preflight from the hosted page (e.g. GitHub Pages)
        if not self.origin_allowed():
            return self.send_error_text("origin not allowed", 403)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_bytes(self, data, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status=200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", status)

    def send_error_text(self, msg, status=500):
        self.send_bytes(str(msg).encode(), "text/plain; charset=utf-8", status)

    def read_body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def read_body_to(self, path):
        remaining = int(self.headers.get("Content-Length", 0))
        with open(path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/") and not self.origin_allowed():
            return self.send_error_text("origin not allowed", 403)
        if path == "/":
            self.path = "/transcriber.html"
        elif path == "/api/voices":
            try:
                return self.send_json(us_voices())
            except Exception as e:
                return self.send_error_text(f"Couldn't load voices (are you online?): {e}", 502)
        elif path == "/api/vc/status":
            return self.send_json({"available": vc is not None, "reason": vc_error, "profile": profile_meta()})
        elif not re.fullmatch(r"/[\w.\- ]+\.(html|js|css|png|svg|ico)", unquote(path), re.I):
            return self.send_error_text("not found", 404)       # only serve the page, never profiles/models
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if not self.origin_allowed():
            return self.send_error_text("origin not allowed", 403)
        try:
            if url.path == "/api/tts":
                body = json.loads(self.read_body())
                text = str(body.get("text", "")).strip()
                voice = str(body.get("voice", "en-US-AriaNeural"))
                rate = str(body.get("rate", "+0%"))
                if not text or not RATE_RE.match(rate) or not VOICE_RE.match(voice):
                    return self.send_error_text("bad request", 400)
                mp3 = tts(text, voice, rate)
                if not body.get("clone"):
                    return self.send_bytes(mp3, "audio/mpeg")
                if vc is None:
                    return self.send_error_text(vc_error or "voice cloning unavailable", 503)
                if not profile_meta():
                    return self.send_error_text("Create your voice profile first.", 409)
                me = my_embedding()
                audio = vc.convert(decode_to_float(mp3), base_embedding(voice), me)
                if body.get("strong"):          # second pass pulls the timbre further toward the profile
                    audio = vc.convert(audio, vc.speaker_embedding(audio)[0], me)
                return self.send_bytes(wav_bytes(audio, VC_SR), "audio/wav")

            if url.path == "/api/vc/profile":
                if vc is None:
                    return self.send_error_text(vc_error or "voice cloning unavailable", 503)
                src = os.path.join(TMP, uuid.uuid4().hex + ".media")
                self.read_body_to(src)
                try:
                    audio = decode_to_float(src)
                finally:
                    os.remove(src)
                if len(audio) < VC_SR * 5:
                    return self.send_error_text("The sample is too short. Use at least 10 seconds of speech.", 400)
                import torch
                emb, secs = vc.speaker_embedding(audio)
                os.makedirs(PROFILE_DIR, exist_ok=True)
                torch.save(emb, os.path.join(PROFILE_DIR, "me.pt"))
                meta = {"seconds": round(secs, 1), "source": q.get("source", ["recording"])[0][:120],
                        "created": time.strftime("%Y-%m-%d %H:%M")}
                with open(os.path.join(PROFILE_DIR, "me.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                return self.send_json(meta)

            if url.path == "/api/vc/profile/delete":
                for n in ("me.pt", "me.json"):
                    p = os.path.join(PROFILE_DIR, n)
                    if os.path.exists(p):
                        os.remove(p)
                return self.send_json({"ok": True})

            if url.path == "/api/upload":
                vid = uuid.uuid4().hex
                self.read_body_to(os.path.join(TMP, vid + ".video"))
                return self.send_json({"id": vid})

            if url.path == "/api/mux":
                vid = q.get("video", [""])[0]
                if not re.fullmatch(r"[0-9a-f]{32}", vid):
                    return self.send_error_text("bad video id", 400)
                video = os.path.join(TMP, vid + ".video")
                if not os.path.exists(video):
                    return self.send_error_text("video not found, upload again", 404)
                keep = max(0.0, min(1.0, float(q.get("keep", ["0"])[0])))
                job = uuid.uuid4().hex
                wav, out = os.path.join(TMP, job + ".wav"), os.path.join(TMP, job + ".mp4")
                self.read_body_to(wav)
                try:
                    mux(video, wav, out, keep)
                    with open(out, "rb") as f:
                        return self.send_bytes(f.read(), "video/mp4")
                finally:
                    for p in (wav, out):
                        if os.path.exists(p):
                            os.remove(p)

            self.send_error_text("not found", 404)
        except Exception as e:
            self.send_error_text(e, 500)


def main():
    init_voice_clone()
    print("Voice cloning: " + ("ready" if vc else vc_error))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"Video Transcriber running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
