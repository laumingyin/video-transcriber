# Video Transcriber & US Voice-over

A local web app that:

1. **Transcribes** speech from a video or audio file (Whisper, running in your browser).
2. **Re-voices** the transcript in a US-English accent and puts it back on the video.
3. Optionally makes the new voice **sound like you**, keeping the US accent (OpenVoice voice conversion, running locally).

Everything can run in one step: pick a video, click **Transcribe + dub in one step**, get a dubbed MP4.

## Use it online

**<https://laumingyin.github.io/video-transcriber/>**

- **Transcribe** works right in the browser, with nothing to install (Chrome or Edge recommended).
- **Voice-over** needs the local helper running on your PC (see *Quick start*). The online page connects to it at `http://localhost:8765`. If the browser asks to allow access to devices on your local network, choose **Allow**.

The helper only accepts requests from its own page and from `https://laumingyin.github.io`. To allow another site (for example a fork's GitHub Pages address), set `ALLOWED_ORIGINS=https://you.github.io` before starting it.

## Quick start (Windows)

1. Install [Python](https://www.python.org/downloads/) 3.13 (3.14 also works).
2. Double-click **`Start Transcriber.bat`**.
   The first run creates a local Python environment and downloads the voice engine (~1.3 GB, a few minutes).
3. The app opens at <http://localhost:8765>. Keep the launcher window open while you use it.

Manual start on other platforms:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps https://github.com/myshell-ai/OpenVoice/archive/refs/heads/main.zip
.venv/bin/python server.py
```

## Using it

**Tab 1 · Transcribe**
- Drop in a video or audio file, choose a model (Tiny / Base / Small) and language.
- **Transcribe only** gives you text with timestamps, downloadable as `.txt`, `.srt` or `.vtt`.
- **Transcribe + dub in one step** continues straight into the voice-over. Tick *Let me review the text* to stop and edit first.

**Tab 2 · Voice-over**
- Load the transcript from tab 1 or any `.srt` / `.vtt` file. Every line is editable.
- Choose one of 17 US voices, the speaking rate, and how much long lines may be sped up to fit.
- **Sound like me:** create a voice profile from your video, an uploaded recording, or a 30-second microphone recording. *Voice match: Strong* gets closer to your voice at 2× the processing time.
- Add the original video to get a dubbed `.mp4`. You can keep the original audio quietly underneath.

## Privacy

| Data | Where it goes |
|---|---|
| Video / audio for transcription | Stays in your browser |
| Video for merging, voice samples, voice profile | Local helper on `127.0.0.1` only |
| Subtitle text | Sent to Microsoft's online TTS service (edge-tts) to create the base US voice |

Voice profiles are stored in `voice_profiles/`, which is git-ignored. Only clone voices you own or have explicit permission to use.

## Files

| File | Purpose |
|---|---|
| `transcriber.html` | The web app (transcription runs fully in the page) |
| `server.py` | Local helper: serves the page, text-to-speech, voice conversion, video merging |
| `voiceclone.py` | Wrapper around the OpenVoice V2 tone-color converter |
| `subs2voice.py` | Command-line alternative: `.srt` / `.vtt` → voice track / dubbed video |
| `Start Transcriber.bat` | One-click setup and launch on Windows |
| `requirements.txt` | Python dependencies (CPU build of PyTorch) |

## Credits

- [Whisper](https://github.com/openai/whisper) via [Transformers.js](https://github.com/huggingface/transformers.js)
- [edge-tts](https://github.com/rany2/edge-tts) for US neural voices
- [OpenVoice V2](https://github.com/myshell-ai/OpenVoice) (MIT) for voice conversion
- FFmpeg via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)
