"""
subs2voice.py - Turn an .srt / .vtt subtitle file into a time-synced US-English voice track,
and optionally put that voice onto the original video.

Setup (once):
    python -m pip install edge-tts imageio-ffmpeg

Examples:
    python subs2voice.py talk.srt                       # -> talk_voice.wav
    python subs2voice.py talk.srt --video talk.mp4      # -> talk_voice.wav + talk_dubbed.mp4
    python subs2voice.py talk.vtt --video talk.mp4 --keep-original 0.15   # original audio quietly underneath
    python subs2voice.py talk.srt --voice en-US-GuyNeural --rate +5%
    python subs2voice.py --list-voices

Note: edge-tts uses Microsoft's online text-to-speech service, so subtitle text is sent to Microsoft.
"""
import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile
import wave

import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 24000          # output sample rate (Hz), mono 16-bit
BPS = 2             # bytes per sample

TIME_RE = re.compile(
    r"((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)


def to_seconds(ts):
    parts = ts.replace(",", ".").split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def parse_subs(path):
    """Parse SRT or VTT into [(start, end, text)]."""
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    cues, i = [], 0
    while i < len(lines):
        m = TIME_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = to_seconds(m.group(1)), to_seconds(m.group(2))
        i += 1
        text = []
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip())
            i += 1
        txt = re.sub(r"<[^>]+>", "", " ".join(text))   # drop <i>, <v Speaker>, etc.
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            cues.append((start, end, txt))
    cues.sort(key=lambda c: c[0])
    return cues


def decode_pcm(mp3_path, tempo=1.0):
    """Decode audio to raw mono s16le PCM at SR, optionally sped up by `tempo`."""
    cmd = [FFMPEG, "-v", "error", "-i", mp3_path]
    if tempo > 1.001:
        cmd += ["-af", f"atempo={tempo:.4f}"]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


async def synth(idx, text, voice, rate, out_dir, sem):
    path = os.path.join(out_dir, f"{idx:05d}.mp3")
    async with sem:
        for attempt in range(4):
            try:
                await edge_tts.Communicate(text, voice, rate=rate).save(path)
                return path
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"TTS failed for cue {idx + 1}: {text!r}: {e}")
                await asyncio.sleep(1.5 * (attempt + 1))


async def synth_all(cues, voice, rate, out_dir, workers):
    sem = asyncio.Semaphore(workers)
    done = 0

    async def one(i, text):
        nonlocal done
        p = await synth(i, text, voice, rate, out_dir, sem)
        done += 1
        print(f"\r  speech: {done}/{len(cues)}", end="", flush=True)
        return p

    paths = await asyncio.gather(*(one(i, c[2]) for i, c in enumerate(cues)))
    print()
    return paths


def build_track(cues, clips, max_speed):
    """Place each clip at its cue time. Clips that don't fit before the next cue are sped up
    (up to max_speed); anything still too long is pushed later so lines never overlap."""
    buf = bytearray()
    cursor = 0.0            # end of last placed clip, in seconds
    sped, shifted = 0, 0
    for i, ((start, end, _), clip) in enumerate(zip(cues, clips)):
        pcm = decode_pcm(clip)
        dur = len(pcm) / (SR * BPS)
        slot = (cues[i + 1][0] - start) if i + 1 < len(cues) else float("inf")
        if dur > slot > 0:
            tempo = min(dur / slot, max_speed)
            pcm = decode_pcm(clip, tempo)
            dur = len(pcm) / (SR * BPS)
            sped += 1
        at = max(start, cursor)
        if at > start + 0.05:
            shifted += 1
        offset = int(at * SR) * BPS
        if len(buf) < offset:
            buf.extend(b"\x00" * (offset - len(buf)))
        buf[offset:offset + len(pcm)] = pcm
        cursor = at + dur
        print(f"\r  placing: {i + 1}/{len(cues)}", end="", flush=True)
    print()
    return bytes(buf), sped, shifted


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(BPS)
        w.setframerate(SR)
        w.writeframes(pcm)


def mux(video, voice_wav, out, keep_original):
    cmd = [FFMPEG, "-y", "-v", "error", "-i", video, "-i", voice_wav]
    if keep_original > 0:
        cmd += ["-filter_complex",
                f"[0:a]volume={keep_original}[bg];[bg][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
                "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and keep_original > 0 and "matches no streams" in r.stderr:
        print("  video has no audio track; using the voice only")
        return mux(video, voice_wav, out, 0)
    if r.returncode != 0:
        sys.exit("ffmpeg failed:\n" + r.stderr)


async def list_voices():
    for v in sorted(await edge_tts.list_voices(), key=lambda v: v["ShortName"]):
        if v["Locale"] == "en-US":
            print(f'{v["ShortName"]:32} {v["Gender"]}')


def main():
    ap = argparse.ArgumentParser(description="Subtitles (.srt/.vtt) -> US-English voice track")
    ap.add_argument("subs", nargs="?", help="input .srt or .vtt file")
    ap.add_argument("--video", help="original video; writes <name>_dubbed.mp4 with the new voice")
    ap.add_argument("--voice", default="en-US-AriaNeural",
                    help="voice name (default en-US-AriaNeural; try en-US-GuyNeural, en-US-JennyNeural, "
                         "en-US-AndrewNeural, en-US-EmmaNeural, en-US-BrianNeural)")
    ap.add_argument("--rate", default="+0%", help='base speaking rate, e.g. "+10%%" or "-5%%"')
    ap.add_argument("--max-speed", type=float, default=1.5,
                    help="max speed-up for lines that are too long for their slot (default 1.5)")
    ap.add_argument("--keep-original", type=float, default=0.0, metavar="VOL",
                    help="keep original audio under the new voice at this volume (0-1, e.g. 0.15)")
    ap.add_argument("--out", help="output .wav path (default <subs>_voice.wav)")
    ap.add_argument("--workers", type=int, default=6, help="parallel TTS requests (default 6)")
    ap.add_argument("--list-voices", action="store_true", help="list available en-US voices and exit")
    a = ap.parse_args()

    if a.list_voices:
        asyncio.run(list_voices())
        return
    if not a.subs:
        ap.error("give a .srt or .vtt file (or --list-voices)")

    cues = parse_subs(a.subs)
    if not cues:
        sys.exit("No subtitle cues found.")
    base = os.path.splitext(a.subs)[0]
    out_wav = a.out or base + "_voice.wav"
    print(f"{len(cues)} cues, voice {a.voice}")

    with tempfile.TemporaryDirectory() as tmp:
        clips = asyncio.run(synth_all(cues, a.voice, a.rate, tmp, a.workers))
        pcm, sped, shifted = build_track(cues, clips, a.max_speed)

    write_wav(out_wav, pcm)
    print(f"Wrote {out_wav} ({len(pcm) / (SR * BPS):.1f}s)")
    if sped:
        print(f"  {sped} line(s) sped up to fit their timing")
    if shifted:
        print(f"  {shifted} line(s) started late because the previous line ran long "
              f"(raise --max-speed or --rate to tighten)")

    if a.video:
        out_mp4 = os.path.splitext(a.video)[0] + "_dubbed.mp4"
        print("Merging with video...")
        mux(a.video, out_wav, out_mp4, a.keep_original)
        print(f"Wrote {out_mp4}")


if __name__ == "__main__":
    main()
