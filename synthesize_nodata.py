import asyncio
import base64
import json
import random
import re
import subprocess
import zlib
from pathlib import Path

import edge_tts
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

SRT_ZLIB_B64 = "".join(p.read_text().strip() for p in sorted(Path(".").glob("srt_part_*.txt")))
VOICE = "zh-CN-YunxiNeural"
RATE = "+8%"
PITCH = "-3Hz"
WORK = Path("work")
SEGMENTS = WORK / "segments"


def ms(ts: str) -> int:
    h, m, rest = ts.split(":")
    s, milli = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(milli)


def parse_srt(text: str):
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        a, b = lines[1].split(" --> ")
        t = "".join(lines[2:]).strip()
        t = re.sub(r"\s+", " ", t)
        t = t.replace("fromAfter", "from After")
        cues.append({"start": ms(a), "end": ms(b), "text": t})
    return cues


def make_groups(cues):
    groups = []
    cur = []
    for cue in cues:
        if cur:
            gap = cue["start"] - cur[-1]["end"]
            chars = sum(len(x["text"]) for x in cur)
            span = cur[-1]["end"] - cur[0]["start"]
            terminal = bool(re.search(r"[。！？!?]$", cur[-1]["text"]))
            if gap > 280 or terminal or chars >= 58 or span >= 12000:
                groups.append(cur)
                cur = []
        cur.append(cue)
    if cur:
        groups.append(cur)
    return groups


def join_text(group):
    out = ""
    for cue in group:
        t = cue["text"].strip()
        if not out:
            out = t
        elif out[-1] in "，、：；（(—-" or t[:1] in "，。！？；：,.!?;:）)":
            out += t
        else:
            out += t
    return out


def atempo_chain(speed: float) -> str:
    vals = []
    while speed > 2.0:
        vals.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        vals.append(0.5)
        speed /= 0.5
    vals.append(speed)
    return ",".join(f"atempo={v:.6f}" for v in vals)


async def synth_one(i, text, sem):
    out = SEGMENTS / f"{i:04d}.mp3"
    async with sem:
        for attempt in range(6):
            try:
                c = edge_tts.Communicate(text=text, voice=VOICE, rate=RATE, pitch=PITCH)
                await c.save(str(out))
                if out.exists() and out.stat().st_size > 1000:
                    return str(out)
            except Exception:
                if attempt == 5:
                    raise
                await asyncio.sleep((attempt + 1) * 2 + random.random() * 2)
    return str(out)


async def synth_all(groups):
    sem = asyncio.Semaphore(4)
    tasks = []
    for i, g in enumerate(groups, 1):
        tasks.append(asyncio.create_task(synth_one(i, join_text(g), sem)))
        await asyncio.sleep(0.04)
    results = []
    for i, task in enumerate(tasks, 1):
        try:
            results.append(await task)
        except Exception as exc:
            print(f"SYNTH_FAIL {i}: {exc}", flush=True)
            results.append(None)
    return results


def trim_edges(seg: AudioSegment) -> AudioSegment:
    if len(seg) < 150:
        return seg
    ranges = detect_nonsilent(seg, min_silence_len=60, silence_thresh=-48)
    if not ranges:
        return seg
    start = max(0, ranges[0][0] - 45)
    end = min(len(seg), ranges[-1][1] + 70)
    return seg[start:end]


def main():
    WORK.mkdir(exist_ok=True)
    SEGMENTS.mkdir(exist_ok=True)
    srt = zlib.decompress(base64.b64decode(SRT_ZLIB_B64)).decode("utf-8")
    cues = parse_srt(srt)
    groups = make_groups(cues)
    print(f"cues={len(cues)} groups={len(groups)}", flush=True)
    paths = asyncio.run(synth_all(groups))
    total = max(c["end"] for c in cues) + 1000
    timeline = AudioSegment.silent(duration=total, frame_rate=24000).set_channels(1).set_sample_width(2)
    metrics = []
    missing = 0
    for i, (group, path) in enumerate(zip(groups, paths), 1):
        start = group[0]["start"]
        target = max(250, group[-1]["end"] - start - 35)
        if not path:
            missing += 1
            continue
        seg = AudioSegment.from_file(path).set_frame_rate(24000).set_channels(1).set_sample_width(2)
        seg = trim_edges(seg)
        original = len(seg)
        speed = 1.0
        if len(seg) > target:
            speed = len(seg) / target * 1.012
            wav = SEGMENTS / f"{i:04d}_fit.wav"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                "-af", atempo_chain(speed), "-ar", "24000", "-ac", "1", str(wav)
            ], check=True)
            seg = AudioSegment.from_file(wav).set_frame_rate(24000).set_channels(1).set_sample_width(2)
            seg = trim_edges(seg)
        if len(seg) > target:
            seg = seg[:target]
        seg = seg.fade_in(min(25, len(seg) // 3)).fade_out(min(35, len(seg) // 3))
        timeline = timeline.overlay(seg, position=start)
        metrics.append({
            "i": i,
            "start": start,
            "target_ms": target,
            "source_ms": original,
            "final_ms": len(seg),
            "speed": round(speed, 3),
            "text": join_text(group),
        })
        if i % 25 == 0:
            print(f"aligned {i}/{len(groups)}", flush=True)
    wav = WORK / "narration.wav"
    timeline.export(wav, format="wav")
    out = Path("output")
    out.mkdir(exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
        "-af", "highpass=f=70,lowpass=f=15500,loudnorm=I=-16:TP=-1.5:LRA=8",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        str(out / "narration.m4a")
    ], check=True)
    (out / "metrics.json").write_text(json.dumps({
        "voice": VOICE,
        "rate": RATE,
        "pitch": PITCH,
        "cues": len(cues),
        "groups": len(groups),
        "missing": missing,
        "items": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "source.srt").write_text(srt, encoding="utf-8")
    print(f"DONE missing={missing}", flush=True)


if __name__ == "__main__":
    main()
