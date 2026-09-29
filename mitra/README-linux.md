# Running Mitra on Linux

The main `README.md` targets a MacBook Pro M1 Max. Three layers of the stack are
Apple-Silicon-only and one Linux-only bug blocks the robot daemon's media
server. This document covers the differences. Everything not mentioned here
works as the main README describes.


---

## What differs from the macOS path

| Layer | macOS | Linux | Why |
|---|---|---|---|
| ASR | `mlx-whisper` | `transformers` | mlx has no non-Darwin wheels |
| Wake ASR | `mlx-whisper` | `transformers` | same |
| TTS | Indic Parler-TTS | VITS (`facebook/mms-tts-hin`) | parler-tts pins `transformers<=4.46.1`, which collides with reachy-mini's `huggingface-hub>=1.17` — not co-installable |
| Compute device | `mps` | `cpu` (see note) | no Metal; 8 GB VRAM is fully used by the LLM |
| Sim launcher | `mjpython -m reachy_mini.daemon.app.main --sim` | `reachy-mini-daemon --sim` | mjpython is a macOS GUI requirement |
| Robot media | WebRTC via daemon | `--no-media` + host mic | GStreamer `webrtcsink` missing on Ubuntu 22.04 |
| Microphone | robot's USB mic | host mic via sounddevice | consequence of the above |

---

## 1. System packages

```bash
sudo apt install python3-venv portaudio19-dev libportaudio2 ffmpeg
```

**If apt reports a broken `v4l2loopback-dkms`:** that package fails to build
against kernel 6.8 and will block every later apt operation. It is unrelated to
Mitra. Remove it:

```bash
sudo apt remove --purge v4l2loopback-dkms
sudo apt --fix-broken install
```

**A note on JACK:** `portaudio19-dev` pulls in `libjack-dev` (jack1), which
forces removal of `libjack-jackd2-0` (jack2). Nothing breaks — jack1 satisfies
the same alternative — but if you run a jack2 setup, install
`libjack-jackd2-dev` instead to keep it. `portaudio19-dev` is optional anyway:
the `sounddevice` wheel bundles its own PortAudio.

---

## 2. Disk space

Budget **at least 15 GB free** before starting. The model downloads are large
and fail confusingly when space runs out (HuggingFace warns but the process
appears to hang):

- Qwen3-VL 8B via Ollama: ~6 GB
- Whisper large-v3-turbo: ~1.6 GB
- Whisper small (wake word): ~500 MB
- MMS-TTS Hindi: ~250 MB
- PyTorch + CUDA wheels: several GB

```bash
df -h ~
```

If tight, `pip cache purge`, `sudo apt clean && sudo apt autoremove --purge`,
and `journalctl --vacuum-size=200M` are the usual wins. Old models in
`~/.cache/huggingface/hub` are often the single biggest item.

---

## 3. Python environment

Pick **one** interpreter and stay on it. The most common failure on this project
is a half-created virtualenv shadowing a working global install: the shell shows
`(.venv)` while `pip` writes to `~/.local`, and every import fails.

If you use a venv, verify it has pip before installing anything:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade      # Ubuntu's venv sometimes ships without pip
python -m pip --version            # must point inside .venv
```

If you skip the venv, use `python3` and `python3 -m pip` explicitly, never bare
`python` or `pip`. To confirm at any time:

```bash
which python3
python3 -c "import sounddevice; print(sounddevice.__file__)"
```

VS Code auto-activates any `.venv` directory it finds. If you delete one,
also set the interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*) or new
terminals will keep re-activating a directory that no longer exists.

---

## 4. Install

```bash
pip install "reachy-mini[mujoco]"
pip install -e '.[agent,wake,vad,speech]'
pip install sounddevice
ollama pull qwen3-vl:8b-instruct
```

**Do not run** the `parler-tts` line from the main README:

```bash
# pip install git+https://github.com/huggingface/parler-tts.git   # macOS only
```

It downgrades `transformers` to 4.46.1 and `huggingface-hub` to 0.36, breaking
`reachy-mini`. If you have already run it:

```bash
pip uninstall -y parler_tts descript-audio-codec descript-audiotools
pip install --upgrade "transformers>=4.57" "huggingface-hub>=1.17"
```

The `asr` extra is a no-op on Linux (its marker is `arm64 and darwin`), so
including or omitting it makes no difference.

Ollama runs as a systemd service after install — `ollama serve` will report
`address already in use`, which is expected and not an error.

---

## 5. Replace four files

These are the Linux-adapted versions. Each keeps the macOS behaviour as its
default, so a single checkout serves both platforms.

| Replace | With | What changed |
|---|---|---|
| `src/audio/asr.py` | [asr.py](PLACEHOLDER_LINK_ASR) | adds a `transformers` backend alongside `mlx` |
| `src/audio/wake.py` | [wake.py](PLACEHOLDER_LINK_WAKE) | same, plus a `backend`/`device` pair on the factory |
| `src/speech/tts.py` | [tts.py](PLACEHOLDER_LINK_TTS) | adds `engine: vits` to skip parler entirely |
| `main.py` | [main.py](PLACEHOLDER_LINK_MAIN) | `--check` probes the configured backend; passes `engine` to TTS |
| `config.yaml` | [config.yaml](PLACEHOLDER_LINK_CONFIG) | Linux values, macOS values kept as inline comments |

If you prefer to patch by hand rather than replace, the changes are additive:
every `if backend == "mlx"` branch gains a `transformers` sibling, and no
existing default changes.

Verify:

```bash
python3 main.py --check
```

Expected on Linux:

```
components:
  ok       reachy-mini (robot/sim)
  ok       strands-agents (agent)
  ok       openwakeword (wake)
  ok       silero-vad (VAD)
  ok       transformers (ASR, backend=transformers)
  ok       transformers (TTS, engine=vits)
ollama:
  ok       server at http://localhost:11434
  ok       model qwen3-vl:8b-instruct
lexicon: 53 seed entries
```

---

## 6. Configure the microphone

The simulator has no microphone, and with `--no-media` (next section) the robot
provides no audio at all. Capture from the host instead.

List devices:

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Use `pulse` rather than a raw `hw:` entry — PulseAudio/PipeWire handles source
selection, and grabbing an ALSA device directly conflicts with anything else
using audio.

```yaml
robot:
  mic_source: built_in
  built_in_mic_device: "pulse"
```

Confirm capture works before running Mitra:

```bash
python3 - <<'EOF'
import sounddevice as sd, numpy as np
print("recording 3s, speak now")
a = sd.rec(3*16000, samplerate=16000, channels=1, device="pulse")
sd.wait()
print("peak:", float(np.abs(a).max()), "rms:", float(np.sqrt((a**2).mean())))
EOF
```

Peak should be comfortably above 0.1. If it is near zero, open `pavucontrol`,
go to *Recording* while the test runs, and confirm the correct source is
selected and its level is up.

---

## 7. Device placement on 8 GB VRAM

With `keep_alive: 30m`, Qwen3-VL 8B occupies roughly 6.5-7 GB. That leaves no
room for the speech models, so the shipped Linux config puts all three on CPU:

| Component | Q4 size | Device |
|---|---|---|
| Qwen3-VL 8B (+ vision, KV cache) | ~6.5-7 GB | GPU, via Ollama |
| Whisper large-v3-turbo (main ASR) | ~1.6 GB | CPU |
| Whisper small (wake) | ~250 MB | CPU |
| MMS-TTS Hindi | ~250 MB | CPU |

If turn latency is unacceptable, move the **main ASR** to `cuda:0` first — it
processes the longest audio, so the speedup is largest — and make room by
shortening `keep_alive` or dropping to `qwen3-vl:4b-instruct`. Keep the wake
detector on CPU regardless: it runs on every gated speech window, continuously.

On a machine with less than 16 GB RAM, or if disk is tight, set
`asr.default: "openai/whisper-small"`. It is already cached by the wake
detector, loads in seconds, and costs some accuracy on longer sentences.

---

## 8. Run

Three terminals.

**Terminal 1 — simulator.** Note `--no-media`:

```bash
reachy-mini-daemon --sim --no-media
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. A MuJoCo window opens. The
Wayland GLFW warning about window position is harmless.

Without `--no-media` the daemon logs:

```
Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?
```

and the media server never binds, so the client's WebRTC connection is refused
with `ConnectionRefusedError: [Errno 111]` even though port 8000 is up. The
plugin lives in `gst-plugins-rs` and is not packaged for Ubuntu 22.04; building
it is possible but unnecessary, since the simulator has no real camera or
microphone to stream.

**Terminal 2 — Ollama.** Already running as a service. Verify only:

```bash
curl -s localhost:11434 && echo
```

**Sanskrit checks (one-time setup).** Mitra validates every reply against a
real Sanskrit lexicon rather than a list of banned words; the data is ~78 MB
and lives outside git:

```bash
pip install 'mitra[sanskrit]'
python3 scripts/fetch_sanskrit_data.py        # vidyut morphology + Cologne dicts
python3 scripts/build_dictionary.py           # index MW/Apte for the review CLI
python3 scripts/eval_grammar.py               # score the checks on real sentences
```

Without it Mitra runs exactly as before, logging one warning at startup.

**Terminal 3 — Mitra:**

```bash
python3 main.py --debug
```

`--debug` also mirrors every spoken line in English, so the console is
readable without Devanagari:

```
INFO mitra: speak: मह्यं गणितं रोचते।
INFO mitra: speak (en): Mathematics is pleasing to me.
```

The gloss is a second, history-free call to the same Ollama model (~0.4 s,
made after playback starts, so it never delays speech) and it translates
literally — a mangled reply reads as mangled English rather than being
quietly repaired. Turn it off with `logging.gloss_english: false`.

First run downloads ~2 GB of Whisper and TTS weights. Progress bars are
suppressed by `HF_HUB_DISABLE_PROGRESS_BARS` in `main.py`; to watch progress,
comment that line out, or run in another terminal:

```bash
watch -n2 'du -sh ~/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo'
```

Loading Whisper large-v3-turbo onto CPU takes several minutes even after the
download completes. **Wait for `Mitra asleep — say the wake word` before
speaking.** Nothing is listening until that line appears.

---

## 9. Known limitations on this path

Both follow from `--no-media`, and both are worth fixing before any demo.

**No camera.** `capture_image()` returns nothing, so vision turns do not work.
Object recognition — the core of FR-2 — cannot be tested on this setup.

**No speaker output.** `speaker_play()` pushes to the daemon's media manager,
which logs `Audio system is not initialized.` and discards the audio. Replies
appear in the `--debug` console but are not audible. The fix mirrors what
`_start_built_in_mic` does for input: roughly fifteen lines adding a
`sounddevice.OutputStream` path to `ReachyRobot.speaker_play`, gated on the same
`mic_source: built_in` flag.

---

## 10. Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'`** — wrong interpreter. See
section 3. `deactivate; hash -r; which python3`.

**Daemon exits with `address already in use`** — an older daemon is still
running. `pkill -f reachy-mini-daemon`, wait, confirm with `ss -ltnp | grep 8000`.

**Startup hangs after `warming up ASR`** — either still downloading or out of
disk. Check both:

```bash
df -h ~
du -sh ~/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo
```

**No `wake check` lines when you speak** — audio is not reaching the detector.
Re-run the mic test in section 6. Also try `scripts/wake_probe.py`, which prints
live RMS against the energy gate.

**`wake check` fires but the transcript is nonsense** — repeated characters
(`RRRRRR...`), stock phrases (`Thank you.`, `you`), or the wake word rendered in
another script (`헤이 미트라`) are all Whisper hallucinating on short clips,
usually because it guessed the wrong language. Pin it in `_hf_transcribe` in
`src/audio/wake.py`:

```python
out = self._pipeline(
    {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000},
    generate_kwargs={"language": "en", "task": "transcribe"},
)
```

The wake detector only ever matches one English word, so there is no reason to
let it auto-detect. Do **not** pin the language in `src/audio/asr.py` — the main
ASR genuinely needs detection across English, Kannada, and Sanskrit.

If the transcript is plausible but simply not matched, note that Whisper spells
the word many ways. `TranscriptWakeDetector._variants` is where to add
spellings.

**Raw microphone level is low** — `pavucontrol`, *Input Devices*, raise the
level until speech clears the meter's midpoint. Whisper hallucinates far more on
quiet input than on loud input.
