"""Thin wrapper over the ``reachy-mini`` SDK: camera, microphone, speaker, head.

All hardware access in Mitra goes through this module (DESIGN §2). The same
wrapper talks to a real Reachy Mini Lite over USB or to the MuJoCo simulation
daemon (``mjpython -m reachy_mini.daemon.app.main --sim`` on macOS,
``reachy-mini-daemon --sim`` elsewhere) — the daemon API is identical, so
nothing above this layer knows which one is running. ``FakeReachy`` is the
in-process test double (no daemon at all).
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from mitra.audio import resample

_PLAYBACK_CHUNK_S = 0.25  # push audio in small chunks so barge-in can stop it
_HOST_OUTPUT_SR = 22050   # used when playing through the host sound card


class ReachyRobot:
    """Connects to the reachy-mini daemon (real robot or --sim)."""

    def __init__(self, mic_chunk_s: float = 0.08, mic_source: str = "robot",
                 built_in_mic_device: str = "MacBook Pro Microphone"):
        """mic_source: "robot" (default) reads the robot's own mic through the
        SDK, and plays speech back through the robot's speaker.

        "built_in" instead captures from a host input device via
        sounddevice/PortAudio AND plays back through the host's default output.
        Two different problems need this same escape hatch:

        - macOS: a macOS 26 (Tahoe) Core Audio regression returns all-zero
          samples for the robot's multichannel USB mic
          (pollen-robotics/reachy_mini#820). Capture is broken, playback works.
        - Linux: the daemon's media server needs the GStreamer webrtc plugin
          (``webrtcsink``), which is not packaged for Ubuntu 22.04, so the
          daemon must run with ``--no-media`` and exposes no audio at all.
          ``get_output_audio_samplerate()`` then returns -1, which makes
          ``resample()`` compute a negative sample count and raise. Hence the
          hardcoded output rate below rather than asking the SDK.

        Camera and head motion always go through the robot regardless.
        """
        try:
            from reachy_mini import ReachyMini
            from reachy_mini.utils import create_head_pose
        except ImportError as e:
            raise ImportError(
                "The 'reachy-mini' package is required for the real/sim robot. "
                "Install with: pip install 'reachy-mini[mujoco]'"
            ) from e
        self._create_head_pose = create_head_pose
        self._mini = ReachyMini()

        # Set BEFORE the samplerate probe below — the probe is what fails when
        # the daemon has no media pipeline.
        self._mic_source = mic_source
        self._use_host_audio = mic_source == "built_in"

        # start_recording/start_playing are no-ops (and may warn) under
        # --no-media; never let them abort the connection.
        for _start in (self._mini.media.start_recording,
                       self._mini.media.start_playing):
            try:
                _start()
            except Exception:
                pass

        if self._use_host_audio:
            self._out_sr = _HOST_OUTPUT_SR
        else:
            self._out_sr = int(self._mini.media.get_output_audio_samplerate())
            if self._out_sr <= 0:
                # Daemon reported no usable speaker (e.g. started --no-media)
                # while mic_source=robot. Fall back rather than crash on the
                # first reply, which is the only place a bad rate surfaces.
                self._out_sr = _HOST_OUTPUT_SR
                self._use_host_audio = True

        self._mic_chunk_s = mic_chunk_s
        self._stop_playback = threading.Event()
        self._playing = threading.Event()
        # Ring buffer + drain thread(s) feed mic_read() identically regardless
        # of source — the SDK's appsink queue holds only ~2 s, so the mic must
        # be drained at realtime even while consumers (wake ASR, Whisper)
        # block, otherwise GStreamer drops samples.
        self._closed = threading.Event()
        self._mic_buf: deque[np.ndarray] = deque()
        self._mic_samples = 0
        self._mic_ready = threading.Condition()
        self._built_in_stream = None
        self._emotions_lib = None  # lazy: only load the ~85-move library if used
        self._dances_lib = None

        if mic_source == "built_in":
            self._in_sr = 16000  # matches mitra.audio.TARGET_SAMPLERATE
            self._start_built_in_mic(built_in_mic_device)
        elif mic_source == "robot":
            self._in_sr = int(self._mini.media.get_input_audio_samplerate())
            self._mic_cap = 30 * self._in_sr  # ring: keep at most 30 s
            threading.Thread(target=self._mic_drain_loop_robot, daemon=True).start()
        else:
            raise ValueError(f"unknown mic_source: {mic_source!r}")

    # --- camera ---

    def camera_read(self) -> np.ndarray:
        """One frame, numpy (H, W, 3) uint8."""
        return self._mini.media.get_frame()

    # --- microphone ---

    @property
    def mic_samplerate(self) -> int:
        return self._in_sr

    def _push_mic_chunk(self, mono: np.ndarray) -> None:
        with self._mic_ready:
            self._mic_buf.append(mono)
            self._mic_samples += len(mono)
            while self._mic_samples > self._mic_cap:  # drop oldest
                self._mic_samples -= len(self._mic_buf.popleft())
            self._mic_ready.notify_all()

    def _mic_drain_loop_robot(self) -> None:
        """Continuously pull ~10 ms buffers from the SDK into our ring buffer.

        get_audio_sample() returns ONE buffered chunk per call (or None), so
        it must be called in a tight loop to keep up with realtime.
        """
        while not self._closed.is_set():
            try:
                samples = self._mini.media.get_audio_sample()
            except Exception:
                time.sleep(0.1)
                continue
            if samples is None or len(samples) == 0:
                time.sleep(0.003)
                continue
            mono = np.asarray(samples, dtype=np.float32)
            if mono.ndim == 2:
                mono = mono.mean(axis=1)
            self._push_mic_chunk(mono)

    def _start_built_in_mic(self, device_name: str) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "mic_source='built_in' needs sounddevice. "
                "Install with: pip install sounddevice"
            ) from e
        self._mic_cap = 30 * self._in_sr

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            mono = np.asarray(indata, dtype=np.float32)
            if mono.ndim == 2:
                mono = mono.mean(axis=1)
            self._push_mic_chunk(mono)

        self._built_in_stream = sd.InputStream(
            device=device_name, channels=1, samplerate=self._in_sr,
            blocksize=int(self._in_sr * 0.02), callback=callback,
        )
        self._built_in_stream.start()

    def mic_read(self) -> np.ndarray:
        """Mono float32 chunk of ~mic_chunk_s from the ring buffer; blocks up
        to ~one chunk length. Slow consumers delay processing, never capture."""
        target = int(self._in_sr * self._mic_chunk_s)
        deadline = time.monotonic() + max(0.2, 2 * self._mic_chunk_s)
        with self._mic_ready:
            while self._mic_samples < target and not self._closed.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._mic_ready.wait(timeout=remaining)
            if not self._mic_buf:
                return np.zeros(0, dtype=np.float32)
            chunks, taken = [], 0
            while self._mic_buf and taken < target:
                chunk = self._mic_buf.popleft()
                chunks.append(chunk)
                taken += len(chunk)
            self._mic_samples -= taken
        return np.concatenate(chunks)

    def flush_mic(self) -> None:
        """Discard whatever's currently buffered. Call this right after
        speaker playback ends: with mic_source="built_in" there's no echo
        cancellation on the raw mic stream, so it keeps recording the robot's
        own voice while it talks — without a flush, that gets drained and fed
        back in as if it were a new user utterance the moment playback stops."""
        with self._mic_ready:
            self._mic_buf.clear()
            self._mic_samples = 0

    # --- speaker ---

    def speaker_play(self, wav: np.ndarray, samplerate: int, block: bool = True) -> None:
        wav = resample(wav, samplerate, self._out_sr)
        self._stop_playback.clear()
        thread = threading.Thread(target=self._playback_worker, args=(wav,), daemon=True)
        self._playing.set()
        thread.start()
        if block:
            thread.join()

    def _playback_worker(self, wav: np.ndarray) -> None:
        if self._use_host_audio:
            self._playback_host(wav)
            return
        chunk = max(1, int(self._out_sr * _PLAYBACK_CHUNK_S))
        try:
            for i in range(0, len(wav), chunk):
                if self._stop_playback.is_set():
                    break
                part = wav[i:i + chunk]
                self._mini.media.push_audio_sample(part.reshape(-1, 1))
                # push is non-blocking; pace at realtime so stop can interrupt
                time.sleep(len(part) / self._out_sr)
        finally:
            self._playing.clear()

    def _playback_host(self, wav: np.ndarray) -> None:
        """Play through the host's default output device.

        sd.play() is non-blocking and sd.stop() cancels it, so barge-in works
        the same way it does on the robot path — no manual chunking needed.
        """
        try:
            import sounddevice as sd

            sd.play(np.asarray(wav, dtype=np.float32), self._out_sr)
            while sd.get_stream().active:
                if self._stop_playback.is_set():
                    sd.stop()
                    break
                time.sleep(0.02)
        except Exception:
            # A failed reply must never kill the run loop (FR-6.4). The
            # orchestrator logs; here we just make sure _playing clears.
            pass
        finally:
            self._playing.clear()

    def speaker_stop(self) -> None:
        self._stop_playback.set()
        if self._use_host_audio:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass

    def speaker_busy(self) -> bool:
        return self._playing.is_set()

    # --- head ---

    def nod(self) -> None:
        """Single brief nod — the wake acknowledgment (FR-1.3, FR-5.1)."""
        down = self._create_head_pose(pitch=12, degrees=True)
        self._mini.goto_target(head=down, duration=0.3)
        self._mini.goto_target(head=self._create_head_pose(), duration=0.3)

    # State-feedback poses (FR-5.3 extension): head angles in degrees,
    # antennas in radians. Tune freely — these only signal state, they carry
    # no meaning to the pipeline.
    POSES = {
        "neutral":   {"head": {}, "antennas": [0.0, 0.0]},
        "listening": {"head": {"pitch": -6}, "antennas": [0.5, -0.5]},   # perked up
        "thinking":  {"head": {"roll": 14}, "antennas": [0.1, -0.6]},    # quizzical tilt
        "asleep":    {"head": {"pitch": 20}, "antennas": [1.1, -1.1]},   # drooped
    }

    def pose(self, name: str, duration: float = 0.5) -> None:
        """Move to a named state pose without blocking the caller."""
        spec = self.POSES[name]

        def _move():
            try:
                self._mini.goto_target(
                    head=self._create_head_pose(**spec["head"], degrees=True),
                    antennas=spec["antennas"],
                    duration=duration,
                )
            except Exception:
                pass  # a missed gesture must never disturb the pipeline

        threading.Thread(target=_move, daemon=True).start()

    # Recorded emotion/dance library (FR-5.3 extension, additive — POSES
    # above is untouched): Pollen's official ~85-move "emotions" and ~19-move
    # "dances" datasets, pre-downloaded by the daemon at startup. Fuller,
    # slower animations (a few seconds, often with sound) than the instant
    # POSES angles — better suited to occasional "moments" (wake greeting,
    # falling asleep, confusion) than to firing on every conversational turn.
    EMOTION_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
    DANCE_LIBRARY = "pollen-robotics/reachy-mini-dances-library"

    def _emotions(self):
        if self._emotions_lib is None:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._emotions_lib = RecordedMoves(self.EMOTION_LIBRARY)
        return self._emotions_lib

    def _dances(self):
        if self._dances_lib is None:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._dances_lib = RecordedMoves(self.DANCE_LIBRARY)
        return self._dances_lib

    def list_emotions(self) -> list[str]:
        """Names playable with play_emotion(), e.g. 'welcoming1', 'curious1',
        'mini-deep-sleep' — see pollen-robotics/reachy-mini-emotions-library."""
        return self._emotions().list_moves()

    def list_dances(self) -> list[str]:
        """Names playable with play_dance() — see
        pollen-robotics/reachy-mini-dances-library."""
        return self._dances().list_moves()

    def play_emotion(self, name: str, block: bool = False, sound: bool = True) -> None:
        """Play one of the recorded emotions by name (list_emotions())."""
        self._play_recorded(self._emotions(), name, block, sound)

    def play_dance(self, name: str, block: bool = False, sound: bool = True) -> None:
        """Play one of the recorded dances by name (list_dances())."""
        self._play_recorded(self._dances(), name, block, sound)

    def _play_recorded(self, library, name: str, block: bool, sound: bool) -> None:
        import asyncio

        try:
            move = library.get(name)
        except Exception:
            return  # unknown/uncached move must never disturb the pipeline

        def _run():
            try:
                asyncio.run(self._mini.async_play_move(move, sound=sound))
            except Exception:
                pass

        if block:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()

    def close(self) -> None:
        self._closed.set()
        self._stop_playback.set()
        self.speaker_stop()
        with self._mic_ready:
            self._mic_ready.notify_all()
        if self._built_in_stream is not None:
            try:
                self._built_in_stream.stop()
                self._built_in_stream.close()
            except Exception:
                pass
        try:
            self._mini.media.stop_recording()
            self._mini.media.stop_playing()
        except Exception:
            pass
        finally:
            self._mini.__exit__(None, None, None)


class FakeReachy:
    """Test double with the same surface as ReachyRobot (DESIGN §2, §9)."""

    def __init__(self, mic_samplerate: int = 16000):
        self._mic_sr = mic_samplerate
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.nods = 0
        self.poses: list[str] = []
        self.emotions: list[str] = []
        self.dances: list[str] = []
        self.flushes = 0
        self.played: list[np.ndarray] = []
        self.stops = 0
        self.closed = False
        self.hold_playback = False  # True → speaker stays busy until speaker_stop()
        self._playing = False
        self._mic_queue: deque[np.ndarray] = deque()

    # camera
    def camera_read(self) -> np.ndarray:
        return self.frame

    # microphone
    @property
    def mic_samplerate(self) -> int:
        return self._mic_sr

    def feed_mic(self, chunk: np.ndarray) -> None:
        self._mic_queue.append(np.asarray(chunk, dtype=np.float32))

    def mic_read(self) -> np.ndarray:
        if self._mic_queue:
            return self._mic_queue.popleft()
        return np.zeros(0, dtype=np.float32)

    def flush_mic(self) -> None:
        self.flushes += 1
        self._mic_queue.clear()

    # speaker
    def speaker_play(self, wav: np.ndarray, samplerate: int, block: bool = True) -> None:
        self.played.append(np.asarray(wav))
        self._playing = self.hold_playback and not block

    def speaker_stop(self) -> None:
        self.stops += 1
        self._playing = False

    def speaker_busy(self) -> bool:
        return self._playing

    # head
    def nod(self) -> None:
        self.nods += 1

    def pose(self, name: str, duration: float = 0.5) -> None:
        self.poses.append(name)

    # recorded emotion/dance library (fake catalogue for tests)
    FAKE_EMOTIONS = ["welcoming1", "mini-deep-sleep", "confused1", "curious1"]
    FAKE_DANCES = ["simple_nod", "yeah_nod"]

    def list_emotions(self) -> list[str]:
        return list(self.FAKE_EMOTIONS)

    def list_dances(self) -> list[str]:
        return list(self.FAKE_DANCES)

    def play_emotion(self, name: str, block: bool = False, sound: bool = True) -> None:
        self.emotions.append(name)

    def play_dance(self, name: str, block: bool = False, sound: bool = True) -> None:
        self.dances.append(name)

    def close(self) -> None:
        self.closed = True