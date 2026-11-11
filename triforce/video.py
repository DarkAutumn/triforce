"""Filming a grid of replays, for ``create-video.py``.

``VideoGrid`` is ``play_grid.ReplayGrid``'s counterpart for a film: the same cells, messages and frames
(``triforce.play_view.Window`` shows either), but stepped one frame at a time instead of played against the
clock, and finite.  The first replays start in the cells in order; a cell whose replay ended shows its result for
``RESULT_PAUSE`` seconds (as on screen), then takes the next replay not yet played; once none is left it holds its
last picture and result.  The film ends when every replay has played and the last result has had its pause.

One cell can be heard (``heard``): its sound is made by the emulator beside the picture
(``_native.ReplayPlayer.set_sound``), and ``VideoGrid.audio()`` hands out exactly the samples of each film frame,
silence while that cell shows a result, so sound and picture stay together however long the film is.

``Encoder`` writes the frames (and the samples) to a video file through ``ffmpeg``: H.264 in the container the
file name asks for (``.mp4``, ``.mkv``, ``.mov``), at the console's frame rate, with AAC sound.
"""

import os
import shutil
import subprocess
import wave
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import _native, replay
from .play_grid import FILES, NES_FPS, RESULT_PAUSE, Show

FRAME_RATE = Fraction(39_375_000, 655_171)      # an NTSC NES's frames per second (60.0988), exactly
SAMPLE_RATE = 48_000
RESULT_FRAMES = round(RESULT_PAUSE * NES_FPS)


def center_cell(cols: int, rows: int) -> int:
    """The cell in the middle of a ``cols`` x ``rows`` grid, or the top left one (0) when there is no middle
    (an even number of columns or rows)."""
    return (rows // 2) * cols + cols // 2 if cols % 2 and rows % 2 else 0


def replay_files(paths: Sequence) -> List[Path]:
    """The replay files ``paths`` name, in order: a file as it is, a directory as its top-level ``*.tfreplay``
    files in name order.  ``ValueError`` for a path that is neither, or a directory without replays."""
    files: List[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix == replay.SUFFIX and p.is_file())
            if not found:
                raise ValueError(f"{path} has no {replay.SUFFIX} files")
            files += found
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"{path} is not a replay file or a directory of them")
    return files


class _Reel:
    """One cell of the film."""
    # pylint: disable=too-few-public-methods

    def __init__(self):
        self.show: Optional[Show] = None
        self.player = None
        self.phase = 'idle'         # idle | playing | result | done
        self.until = 0              # the film frame a result's pause ends at
        self.sent: Optional[dict] = None
        self.shown = 0              # frames shown so far
        self.pixels = None


class VideoGrid:
    """``cells`` cells playing the replay files ``paths`` one film frame per ``step()`` (module doc).  ``heard``:
    the cell whose sound ``audio()`` returns (None: silence).  ``controller``: draw the controller into every
    frame.  ``states_dir``: where a replay recorded from a named state file finds it."""
    # pylint: disable=too-many-instance-attributes

    source = FILES
    fps = NES_FPS

    def __init__(self, cells: int, paths: Sequence, *, heard: Optional[int] = None, controller: bool = False,
                 states_dir=None, rate: int = SAMPLE_RATE):
        # pylint: disable=too-many-arguments
        self.cells = cells
        self.paths = [Path(path) for path in paths]
        if len(self.paths) < cells:
            raise ValueError(f"{cells} cells need at least {cells} replays, got {len(self.paths)}")
        if heard is not None and not 0 <= heard < cells:
            raise ValueError(f"cell {heard} of a grid with {cells} cells")
        self.heard, self.controller, self.rate = heard, controller, rate
        self.states_dir = None if states_dir is None else str(states_dir)
        self.frames = 0             # film frames stepped so far
        self.played = 0             # replays that played to their end
        self._taken = 0             # replays handed to a cell
        self._reels = [_Reel() for _ in range(cells)]
        self._messages: List[dict] = []
        self._pending = np.zeros(0, np.float32)     # the heard cell's samples not yet handed out
        self._samples = 0           # samples handed out so far

    # ------------------------------------------------------------------ what a Window asks of a grid

    def start(self) -> None:
        """Nothing to start: the film moves with ``step``."""

    def stop(self, timeout: float = 0.0) -> None:  # pylint: disable=unused-argument
        """Nothing to stop."""

    def set_fps(self, fps: float) -> None:  # pylint: disable=unused-argument
        """A film has one speed."""

    def set_paused(self, on: bool) -> None:  # pylint: disable=unused-argument
        """A film is not paused."""

    def set_sound(self, cell: Optional[int]) -> None:  # pylint: disable=unused-argument
        """The heard cell is the film's own (``heard``)."""

    def frame(self, cell: int, since: int) -> Optional[Tuple[int, np.ndarray]]:
        reel = self._reels[cell]
        return (reel.shown, reel.pixels) if reel.shown > since else None

    def frame_counters(self) -> List[int]:
        return [reel.shown for reel in self._reels]

    emulated_frames = frame_counters

    def stats(self) -> dict:
        return {'source': FILES, 'feed': None, 'golden': None, 'files': len(self.paths), 'sound_error': None}

    def poll_status(self) -> List[dict]:
        messages, self._messages = self._messages, []
        return messages

    # ------------------------------------------------------------------ the film

    def _load(self, index: int, reel: _Reel) -> None:
        """Puts the next replay not yet played into ``reel``, at the frame watching starts."""
        path = self.paths[self._taken]
        self._taken += 1
        reel.show = Show(path.read_bytes(), FILES, path.name)
        reel.player = _native.ReplayPlayer(reel.show.data, states_dir=self.states_dir)
        reel.player.seek(reel.show.start)
        if index == self.heard:
            reel.player.set_sound(float(self.rate))
        reel.phase, reel.sent = 'playing', None
        self._messages.append({'cell': index, 'event': 'start', 'seed': reel.show.seed, 'source': FILES,
                               'name': path.name})

    def _play(self, index: int, reel: _Reel) -> None:
        """Runs one frame of ``reel``'s replay; at its end the result goes up."""
        player, show = reel.player, reel.show
        if player.step(1, render=True):
            reel.pixels = np.asarray(player.screen(controller=self.controller))
            reel.shown += 1
            if index == self.heard:
                self._pending = np.concatenate([self._pending, player.audio()])
            room = show.status(player.frame)
            if room != reel.sent:
                reel.sent = room
                self._messages.append({'cell': index, 'event': 'room', **room})
        if player.frame >= player.frames:
            self.played += 1
            reel.player = None
            reel.phase, reel.until = 'result', self.frames + RESULT_FRAMES
            self._messages.append({'cell': index, 'event': 'end', 'seed': show.seed, 'result': show.result,
                                   'outcome': show.outcome, 'reason': show.reason})

    def step(self) -> bool:
        """Moves every cell one frame on.  False once the film is over: nothing moved, and nothing will."""
        for index, reel in enumerate(self._reels):
            if reel.phase == 'result' and self.frames >= reel.until:
                reel.phase = 'idle' if self._taken < len(self.paths) else 'done'
            if reel.phase == 'idle' and self._taken < len(self.paths):
                self._load(index, reel)
            if reel.phase == 'playing':
                self._play(index, reel)
        if all(reel.phase == 'done' for reel in self._reels):
            return False
        self.frames += 1
        return True

    def audio(self) -> np.ndarray:
        """The samples (float32, mono, -1..1) of the frame ``step`` just made: every frame's share of ``rate``,
        so the samples handed out always add up to the film's length."""
        want = int(self.frames * self.rate / FRAME_RATE) - self._samples
        out = np.zeros(want, np.float32)
        have = min(want, len(self._pending))
        out[:have] = self._pending[:have]
        self._pending = self._pending[have:]
        self._samples += want
        return out


def ffmpeg() -> str:
    """The ffmpeg program.  ``RuntimeError`` when it is not on the PATH."""
    path = shutil.which('ffmpeg')
    if path is None:
        raise RuntimeError('ffmpeg not found on PATH')
    return path


class Encoder:
    """Writes ``width`` x ``height`` RGBX frames, and with ``sound`` their samples, to the video file ``path``
    (H.264, yuv420p; AAC).  The picture is encoded as the frames come; the sound is kept in a temporary WAV
    file beside the output and joined in ``close``."""

    def __init__(self, path, width: int, height: int, sound: bool = False, rate: int = SAMPLE_RATE):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.path = Path(path)
        self.sound = sound
        self._picture = self.path.with_name(f".{self.path.stem}.picture{self.path.suffix}") if sound else self.path
        self._wave_path = self.path.with_name(f".{self.path.stem}.sound.wav")
        self._wave = None
        if sound:
            self._wave = wave.open(str(self._wave_path), 'wb')
            self._wave.setnchannels(1)
            self._wave.setsampwidth(2)
            self._wave.setframerate(rate)
        # yuv420p needs even sides: an odd one gets a black column or row.
        command = [ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb0',
                   '-s', f"{width}x{height}", '-r', f"{FRAME_RATE.numerator}/{FRAME_RATE.denominator}", '-i', '-',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self._picture)]
        self._encoder = subprocess.Popen(command, stdin=subprocess.PIPE)  # pylint: disable=consider-using-with
        self.frames = 0

    def write(self, frame, samples: Optional[np.ndarray] = None) -> None:
        """One frame (``width * height * 4`` bytes, RGBX) and, with sound, its samples (float32, -1..1)."""
        try:
            self._encoder.stdin.write(frame)
        except BrokenPipeError as error:    # ffmpeg exited early; its reason is on stderr
            raise RuntimeError(f"ffmpeg stopped while writing {self._picture}") from error
        if self._wave is not None and samples is not None:
            self._wave.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype('<i2').tobytes())
        self.frames += 1

    def close(self) -> None:
        """Finishes the file.  ``RuntimeError`` when ffmpeg failed."""
        try:
            self._encoder.stdin.close()
        except BrokenPipeError:
            pass
        failed = self._encoder.wait()
        if self._wave is not None:
            self._wave.close()
        try:
            if failed:
                raise RuntimeError(f"ffmpeg failed with exit code {failed} writing {self._picture}")
            if self.sound:
                command = [ffmpeg(), '-y', '-loglevel', 'error', '-i', str(self._picture), '-i', str(self._wave_path),
                           '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(self.path)]
                done = subprocess.run(command, check=False)
                if done.returncode:
                    raise RuntimeError(f"ffmpeg failed with exit code {done.returncode} adding sound to {self.path}")
        finally:
            if self.sound:
                for leftover in (self._picture, self._wave_path):
                    if os.path.exists(leftover):
                        os.remove(leftover)
