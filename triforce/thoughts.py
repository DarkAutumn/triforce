"""What a replay says the agent was steered to, could do and chose, at any frame: the overlay play-zelda.py and
create-video.py draw over a cell with ``--thoughts`` (docs/architecture.md "Replays").

Everything comes from the replay alone (version 2: its TARGETS, MASKS and DECISIONS, named by its own META): no
route data, network or code of the version that recorded it.  ``Thoughts(recording).at(frame)`` is up to three
lines for people, each the latest record at or before ``frame``:

    FIGHT: monster slot 3 type 0x2a, monster slot 4 type 0x2a -> 8:0x3c
    can: MOVE[N,S,W,E] WAIT SWORD[N,S,W,E] BOMB blocked: tenth
    boss-gleeok: SWORD N 62% hit 3% (temperature) | MOVE S 16 21%, MOVE W 16 9%

the objective and its targets (``[reward-only]``: priced by the reward, not in the observation's target) with the
rooms it leads to; the legal actions (a kind with its legal directions) and the rule behind each blocked kind; the
network that acted, the action it took with its probability, the hit head's P(Link is hit) for it where the network
has one, how the act mode drew it (META ``tables.draws``), and the next most likely actions.  A replay before version
2 has none of these: no lines.
"""

import bisect
from typing import List

from . import _native

TOP = 3     # the most likely actions a decision line names


class Thoughts:
    """A replay's TARGETS, MASKS and DECISIONS by frame (the module doc)."""

    def __init__(self, recording: '_native.Replay', top: int = TOP):
        self.models = [model.get('name') or f"#{index}" for index, model in enumerate(recording.meta.get('models')
                                                                                       or [])]
        self.targets = recording.targets()
        self.masks = recording.masks()
        decisions = recording.decisions()
        self.decisions = {'frame': decisions['frame'].tolist(), 'policy': decisions['policy'].tolist(),
                          'draw': decisions.get('draw') or [None] * len(decisions['frame'])}
        self.choices = recording.choices(top + 1)
        self._target_frames = [record['frame'] for record in self.targets]
        self._mask_frames = [record['frame'] for record in self.masks]

    @staticmethod
    def _latest(frames: list, frame: int) -> int:
        return bisect.bisect_right(frames, frame) - 1

    def at(self, frame: int) -> List[str]:
        """The overlay's lines ``frame`` input frames into the replay (the module doc)."""
        lines = []
        i = self._latest(self._target_frames, frame)
        if i >= 0:
            record = self.targets[i]
            text = f"{record['objective']}: " + (', '.join(t['text'] for t in record['targets']) or '-')
            if record['next_rooms']:
                text += f" -> {' '.join(record['next_rooms'])}"
            lines.append(text)
        i = self._latest(self._mask_frames, frame)
        if i >= 0:
            lines.append(f"can: {self.masks[i]['text']}")
        i = self._latest(self.decisions['frame'], frame)
        if i >= 0:
            policy = self.decisions['policy'][i]
            name = self.models[policy] if 0 <= policy < len(self.models) else f"#{policy}"
            choice = self.choices[i]
            label, p, danger = choice['chosen']
            text = f"{name}: {label}" + ('' if p is None else f" {p:.0%}")
            if danger is not None:
                text += f" hit {danger:.0%}"
            if self.decisions['draw'][i]:
                text += f" ({self.decisions['draw'][i]})"
            others = [f"{other} {q:.0%}" for other, q, _ in choice['top'] if other != label][:TOP - 1]
            if others:
                text += ' | ' + ', '.join(others)
            if choice['vetoed']:
                text += f" | vetoed {', '.join(choice['vetoed'])}"
            lines.append(text)
        return lines
