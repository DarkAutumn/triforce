#!/usr/bin/env python3
"""Records gameplay videos of trained Zelda RL agents as 1920×1080 MP4 files."""

import argparse
import os
import re
import tempfile
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from triforce import ActionSpaceDefinition, ModelKindDefinition, TrainingScenarioDefinition, make_zelda_env
from triforce.action_space import ZeldaActionSpace
from triforce.models import Network
from triforce.observation_wrapper import ENTITY_SLOTS, ENTITY_TYPE_NAMES, infer_obs_kind
from triforce.zelda_enums import (ActionKind, Direction, NES_FULL_WIDTH, NES_FULL_HEIGHT,
                                  NES_CROPPED_WIDTH, NES_CROPPED_HEIGHT,
                                  HUD_TRIM_FULL, HUD_TRIM_CROPPED)


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # pylint: disable=no-member
GAMEPLAY_SCALE = 4

CELL_LABELS = {0: "MOVEMENT", 1: "SWORD / BEAMS", 2: "BOMB", 3: "BOOMERANG"}
CELL_ACTION_KINDS = {
    0: [ActionKind.MOVE], 1: [ActionKind.SWORD, ActionKind.BEAMS],
    2: [ActionKind.BOMBS], 3: [ActionKind.BOOMERANG],
}
CELL_SPRITE_FILES = {0: "walking.gif", 1: "magic-sword.gif", 2: "bomb.gif", 3: "boomerang.gif"}
DIR_ORDER = [Direction.N, Direction.S, Direction.E, Direction.W]

COLOR_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_DIM = (100, 100, 100)
COLOR_ACTIVE_GREEN = (100, 255, 100)
COLOR_ACTIVE_RED = (255, 100, 100)
COLOR_ACTIVE_BLUE = (100, 180, 255)
COLOR_ACTIVE_YELLOW = (255, 255, 100)


# ── Font Loading ────────────────────────────────────────────────

_FONT_CACHE = {}
_FONT_WARNING_SHOWN = False


def _find_firacode_path():
    """Find FiraCode Nerd Font on the system."""
    search_dirs = [os.path.expanduser("~/.local/share/fonts"),
                   "/usr/share/fonts", "/usr/local/share/fonts"]
    preferred = ["Fira Code Regular Nerd Font Complete.ttf",
                 "FiraCodeNerdFont-Regular.ttf",
                 "Fira Code Retina Nerd Font Complete.ttf"]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for pref in preferred:
                if pref in files:
                    return os.path.join(root, pref)
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for f in files:
                if "fira" in f.lower() and "nerd" in f.lower() and f.endswith(".ttf"):
                    return os.path.join(root, f)
    return None


def get_font(size):
    """Get a font at the given size, with FiraCode Nerd Font preferred."""
    global _FONT_WARNING_SHOWN  # pylint: disable=global-statement
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    font_path = _find_firacode_path()
    if font_path:
        font = ImageFont.truetype(font_path, size)
    else:
        if not _FONT_WARNING_SHOWN:
            warnings.warn("FiraCode Nerd Font not found, using default font", stacklevel=2)
            _FONT_WARNING_SHOWN = True
        font = ImageFont.load_default()

    _FONT_CACHE[size] = font
    return font


# ── JET Colormap (ported from game_view.py) ────────────────────

def _build_jet_lut():
    """Build a 256-entry JET colormap lookup table (uint8, shape (256, 3))."""
    lut = np.empty((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        if v < 0.125:
            r, g, b = 0.0, 0.0, 0.5 + v * 4.0
        elif v < 0.375:
            r, g, b = 0.0, (v - 0.125) * 4.0, 1.0
        elif v < 0.625:
            r, g, b = (v - 0.375) * 4.0, 1.0, 1.0 - (v - 0.375) * 4.0
        elif v < 0.875:
            r, g, b = 1.0, 1.0 - (v - 0.625) * 4.0, 0.0
        else:
            r, g, b = 1.0 - (v - 0.875) * 4.0, 0.0, 0.0
        lut[i] = (int(r * 255), int(g * 255), int(b * 255))
    return lut

_JET_LUT = _build_jet_lut()


# ── Sprite Loading ──────────────────────────────────────────────

def _load_sprite_frames(gif_path, scale=3):
    """Load all frames from a GIF, scale up, and return as list of RGBA numpy arrays."""
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = img.convert("RGBA")
            w, h = frame.size
            frame = frame.resize((w * scale, h * scale), Image.Resampling.NEAREST)
            frames.append(np.array(frame))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


# ── AttentionOverlay ────────────────────────────────────────────

class AttentionOverlay:
    """Applies JET heatmap attention overlay onto gameplay frames."""

    def __init__(self, head_index, nes_width, nes_height, hud_px):
        # head_index: 0=combined, 1..N=specific head (1-indexed)
        self._head_index = head_index
        self._nes_width = nes_width
        self._nes_height = nes_height
        self._hud_px = hud_px

    def apply(self, frame, attention_weights):
        """Apply attention heatmap over frame (H, W, 3 uint8). Returns modified copy."""
        if attention_weights is None:
            return frame

        if attention_weights.ndim == 2:
            attention_weights = attention_weights[np.newaxis]

        # Select head
        if self._head_index == 0:
            selected = attention_weights.max(axis=0)
        else:
            idx = min(self._head_index - 1, attention_weights.shape[0] - 1)
            selected = attention_weights[idx]

        # Normalize using global min/max
        w_min = attention_weights.min()
        w_max = attention_weights.max()
        if w_max > w_min:
            normalized = (selected - w_min) / (w_max - w_min)
        else:
            return frame

        # Upsample to gameplay area
        gameplay_h = self._nes_height - self._hud_px
        gameplay_w = self._nes_width
        src_h, src_w = normalized.shape

        row_idx = np.linspace(0, src_h - 1, gameplay_h)
        col_idx = np.linspace(0, src_w - 1, gameplay_w)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')

        r0 = np.floor(row_grid).astype(int).clip(0, src_h - 1)
        r1 = np.minimum(r0 + 1, src_h - 1)
        c0 = np.floor(col_grid).astype(int).clip(0, src_w - 1)
        c1 = np.minimum(c0 + 1, src_w - 1)

        dr = row_grid - r0
        dc = col_grid - c0

        upsampled = (normalized[r0, c0] * (1 - dr) * (1 - dc) +
                     normalized[r1, c0] * dr * (1 - dc) +
                     normalized[r0, c1] * (1 - dr) * dc +
                     normalized[r1, c1] * dr * dc)

        # Apply JET colormap
        indices = np.clip((upsampled * 255).astype(np.uint8), 0, 255)
        heatmap_rgb = _JET_LUT[indices]  # (gameplay_h, gameplay_w, 3)

        # Alpha blend onto gameplay area (below HUD)
        alpha = 140.0 / 255.0
        result = frame.copy()
        gameplay_region = result[self._hud_px:self._nes_height, :gameplay_w].astype(np.float32)
        blended = gameplay_region * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
        result[self._hud_px:self._nes_height, :gameplay_w] = blended.astype(np.uint8)

        return result


# ── ObservationRenderer ─────────────────────────────────────────

class ObservationRenderer:
    """Renders the observation panel as a numpy array using Pillow."""

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._font_body = get_font(18)
        self._font_header = get_font(22)
        self._font_small = get_font(15)

    def render(self, observation):
        """Render observation data to a (height, width, 3) numpy array."""
        img = Image.new("RGB", (self._width, self._height), COLOR_BG)
        draw = ImageDraw.Draw(img)
        y = 10

        # Header
        draw.text((10, y), "OBSERVATION", fill=COLOR_TEXT, font=self._font_header)
        y += 30

        if observation is None:
            return np.array(img)

        # Entities
        y = self._draw_entities(draw, y, observation)
        y += 10

        # Directions
        y = self._draw_directions(draw, y, observation)
        y += 10

        # Booleans
        self._draw_booleans(draw, y, observation)

        return np.array(img)

    def _draw_entities(self, draw, y, obs):
        """Draw entity list."""
        entities = obs.get("entities")
        entity_types = obs.get("entity_types")
        if entities is None:
            return y

        if hasattr(entities, 'cpu'):
            entities = entities.cpu().numpy()
        if entity_types is not None and hasattr(entity_types, 'cpu'):
            entity_types = entity_types.cpu().numpy()

        draw.text((10, y), "Entities", fill=COLOR_DIM, font=self._font_small)
        y += 20

        for i in range(min(ENTITY_SLOTS, entities.shape[0])):
            presence = float(entities[i, 0])
            if presence < 0.5:
                continue

            type_id = int(entity_types[i]) if entity_types is not None else 0
            type_name = ENTITY_TYPE_NAMES.get(type_id, f"Unknown({type_id})")

            props = []
            health = float(entities[i, 3])
            stun = float(entities[i, 4])
            hurts = float(entities[i, 5])
            killable = float(entities[i, 6])

            if health > 0.01:
                props.append(f"HP:{health * 15:.0f}")
            if stun > 0.01:
                props.append(f"Stun:{stun:.2f}")
            if hurts > 0.5:
                props.append("\u26a0")  # ⚠
            if killable > 0.5:
                props.append("\u2694")  # ⚔

            text = f"  {type_name}  {' '.join(props)}"
            draw.text((10, y), text, fill=COLOR_TEXT, font=self._font_body)
            y += 22

        return y

    def _draw_directions(self, draw, y, obs):
        """Draw objective and source directions."""
        info = obs.get("information")
        if info is None:
            return y

        if hasattr(info, 'cpu'):
            info = info.cpu().numpy()

        obj_dirs = self._extract_directions(info, 0)
        src_dirs = self._extract_directions(info, 6)

        draw.text((10, y), f"Objective: {self._format_directions(obj_dirs)}",
                  fill=COLOR_ACTIVE_BLUE, font=self._font_body)
        y += 22
        draw.text((10, y), f"Source:    {self._format_directions(src_dirs)}",
                  fill=COLOR_ACTIVE_YELLOW, font=self._font_body)
        y += 22

        return y

    @staticmethod
    def _extract_directions(info, offset):
        """Extract cardinal directions from info vector."""
        dirs = []
        labels = ["N", "S", "E", "W"]
        for i, label in enumerate(labels):
            val = info[offset + i]
            if hasattr(val, 'item'):
                val = val.item()
            if val > 0:
                dirs.append(label)
        return dirs

    @staticmethod
    def _format_directions(dirs):
        """Format direction list as arrow string."""
        arrow_map = {"N": "\u2191", "S": "\u2193", "E": "\u2192", "W": "\u2190"}
        if not dirs:
            return "---"
        return " ".join(arrow_map.get(d, d) for d in dirs)

    def _draw_booleans(self, draw, y, obs):
        """Draw boolean indicators."""
        info = obs.get("information")
        if info is None:
            return

        if hasattr(info, 'cpu'):
            info = info.cpu().numpy()

        draw.text((10, y), "Status", fill=COLOR_DIM, font=self._font_small)
        y += 20

        indicators = [
            (10, "Enemies", COLOR_ACTIVE_RED),
            (11, "Beams", COLOR_ACTIVE_BLUE),
            (12, "Low HP", COLOR_ACTIVE_RED),
            (13, "Full HP", COLOR_ACTIVE_GREEN),
            (14, "Clock", COLOR_ACTIVE_YELLOW),
        ]

        for idx, label, color in indicators:
            val = info[idx]
            if hasattr(val, 'item'):
                val = val.item()
            active = val > 0
            text_color = color if active else COLOR_DIM
            draw.text((10, y), f"  {label}", fill=text_color, font=self._font_body)
            y += 22


# ── ActionRenderer ──────────────────────────────────────────────

class ActionRenderer:
    """Renders the action probabilities panel with animated GIF sprites."""

    def __init__(self, width, height, fps):
        self._width = width
        self._height = height
        self._fps = fps
        self._font_prob = get_font(16)
        self._font_label = get_font(14)
        self._font_header = get_font(22)
        self._frame_counter = 0

        # Load sprites
        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
        self._sprites = {}
        for cell_id, filename in CELL_SPRITE_FILES.items():
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                self._sprites[cell_id] = _load_sprite_frames(path, scale=3)
            else:
                self._sprites[cell_id] = []

        # Animation timing: walking.gif has 250ms per frame
        self._anim_interval = round(fps * 0.25)  # frames per sprite switch

    def render(self, probabilities, action_mask_desc):
        """Render action probabilities to a (height, width, 3) numpy array."""
        img = Image.new("RGBA", (self._width, self._height), (*COLOR_BG, 255))
        draw = ImageDraw.Draw(img)

        # Header
        draw.text((10, 10), "ACTIONS", fill=COLOR_TEXT, font=self._font_header)

        # Compute per-cell probabilities
        cell_probs = self._compute_cell_probs(probabilities, action_mask_desc)

        # 2x2 grid layout
        cell_w = self._width // 2
        cell_h = (self._height - 40) // 2  # subtract header space
        grid_y_start = 40

        positions = [
            (0, grid_y_start),
            (cell_w, grid_y_start),
            (0, grid_y_start + cell_h),
            (cell_w, grid_y_start + cell_h),
        ]

        for cell_id in range(4):
            cx, cy = positions[cell_id]
            self._draw_cell(img, draw, cx, cy, cell_w, cell_h, cell_id, cell_probs.get(cell_id, {}))

        self._frame_counter += 1
        return np.array(img.convert("RGB"))

    def _compute_cell_probs(self, probabilities, action_mask_desc):
        """Compute renormalized per-cell directional probabilities."""
        if probabilities is None:
            return {}

        # Build mask lookup
        allowed = set()
        if action_mask_desc is not None:
            for action_kind, directions in action_mask_desc:
                for direction in directions:
                    allowed.add((action_kind, direction))

        # Collect all rows with mask status
        rows = []
        for key, value_list in probabilities.items():
            if key == 'value':
                continue
            for direction, prob in value_list:
                is_masked = action_mask_desc is not None and (key, direction) not in allowed
                rows.append((key, direction, prob, is_masked))

        # Renormalize
        unmasked_sum = sum(prob for (_, _, prob, masked) in rows if not masked)

        # Group by cell
        cell_probs = {}
        for cell_id, action_kinds in CELL_ACTION_KINDS.items():
            dir_probs = {}
            for kind in action_kinds:
                for action_kind, direction, prob, is_masked in rows:
                    if action_kind != kind:
                        continue
                    if is_masked:
                        continue
                    normed = (prob / unmasked_sum) if unmasked_sum > 0 else 0.0
                    dir_probs[direction] = dir_probs.get(direction, 0.0) + normed
            cell_probs[cell_id] = dir_probs

        return cell_probs

    def _draw_cell(self, img, draw, cx, cy, cell_w, cell_h, cell_id, dir_probs):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Draw one action cell with sprite and directional probabilities."""
        center_x = cx + cell_w // 2
        center_y = cy + cell_h // 2 - 10  # shift up slightly for label

        # Check if any probability exceeds threshold
        has_active = any(p >= 0.001 for p in dir_probs.values())

        # Draw sprite
        frames = self._sprites.get(cell_id, [])
        if frames:
            if len(frames) > 1 and self._anim_interval > 0:
                frame_idx = (self._frame_counter // self._anim_interval) % len(frames)
            else:
                frame_idx = 0
            sprite = frames[frame_idx]

            # Grey out if inactive
            if not has_active:
                grey = np.mean(sprite[:, :, :3], axis=2, keepdims=True).astype(np.uint8)
                sprite = sprite.copy()
                sprite[:, :, :3] = np.broadcast_to(grey, sprite[:, :, :3].shape)
                sprite[:, :, 3] = sprite[:, :, 3] // 2

            sprite_img = Image.fromarray(sprite, "RGBA")
            sh, sw = sprite.shape[:2]
            paste_x = center_x - sw // 2
            paste_y = center_y - sh // 2
            img.paste(sprite_img, (paste_x, paste_y), sprite_img)

        # Draw directional probabilities
        if has_active:
            sprite_h = frames[0].shape[0] if frames else 40
            sprite_w = frames[0].shape[1] if frames else 40

            for direction in DIR_ORDER:
                prob = dir_probs.get(direction, 0.0)
                if prob < 0.001:
                    continue
                text = f"{prob * 100:.1f}%"
                bbox = self._font_prob.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                if direction == Direction.N:
                    tx = center_x - tw // 2
                    ty = center_y - sprite_h // 2 - th - 4
                elif direction == Direction.S:
                    tx = center_x - tw // 2
                    ty = center_y + sprite_h // 2 + 4
                elif direction == Direction.W:
                    tx = center_x - sprite_w // 2 - tw - 8
                    ty = center_y - th // 2
                else:  # E
                    tx = center_x + sprite_w // 2 + 8
                    ty = center_y - th // 2

                draw.text((tx, ty), text, fill=COLOR_TEXT, font=self._font_prob)

        # Cell label
        label = CELL_LABELS[cell_id]
        bbox = self._font_label.getbbox(label)
        lw = bbox[2] - bbox[0]
        label_color = COLOR_TEXT if has_active else COLOR_DIM
        draw.text((center_x - lw // 2, cy + cell_h - 24), label, fill=label_color, font=self._font_label)


# ── FrameComposer ──────────────────────────────────────────────

class FrameComposer:
    """Composes the 1920×1080 video frame from gameplay + panels."""

    def __init__(self, nes_width, nes_height, show_observation, show_actions, fps):
        self._nes_width = nes_width
        self._nes_height = nes_height

        # Scale gameplay to fill vertical space while maintaining aspect ratio
        self._game_scale = VIDEO_HEIGHT / nes_height
        self._game_w = int(nes_width * self._game_scale)
        self._game_h = VIDEO_HEIGHT

        has_side_panel = show_observation or show_actions
        if has_side_panel:
            # Gameplay on left, side panel on right
            self._game_x = 0
            self._game_y = 0
            self._panel_x = self._game_w
            self._panel_w = VIDEO_WIDTH - self._game_w
        else:
            # Center gameplay
            self._game_x = (VIDEO_WIDTH - self._game_w) // 2
            self._game_y = 0
            self._panel_x = 0
            self._panel_w = 0

        # Split side panel between observation and action renderers
        self._obs_renderer = None
        self._action_renderer = None

        if show_observation and show_actions:
            obs_h = VIDEO_HEIGHT // 2
            act_h = VIDEO_HEIGHT - obs_h
            self._obs_renderer = ObservationRenderer(self._panel_w, obs_h)
            self._obs_y = 0
            self._action_renderer = ActionRenderer(self._panel_w, act_h, fps)
            self._action_y = obs_h
        elif show_observation:
            self._obs_renderer = ObservationRenderer(self._panel_w, VIDEO_HEIGHT)
            self._obs_y = 0
        elif show_actions:
            self._action_renderer = ActionRenderer(self._panel_w, VIDEO_HEIGHT, fps)
            self._action_y = 0

    def compose(self, nes_frame, observation=None, probabilities=None, action_mask_desc=None):
        """Compose a full 1920×1080 frame.

        Args:
            nes_frame: (H, W, 3) uint8 numpy array (NES RGB frame)
            observation: dict with 'entities', 'entity_types', 'information' keys
            probabilities: OrderedDict from get_probabilities()
            action_mask_desc: list of (ActionKind, [Direction, ...]) pairs
        Returns:
            (1080, 1920, 3) uint8 numpy array
        """
        canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

        # Scale gameplay frame
        scaled = cv2.resize(nes_frame, (self._game_w, self._game_h),  # pylint: disable=no-member
                            interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member
        canvas[self._game_y:self._game_y + self._game_h,
               self._game_x:self._game_x + self._game_w] = scaled

        # Render side panels
        if self._obs_renderer is not None:
            obs_panel = self._obs_renderer.render(observation)
            canvas[self._obs_y:self._obs_y + obs_panel.shape[0],
                   self._panel_x:self._panel_x + obs_panel.shape[1]] = obs_panel

        if self._action_renderer is not None:
            act_panel = self._action_renderer.render(probabilities, action_mask_desc)
            canvas[self._action_y:self._action_y + act_panel.shape[0],
                   self._panel_x:self._panel_x + act_panel.shape[1]] = act_panel

        return canvas


# ── VideoRecorder ──────────────────────────────────────────────

class VideoRecorder:
    """Manages OpenCV VideoWriter with temp-file strategy for --only-wins."""

    def __init__(self, output_dir, fps):
        self._output_dir = output_dir
        self._fps = fps
        self._writer = None
        self._temp_path = None
        self._final_path = None

        os.makedirs(output_dir, exist_ok=True)

    def _next_filename(self):
        """Find the next available triforce_NN.mp4 filename."""
        existing = set()
        pattern = re.compile(r'^triforce_(\d+)\.mp4$')
        if os.path.isdir(self._output_dir):
            for f in os.listdir(self._output_dir):
                m = pattern.match(f)
                if m:
                    existing.add(int(m.group(1)))

        n = 0
        while n in existing:
            n += 1
        return os.path.join(self._output_dir, f"triforce_{n:02d}.mp4")

    def start(self):
        """Start recording a new video to a temp file."""
        self._final_path = self._next_filename()

        # Write to temp file in same directory (to allow atomic rename)
        fd, self._temp_path = tempfile.mkstemp(suffix='.mp4', dir=self._output_dir)
        os.close(fd)

        self._writer = cv2.VideoWriter(  # pylint: disable=no-member
            self._temp_path, FOURCC, self._fps, (VIDEO_WIDTH, VIDEO_HEIGHT))

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self._temp_path}")

    def write_frame(self, frame_rgb):
        """Write a single RGB frame (H, W, 3) to the video."""
        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
        self._writer.write(frame_bgr)

    def finish(self, keep):
        """Finish recording. If keep=True, rename to final path. If False, delete."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self._temp_path and os.path.exists(self._temp_path):
            if keep:
                os.rename(self._temp_path, self._final_path)
                result = self._final_path
            else:
                os.remove(self._temp_path)
                result = None
            self._temp_path = None
            return result

        self._temp_path = None
        return None

    def cleanup(self):
        """Clean up any dangling temp file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._temp_path and os.path.exists(self._temp_path):
            os.remove(self._temp_path)
            self._temp_path = None


# ── Probability Computation (ported from environment_bridge.py) ─

def get_probabilities(model, action_space, obs, mask):
    """Compute post-mask action probabilities. Returns OrderedDict."""
    if model.is_multihead:
        return _get_multihead_probabilities(model, action_space, obs, mask)

    with torch.no_grad():
        logits, value = model.forward(obs)

    if mask is not None:
        if mask.dim() < logits.dim():
            mask = mask.unsqueeze(0)
        logits = logits.clone()
        logits[~mask] = -1e9

    probs = torch.nn.functional.softmax(logits, dim=-1)

    result = OrderedDict()
    result['value'] = value

    for i, prob in enumerate(probs.squeeze()):
        taken = action_space.get_action_taken(i)
        result.setdefault(taken.kind, []).append((taken.direction, prob.item()))

    return result


def _get_multihead_probabilities(model, action_space, obs, mask):
    """Compute multihead probabilities."""
    with torch.no_grad():
        action_type_logits, direction_logits, value = model.forward(obs)

    num_action_types = int(model.action_space.nvec[0])

    if mask is not None:
        multihead_mask = mask.squeeze(0)
        type_mask = multihead_mask[:num_action_types].unsqueeze(0)
        dir_mask = multihead_mask[num_action_types:].unsqueeze(0)

        action_type_logits = action_type_logits.clone()
        action_type_logits[~type_mask] = -1e9
        direction_logits = direction_logits.clone()
        direction_logits[~dir_mask] = -1e9

    type_probs = torch.nn.functional.softmax(action_type_logits, dim=-1).squeeze(0)
    dir_probs = torch.nn.functional.softmax(direction_logits, dim=-1).squeeze(0)

    result = OrderedDict()
    result['value'] = value

    for type_idx in range(num_action_types):
        for dir_idx in range(4):
            flat_idx = action_space.multihead_to_flat(type_idx, dir_idx)
            taken = action_space.get_action_taken(flat_idx)
            joint_prob = (type_probs[type_idx] * dir_probs[dir_idx]).item()
            result.setdefault(taken.kind, []).append((taken.direction, joint_prob))

    return result


def get_attention_weights(model, obs):
    """Get attention weights from model. Returns numpy array or None."""
    if not hasattr(model, 'forward_with_attention'):
        return None

    with torch.no_grad():
        was_training = model.training
        model.eval()
        result = model.forward_with_attention(obs)
        if was_training:
            model.train()
        attn = result[-1]
        if attn is None:
            return None
        return attn.squeeze(0).cpu().numpy()


# ── Environment & Model Setup ──────────────────────────────────

def load_model_and_env(model_path, scenario_name, device):
    """Load model and create environment. Returns (network, env, action_space, nes_dims)."""
    # Load model
    metadata = Network.load_metadata(model_path)
    model_kind_name = metadata["model_kind"]
    action_space_name = metadata["action_space_name"]
    if not model_kind_name or not action_space_name:
        raise ValueError(f"Model file {model_path} is missing model_kind or action_space_name metadata.")

    model_kind = ModelKindDefinition.get(model_kind_name)
    obs_space = metadata["obs_space"]
    action_space_shape = metadata["action_space"]
    network = model_kind.network_class(obs_space, action_space_shape,
                                       model_kind=model_kind_name,
                                       action_space_name=action_space_name)
    network.load(model_path)
    network.to(device)
    network.eval()

    # Create environment
    scenario_def = TrainingScenarioDefinition.get(scenario_name)
    asd = ActionSpaceDefinition.get(action_space_name)
    multihead = getattr(network, 'is_multihead', False)
    obs_kind, frame_stack = infer_obs_kind(network.observation_space)

    env = make_zelda_env(scenario_def, asd.actions, render_mode='rgb_array',
                         translation=False, frame_stack=frame_stack,
                         multihead=multihead, obs_kind=obs_kind)

    # Find the ZeldaActionSpace wrapper
    action_space = env
    while not isinstance(action_space, ZeldaActionSpace):
        action_space = action_space.env

    # Detect NES dimensions (full_screen vs cropped)
    from triforce.observation_wrapper import ObservationWrapper  # pylint: disable=import-outside-toplevel
    full_screen = False
    env_walk = env
    while env_walk:
        if isinstance(env_walk, ObservationWrapper):
            full_screen = env_walk.full_screen  # pylint: disable=no-member
            break
        env_walk = getattr(env_walk, 'env', None)

    if full_screen:
        nes_dims = (NES_FULL_WIDTH, NES_FULL_HEIGHT, HUD_TRIM_FULL)
    else:
        nes_dims = (NES_CROPPED_WIDTH, NES_CROPPED_HEIGHT, HUD_TRIM_CROPPED)

    return network, env, action_space, nes_dims


# ── Main Recording Loop ────────────────────────────────────────

def _compose_frame(composer, nes_frame, obs, network, action_space, state, action_mask,
                   show_obs, show_actions, show_attention, attention_overlay):
    """Compose a single video frame with all overlays."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if show_attention and attention_overlay is not None:
        attn = get_attention_weights(network, obs)
        if attn is not None:
            nes_frame = attention_overlay.apply(nes_frame, attn)

    probs = None
    mask_desc = None
    if show_actions:
        mask_for_probs = action_mask.unsqueeze(0) if action_mask is not None else None
        probs = get_probabilities(network, action_space, obs, mask_for_probs)
        mask_desc = action_space.get_allowed_actions(state, action_mask)

    return composer.compose(nes_frame, obs if show_obs else None, probs, mask_desc)


def record_episodes(args):
    """Main recording loop."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    network, env, action_space, nes_dims = load_model_and_env(args.model_path, args.scenario, args.device)
    nes_w, nes_h, hud_px = nes_dims

    show_obs = not args.no_observation
    show_actions = not args.no_model_actions
    show_attention = args.show_attention is not None

    composer = FrameComposer(nes_w, nes_h, show_obs, show_actions, args.fps)

    attention_overlay = None
    if show_attention:
        attention_overlay = AttentionOverlay(args.show_attention, nes_w, nes_h, hud_px)

    recorder = VideoRecorder(args.output_path, args.fps)

    completed = 0
    attempts = 0
    max_attempts = args.count * 20

    try:
        while completed < args.count and attempts < max_attempts:
            attempts += 1
            recorder.start()

            obs, info = env.reset()
            action_mask = (info.get('action_mask', None)
                           if isinstance(info, dict) else info.info.get('action_mask', None))

            # For translation=False, info is the state object
            if not isinstance(info, dict):
                state = info
                action_mask = state.info['action_mask']
                initial_frame = state.info.get('initial_frame', None)
                if initial_frame is not None:
                    frame = _compose_frame(composer, initial_frame, obs, network, action_space,
                                           state, action_mask, show_obs, show_actions,
                                           show_attention, attention_overlay)
                    recorder.write_frame(frame)

            terminated = truncated = False
            end_reason = None

            while not terminated and not truncated:
                # Select action
                mask_batch = action_mask.unsqueeze(0) if action_mask is not None else None
                action = network.get_action(obs, mask_batch)
                action = action.squeeze(0)

                # Compute pre-step data for display
                probs = None
                mask_desc = None
                if show_actions:
                    probs = get_probabilities(network, action_space, obs, mask_batch)
                attn_weights = None
                if show_attention:
                    attn_weights = get_attention_weights(network, obs)

                # Step environment
                obs, _rewards, terminated, truncated, state_change = env.step(action)
                action_mask = state_change.state.info['action_mask']

                if show_actions:
                    mask_desc = action_space.get_allowed_actions(state_change.state, action_mask)

                # Write frames
                for nes_frame in state_change.frames:
                    display_frame = nes_frame
                    if attention_overlay is not None and attn_weights is not None:
                        display_frame = attention_overlay.apply(display_frame, attn_weights)
                    video_frame = composer.compose(
                        display_frame, obs if show_obs else None, probs, mask_desc)
                    recorder.write_frame(video_frame)

                if terminated or truncated:
                    end_reason = state_change.state.info.get('end_reason', '')

            # Determine success
            is_success = isinstance(end_reason, str) and end_reason.startswith("success")

            if args.only_wins and not is_success:
                recorder.finish(keep=False)
                print(f"  Episode {attempts}: discarded (reason: {end_reason})")
            else:
                path = recorder.finish(keep=True)
                completed += 1
                print(f"  Episode {attempts}: saved {path} (reason: {end_reason}) "
                      f"[{completed}/{args.count}]")

        if completed < args.count:
            print(f"Warning: only completed {completed}/{args.count} videos "
                  f"after {max_attempts} attempts.")

    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
        recorder.cleanup()
    finally:
        env.close()


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record gameplay videos of trained Zelda RL agents.")

    parser.add_argument("model_path", type=str,
                        help="Path to a .pt model file.")
    parser.add_argument("scenario", type=str,
                        help="Scenario name from triforce.yaml.")

    parser.add_argument("--output-path", type=str, default="./recording/",
                        help="Directory to save videos (default: ./recording/).")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of videos to record (default: 1).")
    parser.add_argument("--only-wins", action="store_true",
                        help="Only save videos where the scenario ended in success.")
    parser.add_argument("--show-attention", type=int, nargs="?", const=0, default=None,
                        help="Overlay attention heatmap. Optional N selects head (1-indexed).")
    parser.add_argument("--no-observation", action="store_true",
                        help="Disable the observation panel.")
    parser.add_argument("--no-model-actions", action="store_true",
                        help="Disable the action probabilities panel.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device for inference (default: cpu).")
    parser.add_argument("--fps", type=int, choices=[30, 60], default=60,
                        help="Video frame rate (default: 60).")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    print(f"Recording {args.count} video(s) of '{args.scenario}' from {args.model_path}")
    print(f"  Output: {args.output_path}  FPS: {args.fps}  Device: {args.device}")
    if args.only_wins:
        print("  Mode: only wins")
    if args.show_attention is not None:
        head_desc = "combined" if args.show_attention == 0 else f"head {args.show_attention}"
        print(f"  Attention: {head_desc}")

    record_episodes(args)
    print("Done.")


if __name__ == '__main__':
    main()
