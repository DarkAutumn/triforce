"""Observation Space — Neural Network Feature Flow Analysis

Traces the full data flow from observation wrapper through the neural network
to understand how each observation component contributes to the final decision.
Computes parameter counts and feature dimension proportions.

This script does NOT require the NES ROM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
import numpy as np

from triforce.observation_wrapper import (
    BOOLEAN_FEATURES, ENEMY_COUNT, ENEMY_FEATURES, NUM_ENEMY_TYPES,
    ITEM_COUNT, ITEM_FEATURES, PROJECTILE_COUNT, PROJECTILE_FEATURES,
    VIEWPORT_PIXELS,
)
from triforce.models import SharedNatureAgent, NatureCNN, CombinedExtractor


def count_params(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    print("=" * 80)
    print("NEURAL NETWORK FEATURE FLOW ANALYSIS")
    print("=" * 80)

    # Build the observation space as the real code does
    frame_stack = 4
    obs_space = Dict({
        "image": Box(low=0.0, high=1.0,
                     shape=(frame_stack, VIEWPORT_PIXELS, VIEWPORT_PIXELS),
                     dtype=np.float32),
        "enemy_features": Box(low=-1.0, high=1.0,
                              shape=(ENEMY_COUNT, ENEMY_FEATURES),
                              dtype=np.float32),
        "enemy_id": Discrete(ENEMY_COUNT),
        "item_features": Box(low=-1.0, high=1.0,
                             shape=(ITEM_COUNT, ITEM_FEATURES),
                             dtype=np.float32),
        "projectile_features": Box(low=-1.0, high=1.0,
                                   shape=(PROJECTILE_COUNT, PROJECTILE_FEATURES),
                                   dtype=np.float32),
        "information": MultiBinary(BOOLEAN_FEATURES),
    })

    # Action space — typical MOVE+SWORD+BEAMS = 4+4+4 = 12 actions
    action_space = Discrete(12)

    # Build the network
    net = SharedNatureAgent(obs_space, action_space)

    print(f"\n--- Observation Space Shapes ---")
    for key, space in obs_space.items():
        print(f"  {key}: {space.shape if hasattr(space, 'shape') else space}")

    print(f"\n--- Network Architecture ---")
    print(f"  CombinedExtractor → MlpExtractor → action_net/value_net")

    # Parameter counts per component
    cnn_params = count_params(net.base.image_extractor)
    embed_params = count_params(net.base.enemy_embedding)
    mlp_params = count_params(net.mlp_extractor)
    action_params = count_params(net.action_net)
    value_params = count_params(net.value_net)
    total_params = count_params(net)

    print(f"\n--- Parameter Counts ---")
    print(f"  NatureCNN (image extractor):   {cnn_params:>8,}")
    print(f"  Enemy ID embedding:            {embed_params:>8,}")
    print(f"  MLP extractor (policy+value):  {mlp_params:>8,}")
    print(f"  Action net:                    {action_params:>8,}")
    print(f"  Value net:                     {value_params:>8,}")
    print(f"  TOTAL trainable params:        {total_params:>8,}")

    # Feature dimension breakdown after CombinedExtractor
    embedding_dim = 4
    image_linear_size = 256

    dims = {
        'image_cnn': image_linear_size,
        'enemy_features': ENEMY_COUNT * ENEMY_FEATURES,
        'enemy_embeddings': ENEMY_COUNT * embedding_dim,
        'item_features': ITEM_COUNT * ITEM_FEATURES,
        'projectile_features': PROJECTILE_COUNT * PROJECTILE_FEATURES,
        'information': BOOLEAN_FEATURES,
    }

    total_dim = sum(dims.values())

    print(f"\n--- CombinedExtractor Feature Dimensions ---")
    print(f"  {'Component':<25} {'Dims':>5} {'Pct':>6}")
    print(f"  {'-'*25} {'-'*5} {'-'*6}")
    for name, dim in dims.items():
        pct = 100 * dim / total_dim
        print(f"  {name:<25} {dim:>5} {pct:>5.1f}%")
    print(f"  {'-'*25} {'-'*5} {'-'*6}")
    print(f"  {'TOTAL':<25} {total_dim:>5} {'100.0':>5}%")

    print(f"\n  Key insight: Image features dominate at {100*image_linear_size/total_dim:.1f}%")
    print(f"  Structured features are only {100*(total_dim-image_linear_size)/total_dim:.1f}% of input")

    # MLP extractor analysis
    print(f"\n--- MLP Extractor (after CombinedExtractor) ---")
    print(f"  Input: {total_dim} → Policy: 64 → Action logits: {action_space.n}")
    print(f"  Input: {total_dim} → Value:  64 → Value estimate: 1")
    print(f"  Policy: 3 layers (Linear→ReLU→Linear→ReLU→Linear→Tanh)")
    print(f"  Value:  3 layers (Linear→ReLU→Linear→ReLU→Linear→ReLU)")

    # Information vector analysis
    print(f"\n--- Information Vector Breakdown ---")
    info_features = {
        'objective_direction': (4, 'N/S/E/W binary flags'),
        'get_item_flag': (1, 'ITEM or TREASURE objective'),
        'fight_flag': (1, 'FIGHT objective'),
        'source_direction': (4, 'one-hot entry direction'),
        'has_enemies': (1, 'any active enemies?'),
        'has_beams': (1, 'can fire beams?'),
        'low_health': (1, 'health <= 1 heart'),
        'full_health': (1, 'health at max'),
    }

    total_info = 0
    for name, (dim, desc) in info_features.items():
        print(f"  {name:<25} {dim:>2} features — {desc}")
        total_info += dim

    print(f"  {'TOTAL':<25} {total_info:>2} features")

    # What the model cannot distinguish
    print(f"\n" + "=" * 80)
    print("WHAT THE MODEL CANNOT DISTINGUISH")
    print("=" * 80)

    indistinguishable = [
        ("2 hearts vs 5 hearts",
         "Both are health>1 and not full. Only low(<=1) and full are visible.",
         "Impact: Cannot learn risk-averse behavior proportional to health."),
        ("Locked door vs open door",
         "No door state in features. Both look similar in 128px grayscale.",
         "Impact: Agent may repeatedly walk into locked doors."),
        ("Heart drop vs key drop",
         "Item slots have position but no type. Both are identical.",
         "Impact: Agent can't prioritize keys over hearts when keys are needed."),
        ("Gel vs Darknut (in features)",
         "Enemy IDs ARE embedded via nn.Embedding(150,4), so the network CAN learn "
         "to distinguish types IF the embedding learns meaningful groupings.",
         "Impact: This is actually well-designed — embeddings allow learned distinctions."),
        ("Active sword beam vs inactive",
         "No weapon animation state in features. Only image shows this.",
         "Impact: Agent can't know if beams are in flight from features alone."),
        ("Close to objective vs far from objective",
         "Objective is directional only (N/S/E/W). No distance to objective.",
         "Impact: Can't modulate urgency or path-planning based on progress."),
    ]

    for i, (scenario, detail, impact) in enumerate(indistinguishable, 1):
        print(f"\n  {i}. {scenario}")
        print(f"     {detail}")
        print(f"     {impact}")

    # Proposed additions dimensionality
    print(f"\n" + "=" * 80)
    print("PROPOSED ADDITIONS — DIMENSIONALITY IMPACT")
    print("=" * 80)

    proposals = [
        ("wavefront_distance", 1, "Normalized distance to objective via wavefront"),
        ("wavefront_gradient_4dir", 4, "Wavefront-optimal direction (like mini-compass)"),
        ("health_fraction", 1, "link.health / link.max_health as [0,1]"),
        ("local_walkability_5x5", 25, "5x5 binary grid centered on Link's tile"),
        ("weapon_states", 3, "sword/beam/bomb active flags"),
        ("door_states", 4, "N/S/E/W door status (open/locked/barred)"),
        ("item_types", 2, "Item type IDs for the 2 item slots"),
    ]

    print(f"\n  {'Feature':<30} {'Dims':>5} {'Description'}")
    print(f"  {'-'*30} {'-'*5} {'-'*40}")
    total_new = 0
    for name, dim, desc in proposals:
        print(f"  {name:<30} {dim:>5} {desc}")
        total_new += dim

    print(f"  {'-'*30} {'-'*5}")
    print(f"  {'TOTAL new features':<30} {total_new:>5}")
    print(f"  Current structured features: {total_dim - image_linear_size}")
    print(f"  New total structured:        {total_dim - image_linear_size + total_new}")
    print(f"  Increase: {100*total_new/(total_dim - image_linear_size):.1f}% more structured features")
    print(f"  New CombinedExtractor dim:   {total_dim + total_new}")
    print(f"  Image proportion drops:      {100*image_linear_size/(total_dim+total_new):.1f}% (from {100*image_linear_size/total_dim:.1f}%)")


if __name__ == '__main__':
    main()
