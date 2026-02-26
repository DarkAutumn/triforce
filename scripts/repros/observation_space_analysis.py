"""Observation Space Analysis — Static Code Analysis

Analyzes the observation space structure defined in observation_wrapper.py and models.py
to understand what the model sees, compute dimension counts, and identify information gaps.

This script does NOT require the NES ROM — it statically analyzes the observation space
definition from the source code.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from triforce.observation_wrapper import (
    BOOLEAN_FEATURES, DISTANCE_SCALE, VIEWPORT_PIXELS,
    ENEMY_COUNT, ENEMY_FEATURES, NUM_ENEMY_TYPES,
    ITEM_COUNT, ITEM_FEATURES,
    PROJECTILE_COUNT, PROJECTILE_FEATURES,
)

def main():
    print("=" * 80)
    print("OBSERVATION SPACE STRUCTURE ANALYSIS")
    print("=" * 80)

    # 1. Image observation
    frame_stack = 4  # typical default
    print(f"\n--- Image Observation ---")
    print(f"  Viewport size: {VIEWPORT_PIXELS}x{VIEWPORT_PIXELS} pixels (grayscale)")
    print(f"  Frame stack: {frame_stack} frames")
    print(f"  Image shape: ({frame_stack}, {VIEWPORT_PIXELS}, {VIEWPORT_PIXELS})")
    image_params = frame_stack * VIEWPORT_PIXELS * VIEWPORT_PIXELS
    print(f"  Total image values: {image_params:,}")

    # 2. Enemy features
    print(f"\n--- Enemy Features ---")
    print(f"  Slots: {ENEMY_COUNT}")
    print(f"  Features per enemy: {ENEMY_FEATURES}")
    print(f"    [0] presence (0/1)")
    print(f"    [1] closeness (proximity, 0.1-1.0, distance_scale={DISTANCE_SCALE})")
    print(f"    [2-3] direction vector (normalized dx, dy)")
    print(f"    [4-5] facing direction vector (enemy direction enum → vector)")
    print(f"  Total enemy feature values: {ENEMY_COUNT * ENEMY_FEATURES}")
    print(f"  Enemy ID embedding: {ENEMY_COUNT} IDs × {NUM_ENEMY_TYPES} types")
    print(f"  NOTE: Enemy IDs are raw enum values passed as integers, NOT one-hot")
    print(f"         The network uses nn.Embedding(150, 4) to embed them")

    # 3. Item features
    print(f"\n--- Item Features ---")
    print(f"  Slots: {ITEM_COUNT}")
    print(f"  Features per item: {ITEM_FEATURES}")
    print(f"    [0] presence (0/1)")
    print(f"    [1] closeness (proximity)")
    print(f"    [2-3] direction vector (normalized dx, dy)")
    print(f"  Total item feature values: {ITEM_COUNT * ITEM_FEATURES}")
    print(f"  NOTE: Item TYPE is not included — all items look identical")

    # 4. Projectile features
    print(f"\n--- Projectile Features ---")
    print(f"  Slots: {PROJECTILE_COUNT}")
    print(f"  Features per projectile: {PROJECTILE_FEATURES}")
    print(f"    [0] presence (0/1)")
    print(f"    [1] closeness (proximity)")
    print(f"    [2-3] direction vector (normalized dx, dy)")
    print(f"    [4] blockable flag (-1/1)")
    print(f"  Total projectile feature values: {PROJECTILE_COUNT * PROJECTILE_FEATURES}")

    # 5. Information vector
    print(f"\n--- Information Vector ---")
    print(f"  Total features: {BOOLEAN_FEATURES}")
    print(f"    [0-3] Objective direction (N/S/E/W binary)")
    print(f"    [4]   Get item/treasure objective")
    print(f"    [5]   Fight enemies objective")
    print(f"    [6-9] Source direction (entry direction, one-hot)")
    print(f"    [10]  Has active enemies")
    print(f"    [11]  Has beams available")
    print(f"    [12]  Low health (health <= 1)")
    print(f"    [13]  Full health")

    # 6. Summary
    total_structured = (ENEMY_COUNT * ENEMY_FEATURES +
                        ENEMY_COUNT +  # enemy IDs
                        ITEM_COUNT * ITEM_FEATURES +
                        PROJECTILE_COUNT * PROJECTILE_FEATURES +
                        BOOLEAN_FEATURES)

    print(f"\n--- TOTALS ---")
    print(f"  Image: {image_params:,} values")
    print(f"  Structured: {total_structured} values")
    print(f"    Enemy features: {ENEMY_COUNT * ENEMY_FEATURES}")
    print(f"    Enemy IDs: {ENEMY_COUNT}")
    print(f"    Item features: {ITEM_COUNT * ITEM_FEATURES}")
    print(f"    Projectile features: {PROJECTILE_COUNT * PROJECTILE_FEATURES}")
    print(f"    Information: {BOOLEAN_FEATURES}")

    # 7. CombinedExtractor output dimension
    embedding_dim = 4
    image_linear_size = 256
    output_dim = (image_linear_size +
                  ENEMY_COUNT * ENEMY_FEATURES +
                  ENEMY_COUNT * embedding_dim +
                  ITEM_COUNT * ITEM_FEATURES +
                  PROJECTILE_COUNT * PROJECTILE_FEATURES +
                  BOOLEAN_FEATURES)

    print(f"\n--- CombinedExtractor ---")
    print(f"  Image CNN → {image_linear_size} features")
    print(f"  Enemy features flat: {ENEMY_COUNT * ENEMY_FEATURES}")
    print(f"  Enemy ID embeddings flat: {ENEMY_COUNT * embedding_dim}")
    print(f"  Item features flat: {ITEM_COUNT * ITEM_FEATURES}")
    print(f"  Projectile features flat: {PROJECTILE_COUNT * PROJECTILE_FEATURES}")
    print(f"  Information: {BOOLEAN_FEATURES}")
    print(f"  TOTAL combined features: {output_dim}")
    print(f"  → MlpExtractor(input_size={output_dim}) → 64-dim policy, 64-dim value")

    # 8. Information gaps
    print(f"\n" + "=" * 80)
    print("INFORMATION GAPS — What the model does NOT see")
    print("=" * 80)

    gaps = [
        ("Wavefront distance/direction",
         "The critic uses wavefront for movement rewards, but the model never sees it. "
         "The model must learn navigation entirely from image + reward signal."),
        ("Local tile walkability",
         "Wall information is only in the image (128x128 viewport). No structured tile "
         "data is provided. The model learns walls via the -0.25 collision penalty."),
        ("Health as continuous value",
         "Only binary: low_health (<=1) and full_health. The model cannot distinguish "
         "2 hearts from 5 hearts. Health fraction would be a single float [0,1]."),
        ("Enemy type properties",
         "Enemy IDs are embedded (good), but no explicit threat level, ranged attack "
         "capability, or vulnerability information is provided."),
        ("Room/map position",
         "No room ID or dungeon map coordinates. The model doesn't know where it is "
         "beyond the objective direction and source direction."),
        ("Movement history",
         "Only the source direction (entry direction) is tracked. No trajectory or "
         "recently-visited tile information."),
        ("Weapon state",
         "No indication of whether sword, bombs, or beams are currently animating. "
         "The model relies on the image to see weapon animations."),
        ("Door lock/bar status",
         "No structured indication of which doors are locked or barred. The model "
         "must learn this from the image (subtle tile differences)."),
        ("Item type",
         "Items have position/proximity but no type indicator. A bomb drop, heart, "
         "rupee, and key all look the same in the feature vector."),
    ]

    for i, (name, description) in enumerate(gaps, 1):
        print(f"\n  {i}. {name}")
        print(f"     {description}")

    # 9. Proximity function analysis
    print(f"\n" + "=" * 80)
    print("PROXIMITY FUNCTION ANALYSIS")
    print("=" * 80)
    print(f"\n  _distance_to_proximity(distance, scale={DISTANCE_SCALE}, min_closeness=0.1)")
    print(f"  Mapping: distance → proximity [0.1, 1.0]")

    test_distances = [0, 5, 10, 20, 30, 50, 75, 100, 150]
    for d in test_distances:
        if d <= 5:
            prox = 1.0
        elif d >= DISTANCE_SCALE:
            prox = 0.1
        else:
            closeness = 1.0 - ((d - 5) / (DISTANCE_SCALE - 5))
            prox = max(0.1, 0.1 + closeness * (1 - 0.1))
        print(f"    distance={d:>4}px → proximity={prox:.3f}")

    print(f"\n  Problem: proximity is non-linear. distance=5 and distance=0 both → 1.0")
    print(f"  The scale is {DISTANCE_SCALE}px. NES screen is 256x176px gameplay area.")
    print(f"  Enemies at 100px+ all read as 0.1 (minimum). Limited resolution at range.")

    # 10. Observation space vs reward system alignment
    print(f"\n" + "=" * 80)
    print("OBSERVATION-REWARD ALIGNMENT ANALYSIS")
    print("=" * 80)

    misalignments = [
        ("Wavefront rewards ↔ No wavefront obs",
         "The critic gives ±0.05/0.06 based on wavefront distance changes, but the "
         "model never sees the wavefront. It must discover the reward-maximizing path "
         "purely from trial-and-error on each room layout."),
        ("Wall collision penalty ↔ No tile obs",
         "The critic penalizes -0.25 for wall collisions, but walkability is only "
         "visible in the 128px viewport image. The model must learn tile layouts "
         "from pixel patterns — which it CAN do, but structured data would be faster."),
        ("Danger zone penalty ↔ Enemy proximity only",
         "The critic uses per-enemy tile proximity to compute danger, but the model "
         "only sees 4 enemy slots with a scalar proximity. The danger calculation is "
         "more nuanced than what the model can derive from its features."),
        ("Health-scaled rewards ↔ Binary health obs",
         "Health loss penalty is flat (-0.75). But with only binary health info, the "
         "model can't learn risk-aversion that scales with remaining health."),
        ("Equipment rewards ↔ No equipment obs",
         "The model gets REWARD_MAXIMUM for equipment, but doesn't know what equipment "
         "it currently has (beyond beams_available). Can't plan for equipment-dependent "
         "strategies."),
    ]

    for i, (name, description) in enumerate(misalignments, 1):
        print(f"\n  {i}. {name}")
        print(f"     {description}")


if __name__ == '__main__':
    main()
