"""
Stage 3 Evaluation Script — Precision Landing

Loads the trained Stage 3 model and runs N episodes headless through the
game-identical CurriculumGameEnv to measure landing performance.

Usage:
    python evaluate_stage3.py                     # 100 episodes, no vis
    python evaluate_stage3.py --episodes 200      # 200 episodes
    python evaluate_stage3.py --visualize          # watch the agent land
    python evaluate_stage3.py --seed 999           # specific seed
"""

import argparse
import math
import numpy as np
import os
import time

from train_in_game import CurriculumGameEnv
from double_dqn_agent import DoubleDQNAgent


def evaluate(seed=999, episodes=100, save_dir='models', visualize=False):
    model_path = os.path.join(save_dir, f'lateral_stage3_seed{seed}_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, f'lateral_stage3_seed{seed}_final.pth')
    if not os.path.exists(model_path):
        print(f"No Stage 3 model found for seed {seed} in {save_dir}")
        return

    env = CurriculumGameEnv(stage=3, render_mode='human' if visualize else None)
    agent = DoubleDQNAgent(state_size=9, action_size=7, seed=seed, hidden_size=256)
    agent.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Running {episodes} evaluation episodes (eps=0.0)\n")

    # Tracking
    landed = 0
    crashed = 0
    out_of_fuel = 0
    truncated_count = 0
    on_target = 0
    landing_speeds = []
    landing_angles = []
    landing_dist_x = []
    landing_steps = []
    landing_fuel = []
    all_scores = []

    start = time.time()

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(obs, eps=0.0)
            obs, reward, terminated, trunc, info = env.step(action)
            total_reward += reward
            done = terminated or trunc

            if visualize:
                env.render()

        all_scores.append(total_reward)
        status = info.get('landing_status', 'flying')

        if status == 'landed':
            landed += 1
            if info.get('on_target_pad', False):
                on_target += 1
            speed = math.sqrt(info.get('velocity_x', 0)**2 + info.get('velocity_y', 0)**2)
            landing_speeds.append(speed)
            landing_angles.append(abs(info.get('angle', 0)))
            landing_steps.append(info.get('steps', 0))
            landing_fuel.append(info.get('fuel_percent', 0))
            # Compute dist_x from pad center
            if hasattr(env, 'env') and hasattr(env.env, 'target_pad') and hasattr(env.env, 'lander'):
                pos = env.env.lander.descent_stage.position
                pad = env.env.target_pad
                dx = abs(pos.x - (pad["x1"] + pad["x2"]) / 2.0)
                landing_dist_x.append(dx)
        elif status == 'crashed':
            crashed += 1
        elif status == 'out_of_fuel':
            out_of_fuel += 1
        else:
            truncated_count += 1

        if ep % 10 == 0 or ep == 1:
            pct = 100.0 * landed / ep
            print(f"  Ep {ep:4d}/{episodes} | Landed: {landed} ({pct:.1f}%) | "
                  f"Crashed: {crashed} | Truncated: {truncated_count} | "
                  f"Score: {total_reward:.0f}")

    elapsed = time.time() - start

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"STAGE 3 EVALUATION — SEED {seed}")
    print("=" * 60)
    print(f"Episodes:      {episodes}")
    print(f"Landed:        {landed} ({100.0 * landed / episodes:.1f}%)")
    print(f"  On target:   {on_target} ({100.0 * on_target / episodes:.1f}%)")
    print(f"  Off target:  {landed - on_target}")
    print(f"Crashed:       {crashed} ({100.0 * crashed / episodes:.1f}%)")
    print(f"Out of fuel:   {out_of_fuel}")
    print(f"Truncated:     {truncated_count}")
    print(f"Avg score:     {np.mean(all_scores):.1f}")

    if landing_speeds:
        print(f"\n--- Landing Quality ---")
        print(f"Avg speed:     {np.mean(landing_speeds):.2f} m/s  (max {np.max(landing_speeds):.2f})")
        print(f"Avg angle:     {np.mean(landing_angles):.1f} deg  (max {np.max(landing_angles):.1f})")
        print(f"Avg steps:     {np.mean(landing_steps):.0f}  (min {np.min(landing_steps):.0f}, max {np.max(landing_steps):.0f})")
        print(f"Avg fuel left: {np.mean(landing_fuel):.1f}%")
        if landing_dist_x:
            print(f"Avg dist to pad: {np.mean(landing_dist_x):.2f} m  (max {np.max(landing_dist_x):.2f})")
        # Quality breakdown
        soft = sum(1 for s in landing_speeds if s < 1.0)
        gentle = sum(1 for s in landing_speeds if 1.0 <= s < 2.0)
        hard = sum(1 for s in landing_speeds if s >= 2.0)
        print(f"\nLanding breakdown:")
        print(f"  Soft (<1 m/s):     {soft} ({100.0 * soft / landed:.1f}%)")
        print(f"  Gentle (1-2 m/s):  {gentle} ({100.0 * gentle / landed:.1f}%)")
        print(f"  Hard (>2 m/s):     {hard} ({100.0 * hard / landed:.1f}%)")

    print(f"\nEval time: {elapsed:.1f}s ({elapsed / episodes:.2f}s/ep)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 landing model")
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    evaluate(seed=args.seed, episodes=args.episodes,
             save_dir=args.save_dir, visualize=args.visualize)
