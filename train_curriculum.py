"""
Curriculum Learning for Apollo Lander

Staged training approach:
  Stage 1: Learn to stabilize (start tilted, just get upright)
  Stage 2: Learn to hover (upright + arrest descent)
  Stage 3: Learn to position (hover + move over pad)
  Stage 4: Full task (position + land)

Each stage gradually increases difficulty and builds on previous skills.

Usage:
    python train_curriculum.py --seed 999
    python train_curriculum.py --seed 999 --stage 2  # Start from stage 2
    python train_curriculum.py --seed 999 --visualize
"""

import argparse
import numpy as np
import os
import json
from collections import deque
import time
import math

from apollo_lander_env import ApolloLanderEnv
from double_dqn_agent import DoubleDQNAgent


class CurriculumEnv:
    """
    Wrapper that modifies the base environment for curriculum learning.

    Stages:
        1: Stabilization only - start tilted, success = get upright
        2: Hover - start tilted, success = upright + hovering
        3: Position + descent - success = upright + slow + near pad + below 20m altitude
        4: Full landing - complete task (land on surface)
    """

    def __init__(self, stage=1, render_mode=None):
        # Spring constants vary by curriculum stage:
        # Stage 1: only rotation springs, no pad/descent (focus on getting upright)
        # Stage 2: add ascend penalty + mild descent tracking (prevent fly-up)
        # Stage 3-4: full altitude-proportional descent profile for controlled landing
        spring_configs = {
            1: {'spring_k_rotation': 25.0, 'spring_k_pad': 0.0, 'spring_c_damping': 40.0,
                'spring_c_descent': 0.0, 'spring_c_ascend': 0.0,
                'descent_max_rate': 0.0, 'spring_pad_rest_length': 15.0},
            2: {'spring_k_rotation': 22.0, 'spring_k_pad': 0.5, 'spring_c_damping': 35.0,
                'spring_c_descent': 2.0, 'spring_c_ascend': 4.0,
                'descent_max_rate': 0.0, 'spring_pad_rest_length': 15.0},
            3: {'spring_k_rotation': 20.0, 'spring_k_pad': 1.5, 'spring_c_damping': 30.0,
                'spring_c_descent': 3.0, 'spring_c_ascend': 5.0,
                'descent_max_rate': 2.5, 'spring_pad_rest_length': 15.0},
            4: {'spring_k_rotation': 20.0, 'spring_k_pad': 2.0, 'spring_c_damping': 30.0,
                'spring_c_descent': 3.0, 'spring_c_ascend': 5.0,
                'descent_max_rate': 2.5, 'spring_pad_rest_length': 15.0},
        }
        sc = spring_configs[stage]
        self.base_env = ApolloLanderEnv(
            render_mode=render_mode,
            spring_k_rotation=sc['spring_k_rotation'],
            spring_k_pad=sc['spring_k_pad'],
            spring_c_damping=sc['spring_c_damping'],
            spring_c_descent=sc['spring_c_descent'],
            spring_c_ascend=sc['spring_c_ascend'],
            descent_max_rate=sc['descent_max_rate'],
            spring_pad_rest_length=sc['spring_pad_rest_length'],
        )
        self.stage = stage
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        # Stage-specific parameters
        # graduation_threshold: success rate needed to advance to next stage
        # min_episodes: minimum episodes before graduation check
        self.stage_configs = {
            1: {  # Stabilization - get upright and STAY upright
                'max_angle': np.pi/4,      # +/- 45 degree start (achievable range)
                'max_steps': 500,          # Short episodes
                'success_angle': 0.26,     # ~15 degrees
                'success_angular_vel': 0.3, # Must be mostly stable
                'success_reward': 200.0,
                'graduation_threshold': 70.0,  # 70% success to advance
                'min_episodes': 200,
            },
            2: {  # Hover - upright + controlled descent
                'max_angle': np.pi/3,      # +/- 60 degree start
                'max_steps': 800,
                'success_angle': 0.17,
                'success_vel_y': 1.0,      # Descent under control
                'success_angular_vel': 0.2,
                'success_reward': 250.0,
                'graduation_threshold': 65.0,
                'min_episodes': 200,
            },
            3: {  # Position - descend near pad
                'max_angle': np.pi/4,      # +/- 45 degree start
                'max_steps': 1500,         # More time to descend
                'success_angle': 0.17,
                'success_vel_y': 2.0,      # Compatible with descent profile at 30m (target=-1.5)
                'success_dist_x': 10.0,    # Over pad (relaxed)
                'success_angular_vel': 0.2,
                'success_altitude': 30.0,  # Must descend below 30m (20m from spawn)
                'success_reward': 300.0,
                'graduation_threshold': 40.0,  # Lowered: descent skills transfer well even at 37%
                'min_episodes': 300,
            },
            4: {  # Full landing - complete the task
                'max_angle': np.pi/3,      # +/- 60 degrees (gentle step from Stage 3's 45°)
                'max_steps': 2000,
                'success_reward': 400.0,   # Landing handled by base env
                'graduation_threshold': 50.0,  # 50% landing rate is good!
                'min_episodes': 500,
            }
        }

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)

        config = self.stage_configs[self.stage]

        # Adjust initial angle based on stage
        if self.stage < 4:
            max_angle = config['max_angle']
            initial_angle = self.base_env.np_random.uniform(-max_angle, max_angle)

            descent = self.base_env.lander.descent_stage
            ascent = self.base_env.lander.ascent_stage

            # Compute the ascent-to-descent offset at the current (upright) position
            ascent_offset_y = ascent.position.y - descent.position.y
            ascent_offset_x = ascent.position.x - descent.position.x

            # Set both body angles
            descent.angle = initial_angle
            ascent.angle = initial_angle

            # Reposition ascent stage to maintain correct relative position
            # Rotate the offset vector by the new angle
            sin_a = math.sin(initial_angle)
            cos_a = math.cos(initial_angle)
            new_offset_x = ascent_offset_x * cos_a - ascent_offset_y * sin_a
            new_offset_y = ascent_offset_x * sin_a + ascent_offset_y * cos_a
            from Box2D import b2Vec2
            ascent.position = b2Vec2(
                descent.position.x + new_offset_x,
                descent.position.y + new_offset_y
            )

            # Zero out velocities to start clean
            descent.angularVelocity = 0.0
            ascent.angularVelocity = 0.0
            descent.linearVelocity = (0, 0)
            ascent.linearVelocity = (0, 0)

            # Reset angle tracking to match new initial angle
            # Box2D angle convention: positive = CCW; our convention: positive = tilted right
            # cumulative_angle uses negated convention, so negate the Box2D angle
            self.base_env.cumulative_angle = -initial_angle
            self.base_env.prev_raw_angle = initial_angle
            self.base_env.prev_shaping = None  # Recompute on first step

        # Reset step counter
        self.steps = 0
        self.max_steps = config['max_steps']
        self._logged_success = False  # Reset success logging flag
        self._action_counts = {i: 0 for i in range(5)}  # Reset action tracking
        self._stable_steps = 0  # Consecutive steps meeting success criteria
        self._episode_had_success = False  # Track if stability was ever achieved

        return self.base_env._get_observation(), info

    def _get_terrain_height(self, x):
        """Get terrain height at x position."""
        from apollo_lander_env import get_terrain_height_at
        return get_terrain_height_at(x, self.base_env.terrain_pts)

    # Action names for logging (always-on engine with throttle control)
    # RCS_L rotates craft LEFT (use when tilted right, angle > 0)
    # RCS_R rotates craft RIGHT (use when tilted left, angle < 0)
    ACTION_NAMES = {
        0: "NOOP",
        1: "RCS_L",    # Rotate LEFT (correct for angle > 0)
        2: "THR_UP",   # Increase throttle +5%
        3: "THR_DN",   # Decrease throttle -5%
        4: "RCS_R",    # Rotate RIGHT (correct for angle < 0)
    }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        self.steps += 1

        # Track actions for analysis
        if not hasattr(self, '_action_counts'):
            self._action_counts = {i: 0 for i in range(5)}
        self._action_counts[action] = self._action_counts.get(action, 0) + 1

        # Detailed logging every N steps (adjustable)
        log_interval = 50  # Log every 50 steps
        if self.steps % log_interval == 0 or self.steps <= 10:
            angle = self.base_env.cumulative_angle
            angle_deg = math.degrees(angle)
            # Negate to match our convention: positive = rotating RIGHT
            angular_vel = -self.base_env.lander.descent_stage.angularVelocity
            vel = self.base_env.lander.descent_stage.linearVelocity
            pos = self.base_env.lander.descent_stage.position
            terrain_h = self._get_terrain_height(pos.x)
            altitude = pos.y - terrain_h

            action_name = self.ACTION_NAMES.get(action, f"?{action}")
            throttle_pct = int(self.base_env.throttle * 100)

            # Show recent action distribution
            total_actions = sum(self._action_counts.values())
            action_dist = " ".join([f"{self.ACTION_NAMES[i]}:{100*self._action_counts[i]//total_actions}%"
                                    for i in range(5) if self._action_counts[i] > 0])

            print(f"  [S{self.steps:4d}] Act={action_name:5s} | "
                  f"Ang={angle_deg:+6.1f}° AngV={angular_vel:+5.2f} | "
                  f"Alt={altitude:5.1f}m VelY={vel.y:+5.1f} | "
                  f"Thr={throttle_pct}% | {action_dist}")

        config = self.stage_configs[self.stage]

        # Check stage-specific success conditions
        # Note: In stages 1-3, we give bonus reward but DON'T terminate early
        # This allows the agent to practice maintaining the stable state longer

        # Number of consecutive stable steps needed for early termination
        SUSTAINED_STEPS = 30

        if self.stage == 1:  # Stabilization
            angle = abs(self.base_env.lander.descent_stage.angle)
            angular_vel = abs(self.base_env.lander.descent_stage.angularVelocity)
            angle_deg = math.degrees(angle)
            success_ang_vel = config.get('success_angular_vel', 0.2)

            is_stable = angle < config['success_angle'] and angular_vel < success_ang_vel
            if is_stable:
                self._stable_steps += 1
                reward += config['success_reward'] / 100.0  # Per-step bonus
                info['stage_success'] = True
                if not self._logged_success:
                    print(f"  [STABLE] Ep {self.base_env.episode_count} Step {self.steps}: "
                          f"Angle={angle_deg:.1f}° AngVel={angular_vel:.2f} (streak: {self._stable_steps})")
                    self._logged_success = True
            else:
                self._stable_steps = 0
                self._logged_success = False

            # Sustained stability → early success termination
            if self._stable_steps >= SUSTAINED_STEPS:
                reward += config['success_reward']  # Big bonus for sustained stability
                terminated = True
                info['stage_success'] = True
                print(f"  [GRADUATED] Ep {self.base_env.episode_count} Step {self.steps}: "
                      f"SUSTAINED STABILITY for {SUSTAINED_STEPS} steps!")

        elif self.stage == 2:  # Hover
            angle = abs(self.base_env.lander.descent_stage.angle)
            vel_y = abs(self.base_env.lander.descent_stage.linearVelocity.y)
            angular_vel = abs(self.base_env.lander.descent_stage.angularVelocity)
            angle_deg = math.degrees(angle)
            success_ang_vel = config.get('success_angular_vel', 0.2)

            is_stable = (angle < config['success_angle'] and
                         vel_y < config['success_vel_y'] and
                         angular_vel < success_ang_vel)
            if is_stable:
                self._stable_steps += 1
                reward += config['success_reward'] / 100.0
                info['stage_success'] = True
                if not self._logged_success:
                    print(f"  [STABLE] Ep {self.base_env.episode_count} Step {self.steps}: "
                          f"HOVERING | Angle={angle_deg:.1f}° VelY={vel_y:.2f} AngVel={angular_vel:.2f} (streak: {self._stable_steps})")
                    self._logged_success = True
            else:
                self._stable_steps = 0
                self._logged_success = False

            if self._stable_steps >= SUSTAINED_STEPS:
                reward += config['success_reward']
                terminated = True
                info['stage_success'] = True
                print(f"  [GRADUATED] Ep {self.base_env.episode_count} Step {self.steps}: "
                      f"SUSTAINED HOVER for {SUSTAINED_STEPS} steps!")

        elif self.stage == 3:  # Position + descent
            angle = abs(self.base_env.lander.descent_stage.angle)
            vel_y = abs(self.base_env.lander.descent_stage.linearVelocity.y)
            angular_vel = abs(self.base_env.lander.descent_stage.angularVelocity)
            angle_deg = math.degrees(angle)
            success_ang_vel = config.get('success_angular_vel', 0.2)

            pos = self.base_env.lander.descent_stage.position
            target_x = (self.base_env.target_pad["x1"] + self.base_env.target_pad["x2"]) / 2.0
            dist_x = abs(pos.x - target_x)

            # Altitude check: agent must actually descend, not hover at spawn height
            terrain_h = self._get_terrain_height(pos.x)
            altitude = pos.y - terrain_h
            max_alt = config.get('success_altitude', 999.0)

            is_stable = (angle < config['success_angle'] and
                         vel_y < config['success_vel_y'] and
                         dist_x < config['success_dist_x'] and
                         angular_vel < success_ang_vel and
                         altitude < max_alt)
            if is_stable:
                self._stable_steps += 1
                reward += config['success_reward'] / 100.0
                info['stage_success'] = True
                if not self._logged_success:
                    print(f"  [STABLE] Ep {self.base_env.episode_count} Step {self.steps}: "
                          f"POSITIONED | Angle={angle_deg:.1f}° DistX={dist_x:.1f}m Alt={altitude:.1f}m AngVel={angular_vel:.2f} (streak: {self._stable_steps})")
                    self._logged_success = True
            else:
                self._stable_steps = 0
                self._logged_success = False

            if self._stable_steps >= SUSTAINED_STEPS:
                reward += config['success_reward']
                terminated = True
                info['stage_success'] = True
                print(f"  [GRADUATED] Ep {self.base_env.episode_count} Step {self.steps}: "
                      f"SUSTAINED POSITION for {SUSTAINED_STEPS} steps!")

        # Stage 4 uses base environment's termination

        # Truncate if max steps reached
        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


def train_stage(stage, seed, episodes, save_dir='models', visualize=False,
                load_previous=True):
    """Train a single curriculum stage."""

    print(f"\n{'='*70}")
    print(f"CURRICULUM STAGE {stage} - SEED {seed}")
    print(f"{'='*70}")

    stage_names = {
        1: "STABILIZATION (get upright)",
        2: "HOVER (upright + arrest descent)",
        3: "POSITION (hover over pad)",
        4: "FULL LANDING (complete task)"
    }
    print(f"Stage: {stage_names[stage]}")
    print(f"Episodes: {episodes}")
    print("="*70)

    os.makedirs(save_dir, exist_ok=True)

    env = CurriculumEnv(stage=stage, render_mode='human' if visualize else None)

    # Stage-specific hyperparameters:
    # Stages 1-3: faster learning (5e-4) with standard buffer
    # Stage 4: slower learning (2e-4) with larger buffer to prevent catastrophic forgetting
    agent_configs = {
        1: {'lr': 5e-4, 'buffer_size': int(1e5)},
        2: {'lr': 5e-4, 'buffer_size': int(1e5)},
        3: {'lr': 5e-4, 'buffer_size': int(1e5)},
        4: {'lr': 2e-4, 'buffer_size': int(2e5)},
    }
    ac = agent_configs[stage]

    agent = DoubleDQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        seed=seed,
        hidden_size=256,
        lr=ac['lr'],
        batch_size=128,
        gamma=0.99,
        tau=1e-3,
        buffer_size=ac['buffer_size'],
        update_every=4
    )

    # Load weights from previous stage if available
    if load_previous and stage > 1:
        prev_path = os.path.join(save_dir, f'curriculum_stage{stage-1}_seed{seed}_best.pth')
        if os.path.exists(prev_path):
            agent.load(prev_path)
            print(f"Loaded weights from stage {stage-1}")

    scores = []
    scores_window = deque(maxlen=100)
    successes = deque(maxlen=100)

    # Stage-specific epsilon scheduling:
    # Stage 1: high exploration (learning from scratch)
    # Stage 2-3: moderate exploration (building on previous skills)
    # Stage 4: low exploration (exploit learned skills, random actions destroy landing)
    eps_configs = {
        1: {'start': 1.0, 'end': 0.05, 'decay': 0.997},
        2: {'start': 0.5, 'end': 0.05, 'decay': 0.997},
        3: {'start': 0.4, 'end': 0.03, 'decay': 0.996},
        4: {'start': 0.2, 'end': 0.02, 'decay': 0.995},
    }
    ec = eps_configs[stage]
    eps = ec['start']
    eps_end = ec['end']
    eps_decay = ec['decay']

    best_success_rate = 0
    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        done = False

        while not done:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if visualize:
                env.render()

        # Track success
        success = info.get('stage_success', False) or info.get('landing_status') == 'landed'
        successes.append(1 if success else 0)

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        avg = np.mean(scores_window)
        success_rate = 100 * sum(successes) / len(successes)

        # Save best model based on success rate
        if episode >= 50 and success_rate > best_success_rate:
            best_success_rate = success_rate
            agent.save(os.path.join(save_dir, f'curriculum_stage{stage}_seed{seed}_best.pth'))

        if episode % 50 == 0:
            elapsed = time.time() - start_time
            config = env.stage_configs[stage]
            grad_thresh = config.get('graduation_threshold', 70.0)
            print(f'Episode {episode:4d}/{episodes} | Avg: {avg:7.1f} | '
                  f'Success: {success_rate:5.1f}% | Best: {best_success_rate:5.1f}% | '
                  f'Grad@{grad_thresh:.0f}% | Eps: {eps:.3f} | {elapsed/60:.1f}m')

        # Early graduation: move to next stage if success rate threshold met
        config = env.stage_configs[stage]
        grad_thresh = config.get('graduation_threshold', 70.0)
        min_eps = config.get('min_episodes', 200)
        if episode >= min_eps and success_rate >= grad_thresh:
            print(f"\n{'='*70}")
            print(f"STAGE {stage} GRADUATED at episode {episode}!")
            print(f"Success Rate: {success_rate:.1f}% >= {grad_thresh:.0f}% threshold")
            print(f"{'='*70}")
            break

    # Save final model
    agent.save(os.path.join(save_dir, f'curriculum_stage{stage}_seed{seed}_final.pth'))
    env.close()

    elapsed = time.time() - start_time
    final_success = 100 * sum(successes) / len(successes)

    print(f'\n{"="*70}')
    print(f'STAGE {stage} COMPLETE - SEED {seed}')
    print(f'{"="*70}')
    print(f'Episodes: {episode}')
    print(f'Final Success Rate: {final_success:.1f}%')
    print(f'Best Success Rate: {best_success_rate:.1f}%')
    print(f'Training Time: {elapsed/60:.1f} minutes')
    print(f'{"="*70}')

    return {
        'stage': stage,
        'seed': seed,
        'episodes': episode,
        'final_success_rate': final_success,
        'best_success_rate': best_success_rate,
        'train_time': elapsed
    }


def train_full_curriculum(seed, episodes_per_stage=500, save_dir='models',
                          visualize=False, start_stage=1):
    """Train through all curriculum stages."""

    print("\n" + "="*70)
    print("FULL CURRICULUM TRAINING")
    print("="*70)
    print(f"Seed: {seed}")
    print(f"Episodes per stage: {episodes_per_stage}")
    print(f"Starting from stage: {start_stage}")
    print("="*70)

    all_results = []
    total_start = time.time()

    for stage in range(start_stage, 5):
        result = train_stage(
            stage=stage,
            seed=seed,
            episodes=episodes_per_stage,
            save_dir=save_dir,
            visualize=visualize,
            load_previous=(stage > start_stage)
        )
        all_results.append(result)

        # Check if stage was mastered
        if result['best_success_rate'] < 50:
            print(f"\nWARNING: Stage {stage} not mastered (only {result['best_success_rate']:.1f}%)")
            print("Consider more training before proceeding.")

    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("CURRICULUM TRAINING SUMMARY")
    print("="*70)
    for r in all_results:
        print(f"  Stage {r['stage']}: {r['best_success_rate']:.1f}% success in {r['episodes']} episodes")
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print("="*70)

    # Save summary
    summary_path = os.path.join(save_dir, f'curriculum_summary_seed{seed}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'seed': seed,
            'total_time_minutes': total_time / 60,
            'results': all_results
        }, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Curriculum learning for Apollo Lander')
    parser.add_argument('--seed', type=int, default=999, help='Random seed')
    parser.add_argument('--stage', type=int, default=1, help='Start from this stage (1-4)')
    parser.add_argument('--episodes', type=int, default=1000, help='Episodes per stage')
    parser.add_argument('--single-stage', action='store_true', help='Only train specified stage')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    parser.add_argument('--save-dir', type=str, default='models', help='Model directory')

    args = parser.parse_args()

    if args.single_stage:
        train_stage(
            stage=args.stage,
            seed=args.seed,
            episodes=args.episodes,
            save_dir=args.save_dir,
            visualize=args.visualize
        )
    else:
        train_full_curriculum(
            seed=args.seed,
            episodes_per_stage=args.episodes,
            save_dir=args.save_dir,
            visualize=args.visualize,
            start_stage=args.stage
        )
