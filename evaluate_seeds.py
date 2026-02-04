"""
Seed Evaluation Script for Apollo Lander Double-DQN

Trains multiple Double-DQN agents with different seeds and evaluates them
to find the best 3 seeds for the triple ensemble.

Usage:
    python evaluate_seeds.py                    # Train and evaluate default seeds
    python evaluate_seeds.py --seeds 42 123 456 789 999  # Specific seeds
    python evaluate_seeds.py --quick            # Quick evaluation (fewer episodes)
    python evaluate_seeds.py --test-only        # Only test existing models

Seeds to try (from previous experiments):
    Known good: 42, 456, 999
    Known bad: 123
    Untested: 789, 1111, 2222, 3333, 4444, 5555
"""

import argparse
import numpy as np
import os
import json
from datetime import datetime
from collections import deque
import time

from apollo_lander_env import ApolloLanderEnv
from double_dqn_agent import DoubleDQNAgent


def train_and_evaluate_seed(seed, train_episodes=600, test_episodes=50, save_dir='models'):
    """
    Train a Double-DQN agent with a specific seed and evaluate its performance.

    Returns:
        dict: Results including average scores, landing rates, etc.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING SEED {seed}")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    # Training phase
    env = ApolloLanderEnv(render_mode=None)
    agent = DoubleDQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        seed=seed,
        hidden_size=256
    )

    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    train_landings = 0
    train_crashes = 0
    start_time = time.time()
    best_avg = -float('inf')

    for episode in range(1, train_episodes + 1):
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

        if info.get('landing_status') == 'landed':
            train_landings += 1
        elif info.get('landing_status') == 'crashed':
            train_crashes += 1

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        avg = np.mean(scores_window)
        if avg > best_avg and episode >= 100:
            best_avg = avg
            agent.save(os.path.join(save_dir, f'apollo_ddqn_seed{seed}_best.pth'))

        if episode % 100 == 0:
            print(f'  Episode {episode}/{train_episodes} | Avg: {avg:.2f} | '
                  f'Landings: {train_landings} | Best: {best_avg:.2f}')

    train_time = time.time() - start_time
    agent.save(os.path.join(save_dir, f'apollo_ddqn_seed{seed}_final.pth'))
    env.close()

    # Testing phase
    print(f"\n  Testing seed {seed} ({test_episodes} episodes)...")
    env = ApolloLanderEnv(render_mode=None)

    # Load best model for testing
    best_path = os.path.join(save_dir, f'apollo_ddqn_seed{seed}_best.pth')
    if os.path.exists(best_path):
        agent.load(best_path)

    test_scores = []
    test_landings = 0
    on_pad_landings = 0

    for ep in range(test_episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

        test_scores.append(score)
        if info.get('landing_status') == 'landed':
            test_landings += 1
            if info.get('on_target_pad', False):
                on_pad_landings += 1

    env.close()

    # Compile results
    results = {
        'seed': seed,
        'train_episodes': train_episodes,
        'train_time_minutes': train_time / 60,
        'train_final_avg': np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
        'train_best_avg': best_avg,
        'train_landing_rate': 100 * train_landings / train_episodes,
        'test_episodes': test_episodes,
        'test_avg_score': np.mean(test_scores),
        'test_std_score': np.std(test_scores),
        'test_min_score': np.min(test_scores),
        'test_max_score': np.max(test_scores),
        'test_landing_rate': 100 * test_landings / test_episodes,
        'test_on_pad_rate': 100 * on_pad_landings / test_episodes,
        # Composite score for ranking (weighted combination)
        'composite_score': (np.mean(test_scores) * 0.4 +
                          best_avg * 0.3 +
                          (100 * test_landings / test_episodes) * 1.5)
    }

    print(f"\n  Seed {seed} Results:")
    print(f"    Train Best Avg: {results['train_best_avg']:.2f}")
    print(f"    Test Avg Score: {results['test_avg_score']:.2f} (+/- {results['test_std_score']:.2f})")
    print(f"    Test Landing Rate: {results['test_landing_rate']:.1f}%")
    print(f"    On-Pad Landing Rate: {results['test_on_pad_rate']:.1f}%")
    print(f"    Composite Score: {results['composite_score']:.2f}")

    return results


def test_existing_model(seed, test_episodes=50, save_dir='models'):
    """Test an existing trained model without retraining."""
    model_path = os.path.join(save_dir, f'apollo_ddqn_seed{seed}_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, f'apollo_ddqn_seed{seed}_final.pth')
    if not os.path.exists(model_path):
        print(f"  No model found for seed {seed}")
        return None

    print(f"\n  Testing existing model for seed {seed}...")

    env = ApolloLanderEnv(render_mode=None)
    agent = DoubleDQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        seed=seed
    )
    agent.load(model_path)

    test_scores = []
    test_landings = 0
    on_pad_landings = 0

    for ep in range(test_episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

        test_scores.append(score)
        if info.get('landing_status') == 'landed':
            test_landings += 1
            if info.get('on_target_pad', False):
                on_pad_landings += 1

    env.close()

    results = {
        'seed': seed,
        'test_episodes': test_episodes,
        'test_avg_score': np.mean(test_scores),
        'test_std_score': np.std(test_scores),
        'test_min_score': np.min(test_scores),
        'test_max_score': np.max(test_scores),
        'test_landing_rate': 100 * test_landings / test_episodes,
        'test_on_pad_rate': 100 * on_pad_landings / test_episodes,
        'composite_score': (np.mean(test_scores) * 0.4 + (100 * test_landings / test_episodes) * 1.5)
    }

    print(f"    Test Avg Score: {results['test_avg_score']:.2f} (+/- {results['test_std_score']:.2f})")
    print(f"    Test Landing Rate: {results['test_landing_rate']:.1f}%")
    print(f"    On-Pad Landing Rate: {results['test_on_pad_rate']:.1f}%")

    return results


def evaluate_seeds(seeds, train_episodes=600, test_episodes=50, save_dir='models', test_only=False):
    """
    Evaluate multiple seeds and rank them.

    Args:
        seeds: List of seeds to evaluate
        train_episodes: Training episodes per seed
        test_episodes: Test episodes per seed
        save_dir: Directory for models
        test_only: If True, only test existing models

    Returns:
        List of results sorted by composite score (best first)
    """
    print("\n" + "="*70)
    print("SEED EVALUATION FOR APOLLO LANDER DOUBLE-DQN")
    print("="*70)
    print(f"Seeds to evaluate: {seeds}")
    print(f"{'Test only mode' if test_only else f'Training episodes: {train_episodes}'}")
    print(f"Test episodes: {test_episodes}")
    print("="*70)

    all_results = []
    start_time = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Processing seed {seed}...")

        if test_only:
            result = test_existing_model(seed, test_episodes, save_dir)
        else:
            result = train_and_evaluate_seed(seed, train_episodes, test_episodes, save_dir)

        if result:
            all_results.append(result)

    # Sort by composite score (descending)
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)

    total_time = time.time() - start_time

    # Print rankings
    print("\n" + "="*70)
    print("SEED RANKINGS (Best to Worst)")
    print("="*70)
    print(f"{'Rank':<6}{'Seed':<8}{'Test Avg':<12}{'Landing %':<12}{'On-Pad %':<12}{'Composite':<12}")
    print("-"*70)

    for i, r in enumerate(all_results):
        print(f"{i+1:<6}{r['seed']:<8}{r['test_avg_score']:<12.2f}"
              f"{r['test_landing_rate']:<12.1f}{r['test_on_pad_rate']:<12.1f}"
              f"{r['composite_score']:<12.2f}")

    print("="*70)
    print(f"Total evaluation time: {total_time/60:.1f} minutes")

    # Recommend top 3 for ensemble
    if len(all_results) >= 3:
        top_3 = all_results[:3]
        print(f"\n{'='*70}")
        print("RECOMMENDED SEEDS FOR TRIPLE ENSEMBLE")
        print("="*70)
        for i, r in enumerate(top_3):
            print(f"  Agent {i}: Seed {r['seed']}")
            print(f"           Test Avg: {r['test_avg_score']:.2f}")
            print(f"           Landing Rate: {r['test_landing_rate']:.1f}%")
        print("="*70)

        recommended_seeds = [r['seed'] for r in top_3]
        print(f"\nTo train triple ensemble with these seeds:")
        print(f"  Seeds to use: {recommended_seeds}")

    # Save results to JSON
    results_file = os.path.join(save_dir, f'seed_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'seeds_tested': seeds,
            'train_episodes': train_episodes if not test_only else 'N/A',
            'test_episodes': test_episodes,
            'total_time_minutes': total_time / 60,
            'results': all_results,
            'recommended_top_3': [r['seed'] for r in all_results[:3]] if len(all_results) >= 3 else None
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate seeds for Apollo Lander Double-DQN')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 999, 1111, 2222],
                       help='Seeds to evaluate')
    parser.add_argument('--train-episodes', type=int, default=600,
                       help='Training episodes per seed')
    parser.add_argument('--test-episodes', type=int, default=50,
                       help='Test episodes per seed')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation (300 train, 20 test)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing models')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory for models')

    args = parser.parse_args()

    if args.quick:
        args.train_episodes = 300
        args.test_episodes = 20
        print("Quick mode: 300 train episodes, 20 test episodes per seed")

    results = evaluate_seeds(
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        save_dir=args.save_dir,
        test_only=args.test_only
    )
