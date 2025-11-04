"""
Performance analysis utilities for model evaluation.
This module contains all performance analysis functions extracted from the original notebook.
"""

import numpy as np
import json


def analyze_training_performance(
    loss_test_history,
    loss_last_test_history,
    acc_test_history,
    acc_last_test_history,
    iteration_history,
    total_time,
    niter,
    n_final=10,
    save_results=True,
    results_path='training_results.json'
):
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Convert to numpy arrays for easier computation
    loss_test_history = np.array(loss_test_history)
    loss_last_test_history = np.array(loss_last_test_history)
    acc_test_history = np.array(acc_test_history)
    acc_last_test_history = np.array(acc_last_test_history)
    iteration_history = np.array(iteration_history)

    # === Best Test Loss ===
    best_test_loss_idx = np.argmin(loss_test_history)
    best_test_loss = loss_test_history[best_test_loss_idx]
    best_test_loss_iter = iteration_history[best_test_loss_idx]

    print(f"\nBest Test Loss:    {best_test_loss:.4f}  (at iteration {best_test_loss_iter:,})")

    # === Best Test Last Character Loss ===
    best_test_loss_last_idx = np.argmin(loss_last_test_history)
    best_test_loss_last = loss_last_test_history[best_test_loss_last_idx]
    best_test_loss_last_iter = iteration_history[best_test_loss_last_idx]

    print(f"Best Last Char Loss:    {best_test_loss_last:.4f}  (at iteration {best_test_loss_last_iter:,})")

    # === Best Test Accuracy ===
    best_test_acc_idx = np.argmax(acc_test_history)
    best_test_acc = acc_test_history[best_test_acc_idx]
    best_test_acc_iter = iteration_history[best_test_acc_idx]

    print(f"Best Test Acc:     {100*best_test_acc:.2f}%  (at iteration {best_test_acc_iter:,})")

    # === Best Last Character Accuracy ===
    best_test_acc_last_idx = np.argmax(acc_last_test_history)
    best_test_acc_last = acc_last_test_history[best_test_acc_last_idx]
    best_test_acc_last_iter = iteration_history[best_test_acc_last_idx]

    print(f"Best Last Char Acc: {100*best_test_acc_last:.2f}%  (at iteration {best_test_acc_last_iter:,})")

    # === Final Average Metrics (last n_final checkpoints) ===
    print(f"\nFinal Average (last {n_final} checkpoints):")

    n_final = min(n_final, len(loss_test_history))
    final_loss = loss_test_history[-n_final:]
    final_loss_last = loss_last_test_history[-n_final:]
    final_acc = acc_test_history[-n_final:]
    final_acc_last = acc_last_test_history[-n_final:]

    print(f"  Test Loss:       {final_loss.mean():.4f} ± {final_loss.std():.4f}")
    print(f"  Last Char Loss:  {final_loss_last.mean():.4f} ± {final_loss_last.std():.4f}")
    print(f"  Test Accuracy:   {100*final_acc.mean():.2f}% ± {100*final_acc.std():.2f}%")
    print(f"  Last Char Acc:   {100*final_acc_last.mean():.2f}% ± {100*final_acc_last.std():.2f}%")

    # === Training Efficiency ===
    print("\nTraining Time:")
    print(f"  Total:           {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Per Iteration:   {total_time/niter:.4f} seconds")
    print(f"  Total Iterations: {niter:,}")

    # === Convergence Analysis ===
    print("\nConvergence Analysis:")
    n_window = max(5, len(loss_test_history) // 4)
    if len(loss_test_history) >= 2 * n_window:
        recent_loss = np.mean(loss_test_history[-n_window:])
        earlier_loss = np.mean(loss_test_history[-2*n_window:-n_window])
        still_improving = recent_loss < earlier_loss
        print(f"  Still Improving:    {'Yes' if still_improving else 'No (plateau)'}")
    else:
        print("  Still Improving:    N/A (insufficient data)")

    print("=" * 60)

    # Prepare results dictionary
    results = {
        "best": {
            "test_loss": float(best_test_loss),
            "test_loss_iter": int(best_test_loss_iter),
            "test_loss_last": float(best_test_loss_last),
            "test_loss_last_iter": int(best_test_loss_last_iter),
            "test_acc": float(best_test_acc),
            "test_acc_iter": int(best_test_acc_iter),
            "test_acc_last": float(best_test_acc_last),
            "test_acc_last_iter": int(best_test_acc_last_iter),
        },
        "final_avg": {
            "test_loss_mean": float(final_loss.mean()),
            "test_loss_std": float(final_loss.std()),
            "test_loss_last_mean": float(final_loss_last.mean()),
            "test_loss_last_std": float(final_loss_last.std()),
            "test_acc_mean": float(final_acc.mean()),
            "test_acc_std": float(final_acc.std()),
            "test_acc_last_mean": float(final_acc_last.mean()),
            "test_acc_last_std": float(final_acc_last.std()),
            "n_checkpoints": int(n_final),
        },
        "training": {
            "total_time": float(total_time),
            "total_iterations": int(niter),
            "time_per_iter": float(total_time/niter),
        }
    }

    # Save results to file
    if save_results:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to '{results_path}'")

    return results


def get_summary_metrics(results):
    best_test_loss = results['best']['test_loss']
    best_test_loss_iter = results['best']['test_loss_iter']
    best_test_loss_last = results['best']['test_loss_last']
    best_test_loss_last_iter = results['best']['test_loss_last_iter']
    best_test_acc = results['best']['test_acc']
    best_test_acc_iter = results['best']['test_acc_iter']
    best_test_acc_last = results['best']['test_acc_last']
    best_test_acc_last_iter = results['best']['test_acc_last_iter']

    # For final metrics, we need to return numpy arrays for plotting
    final_loss_mean = results['final_avg']['test_loss_mean']
    final_loss_std = results['final_avg']['test_loss_std']
    final_acc_mean = results['final_avg']['test_acc_mean']
    final_acc_std = results['final_avg']['test_acc_std']
    final_acc_last_mean = results['final_avg']['test_acc_last_mean']
    final_acc_last_std = results['final_avg']['test_acc_last_std']

    # Create dummy arrays for compatibility with plotting functions
    n_final = results['final_avg']['n_checkpoints']
    final_loss = np.full(n_final, final_loss_mean)
    final_acc = np.full(n_final, final_acc_mean)
    final_acc_last = np.full(n_final, final_acc_last_mean)

    return (best_test_loss, best_test_loss_iter, best_test_loss_last, best_test_loss_last_iter, best_test_acc, best_test_acc_iter,
            best_test_acc_last, best_test_acc_last_iter, final_loss, final_acc, final_acc_last)
