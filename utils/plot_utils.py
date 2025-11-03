"""
Plotting utilities for training visualization.
This module contains all plotting functions extracted from the original notebook.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(
    time_history,
    loss_history,
    time_test_history,
    loss_test_history,
    loss_last_test_history,
    iteration_history,
    acc_test_history,
    acc_last_test_history,
    save_path='training_curves.pdf'
):  
    # Create a comprehensive figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(9, 10))

    # ===== Plot 1: Loss Curves =====
    ax1 = axes[0, 0]
    ax1.plot(time_history, loss_history, '-', label='Train', color="blue", alpha=0.6)
    ax1.plot(time_test_history, loss_test_history, '-', label='Test', lw=2, color="red")
    # Mark best test loss
    best_idx = np.argmin(loss_test_history)
    ax1.scatter(time_test_history[best_idx], loss_test_history[best_idx],
               s=100, color='red', marker='*', zorder=5,
               label=f'Best: {loss_test_history[best_idx]:.4f}')
    ax1.set_xlabel("Time (seconds)", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc='upper right')
    ax1.set_title("Training & Test Loss", fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)

    # ===== Plot 2: Last Char. Loss Curves =====
    ax2 = axes[0, 1]
    ax2.plot(time_history, loss_history, '-', label='Train', color="blue", alpha=0.6)
    ax2.plot(time_test_history, loss_last_test_history, '-', label='Test', lw=2, color="red")
    # Mark best test loss
    best_idx = np.argmin(loss_last_test_history)
    ax2.scatter(time_test_history[best_idx], loss_last_test_history[best_idx],
               s=100, color='red', marker='*', zorder=5,
               label=f'Best: {loss_last_test_history[best_idx]:.4f}')
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(loc='upper right')
    ax2.set_title("Training & Test Last Char Loss", fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)


    # ===== Plot 3: Accuracy Curves =====
    ax3 = axes[1, 0]
    ax3.plot(iteration_history, [100*x for x in acc_test_history], '-o',
            lw=2, markersize=3, color="green", label='Total Acc')
    # Mark best accuracy
    best_acc_idx = np.argmax(acc_test_history)
    ax3.scatter(iteration_history[best_acc_idx], 100*acc_test_history[best_acc_idx],
               s=100, color='red', marker='*', zorder=5,
               label=f'Best: {100*acc_test_history[best_acc_idx]:.2f}%')
    # Add final average line
    n_final = min(10, len(acc_test_history))
    final_avg = np.mean(acc_test_history[-n_final:]) * 100
    ax3.axhline(y=final_avg, color='orange', linestyle='--', alpha=0.7,
               label=f'Final Avg: {final_avg:.2f}%')
    ax3.set_xlabel("Iteration", fontsize=11)
    ax3.set_ylabel("Total Accuracy (%)", fontsize=11)
    ax3.legend(loc='lower right')
    ax3.set_title("Test Total Accuracy", fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    # ===== Plot 4: Last Character Accuracy =====
    ax4 = axes[1, 1]
    ax4.plot(iteration_history, [100*x for x in acc_last_test_history], '-s',
            lw=2, markersize=3, color="purple", label='Last Char Acc')
    # Mark best
    best_last_idx = np.argmax(acc_last_test_history)
    ax4.scatter(iteration_history[best_last_idx], 100*acc_last_test_history[best_last_idx],
               s=100, color='#E69F00', marker='*', zorder=5,
               label=f'Best: {100*acc_last_test_history[best_last_idx]:.2f}%')
    # Add final average line
    final_last_avg = np.mean(acc_last_test_history[-n_final:]) * 100
    ax4.axhline(y=final_last_avg, color='orange', linestyle='--', alpha=0.7,
               label=f'Final Avg: {final_last_avg:.2f}%')
    ax4.set_xlabel("Iteration", fontsize=11)
    ax4.set_ylabel("Last Character Accuracy (%)", fontsize=11)
    ax4.legend(loc='lower right')
    ax4.set_title("Test Last Character Accuracy", fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)

    # ===== Plot 5: Loss Convergence (smoothed) =====
    ax5 = axes[2, 0]
    # Smooth the test loss curve
    window_size = max(1, len(loss_test_history) // 10)
    if window_size > 1:
        smoothed_loss = np.convolve(loss_test_history,
                                    np.ones(window_size)/window_size,
                                    mode='valid')
        smoothed_iters = iteration_history[window_size-1:]
        ax5.plot(smoothed_iters, smoothed_loss, '-', lw=2.5, color='darkred',
                label=f'Smoothed (window={window_size})')
    ax5.plot(iteration_history, loss_test_history, '-', alpha=0.3, color='red',
            label='Raw Test Loss')
    ax5.set_xlabel("Iteration", fontsize=11)
    ax5.set_ylabel("Test Loss", fontsize=11)
    ax5.legend(loc='upper right')
    ax5.set_title("Loss Convergence (Smoothed)", fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to '{save_path}'")
    plt.show()