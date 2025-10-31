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
    iteration_history,
    acc_test_history,
    acc_last_test_history,
    save_path='training_curves.png'
):
    """
    Create comprehensive training visualization with multiple subplots.
    """
    # Create a comprehensive figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

    # ===== Plot 2: Accuracy Curves =====
    ax2 = axes[0, 1]
    ax2.plot(iteration_history, [100*x for x in acc_test_history], '-o',
            lw=2, markersize=3, color="green", label='Test Acc')
    # Mark best accuracy
    best_acc_idx = np.argmax(acc_test_history)
    ax2.scatter(iteration_history[best_acc_idx], 100*acc_test_history[best_acc_idx],
               s=100, color='green', marker='*', zorder=5,
               label=f'Best: {100*acc_test_history[best_acc_idx]:.2f}%')
    # Add final average line
    n_final = min(10, len(acc_test_history))
    final_avg = np.mean(acc_test_history[-n_final:]) * 100
    ax2.axhline(y=final_avg, color='orange', linestyle='--', alpha=0.7,
               label=f'Final Avg: {final_avg:.2f}%')
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax2.legend(loc='lower right')
    ax2.set_title("Test Accuracy", fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # ===== Plot 3: Last Character Accuracy =====
    ax3 = axes[1, 0]
    ax3.plot(iteration_history, [100*x for x in acc_last_test_history], '-s',
            lw=2, markersize=3, color="purple", label='Last Char Acc')
    # Mark best
    best_last_idx = np.argmax(acc_last_test_history)
    ax3.scatter(iteration_history[best_last_idx], 100*acc_last_test_history[best_last_idx],
               s=100, color='purple', marker='*', zorder=5,
               label=f'Best: {100*acc_last_test_history[best_last_idx]:.2f}%')
    # Add final average line
    final_last_avg = np.mean(acc_last_test_history[-n_final:]) * 100
    ax3.axhline(y=final_last_avg, color='orange', linestyle='--', alpha=0.7,
               label=f'Final Avg: {final_last_avg:.2f}%')
    ax3.set_xlabel("Iteration", fontsize=11)
    ax3.set_ylabel("Last Character Accuracy (%)", fontsize=11)
    ax3.legend(loc='lower right')
    ax3.set_title("Last Character Prediction", fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    # ===== Plot 4: Loss Convergence (smoothed) =====
    ax4 = axes[1, 1]
    # Smooth the test loss curve
    window_size = max(1, len(loss_test_history) // 10)
    if window_size > 1:
        smoothed_loss = np.convolve(loss_test_history,
                                    np.ones(window_size)/window_size,
                                    mode='valid')
        smoothed_iters = iteration_history[window_size-1:]
        ax4.plot(smoothed_iters, smoothed_loss, '-', lw=2.5, color='darkred',
                label=f'Smoothed (window={window_size})')
    ax4.plot(iteration_history, loss_test_history, '-', alpha=0.3, color='red',
            label='Raw Test Loss')
    ax4.set_xlabel("Iteration", fontsize=11)
    ax4.set_ylabel("Test Loss", fontsize=11)
    ax4.legend(loc='upper right')
    ax4.set_title("Loss Convergence (Smoothed)", fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to '{save_path}'")
    plt.show()


def plot_summary_table(
    best_test_loss,
    best_test_loss_iter,
    best_test_acc,
    best_test_acc_iter,
    best_test_acc_last,
    best_test_acc_last_iter,
    final_loss,
    final_acc,
    final_acc_last
):
    """
    Create a summary table showing best and final average metrics.
    """
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Prepare summary data
    summary_data = [
        ['Metric', 'Best Value', 'Iteration', 'Final Avg', 'Std'],
        ['Test Loss', f'{best_test_loss:.4f}', f'{best_test_loss_iter:,}',
         f'{final_loss.mean():.4f}', f'±{final_loss.std():.4f}'],
        ['Test Accuracy', f'{100*best_test_acc:.2f}%', f'{best_test_acc_iter:,}',
         f'{100*final_acc.mean():.2f}%', f'±{100*final_acc.std():.2f}%'],
        ['Last Char Acc', f'{100*best_test_acc_last:.2f}%', f'{best_test_acc_last_iter:,}',
         f'{100*final_acc_last.mean():.2f}%', f'±{100*final_acc_last.std():.2f}%'],
    ]

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.show()
