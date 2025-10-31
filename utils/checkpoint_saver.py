"""
Checkpoint utilities for saving model checkpoints.
"""

from flax.training import checkpoints


def save_collected_checkpoints(checkpoints_to_save, stage_dir, best_dir):
    """
    Save all checkpoints.
    Args:
        checkpoints_to_save: A dictionary containing the checkpoints to save
            {
                'last_epoch': (step, checkpoint_state) or None,
                'best_loss': (step, checkpoint_state) or None,
                'best_acc': (step, checkpoint_state) or None,
                'best_acc_last': (step, checkpoint_state) or None
            }
        stage_dir: stage checkpoint directory
        best_dir: best checkpoint directory
    """

    print("SAVING CHECKPOINTS")

    # save last epoch checkpoint
    if checkpoints_to_save.get('last_epoch') is not None:
        step, ckpt_state = checkpoints_to_save['last_epoch']
        print(f"\t Saving last epoch checkpoint (step {step:,})...")
        checkpoints.save_checkpoint(
            ckpt_dir=str(stage_dir),
            target=ckpt_state,
            step=step,
            prefix='last_ckpt_',
            keep=1,
            overwrite=True,
        )
        print(f"\t Saved (loss: {ckpt_state['test_loss']:.4f})")

    # save best loss checkpoint
    if checkpoints_to_save.get('best_loss') is not None:
        step, ckpt_state = checkpoints_to_save['best_loss']
        print(f"\t Saving best loss checkpoint (step {step:,})...")
        checkpoints.save_checkpoint(
            ckpt_dir=str(best_dir),
            target=ckpt_state,
            step=step,
            prefix='best_loss_',
            keep=1,
            overwrite=True,
        )
        print(f"\t Saved (loss: {ckpt_state['test_loss']:.4f})")

    # save best accuracy checkpoint
    if checkpoints_to_save.get('best_acc') is not None:
        step, ckpt_state = checkpoints_to_save['best_acc']
        print(f"\t Saving best accuracy checkpoint (step {step:,})...")
        checkpoints.save_checkpoint(
            ckpt_dir=str(best_dir),
            target=ckpt_state,
            step=step,
            prefix='best_acc_total_',
            keep=1,
            overwrite=True,
        )
        print(f"\t Saved (acc: {100*ckpt_state['test_acc']:.2f}%)")

    # save best last-char accuracy checkpoint
    if checkpoints_to_save.get('best_acc_last') is not None:
        step, ckpt_state = checkpoints_to_save['best_acc_last']
        print(f"\t Saving best last-char accuracy checkpoint (step {step:,})...")
        checkpoints.save_checkpoint(
            ckpt_dir=str(best_dir),
            target=ckpt_state,
            step=step,
            prefix='best_acc_last_',
            keep=1,
            overwrite=True,
        )
        print(f"\t Saved (acc_last: {100*ckpt_state['test_acc_last']:.2f}%)")

    print("ALL CHECKPOINTS SAVED")
