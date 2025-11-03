"""
Checkpoint utilities for saving model checkpoints.
"""

from flax.training import checkpoints


def save_collected_checkpoints(checkpoints_to_save, stage_dir, best_dir):
    # Save various collected checkpoints to specified directories.
    print("SAVING CHECKPOINTS")

    for i, (step, ckpt_state) in enumerate(checkpoints_to_save.get('stage', [])):
        print(f"\t Saving stage checkpoint {i+1} (step {step:,})...")
        checkpoints.save_checkpoint(
            ckpt_dir=str(stage_dir),
            target=ckpt_state,
            step=step,
            prefix=f'stage_{i+1}_',
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

    # save best loss_last checkpoint
    if checkpoints_to_save.get('best_loss_last') is not None:
        step, ckpt_state = checkpoints_to_save['best_loss_last']
        print(f"\t Saving best last-character loss checkpoint (step {step:,})...")
        checkpoints.save_checkpoint(
            ckpt_dir=str(best_dir),
            target=ckpt_state,
            step=step,
            prefix='best_loss_last_',
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
