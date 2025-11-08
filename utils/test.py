from . import eval
import jax.numpy as jnp
import numpy as np


def test_checkpoint(model, param, test_data):
    data_length = len(test_data)
    batch_size = 1024
    sequence_length = 32
    num_blocks = (data_length - 1) // sequence_length
    num_batches = max(1, num_blocks // batch_size)

    total_loss = 0.0
    total_tok = 0
    correct_all = 0
    correct_last = 0
    total_seq = 0
    print("Starting evaluation on test set...")
    for i in range(num_batches):
        test_batch = get_batch_test(test_data, i, batch_size, sequence_length)
        if test_batch is None:
            break
        test_inputs, test_targets = test_batch
        test_logits = model.apply({'params': param}, test_inputs, deterministic=True)

        B_eff, T = test_inputs.shape
        loss_all = eval.loss_all(test_logits, test_targets)
        acc_all, acc_last = eval.accuracy(test_logits, test_targets)
        total_loss += loss_all.item() * B_eff * T
        total_tok += B_eff * T
        correct_all += acc_all.item() * B_eff * T
        correct_last += acc_last.item() * B_eff
        total_seq += B_eff
    print("Finished evaluation on test set...")
    avg_loss = total_loss / total_tok
    perplexity = np.exp(avg_loss)
    bpc = eval.bits_per_character(avg_loss)
    overall_acc = correct_all / total_tok
    last_char_acc = correct_last / total_seq
    print("Test Set Evaluation:")
    print(f"\t \tPerplexity: {perplexity:.4f}")
    print(f"\t \tBits per Character: {bpc:.4f}")
    print(f"\t \tOverall Accuracy: {overall_acc*100:.2f}%")
    print(f"\t \tLast Character Accuracy: {last_char_acc*100:.2f}%")
    return perplexity, bpc, overall_acc, last_char_acc


def get_batch_test(text_int, it, B, T):
    # get batch in order of size B x T
    starts = np.arange(it * B, (it + 1) * B) * T
    starts = starts[starts + T < len(text_int)]
    if starts.size == 0:
        return None
    x = jnp.stack([text_int[s:s + T] for s in starts])
    y = jnp.stack([text_int[s + 1:s + 1 + T] for s in starts])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)
