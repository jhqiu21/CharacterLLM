# Logger utility for logging training and evaluation metrics


class Logger:
    def __init__(self, epoch: int):
        self.iteration = []
        # Training metrics
        self.loss_all_train = []
        self.loss_last_train = []
        self.loss_weighted_train = []
        # Evaluation loss metrics
        self.loss_all_val = []
        self.loss_last = []
        # Evaluation accuracy metrics
        self.acc = []
        self.acc_last = []
        # Other metrics
        self.perplexity = []
        self.bpc = []
        self.pred_entropy = []
        self.ece = []
        self.step = list(range(epoch))

    def log_train(self, metrics):
        self.loss_weighted_train.append(metrics['loss_train'])
        self.loss_all_train.append(metrics['loss_all'])
        self.loss_last_train.append(metrics['loss_last'])

    def log_eval(self, iteration: int, test_metrics):
        self.iteration.append(iteration)
        self.loss_all_val.append(test_metrics['loss_all'])
        self.loss_last.append(test_metrics['loss_last'])
        self.acc.append(test_metrics['acc'])
        self.acc_last.append(test_metrics['acc_last'])
        self.perplexity.append(test_metrics['perplexity'])
        self.bpc.append(test_metrics['bpc'])
        self.pred_entropy.append(test_metrics['pred_entropy'])
        self.ece.append(test_metrics['ece'][0])

    def print_metrics(self, time_since_start):
        print("")
        print(f"iteration {self.iteration[-1]}  time: {time_since_start:.1f} seconds")
        print(f"\t \t loss (weighted train): {self.loss_weighted_train[-1]:.4f}")
        print(f"\t \t loss (train :: test): {self.loss_all_train[-1]:.4f} :: {self.loss_all_val[-1]:.4f}")
        print(f"\t \t loss (last character): {self.loss_last[-1]:.4f}")
        print(f"\t \t accuracy: {100*self.acc[-1]:.1f}%")
        print(f"\t \t accuracy (last character): {100*self.acc_last[-1]:.1f}%")
        print(f"\t \t perplexity: {self.perplexity[-1]:.4f}")
        print(f"\t \t bits-per-character: {self.bpc[-1]:.4f}")
        print(f"\t \t ECE: {self.ece[0]:.4f}")
        print(f"\t \t pred_entropy: {self.pred_entropy[-1]:.4f}")

    def isBest(self, test_metrics):
        if not self.loss_all_val:
            # First evaluation is always the best
            return True, True, True

        is_best_loss_all = test_metrics['loss_all'] <= min(self.loss_all_val, default=float('inf'))
        is_best_acc = test_metrics['acc'] >= max(self.acc, default=float('-inf'))
        is_best_acc_last = test_metrics['acc_last'] >= max(self.acc_last, default=float('-inf'))
        return is_best_loss_all, is_best_acc, is_best_acc_last

    def __repr__(self):
        n_train = len(self.loss_weighted_train)
        n_eval = len(self.iteration)
        return f"Logger(train_steps={n_train}, eval_steps={n_eval})"

    def get_metrics_history(self):
        return {
            'step': self.step,
            'iteration': self.iteration,
            'loss_all_train': self.loss_all_train,
            'loss_weighted_train': self.loss_weighted_train,
            'loss_all_val': self.loss_all_val,
            'loss_last': self.loss_last,
            'acc': self.acc,
            'acc_last': self.acc_last,
            'perplexity': self.perplexity,
            'bpc': self.bpc,
            'pred_entropy': self.pred_entropy,
            'ece': self.ece,
        }