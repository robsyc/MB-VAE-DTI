import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
import wandb

class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    # TODO: This is important
    # lambda_train: [0, 1, 0] is passed on to this loss function by DiffMS
    # lambda_train: [1, 5, 0] is passed on to this loss function by DiGress <- what we also use
    def __init__(self, lambda_train = [1, 5, 0]):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)

            self.reset()
        
        # DiffMS Default: 0 * loss_X + 1 * loss_E + 0 * loss_y
        # DiGress Default: 1 * loss_X + 5 * loss_E + 0 * loss_y
        return self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E + self.lambda_train[2] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log