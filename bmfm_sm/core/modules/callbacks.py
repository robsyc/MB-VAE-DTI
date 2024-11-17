import copy
import logging
import os

import matplotlib.pyplot as plt
import mplcursors
import pytorch_lightning as pl


class LogModelParameterChanges(pl.Callback):
    def __init__(self):
        logging.info("Initialized LogModelNormChanges callback")
        self.epoch = 0
        self.l2_norms = []
        self.l2_norms_layer = {}

    def on_train_start(self, trainer, pl_module):
        logging.info("Training is starting")
        self.initial_state_dict = copy.deepcopy(pl_module.model.state_dict())

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch += 1
        logging.info(f"Current epoch {self.epoch}")

    def on_train_epoch_end(self, trainer, pl_module):
        self.update_norms(pl_module)

    def on_train_end(self, trainer, pl_module):
        logging.info(f"Training is ending {self.l2_norms_layer}")
        plt.figure(figsize=(20, 12))
        lines = []
        for layer in self.l2_norms_layer.keys():
            (line,) = plt.plot(
                range(self.epoch), self.l2_norms_layer[layer], label=f"{layer}"
            )
            lines.append(line)

        plt.xlabel("Epoch")
        plt.ylabel("L2 Norm of Parameter Differences")

        mplcursors.cursor(hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Epoch: {sel.target[0]}, L2 Norm: {sel.target[1]:.4f}"
            ),
        )

        plt.legend()
        log_version = os.path.basename(trainer.log_dir)
        png_path = os.path.join(
            trainer.default_root_dir, f"parameter_norms_{log_version}.png"
        )
        plt.savefig(png_path)

    def update_norms(self, pl_module):
        trained_state_dict = pl_module.model.state_dict()
        total_difference = 0.0
        for key in self.initial_state_dict:
            diff = (
                (self.initial_state_dict[key] - trained_state_dict[key])
                .pow(2)
                .sum()
                .item()
            )
            if key in self.l2_norms_layer:
                self.l2_norms_layer[key].append(diff**0.5)
            else:
                self.l2_norms_layer[key] = [diff**0.5]
            total_difference += diff

        l2_norm = total_difference**0.5
        self.l2_norms.append(l2_norm)
