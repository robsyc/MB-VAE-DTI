# -----------------------------------------------------------------------------
# Parts of this file incorporate code from the following open-source project(s):
#   Source: https://github.com/meyresearch/DL_protein_ligand_affinity/blob/main/dnn.py
#   License: License: MIT License
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.predictive.modules.finetune_lightning_module import FineTuneLightningModule


class GorantlaPredictionHead(nn.Module):
    def __init__(self, input_dim, finetuning_args):
        super().__init__()
        self.ligand_fc = nn.Linear(input_dim, finetuning_args["output_dim"])
        self.fc1 = nn.Linear(2 * finetuning_args["output_dim"], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, finetuning_args["n_output"])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(finetuning_args["dropout"])

    def forward(self, ligand_embedding, target_embedding):
        ligand_embedding = self.ligand_fc(ligand_embedding)
        xc = torch.cat((ligand_embedding, target_embedding), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out


class LigandTargetFineTuneLightningModule(FineTuneLightningModule):
    def __init__(
        self,
        base_model_class,
        model_params,
        task_type="classification",
        num_tasks=617,
        checkpoint_path=None,
        lr=0.00001,
        weight_decay=0.01,
        finetuning_args={},
    ):
        super().__init__(
            base_model_class,
            model_params,
            task_type,
            num_tasks,
            checkpoint_path,
            lr,
            weight_decay,
            finetuning_args,
        )

        self.batch_index = ns.FIELD_LABEL
        self.use_bmfm_pred_head = finetuning_args["use_bmfm_pred_head"]
        self.target_model = CNN_esm(
            finetuning_args["n_output"],
            finetuning_args["output_dim"],
            finetuning_args["dropout"],
            finetuning_args["kernel_sizes"],
            finetuning_args["input_dim"],
        )  # .float()

        self.ligand_target_pred_head = GorantlaPredictionHead(
            self.model.get_embed_dim(), finetuning_args
        )

    def get_ligand_target_batch(self, batch):
        target_batch = batch[ns.FIELD_DATA_PROTEIN_EMB]
        ligand_batch = batch  # self.get_ligand_batch(batch,ret_index=False)
        return (target_batch, ligand_batch)

    def forward(self, batch):
        batch = self.get_ligand_target_batch(batch)
        target_embedding = self.target_model.forward(batch[0]).to(torch.float)
        logits, coeffs = self.model.forward0(batch[1])
        ligand_embedding = logits
        out = self.ligand_target_pred_head(ligand_embedding, target_embedding)
        return out, coeffs


# Ref: https://github.com/meyresearch/DL_protein_ligand_affinity/blob/main/dnn.py
class CNN_esm(nn.Module):
    def __init__(
        self,
        n_output=1,
        output_dim=128,
        dropout=0.2,
        kernel_sizes=[4, 8, 12],
        input_dim=1280,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_output = n_output

        self.pro_conv1 = nn.Conv1d(
            input_dim, 32, kernel_sizes[0], padding=kernel_sizes[0] // 2
        )  # ,dtype=torch.float)
        self.pro_conv2 = nn.Conv1d(
            32, 64, kernel_sizes[1], padding=kernel_sizes[1] // 2
        )  # ,dtype=torch.float)
        self.pro_conv3 = nn.Conv1d(
            64, 128, kernel_sizes[2], padding=kernel_sizes[2] // 2
        )  # ,dtype=torch.float)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.pro_fc_g1 = nn.Linear(128, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        target_x = batch
        c = target_x.shape

        shape = int((c[0] * c[1] * c[2]) / self.input_dim)
        target_x = target_x.view(shape, self.input_dim, 1)

        xt = self.pro_conv1(target_x)
        xt = self.relu(xt)

        xt = self.pro_conv2(xt)
        xt = self.relu(xt)

        xt = self.pro_conv3(xt)
        xt = self.relu(xt)

        xt = self.pool(xt).squeeze(-1)

        xt = self.relu(self.pro_fc_g1(xt))

        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        return xt
