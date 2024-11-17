import torch
import torch.nn.functional as F
from torch import nn

from bmfm_sm.core.data_modules.namespace import TaskType
from bmfm_sm.core.modules.base_pretrained_model import (
    BasePredictionHead,
    MultiTaskPredictionHead,
)


class PredictionHeadForMLM(BasePredictionHead):
    def __init__(self, n_embd, n_vocab):
        super().__init__("mlm")
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor

    def compute_loss(
        self, batch_labels, batch_predictions, idxl, chunk
    ) -> torch.Tensor:
        logits = batch_predictions
        targets = batch_labels
        loss = 0
        loss_tmp = 0
        if targets is not None:
            # -- mle loss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
            loss_tmp = true_token_lprobs / len(idxl)
        if chunk < len(idxl) - 1:
            loss_tmp.backward()
            loss += loss_tmp.detach()
        else:
            loss += loss_tmp
        return loss


class PredictionHeadForClassification(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_classes: int,
        activation: str = "ReLU",
        dropout_prob=0,
    ):
        """
        Args:
        ----
            hidden_size (int): the size of each layer
            num_layers (int): number of (linear+(optional)dropout+activation) layers
            num_classes (int): number of output classes
            activation (str): type of activations for layers in between
            dropout_prob (float): probability for the Dropout layers (if set to 0, dropout will not be used at all).

        """
        super().__init__()

        self.layers = 0

        # Setting the linear+activation layers
        for _ in range(num_layers - 1):
            lin_layer = torch.nn.Linear(hidden_size, hidden_size)
            setattr(self, f"layer{self.layers}", lin_layer)
            self.layers += 1

            if dropout_prob > 0:
                setattr(self, f"layer{self.layers}", nn.Dropout(dropout_prob))
                self.layers += 1

            activation_layer = getattr(torch.nn, activation)()
            setattr(self, f"layer{self.layers}", activation_layer)
            self.layers += 1

        # Last layer to the output space
        layer = torch.nn.Linear(hidden_size, num_classes)
        setattr(self, f"layer{self.layers}", layer)
        self.layers += 1

    @property
    def last_linear_layer(self):
        return getattr(self, f"layer{self.layers - 1}")

    def forward(self, x):
        output = x[:]  # Making a copy of x
        for i in range(self.layers):
            output = getattr(self, f"layer{i}")(output)
        return output


# Takes an input and returns a multi-head classification output. Example use case: Want to predict cluster label for k_100, k_1000 and k_3000 clysters
class PredictionHeadForMultiClassification(nn.Module):
    def __init__(
        self, hidden_size, num_layers, output_dims, activation="ReLU", dropout_prob=0.2
    ):
        """
        Args:
        ----
            hidden_size (int): the size of each layer (will also be used as the input dimensionality)
            num_layers (int): number of (linear+activation) layers
            output_dims (list): list with the number of output classes for each of the prediction heads. E.g. for 3 heads, could pass in [100, 1000, 3000]
            activation (str): type of activations for layers in between
            dropout_prob (float): probability for the Dropout layer (if set to 0, dropout will not be used at all).

        """
        super().__init__()
        self.num_heads = len(output_dims)

        for i in range(self.num_heads):
            layer = PredictionHeadForClassification(
                num_layers,
                hidden_size,
                num_classes=output_dims[i],
                activation=activation,
                dropout_prob=dropout_prob,
            )
            setattr(self, f"head_{i}", layer)

    def forward(self, x):
        classification_outputs = [
            getattr(self, f"head_{i}")(x) for i in range(self.num_heads)
        ]
        return classification_outputs


class PredictionHeadForClusters(BasePredictionHead):
    def __init__(self, cluster_labels, input_dim=512):
        """:param cluster_labels is a list with strings for all the types of cluster labels to predict on E.g. ['k_100', 'k_1000', 'k_5000']"""
        super().__init__("clusters")

        self.cluster_dims = [int(cluster.split("k_")[1]) for cluster in cluster_labels]
        self.multi_class_classifier = PredictionHeadForMultiClassification(
            hidden_size=input_dim,
            num_layers=1,
            output_dims=self.cluster_dims,
            dropout_prob=0,
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.multi_class_classifier(x)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        cluster_loss = 0
        for i in range(len(batch_labels)):
            label = batch_labels[i]
            pred = batch_predictions[i]
            cluster_loss += self.loss(pred, label)
        return cluster_loss


class PredictionHeadForJigsawImage(BasePredictionHead):
    def __init__(self, input_dim=512, jigsaw_classes=101):
        super().__init__("jigsaw")
        self.jigsaw_classifier = MultiTaskPredictionHead(
            input_dim,
            num_tasks=1,
            task_type=TaskType.CLASSIFICATION,
            num_classes_per_task=jigsaw_classes,
            head=None,
        )

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.jigsaw_classifier(x).squeeze()

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(batch_predictions, batch_labels)


class PredictionHeadForMaskedContrastiveLearningImage(BasePredictionHead):
    def __init__(self):
        super().__init__("mcl")

    def forward(self, x):
        pass

    def compute_loss(self, hidden_feat_nonmask, hidden_feat_mask):
        return (hidden_feat_nonmask - hidden_feat_mask).pow(2).sum(axis=1).sqrt().mean()


class PredictionHeadForRationalityClassifImage(BasePredictionHead):
    def __init__(self, input_dim=512):
        super().__init__("rationality")
        self.fc = MultiTaskPredictionHead(
            input_dim,
            num_tasks=1,
            task_type=TaskType.CLASSIFICATION,
            num_classes_per_task=2,
            head=None,
        )  # linear

        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

        self.loss = torch.nn.NLLLoss()

    def forward(self, x):
        return self.logic(self.fc(x).squeeze())

    def compute_loss(self, jigsaw_ration_pred, nonjigsaw_ration_pred, jigsaw_label):
        # Jigsaw label > 0 --> Jumbled --> Sets that as 0s (irrational) and else as 1s (rational)
        jigsaw_ration_true = torch.where(jigsaw_label > 0, 0, 1).long().cpu()
        nonjigsaw_ration_true = torch.ones(nonjigsaw_ration_pred.shape[0]).long().cpu()
        jigsaw_ration_pred_clone = jigsaw_ration_pred.detach().cpu()
        nonjigsaw_ration_pred_clone = nonjigsaw_ration_pred.detach().cpu()
        matcher_loss = self.loss(
            jigsaw_ration_pred_clone, jigsaw_ration_true
        ) + self.loss(nonjigsaw_ration_pred_clone, nonjigsaw_ration_true)

        return matcher_loss


class MultiLayerPerceptronNeMo(torch.nn.Module):
    """
    Credit to https://github.com/NVIDIA/NeMo (Nvidia NeMo Repository).

    A simple MLP that can either be used independently or put on top
    of pretrained models (such as BERT) and act as a classifier.

    Args:
    ----
        hidden_size (int): the size of each layer
        num_classes (int): number of output classes
        num_layers (int): number of layers
        activation (str): type of activations for layers in between
        log_softmax (bool): whether to add a log_softmax layer before output

    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = "relu",
        log_softmax: bool = True,
    ):
        super().__init__()
        self.layers = 0
        for _ in range(num_layers - 1):
            layer = torch.nn.Linear(hidden_size, hidden_size)
            setattr(self, f"layer{self.layers}", layer)
            setattr(self, f"layer{self.layers + 1}", getattr(torch, activation))
            self.layers += 2
        layer = torch.nn.Linear(hidden_size, num_classes)
        setattr(self, f"layer{self.layers}", layer)
        self.layers += 1
        self.log_softmax = log_softmax

    @property
    def last_linear_layer(self):
        return getattr(self, f"layer{self.layers - 1}")

    def forward(self, x):
        output_states = x[:]  # Making a copy of x
        for i in range(self.layers):
            output_states = getattr(self, f"layer{i}")(output_states)

        if self.log_softmax:
            output_states = torch.log_softmax(output_states, dim=-1)
        return output_states


class PredictionHeadForNodeRankHomology(BasePredictionHead):
    def __init__(self, input_dim=512, num_heads=2, num_layers=1):
        super().__init__("NodeRankHomology")
        self.multi_class_regression = MultiTaskPredictionHead(
            input_dim, num_tasks=num_heads, task_type=TaskType.REGRESSION, head=None
        )  # linear
        self.loss = torch.nn.MSELoss(reduction="none")

    def forward(self, x):
        return self.multi_class_regression(x)

    def compute_loss(self, batch_predictions, batch_labels) -> torch.Tensor:
        return self.loss(batch_predictions, batch_labels)


class PredictionHeadForMaskedGraphProperties(BasePredictionHead):
    def __init__(self, num_classes_per_task, input_dim=512):
        super().__init__("GraphProperties")
        self.mlp = MultiTaskPredictionHead(
            input_dim,
            num_tasks=len(num_classes_per_task),
            task_type=TaskType.CLASSIFICATION,
            head="mlp",
            num_classes_per_task=num_classes_per_task,
        )
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x):
        return self.mlp(x)

    def compute_loss(self, batch_predictions, batch_labels):
        return self.loss(batch_predictions, batch_labels)


class PredictionHeadForMaskedEdges(BasePredictionHead):
    def __init__(self, num_classes, input_dim=512):
        super().__init__("EdgeMasking")
        self.mlp = MultiTaskPredictionHead(
            input_dim,
            num_tasks=1,
            task_type=TaskType.CLASSIFICATION,
            head="mlp",
            num_classes_per_task=num_classes,
        )
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x):
        return self.mlp(x)

    def compute_loss(self, batch_predictions, batch_labels):
        return self.loss(batch_predictions, batch_labels)


### UTILITY FUNCTIONS


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
