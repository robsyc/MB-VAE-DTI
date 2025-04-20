# -----------------------------------------------------------------------------
# Parts of this file incorporate code from the following open-source projects:
#
# Description: Code for GIN and GCN are adapted from:
# Source: https://github.com/snap-stanford/pretrain-gnns/blob/08f126ac13623e551a396dd5e511d766f9d4f8ff/chem/model.py
# License: MIT License
#
# Description: Code for TrimNet is adapted from:
# Source: https://github.com/yvquanli/TrimNet/blob/master/trimnet_drug/source/model.py
# License: MIT License
#
# Description: Code for TokenGT is adapted from:
# Source:https://github.com/jw9730/tokengt/tree/main/large-scale-regression/tokengt
# License: MIT License
#
# Description: Code for AttentiveFP is adapted from:
# https://pytorch-geometric.readthedocs.io/en/2.3.0/generated/torch_geometric.nn.models.AttentiveFP.html#torch_geometric.nn.models.AttentiveFP
# License: MIT License
# -----------------------------------------------------------------------------

import math
from collections.abc import Callable

import torch as T
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, Parameter
from torch.nn.functional import leaky_relu
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import (
    AttentiveFP,
    GlobalAttention,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

from bmfm_sm.core.data_modules.fuse_ops.ops_graphs import OpSmilesToGraph
from bmfm_sm.core.data_modules.namespace import Modality
from bmfm_sm.core.modules.base_pretrained_model import BaseModel


class AttentiveFPModel(BaseModel, AttentiveFP):
    def __init__(self, **kwargs):
        super().__init__(modality=Modality.GRAPH, **kwargs)

    # Create forward method that takes batch as argument
    def forward0(self, batch):
        # Extract needed parameters and call the original forward method
        yh = self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
        return yh

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.forward0(x)

    def get_embed_dim(self):
        return self.hidden_channels


class MultiHeadTripletAttention(MessagePassing):
    def __init__(
        self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)  # aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(
            torch.Tensor(heads * node_channels, node_channels)
        )
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr, size=size
        )

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s
        # time consuming 12.14s
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        # return x_j * alpha
        # return self.prelu(alpha * e_ij * x_j)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return "{node_channels}, {node_channels}, heads={heads}".format(**self.__dict__)


class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super().__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(
            dim, edge_dim, heads
        )  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            m = F.celu(self.conv.forward(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.ln(x.squeeze(0))
        return x


class TrimNetModel(BaseModel):
    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden_channels=32,
        num_layers=3,
        heads=4,
        dropout=0.1,
        out_channels=2,
    ):
        super().__init__(Modality.GRAPH)
        self.depth = num_layers
        self.dropout = dropout
        self.lin0 = Linear(in_channels, hidden_channels)
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList(
            [Block(hidden_channels, edge_dim, heads) for i in range(num_layers)]
        )
        self.set2set = Set2Set(hidden_channels, processing_steps=3)
        # self.lin1 = torch.nn.Linear(2 * hidden_dim, 2)
        # self.out = torch.nn.Sequential(
        #    torch.nn.Linear(2 * hidden_channels, 512),
        #    torch.nn.LayerNorm(512),
        #    torch.nn.ReLU(inplace=True),
        #    torch.nn.Dropout(p=self.dropout),
        #    torch.nn.Linear(512, out_channels)
        # )
        self.lin2 = torch.nn.Linear(2 * hidden_channels, hidden_channels)

    def forward(self, data):
        x = F.celu(self.lin0(data.x))
        for conv in self.convs:
            x = x + F.dropout(
                conv(x, data.edge_index, data.edge_attr),
                p=self.dropout,
                training=self.training,
            )
        x = self.set2set(x, data.batch)
        # (B, 2H) -> (B, H)
        x = self.lin2(F.dropout(x, p=self.dropout, training=self.training))
        # x = self.out(F.dropout(x, p=self.dropout, training=self.training))
        return x

    def forward0(self, batch):
        return self.forward(batch)

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.forward0(x)

    def get_embed_dim(self):
        return self.hidden_channels


# num_atom_type = 120 #including the extra mask tokens
# num_chirality_tag = 3

# num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
# num_bond_direction = 3
# Derive possible dimensions from definitions (instead of hard coded)

# including the extra mask tokens
num_atom_type = len(OpSmilesToGraph.allowable_features["possible_atomic_num_list"]) + 2
num_chirality_tag = len(OpSmilesToGraph.allowable_features["possible_chirality_list"])
# including aromatic and self-loop edge, and extra masked tokens
num_bond_type = len(OpSmilesToGraph.allowable_features["possible_bonds"]) + 2
num_bond_direction = len(OpSmilesToGraph.allowable_features["possible_bond_dirs"])

# Note that this code makes the following assumptions (we adhere to these in the OpSmilesToGraph function):
# - the first two nodes (x) features are: atom_type and chirality_tag
# - the first two edge (edge_attr) features are: bond_type and bond_direction


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
    ----
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826

    """

    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # returns a edge_index, edge_attr tuple
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # second shape dimension must match edge_attr.size(1)
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        # Looks like propagate arguments have changed, name them to ensure no confusion
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # returns a edge_index, edge_attr tuple
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # second shape dimension must match edge_attr.size(1)
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        # Looks like propagate arguments have changed, name them to ensure no confusion
        return self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_embeddings, norm=norm
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # returns a edge_index, edge_attr tuple
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # second shape dimension must match edge_attr.size(1)
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        # Looks like propagate arguments have changed, name them to ensure no confusion
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # returns a edge_index, edge_attr tuple
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # second shape dimension must match edge_attr.size(1)
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.linear(x)

        # Looks like propagate arguments have changed, name them to ensure no confusion
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """
    Args:
    ----
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat.

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(BaseModel):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
    ----
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        num_tasks,
        JK="last",
        drop_ratio=0.1,
        graph_pooling="mean",
        gnn_type="gin",
    ):
        super().__init__(Modality.GRAPH)
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1)
                )
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * self.emb_dim, self.num_tasks
            )

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        # return self.graph_pred_linear(self.pool(node_representation, batch))
        return self.pool(node_representation, batch)

    def get_embeddings(self, batch):
        # Extract needed parameters and call the original forward method
        batch.x = batch.x.to(torch.int64)
        batch.edge_attr = batch.edge_attr.to(torch.int64)
        with torch.no_grad():
            return self.forward(batch)

    def get_embed_dim(self):
        return self.emb_dim


###############################################################################################################


class GCNModel(GNN_graphpred):
    def __init__(
        self,
        num_layer,
        emb_dim,
        num_tasks,
        JK="last",
        drop_ratio=0.1,
        graph_pooling="mean",
        gnn_type="gcn",
    ):
        super().__init__(
            num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling, gnn_type
        )

    # Create forward method that takes batch as argument
    def forward0(self, batch):
        # Extract needed parameters and call the original forward method
        batch.x = batch.x.to(torch.int64)
        batch.edge_attr = batch.edge_attr.to(torch.int64)
        yh = self.forward(batch)
        return yh


###############################################################################################################


class GINModel(GNN_graphpred):
    def __init__(
        self,
        num_layer,
        emb_dim,
        num_tasks,
        JK="last",
        drop_ratio=0.1,
        graph_pooling="mean",
        gnn_type="gin",
    ):
        super().__init__(
            num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling, gnn_type
        )

    # Create forward method that takes batch as argument
    def forward0(self, batch):
        # Extract needed parameters and call the original forward method
        batch.x = batch.x.to(torch.int64)
        batch.edge_attr = batch.edge_attr.to(torch.int64)
        yh = self.forward(batch)
        return yh


#################################################################################
###### Graph2dBaseModel (Super class) and Graph2dModel (Child class) ############
#################################################################################


class Graph2dBaseModel(BaseModel):
    """The current version does not support fine tuning nor loading a pretrained model."""

    # self.encoder_embed_dim
    # encoder = Graph2dEncoder(args)

    def __init__(
        self,
        encoder_embed_dim=768,
        encoder_layers=12,
        encoder_attention_heads=32,
        encoder_ffn_embed_dim=768,
        dropout=0.0,
        attention_dropout=0.1,
        act_dropout=0.1,
        activation_fn="gelu",
        encoder_normalize_before=True,
        apply_graphormer_init=True,
        share_encoder_input_output_embed=False,
        prenorm=True,
        postnorm=False,
        rand_node_id=False,
        rand_node_id_dim=64,
        orf_node_id=False,
        orf_node_id_dim=64,
        lap_node_id=True,
        lap_node_id_k=16,
        lap_node_id_sign_flip=True,
        lap_node_id_eig_dropout=0.2,
        type_id=True,
        stochastic_depth=True,
        performer=False,
        performer_finetune=False,
        performer_nb_features=None,
        performer_feature_redraw_interval=1000,
        performer_generalized_attention=False,
        return_attention=False,
        load_softmax=False,
        num_tasks=-1,  # SD changed
        num_atoms=512 * 9,
        # num_in_degree = 512,
        # num_out_degree= 512,
        num_edges=512 * 3,
        # num_spatial = 512,
        # num_edge_dis = 128,
        # edge_type = "multi_hop",
        # multi_hop_max_dist = 5,
        **kwargs,
    ):
        super().__init__(Modality.GRAPH)

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.act_dropout = act_dropout

        self.activation_fn = activation_fn
        self.encoder_normalize_before = encoder_normalize_before
        self.apply_graphormer_init = apply_graphormer_init
        self.share_encoder_input_output_embed = share_encoder_input_output_embed
        self.prenorm = prenorm
        self.postnorm = postnorm

        self.rand_node_id = rand_node_id
        self.rand_node_id_dim = rand_node_id_dim
        self.orf_node_id = orf_node_id
        self.orf_node_id_dim = orf_node_id_dim
        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip
        self.lap_node_id_eig_dropout = lap_node_id_eig_dropout
        self.type_id = type_id

        self.stochastic_depth = stochastic_depth

        self.performer = performer
        self.performer_finetune = performer_finetune
        self.performer_nb_features = performer_nb_features
        self.performer_feature_redraw_interval = performer_feature_redraw_interval
        self.performer_generalized_attention = performer_generalized_attention

        self.return_attention = return_attention
        self.load_softmax = (
            load_softmax  # remove_head is set to true during fine-tuning
        )

        self.num_atoms = num_atoms
        # self.num_in_degree = num_in_degree
        # self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        # self.num_spatial = num_spatial
        # self.num_edge_dis = num_edge_dis
        self.num_classes = num_tasks  # SD changed...
        # self.edge_type = edge_type
        # self.multi_hop_max_dist = multi_hop_max_dist
        self.task_type = kwargs.get("task_type", "regression")
        # self._setup()
        self._arcitecture()

    def _arcitecture(self):
        assert not (self.prenorm and self.postnorm)
        assert self.prenorm or self.postnorm

        if self.prenorm:
            layernorm_style = "prenorm"
        elif self.postnorm:
            layernorm_style = "postnorm"
        else:
            raise NotImplementedError

        self.graph_encoder = Graph2dEncoder(
            num_atoms=self.num_atoms,
            # num_in_degree=self.num_in_degree,
            # num_out_degree=self.num_out_degree,
            num_edges=self.num_edges,
            # num_spatial=self.num_spatial,
            # num_edge_dis=self.num_edge_dis,
            # edge_type=self.edge_type,
            # multi_hop_max_dist=self.multi_hop_max_dist,
            # for tokenization
            rand_node_id=self.rand_node_id,
            rand_node_id_dim=self.rand_node_id_dim,
            orf_node_id=self.orf_node_id,
            orf_node_id_dim=self.orf_node_id_dim,
            lap_node_id=self.lap_node_id,
            lap_node_id_k=self.lap_node_id_k,
            lap_node_id_sign_flip=self.lap_node_id_sign_flip,
            lap_node_id_eig_dropout=self.lap_node_id_eig_dropout,
            type_id=self.type_id,
            #
            stochastic_depth=self.stochastic_depth,
            performer=self.performer,
            performer_finetune=self.performer_finetune,
            performer_nb_features=self.performer_nb_features,
            performer_feature_redraw_interval=self.performer_feature_redraw_interval,
            performer_generalized_attention=self.performer_generalized_attention,
            num_encoder_layers=self.encoder_layers,
            embedding_dim=self.encoder_embed_dim,
            ffn_embedding_dim=self.encoder_ffn_embed_dim,
            num_attention_heads=self.encoder_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.act_dropout,
            encoder_normalize_before=self.encoder_normalize_before,
            layernorm_style=layernorm_style,
            apply_graphormer_init=self.apply_graphormer_init,
            activation_fn=self.activation_fn,
            return_attention=self.return_attention,
        )

        self.share_input_output_embed = self.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        self.masked_lm_pooler = nn.Linear(
            self.encoder_embed_dim, self.encoder_embed_dim
        )
        self.lm_head_transform_weight = nn.Linear(
            self.encoder_embed_dim, self.encoder_embed_dim
        )
        self.activation_fn = get_activation_fn(self.activation_fn)
        self.layer_norm = nn.LayerNorm(self.encoder_embed_dim)

        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(T.zeros(1))
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    self.encoder_embed_dim, self.num_classes, bias=False
                )
            else:
                raise NotImplementedError

    def forward(self, data, **kwargs):
        # Ref: https://github.com/jw9730/tokengt/blob/d2aba6d0998ab276e2f6cac9b09c4d6feccb7d0f/large-scale-regression/tokengt/models/tokengt.py#L196
        # layernorm_style = "prenorm" / "postnorm"
        # self.encoder(data, **kwargs)
        inner_states, graph_rep, attn_dict, padded_mask_dict = self.graph_encoder(
            data, perturb=kwargs.get("perturb", None)
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        # project masked tokens only
        masked_tokens = kwargs.get("masked_tokens", None)
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        # This is inserted for classification task by SD...
        if self.task_type == "classification":
            x = T.sigmoid(x)

        # Determine what results to return
        results = self.extract_specified_embeddings(x, padded_mask_dict, kwargs)

        if self.return_attention:
            return results, attn_dict
        elif kwargs.get("return_mask", None):
            return results, padded_mask_dict
        else:
            return results

    def get_embeddings(self, x):
        with T.no_grad():
            return self.forward(data=x)

    def get_embed_dim(self):
        return self.encoder_embed_dim

    @staticmethod
    # Determine what token embeddings to return. The options are:
    # return_all: return all token embeddings
    # return_graph: return just the graph token embedding (molecule) (default)
    # return_null: return just the null token embedding
    # return_node: return just the node token embeddings (atoms)
    # return_edge: return just the edge token embeddings (bonds)
    #
    # padded_mask_dict = {'padding_mask': padding_mask, 'padded_node_mask': padded_node_mask, 'padded_edge_mask': padded_edge_mask}
    def extract_specified_embeddings(x, padded_mask_dict, kwargs):
        results = None
        if kwargs.get("return_all", None):
            results = x[:, :, :]
        elif kwargs.get("return_graph", None):
            results = x[:, 0, :]
        elif kwargs.get("return_null", None):
            results = x[:, 1, :]
        elif kwargs.get("return_nodes", None):
            x = x[:, 2:, :]
            mask0 = padded_mask_dict["padded_node_mask"]
            mask1 = mask0.unsqueeze(-1).expand(x.size())
            results = x * mask1
        elif kwargs.get("return_edges", None):
            x = x[:, 2:, :]
            mask0 = padded_mask_dict["padded_edge_mask"]
            mask1 = mask0.unsqueeze(-1).expand(x.size())
            results = x * mask1
        else:
            results = x[:, 0, :]
        return results


###########################################
########## Graph2dEncoder ############
###########################################


class Graph2dEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        # num_in_degree: int,
        # num_out_degree: int,
        num_edges: int,
        # num_spatial: int,
        # num_edge_dis: int,
        # edge_type: str,
        # multi_hop_max_dist: int,
        rand_node_id: bool = False,
        rand_node_id_dim: int = 64,
        orf_node_id: bool = False,
        orf_node_id_dim: int = 64,
        lap_node_id: bool = False,
        lap_node_id_k: int = 8,
        lap_node_id_sign_flip: bool = False,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = False,
        stochastic_depth: bool = False,
        performer: bool = False,
        performer_finetune: bool = False,
        performer_nb_features: int = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        performer_auto_check_redraw: bool = True,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "postnorm",
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        # self.dropout_module = FairseqDropout(
        #     dropout, module_name=self.__class__.__name__
        # )
        self.dropout_module = nn.Dropout(
            dropout,  # module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        self.performer = performer
        self.performer_finetune = performer_finetune

        self.graph_feature = GraphFeatureTokenizer(
            num_atoms=num_atoms,
            num_edges=num_edges,
            rand_node_id=rand_node_id,
            rand_node_id_dim=rand_node_id_dim,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )
        self.performer_finetune = performer_finetune
        self.embed_scale = embed_scale

        if q_noise > 0:
            raise ValueError("q_noise > 0 not supported")
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)  # , export=export)
        else:
            self.emb_layer_norm = None

        if layernorm_style == "prenorm":
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)  # , export=export)

        if self.layerdrop > 0.0:
            raise ValueError("layerdrop > 0 not supported")
        else:
            self.layers = nn.ModuleList([])

        if stochastic_depth:
            assert layernorm_style == "prenorm"  # only for residual nets

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                performer_nb_features,
                performer_generalized_attention,
                performer_auto_check_redraw,
                performer_feature_redraw_interval,
            )
            self.performer = False
            performer = False
            performer_nb_features = None
            performer_generalized_attention = False
            performer_auto_check_redraw = False
            performer_feature_redraw_interval = None

        self.layers.extend(
            [
                self.build_graph2d_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    encoder_layers=num_encoder_layers,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=(
                        (0.1 * (layer_idx + 1) / num_encoder_layers)
                        if stochastic_depth
                        else 0
                    ),
                    performer=performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_graph2d_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        encoder_layers,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        drop_path,
        performer,
        performer_nb_features,
        performer_generalized_attention,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        layernorm_style,
        return_attention,
    ):
        return Graph2dEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            layernorm_style=layernorm_style,
            return_attention=return_attention,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only=False,  # : bool
        token_embeddings=None,  # : Optional[T.Tensor]
        attn_mask=None,  # : Optional[T.Tensor]
    ):
        if token_embeddings is not None:
            raise NotImplementedError
        else:
            (
                x,
                padding_mask,
                padded_index,
                padded_node_mask,
                padded_edge_mask,
            ) = self.graph_feature(batched_data, perturb)

        # x: B x T x C

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        if attn_mask is not None:
            raise NotImplementedError

        attn_dict = {"maps": {}, "padded_index": padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=None,
            )
            if not last_state_only:
                inner_states.append(x)
            attn_dict["maps"][i] = attn

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        padded_mask_dict = {
            "padding_mask": padding_mask,
            "padded_node_mask": padded_node_mask,
            "padded_edge_mask": padded_edge_mask,
        }

        if self.traceable:
            return T.stack(inner_states), graph_rep, attn_dict, padded_mask_dict
        else:
            return inner_states, graph_rep, attn_dict, padded_mask_dict


###########################################
######## Graph2dEncoderLayer #########
###########################################


class Graph2dEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        encoder_layers: int = 12,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        drop_path: float = 0.0,
        performer: bool = False,
        performer_nb_features: int = None,
        performer_generalized_attention: bool = False,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        layernorm_style: str = "postnorm",
        return_attention: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.layernorm_style = layernorm_style
        self.return_attention = return_attention

        # self.dropout_module = FairseqDropout(
        #     dropout, module_name=self.__class__.__name__
        # )
        self.dropout_module = nn.Dropout(
            dropout,  # module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            attention_dropout=attention_dropout,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)  # , export=export)

        # drop path for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.feedforward = self.build_FFN(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise,
            qn_block_size,
            activation_fn,
            activation_dropout,
            dropout,
            module_name=self.__class__.__name__,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)  # , export=export)

        # drop path for stochastic depth
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def build_FFN(
        self,
        embedding_dim,
        ffn_embedding_dim,
        q_noise,
        qn_block_size,
        activation_fn,
        activation_dropout,
        dropout,
        module_name,
    ):
        return FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout,
            module_name=module_name,
        )

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        performer,
        performer_nb_features,
        performer_generalized_attention,
        attention_dropout,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        if performer:
            raise ValueError("Performer not supported")
        else:
            return MultiheadAttention(
                embed_dim,
                num_attention_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )

    def forward(
        self,
        x,  # : T.Tensor
        self_attn_bias=None,  # : Optional[T.Tensor]
        self_attn_mask=None,  # : Optional[T.Tensor]
        self_attn_padding_mask=None,  # : Optional[T.Tensor]
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        if self.layernorm_style == "prenorm":
            residual = x
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_bias=self_attn_bias,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = self.drop_path1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = self.drop_path2(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_bias=self_attn_bias,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        else:
            raise NotImplementedError
        return x, attn


###########################################
########## GraphFeatureTokenizer ##########
###########################################


class GraphFeatureTokenizer(nn.Module):
    """Compute node and edge features for each node and edge in the graph."""

    def __init__(
        self,
        num_atoms,
        num_edges,
        rand_node_id,
        rand_node_id_dim,
        orf_node_id,
        orf_node_id_dim,
        lap_node_id,
        lap_node_id_k,
        lap_node_id_sign_flip,
        lap_node_id_eig_dropout,
        type_id,
        hidden_dim,
        n_layers,
    ):
        super().__init__()

        self.encoder_embed_dim = hidden_dim

        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edges, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.null_token = nn.Embedding(1, hidden_dim)  # this is optional

        self.rand_node_id = rand_node_id
        self.rand_node_id_dim = rand_node_id_dim
        self.orf_node_id = orf_node_id
        self.orf_node_id_dim = orf_node_id_dim
        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip

        self.type_id = type_id

        if self.rand_node_id:
            self.rand_encoder = nn.Linear(2 * rand_node_id_dim, hidden_dim, bias=False)

        if self.lap_node_id:
            self.lap_encoder = nn.Linear(2 * lap_node_id_k, hidden_dim, bias=False)
            self.lap_eig_dropout = (
                nn.Dropout2d(p=lap_node_id_eig_dropout)
                if lap_node_id_eig_dropout > 0
                else None
            )

        if self.orf_node_id:
            self.orf_encoder = nn.Linear(2 * orf_node_id_dim, hidden_dim, bias=False)

        if self.type_id:
            self.order_encoder = nn.Embedding(2, hidden_dim)

        self.apply(lambda module: self.init_params(module, n_layers=n_layers))

    @staticmethod
    def init_params(module, n_layers):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    @staticmethod
    def get_batch(
        node_feature, edge_index, edge_feature, node_num, edge_num, perturb=None
    ):
        """
        :param node_feature: Tensor([sum(node_num), D])
        :param edge_index: LongTensor([2, sum(edge_num)])
        :param edge_feature: Tensor([sum(edge_num), D])
        :param node_num: list
        :param edge_num: list
        :param perturb: Tensor([B, max(node_num), D])
        :return: padded_index: LongTensor([B, T, 2]), padded_feature: Tensor([B, T, D]), padding_mask: BoolTensor([B, T])
        """
        seq_len = [n + e for n, e in zip(node_num, edge_num)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len)
        max_n = max(node_num)
        device = edge_index.device

        token_pos = T.arange(max_len, device=device)[None, :].expand(
            b, max_len
        )  # [B, T]

        # seq_len = T.tensor(seq_len, device=device, dtype=T.long)[:, None]  # [B, 1]
        # node_num = T.tensor(node_num, device=device, dtype=T.long)[:, None]  # [B, 1]
        # edge_num = T.tensor(edge_num, device=device, dtype=T.long)[:, None]  # [B, 1]

        seq_len = T.LongTensor(seq_len)[:, None].to(device)  # [B, 1]
        node_num = node_num.clone()[:, None].to(device)  # [B, 1]
        edge_num = edge_num.clone()[:, None].to(device)  # [B, 1]

        node_index = T.arange(max_n, device=device, dtype=T.long)[None, :].expand(
            b, max_n
        )  # [B, max_n]
        node_index = node_index[None, node_index < node_num].repeat(
            2, 1
        )  # [2, sum(node_num)]

        padded_node_mask = T.less(token_pos, node_num)
        padded_edge_mask = T.logical_and(
            T.greater_equal(token_pos, node_num), T.less(token_pos, node_num + edge_num)
        )

        padded_index = T.zeros(b, max_len, 2, device=device, dtype=T.long)  # [B, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        padded_index[padded_edge_mask, :] = edge_index.t()

        if perturb is not None:
            perturb_mask = padded_node_mask[:, :max_n]  # [B, max_n]
            node_feature = node_feature + perturb[perturb_mask].type(
                node_feature.dtype
            )  # [sum(node_num), D]

        padded_feature = T.zeros(
            b, max_len, d, device=device, dtype=node_feature.dtype
        )  # [B, T, D]
        padded_feature[padded_node_mask, :] = node_feature
        padded_feature[padded_edge_mask, :] = edge_feature

        padding_mask = T.greater_equal(token_pos, seq_len)  # [B, T]
        return (
            padded_index,
            padded_feature,
            padding_mask,
            padded_node_mask,
            padded_edge_mask,
        )

    @staticmethod
    @T.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = T.arange(max_n, device=device, dtype=T.long)[None, :].expand(
            b, max_n
        )  # [B, max_n]
        # node_num = T.tensor(node_num, device=device, dtype=T.long)[:, None]  # [B, 1]
        node_num = node_num.clone()[:, None].to(device)  # [B, 1]
        node_mask = T.less(node_index, node_num)  # [B, max_n]
        return node_mask

    @staticmethod
    @T.no_grad()
    def get_random_sign_flip(eigvec, node_mask):
        b, max_n = node_mask.size()
        d = eigvec.size(1)

        sign_flip = T.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
        sign_flip = sign_flip[node_mask]
        return sign_flip

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec

    @staticmethod
    @T.no_grad()
    def get_orf_batched(node_mask, dim, device, dtype):
        b, max_n = node_mask.size(0), node_mask.size(1)
        orf = gaussian_orthogonal_random_matrix_batched(
            b, dim, dim, device=device, dtype=dtype
        )  # [B, D, D]
        orf = orf[:, None, ...].expand(b, max_n, dim, dim)  # [B, max(n_node), D, D]
        orf = orf[node_mask]  # [sum(n_node), D, D]
        return orf

    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, 2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = T.zeros(
            b, max_n, d, device=node_id.device, dtype=node_id.dtype
        )  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
        index_embed = index_embed.view(b, max_len, 2 * d)
        return index_embed

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = T.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def add_special_tokens(self, padded_feature, padding_mask):
        """
        :param padded_feature: Tensor([B, T, D])
        :param padding_mask: BoolTensor([B, T])
        :return: padded_feature: Tensor([B, 2/3 + T, D]), padding_mask: BoolTensor([B, 2/3 + T])
        """
        b, _, d = padded_feature.size()

        num_special_tokens = 2
        graph_token_feature = self.graph_token.weight.expand(b, 1, d)  # [1, D]
        null_token_feature = self.null_token.weight.expand(
            b, 1, d
        )  # [1, D], this is optional
        special_token_feature = T.cat(
            (graph_token_feature, null_token_feature), dim=1
        )  # [B, 2, D]
        special_token_mask = T.zeros(
            b, num_special_tokens, dtype=T.bool, device=padded_feature.device
        )

        padded_feature = T.cat(
            (special_token_feature, padded_feature), dim=1
        )  # [B, 2 + T, D]
        padding_mask = T.cat((special_token_mask, padding_mask), dim=1)  # [B, 2 + T]
        return padded_feature, padding_mask

    def forward(self, batched_data, perturb=None):
        node_data = batched_data["node_data"]
        lap_eigvec = batched_data["lap_eigvec"]
        edge_index = batched_data["edge_index"]
        edge_data = batched_data["edge_data"]
        node_num = batched_data["node_num"]  # # Populated in collate
        edge_num = batched_data["edge_num"]  # Populated in collate

        node_feature = self.atom_encoder(node_data).sum(-2)  # [sum(n_node), D]
        edge_feature = self.edge_encoder(edge_data).sum(-2)  # [sum(n_edge), D]
        device = node_feature.device
        dtype = node_feature.dtype

        (
            padded_index,
            padded_feature,
            padding_mask,
            padded_node_mask,
            padded_edge_mask,
        ) = self.get_batch(
            node_feature, edge_index, edge_feature, node_num, edge_num, perturb
        )
        node_mask = self.get_node_mask(
            node_num, node_feature.device
        )  # [B, max(n_node)]

        if self.rand_node_id:
            rand_node_id = T.rand(
                sum(node_num), self.rand_node_id_dim, device=device, dtype=dtype
            )  # [sum(n_node), D]
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(
                rand_node_id, node_mask, padded_index
            )  # [B, T, 2D]
            padded_feature = padded_feature + self.rand_encoder(rand_index_embed)

        if self.orf_node_id:
            b, max_n = len(node_num), max(node_num)
            orf = gaussian_orthogonal_random_matrix_batched(
                b, max_n, max_n, device=device, dtype=dtype
            )  # [b, max(n_node), max(n_node)]
            orf_node_id = orf[node_mask]  # [sum(n_node), max(n_node)]
            if self.orf_node_id_dim > max_n:
                # [sum(n_node), Do]
                orf_node_id = F.pad(
                    orf_node_id, (0, self.orf_node_id_dim - max_n), value=float("0")
                )
            else:
                # [sum(n_node), Do]
                orf_node_id = orf_node_id[..., : self.orf_node_id_dim]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            orf_index_embed = self.get_index_embed(
                orf_node_id, node_mask, padded_index
            )  # [B, T, 2Do]
            padded_feature = padded_feature + self.orf_encoder(orf_index_embed)

        if self.lap_node_id:
            lap_dim = lap_eigvec.size(-1)
            if self.lap_node_id_k > lap_dim:
                # [sum(n_node), Dl]
                eigvec = F.pad(
                    lap_eigvec, (0, self.lap_node_id_k - lap_dim), value=float("0")
                )
            else:
                # [sum(n_node), Dl]
                eigvec = lap_eigvec[:, : self.lap_node_id_k]
            if self.lap_eig_dropout is not None:
                eigvec = self.lap_eig_dropout(eigvec[..., None, None]).view(
                    eigvec.size()
                )
            lap_node_id = self.handle_eigvec(
                eigvec, node_mask, self.lap_node_id_sign_flip
            )
            lap_index_embed = self.get_index_embed(
                lap_node_id, node_mask, padded_index
            )  # [B, T, 2Dl]
            padded_feature = padded_feature + self.lap_encoder(lap_index_embed)

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        padded_feature, padding_mask = self.add_special_tokens(
            padded_feature, padding_mask
        )  # [B, 2+T, D], [B, 2+T]

        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float("0"))
        # [B, 2+T, D], [B, 2+T], [B, T, 2]
        return (
            padded_feature,
            padding_mask,
            padded_index,
            padded_node_mask,
            padded_edge_mask,
        )


@T.no_grad()
def orthogonal_matrix_chunk_batched(bsz, cols, device=None):
    unstructured_block = T.randn((bsz, cols, cols), device=device)
    q, r = T.linalg.qr(unstructured_block, mode="reduced")
    return q.transpose(2, 1)  # [bsz, cols, cols]


@T.no_grad()
def gaussian_orthogonal_random_matrix_batched(
    nb_samples, nb_rows, nb_columns, device=None, dtype=T.float32
):
    """Create 2D Gaussian orthogonal matrix."""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = T.cat(block_list, dim=1).type(dtype)
    final_matrix = F.normalize(final_matrix, p=2, dim=2)
    return final_matrix


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    # x: T x B x C
    keep_prob = 1 - drop_prob
    random_tensor = x.new_empty(1, x.size(1), 1).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        q_noise,
        qn_block_size,
        activation_fn,
        activation_dropout,
        dropout,
        module_name,
    ):
        super().__init__()
        # self.fc1 = quant_noise(nn.Linear(embedding_dim, ffn_embedding_dim), q_noise, qn_block_size)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        # self.activation_dropout_module = FairseqDropout(
        #     activation_dropout, module_name=module_name
        # )
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        # self.fc2 = quant_noise(nn.Linear(ffn_embedding_dim, embedding_dim), q_noise, qn_block_size)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        # self.dropout_module = FairseqDropout(
        #     dropout, module_name=module_name
        # )
        self.dropout_module = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


def init_graphormer_params(module):
    """Initialize the weights specific to the Graphormer Model."""

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


############


class MultiheadAttention(nn.Module):
    """
    Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        attention_dropout=0.0,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.self_attention = self_attention
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        assert self.self_attention, "Only support self attention"
        assert (
            not self.self_attention or self.qkv_same_dim
        ), "Self-attention requires QKV to be of the same size"
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # self.attention_dropout_module = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)
        # self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.attention_dropout_module = nn.Dropout(attention_dropout)
        self.dropout_module = nn.Dropout(dropout)
        # self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        # self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        # self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        # self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()
        # self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: T.Tensor | None,
        value: T.Tensor | None,
        attn_bias: T.Tensor | None,
        key_padding_mask: T.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: T.Tensor | None = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> tuple[T.Tensor, T.Tensor | None]:
        """
        Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not T.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)  # [T, B, D]
        k = self.k_proj(query)  # [T, B, D]
        v = self.v_proj(query)  # [T, B, D]
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = T.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask[:, None, None, :].to(T.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        # attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)  # [bsz * num_heads, tgt_len, src_len]
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=T.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        attn = T.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        attn_weights: T.Tensor | None = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(
                1, 0
            )  # [num_heads, bsz, tgt_len, src_len]
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class EncLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        dim_v,
        dim_ff,
        n_heads,
        dropout=0.0,
        drop_mu=0.0,
        return_attn=False,
    ):
        super().__init__()
        self.return_attn = return_attn
        self.add = Add()
        self.ln = Apply(nn.LayerNorm(dim_in))
        self.attn = SelfAttn(
            n_heads=n_heads, d_in=dim_in, d_out=dim_in, d_qk=dim_qk, d_v=dim_v
        )  # <-
        self.ffn = Apply(
            nn.Sequential(
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout, inplace=True),
                nn.Linear(dim_ff, dim_in),
            )
        )

    def forward(self, G):
        h = self.ln(G)
        attn_score, h = self.attn(h)
        G = self.add(G, h)
        h = self.ffn(G)
        return (attn_score, self.add(G, h)) if self.return_attn else self.add(G, h)


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        dim_in,
        dim_out,
        dim_hidden,
        dim_qk,
        dim_v,
        dim_ff,
        n_heads,
        drop_input=0.0,
        dropout=0.0,
        drop_mu=0.0,
        last_layer_n_heads=16,
    ):
        super().__init__()
        assert last_layer_n_heads >= 16
        self.input = Apply(
            nn.Sequential(
                nn.Linear(dim_in, dim_hidden), nn.Dropout(drop_input, inplace=True)
            )
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                EncLayer(
                    dim_hidden,
                    dim_qk,
                    dim_v,
                    dim_ff,
                    n_heads,
                    dropout,
                    drop_mu,
                    return_attn=False,
                )
            )
        layers.append(
            EncLayer(
                dim_hidden,
                dim_qk,
                dim_v,
                dim_ff,
                last_layer_n_heads,
                dropout,
                drop_mu,
                return_attn=True,
            )
        )
        self.layers = nn.Sequential(*layers)

        self.output = Apply(
            nn.Sequential(nn.LayerNorm(dim_hidden), nn.Linear(dim_hidden, dim_out))
        )

    def forward(self, G):  # G.values: [bsize, max(n+e), 2*dim_hidden]
        G = self.input(G)  # G.values: [bsize, max(n+e), dim_hidden]
        # attn_score: [bsize, last_layer_n_heads, |E|, |E|]
        attn_score, G = self.layers(G)
        # G.values: [bsize, max(n+e), dim_hidden]
        # attn_score : [bsize, last_layer_n_heads, |E|, |E|]   # self.output(G).values: [bsize, max(n+e), dim_out]
        return attn_score, self.output(G)


class Apply(nn.Module):
    def __init__(self, f: Callable[[T.Tensor], T.Tensor], skip_masking=False):
        super().__init__()
        self.f = f
        self.skip_masking = skip_masking

    # def forward(self, G: Union[T.Tensor, B]) -> Union[T.Tensor, B]:
    def forward(self, G):
        return apply(G, self.f, self.skip_masking)


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    # def forward(G1: Union[T.Tensor, B], G2: Union[T.Tensor, B]) -> Union[T.Tensor, B]:
    def forward(G1, G2):
        return add_batch(G1, G2)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: T.Tensor):
        # Q, K: (bsize, nheads, |E|, d_qk)
        # V: (bsize, n_heads, |E|, d_v)
        # mask: = (bsize, 1, 1, |E|)

        dim_qk = Q.size(-1)
        # tensor(bsize, nheads, |E|, |E|)
        attn_score = T.matmul(Q, K.transpose(2, 3)) / math.sqrt(dim_qk)
        # tensor(bsize, nheads, |E|, |E|)
        attn_score = attn_score.masked_fill(~mask, -1e9)
        # tensor(bsize, nheads, |E|, |E|)
        attn_score = F.softmax(attn_score, dim=-1)
        output = T.matmul(attn_score, V)  # tensor(bsize, nheads, |E|, d_v)
        # attn_score: tensor(bsize, nheads, |E|, |E|), output: tensor(bsize, nheads, |E|, d_v)
        return attn_score, output


class SelfAttn(nn.Module):
    def __init__(self, n_heads=15, d_in=64, d_out=64, d_qk=512, d_v=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_out = d_out
        self.d_qk = d_qk
        self.d_v = d_v
        self.scaled_dot_attention = ScaledDotProductAttention()
        self.fc1 = nn.Linear(d_in, 2 * n_heads * d_qk)
        self.fc_v = nn.Linear(d_in, n_heads * d_v)
        self.fc_out = nn.Linear(n_heads * d_v, d_out)

    def forward(self, G):  # G.values: [bsize, |E|, d_in)
        bsize, e, _ = G.values.shape
        h = self.fc1(G.values)  # Tensor(bsize, |E|, 2*n_heads*d_qk)
        # (bsize, |E|, n_heads, d_qk)
        Q = h[..., : self.n_heads * self.d_qk].view(bsize, e, self.n_heads, self.d_qk)
        # (bsize, |E|, n_heads, d_qk)
        K = h[..., self.n_heads * self.d_qk :].view(bsize, e, self.n_heads, self.d_qk)

        V = self.fc_v(G.values)  # (bsize, |E|, n_heads*d_v)
        V = V.masked_fill(~G.mask.unsqueeze(-1), 0)
        # (bsize, |E|, n_heads, d_v)
        V = V.view(bsize, e, self.n_heads, self.d_v)

        Q = Q.transpose(1, 2)  # (bsize, n_heads, |E|, d_qk)
        K = K.transpose(1, 2)  # (bsize, n_heads, |E|, d_qk)
        V = V.transpose(1, 2)  # (bsize, n_heads, |E|, d_v)

        G_mask = G.mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, |E|)
        # prod_attn: tensor(bsize, n_heads, |E|, d_v); attn_score: tensor(bsize, nheads, |E|, |E|)
        attn_score, prod_attn = self.scaled_dot_attention(Q, K, V, mask=G_mask)

        # tensor(bsize, |E|, n_heads, d_v)
        prod_attn = prod_attn.transpose(1, 2).contiguous()
        # tensor(bsize, |E|, n_heads * d_v)
        prod_attn = prod_attn.view(bsize, e, -1)

        output = self.fc_out(prod_attn)  # tensor(bsize, |E|, d_out)
        return attn_score, batch_like(G, output, skip_masking=False)


class Batch:
    indices: None | T.LongTensor
    values: T.Tensor
    n_nodes: list
    n_edges: None | list
    device: T.device
    mask: T.BoolTensor
    node_mask: T.BoolTensor
    null_node: bool

    def __init__(
        self,
        indices: None | T.LongTensor,
        values: T.Tensor,
        n_nodes: list,
        n_edges: None | list,
        mask: T.BoolTensor = None,
        skip_masking: bool = False,
        node_mask: T.BoolTensor = None,
        null_node=False,
    ):
        """
        a mini-batch of sparse (hyper)graphs
        :param indices: LongTensor([B, |E|, k])
        :param values: Tensor([B, |E|, D])
        :param n_nodes: List([n1, ..., nb])
        :param n_edges: List([|E1|, ..., |Eb|])  Number of 1-edges + 2-edges
        :param mask: BoolTensor([B, |E|])
        :param skip_masking:
        :parem node_mask
        :param: null_node.
        """
        # caution: to reduce overhead, we assume a specific organization of indices: see comment in get_diag()
        # we also assume that indices are already well-masked (invalid entries are zero): see comment in self.apply_mask()
        self.indices = indices  # [B, |E|, k] or None
        self.values = values  # [B, |E|, D]
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.device = values.device
        self.order = 1 if indices is None else indices.size(-1)
        assert self.order in (1, 2)
        self.node_mask = (
            get_mask(T.tensor(n_nodes, dtype=T.long, device=self.device), max(n_nodes))
            if node_mask is None
            else node_mask
        )  # [B, N]
        if self.order == 1:
            self.mask = self.node_mask
        else:
            self.mask = (
                get_mask(
                    T.tensor(n_edges, dtype=T.long, device=self.device), max(n_edges)
                )
                if mask is None
                else mask
            )  # [B, |E|]
        if not skip_masking:
            # set invalid values to 0
            self.apply_mask(0)

        self.null_node = null_node

    def __repr__(self):
        return f"Batch(indices {list(self.indices.size())}, values {list(self.values.size())}"

    def to(self, device: str | T.device) -> "Batch":
        if self.indices is not None:
            self.indices = self.indices.to(device)
        self.values = self.values.to(device)
        self.mask = self.mask.to(device)
        self.node_mask = self.node_mask.to(device)
        self.mask2d = self.mask2d.to(device)
        self.device = self.values.device
        return self

    def apply_mask(self, value=0.0) -> None:
        # mask out invalid tensor elements
        self.values = masked_fill(self.values, self.mask, value)


def apply(
    G: T.Tensor | Batch, f: Callable[[T.Tensor], T.Tensor], skip_masking=False
) -> T.Tensor | Batch:
    if isinstance(G, T.Tensor):
        return f(G)
    return batch_like(G, f(G.values), skip_masking)


def add_batch(G1: T.Tensor | Batch, G2: T.Tensor | Batch) -> T.Tensor | Batch:
    # add features of two batched graphs with identical edge structures
    if isinstance(G1, Batch) and isinstance(G2, Batch):
        assert G1.order == G2.order
        assert G1.n_nodes == G2.n_nodes
        assert G1.n_edges == G2.n_edges
        return batch_like(G1, G1.values + G2.values, skip_masking=True)
    else:
        assert isinstance(G1, T.Tensor)
        assert isinstance(G2, T.Tensor)
        assert G1.size() == G2.size()
        return G1 + G2


def make_batch_concatenated(
    node_feature: T.Tensor,
    edge_index: T.LongTensor,
    edge_feature: T.Tensor,
    n_nodes: list,
    n_edges: list,
    null_params: dict,
) -> Batch:
    """
    :param node_feature: Tensor([sum(n), Dv])
    :param edge_index: LongTensor([2, sum(e)])
    :param edge_feature: Tensor([sum(e), De])
    :param n_nodes: list
    :param n_edges: list
    :parem null_params: dict
    """
    assert (
        len(node_feature.size())
        == len(edge_index.size())
        == len(edge_feature.size())
        == 2
    )
    use_null_node = null_params["use_null_node"]
    null_feat = null_params["null_feature"]  # [1, shared_dim]

    bsize = len(n_nodes)
    node_dim = node_feature.size(-1)
    edge_dim = edge_feature.size(-1)
    assert node_dim == edge_dim
    shared_dim = node_dim
    device = node_feature.device
    dtype = node_feature.dtype
    node_feature.size(0)  # sum(n)
    edge_feature.size(0)  # sum(e)
    # unpack nodes
    idx = T.arange(max(n_nodes), device=device)
    idx = idx[None, :].expand(bsize, max(n_nodes))  # [B, N]
    node_index = T.arange(max(n_nodes), device=device, dtype=T.long)
    node_index = node_index[None, :, None].expand(bsize, max(n_nodes), 2)  # [B, N, 2]
    node_num_vec = T.tensor(n_nodes, device=device)[:, None]  # [B, 1]
    unpacked_node_index = node_index[idx < node_num_vec]  # [N, 2]
    unpacked_node_feature = node_feature  # [sum(n), Dv]
    # unpack edges
    edge_num_vec = T.tensor(n_edges, device=device)[:, None]  # [B, 1]
    unpacked_edge_index = edge_index.t()  # [|E|, 2]
    unpacked_edge_feature = edge_feature  # [sum(e), De]

    if not use_null_node:
        # compose tensor
        n_edges_ = [n + e for n, e in zip(n_nodes, n_edges)]
        max_size = max(n_edges_)
        edge_index_ = T.zeros(
            bsize, max_size, 2, device=device, dtype=T.long
        )  # [B, N + |E|, 2]
        # [B, N + |E|, shared_dim]
        edge_feature_ = T.zeros(bsize, max_size, shared_dim, device=device, dtype=dtype)
        full_index = T.arange(max_size, device=device)[None, :].expand(
            bsize, max_size
        )  # [B, N + |E|]

        node_mask = full_index < node_num_vec  # [B, N + |E|]
        edge_mask = (node_num_vec <= full_index) & (
            full_index < node_num_vec + edge_num_vec
        )  # [B, N + |E|]
        edge_index_[node_mask] = unpacked_node_index
        edge_index_[edge_mask] = unpacked_edge_index
        edge_feature_[node_mask] = unpacked_node_feature
        edge_feature_[edge_mask] = unpacked_edge_feature
        # setup batch
        return Batch(edge_index_, edge_feature_, n_nodes, n_edges_, null_node=False)
    else:
        # compose tensor
        n_edges_ = [n + e + 1 for n, e in zip(n_nodes, n_edges)]
        total_edges_num_vec = T.tensor(n_edges_, device=device)[:, None]  # [B, 1]
        new_n_nodes = [n + 1 for n in n_nodes]

        unpacked_null_index = []
        for i in range(bsize):
            unpacked_null_index.append([n_nodes[i], n_nodes[i]])
        unpacked_null_index = T.tensor(unpacked_null_index, device=device)  # [B, 2]

        max_size = max(n_edges_)  # N + |E| + 1
        edge_index_ = T.zeros(
            bsize, max_size, 2, device=device, dtype=T.long
        )  # [B, N+|E|+1, 2]
        edge_feature_ = T.zeros(
            bsize, max_size, shared_dim, device=device, dtype=dtype
        )  # [B, N+|E|+1, D]
        full_index = T.arange(max_size, device=device)[None, :].expand(
            bsize, max_size
        )  # [B, N+|E|+1]

        # node_num_vec: [B, 1]
        node_mask = full_index < node_num_vec  # [B, N+|E|+1]
        edge_mask = (node_num_vec <= full_index) & (
            full_index < node_num_vec + edge_num_vec
        )  # [B, N+|E|+1]
        null_mask = (node_num_vec + edge_num_vec <= full_index) & (
            full_index < total_edges_num_vec
        )  # [B, N+|E|+1]

        # unpacked_node_index: [?, 2]
        edge_index_[node_mask] = unpacked_node_index
        edge_index_[edge_mask] = unpacked_edge_index
        edge_index_[null_mask] = unpacked_null_index
        #
        unpacked_null_feature = null_feat.repeat(bsize, 1)
        edge_feature_[node_mask] = unpacked_node_feature
        edge_feature_[edge_mask] = unpacked_edge_feature
        edge_feature_[null_mask] = unpacked_null_feature
        # we let full 0 for the feature of null nodes
        return Batch(edge_index_, edge_feature_, new_n_nodes, n_edges_, null_node=True)


def batch_like(G: Batch, values: T.Tensor, skip_masking=False) -> Batch:
    return Batch(
        G.indices,
        values,
        G.n_nodes,
        G.n_edges,
        G.mask,
        skip_masking,
        G.node_mask,
        G.null_node,
    )


def add_null_token(G: Batch, null_feat: T.tensor):
    # null_feat: [1, shared_dim]
    indices, values, n_nodes, n_edges12 = G.indices, G.values, G.n_nodes, G.n_edges
    # mask: [B, E];  indices: [B, |E|, 2];   values: [B, |E|, D];
    B, E, D = G.values.shape
    device = G.device

    n_edges2 = [n_edges12[i] - n_nodes[i] for i in range(B)]

    new_n_nodes = [x + 1 for x in n_nodes]  # number of 1-edges
    # number of 1-edges + number of 2-edges
    new_n_edges12 = [x + 1 for x in n_edges12]

    unpack_null_index = T.zeros(B, 2, dtype=T.long, device=device)  # [B, 2]
    unpack_null_index[:, 0] = T.tensor(n_nodes)
    unpack_null_index[:, 1] = T.tensor(n_nodes)

    full_index = T.arange(E + 1, device=device)[None, :].expand(B, E + 1)  # [B, E+1]
    node_num_vec = T.tensor(n_nodes, device=device)[:, None]
    edge_num_vec = T.tensor(n_edges2, device=device)[:, None]
    null_num_vec = T.ones(B, device=device)[:, None]
    null_mask = (node_num_vec + edge_num_vec <= full_index) & (
        full_index < node_num_vec + edge_num_vec + null_num_vec
    )

    new_indices = T.zeros(B, E + 1, 2, device=device, dtype=T.long)  # [B, E+1, 2]
    new_indices[:, :E, :] = indices
    new_indices[null_mask] = unpack_null_index

    new_values = T.zeros(B, E + 1, D, device=device, dtype=values.dtype)  # [B, E+1, D]
    new_values[:, :E, :] = values
    null_feat = null_feat.repeat(B, 1)  # [B, D]
    new_values[null_mask] = null_feat

    return Batch(
        new_indices,
        new_values,
        new_n_nodes,
        new_n_edges12,
        null_node=True,
        skip_masking=True,
    )


###########


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`."""
    if activation == "relu":
        return F.relu
    # elif activation == "relu_squared":
    #     return relu_squared
    elif activation == "gelu":
        return F.gelu
    # elif activation == "gelu_fast":
    #     deprecation_warning(
    #         "--activation-fn=gelu_fast has been renamed to gelu_accurate"
    #     )
    #     return gelu_accurate
    # elif activation == "gelu_accurate":
    #     return gelu_accurate
    elif activation == "tanh":
        return T.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return nn.SiLU
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def get_mask(sizes: T.LongTensor, max_sizes: int) -> T.BoolTensor:
    idx = T.arange(max_sizes, device=sizes.device)
    return idx[None, :] < sizes[:, None]  # [B, N]


def masked_fill(x: T.Tensor, mask: T.BoolTensor, value: float) -> T.Tensor:
    return x.clone().masked_fill_(~mask.unsqueeze(-1), value)


class Graph2dModel(Graph2dBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Create forward method that takes batch as argument
    def forward0(self, batch):
        yh = self.forward(data=batch)
        return yh

    def get_embeddings(self, x):
        return super().get_embeddings(x)
