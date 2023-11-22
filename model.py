import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import EdgeGATConv

class DialogueModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes1, num_classes2):
        super(DialogueModel, self).__init__()
        self.treelstm = TreeLSTM(in_feats, in_feats)
        self.egatconv = EdgeGATConv(in_feats, in_feats, in_feats//6, 6, allow_zero_in_degree=True)
        self.linear1 = nn.Linear(in_feats, num_classes1)
        self.linear2 = nn.Linear(in_feats, num_classes2)
        self.dim = in_feats

    def forward(self, g, tree, features):

        n = g.number_of_nodes()
        h = torch.zeros((n, self.dim))
        c = torch.zeros((n, self.dim))
        features = self.treelstm(tree, features, h, c)
        x = self.egatconv(g, features, g.edata["embeddings"]).reshape(features.shape[0], -1)
        x = F.relu(x)
        output1 = F.softmax(self.linear1(x), dim=1)  # Softmax activation for multi-class classification
        output2 = F.softmax(self.linear2(x), dim=1)  # Softmax activation for multi-class classification
        
        return output1, output2

class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout = 0.1):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, g, features, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        h : Tensor
            The features of each node.
        """
        embeds = features
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) 
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        return h

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox["h"], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}