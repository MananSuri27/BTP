import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd


class Node(DGLDataset):
    def __init__(self, id, type, edges, edge_types, nodes, label1, label2):
        self.id = id
        self.type = type
        self.edges = edges[id]
        self.edge_types = edge_types[id]
        self.node_embeddings  = nodes[id]
        self.label1 = label1[id]
        self.label2 = label2[id]

    def process(self):
        nodes_data = self.node_embeddings
        edges_data = self.edges

        edges_src = [edge[1] for edge in edges_data]
        edges_src = [edge[0] for edge in edges_data]

        nodes_data['x'] = nodes_data['x'].map(str)


        label_da = torch.from_numpy(self.label1.to_numpy())
        label_er = torch.from_numpy(self.label2.to_numpy())

        edges_dst = torch.from_numpy(edges_dst.to_numpy())
        edges_src = torch.from_numpy(edges_src.to_numpy())

        print(self.id, self.type, nodes_data.shape[0])

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_data))
        self.graph.ndata['embedding'] = nodes_data.to(dtype=torch.float32)
        self.graph.ndata['label_da'] = label_da
        self.graph.ndata['label_er'] = label_er
        


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1