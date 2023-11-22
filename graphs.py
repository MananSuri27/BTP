import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
from dgl.data import DGLDataset
import pickle
import numpy as np

from transformers import RobertaTokenizer, RobertaModel
import torch

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# List of sentences
def embed(sentences):
    # Tokenize and encode the sentences
    encoded_inputs = tokenizer(sentences, return_tensors='pt', padding='max_length', truncation=True, max_length=64)

    # Forward pass to get the embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract the embeddings
    embeddings = outputs.last_hidden_state[:,0,:]

    # Stack the embeddings to create a 2D tensor
    stacked_embeddings = torch.stack([embedding.squeeze() for embedding in embeddings])


    return stacked_embeddings


edge_dict = {
    0: 'Comment', 1: 'Contrast', 2: 'Correction', 3: 'Question-answer_pair',
    4: 'Parallel', 5: 'Acknowledgement', 6: 'Elaboration', 7: 'Clarification_question',
    8: 'Conditional', 9: 'Continuation', 10: 'Result', 11: 'Explanation',
    12: 'Q-Elab', 13: 'Alternation', 14: 'Narration', 15: 'Background', 16: 'Break'
}


edge_type_list = [edge_dict[k] for k in edge_dict]

edge_embeddings = embed(edge_type_list)





def read_file(file_path):
    # Read the file and convert each line to a list of integers
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert each line to a list of integers
    list_of_lists = [list(map(int, line.strip().split())) for line in lines]
    return list_of_lists

def read_conversations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert each line to a list of integers
    list_of_lists = [list(map(str, line.strip().split("<utterance>")))[1:] for line in lines]

    return list_of_lists



class DialogueDataset(DGLDataset):
    def __init__(self, split):
        self.split = split
        self.nodes = read_conversations(f"/Users/manansuri/Desktop/Research/btp/{self.split}/dailydialog_filter.txt")
        self.edge_type =  pickle.load(open(f"/Users/manansuri/Desktop/Research/btp/{self.split}/predict_relation.pkl","rb"))
        self.edges = pickle.load(open(f"/Users/manansuri/Desktop/Research/btp/{self.split}/predict_link.pkl","rb"))
        super().__init__(name="synthetic")

    def process(self):
        edges =  self.edges
        
        nodes = self.nodes
        labels1 = read_file(f"/Users/manansuri/Desktop/Research/btp/{self.split}/dialogue_act_filter.txt")
        labels2 = read_file(f"/Users/manansuri/Desktop/Research/btp/{self.split}/dialogue_emotion_filter.txt")
        
        self.trees = []
        self.graphs = []
        self.labels = []
        num_graphs = len(nodes)


        # For each graph ID...
        for i in range(num_graphs):
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges[i]
            edges_of_id = [edges_of_id[i] for i in range(len(edges_of_id)) if 0 not in edges_of_id[i]]
            src = np.array([e[1]-1 for e in edges_of_id])
            dst = np.array([e[0]-1 for e in edges_of_id])
            num_nodes = len(nodes[i])

    
            label_da =  torch.from_numpy(np.array([x-1 for x in labels1[i]]))
            label_er = torch.from_numpy(np.array(labels2[i]))

     
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            # g = dgl.add_self_loop(g)
            g.ndata['label_da'] = label_da
            g.ndata['label_er'] = label_er

            self.graphs.append(g)

            # tree
            tree_edges = []
            # types of edges: sequential
            for i in range(num_nodes-1):
                tree_edges.append([i, i+1])
            
            for i in range(num_nodes):
                if i+2<num_nodes:
                    tree_edges.append([i, i+2])
            
            tree_src = np.array([e[1] for e in tree_edges])
            tree_dst = np.array([e[0] for e in tree_edges])

            tree = dgl.graph((tree_src, tree_dst), num_nodes =num_nodes)
            self.trees.append(tree)


            

            # between same user
            


    def __getitem__(self, i):
        g = self.graphs[i]
        nodes = self.nodes[i]
        nodes = ["".join(n.split(":")[1:]).strip() for n in nodes]
        embeds = embed(nodes)
        edge_types = self.edge_type[i]
        edge_types = [edge_types[j] for j in range(len(self.edges[i])) if 0 not in self.edges[i][j]]
        g.edata["embeddings"] = edge_embeddings[edge_types]
        g.ndata["embeddings"] = embeds
        return g, self.trees[i]

    def __len__(self):
        return len(self.graphs)

