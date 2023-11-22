from node import Node
import pickle

import os

modes = ["test", "train", "val"]

def read_file(file_path, dtype):
    # Read the file and convert each line to a list of integers
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert each line to a list of integers
    list_of_lists = [list(map(int, line.strip().split())) for line in lines]
    return list_of_lists

def generateDGL(mode):
    graphs = []



for mode in modes:
    generateDGL(mode)