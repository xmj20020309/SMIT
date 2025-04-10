import argparse
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

model_dict = {
    "gpt-3.5-turbo": "G35",
    "gpt-4o": "G4o",
    "claude-3-opus": "C3",
    "claude-3.5-sonnet": "C35",
    "gemini-pro": "GP"
}

# 新添加的映射字典，用于热力图标签
heatmap_labels = {
    "gpt-3.5-turbo": "G3.5",
    "gpt-4o": "G4o",
    "claude-3-opus": "C3O",
    "claude-3.5-sonnet": "C3.5S",
    "gemini-pro": "G-Pro"
}

if __name__ == '__main__':

    # Define parameter configuration list
    arguments = [
        {"flags": ["--src_lan"], "kwargs": {"type": str, "default": '', "help": "source language"}},
        {"flags": ["--tgt_lan"], "kwargs": {"type": str, "default": '', "help": "target language"}},
        {"flags": ["--forward"], "kwargs": {"type": str, "default": '', "help": "forward translation method"}},
        {"flags": ["--backward"], "kwargs": {"type": str, "default": '', "help": "backward translation method"}},
        {"flags": ["--metric"], "kwargs": {"type": str, "default": '', "help": "selection metric"}},
        {"flags": ["--models"], "kwargs": {"nargs": '+', "help": "LLMs"}},
    ]

    # Initialize the parser
    parser = argparse.ArgumentParser('Command-line script to use')

    # adding parameters
    for arg in arguments:
        parser.add_argument(*arg["flags"], **arg["kwargs"])

    # Parsing parameters
    args = parser.parse_args()

    # print parameters
    print(args.src_lan, args.tgt_lan)

    backward_scores = []

    # Processing backward translation models
    for model in args.models:
        # Constructing file paths
        score_file_backward = (
            f"datasets/{args.src_lan}-{args.tgt_lan}-new/"
            f"{model_dict[model]}_{args.forward}_{args.backward}.{args.metric}"
        )

        # Reading values from a file
        with open(score_file_backward, 'r') as file:
            values = [float(line.split()[0]) for line in file]
            backward_scores.append(values)

    max_backward_index = []

    # Iterate through the index of each column (i.e., for each sentence)
    for i in range(len(backward_scores[0])):
        column_values = [item[i] for item in backward_scores]  # Extract the value of each column
        max_index = column_values.index(max(column_values))  # Find the index of the max value
        max_backward_index.append(max_index)

    forward_scores = []

    # Processing forward translation models
    for model in args.models:
        # Constructing file paths
        score_file_forward = (
            f"datasets/{args.src_lan}-{args.tgt_lan}-new"
            f"/{model_dict[model]}_{args.forward}.{args.metric}"
        )

        # Read the file and extract the values
        with open(score_file_forward, 'r') as file:
            values = [float(line.split()[0]) for line in file]

        # Add to forward_scores
        forward_scores.append(values)

    # Create a matrix to store the count of correct and incorrect choices
    result_matrix = np.zeros((len(args.models), len(args.models)), dtype=int)

    # Iterate over each sentence
    for i in range(len(forward_scores[0])):
        # For each sentence, find the model with the highest forward translation score
        column_values = [item[i] for item in forward_scores]
        max_forward_index = column_values.index(max(column_values))
        best_forward_model = args.models[max_forward_index]

        # Check if the model selected in backward translation is the same as the best forward model
        selected_model_for_backward = args.models[max_backward_index[i]]

        # Find the indices for both forward and backward model selection
        forward_model_index = args.models.index(best_forward_model)
        backward_model_index = args.models.index(selected_model_for_backward)

        # Increment the corresponding cell in the result matrix
        result_matrix[forward_model_index, backward_model_index] += 1

    # 创建带有自定义标签的DataFrame
    custom_labels = [heatmap_labels[model] for model in args.models]
    df = pd.DataFrame(result_matrix, index=custom_labels, columns=custom_labels)

    # Print the DataFrame (this will show a table format)
    print("Correspondence between the selected model and the actual model:")
    print(df)



