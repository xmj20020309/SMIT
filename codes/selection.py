import argparse
from collections import Counter
from numpy import *

model_dict = {
    "gpt-3.5-turbo": "G35",
    "gpt-4o":"G4o",
    "claude-3-opus":"C3",
    "claude-3.5-sonnet":"C35",
    "gemini-pro":"GP"
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

    # Processing models
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

    # Iterate through the index of each column
    for i in range(len(backward_scores[0])):
        # Extract the value of each column
        column_values = [item[i] for item in backward_scores]
        # Find the index of the maximum value in a column
        max_index = column_values.index(max(column_values))
        max_backward_index.append(max_index)

    # Count the maximum value index
    max_backward_count = Counter(max_backward_index)

    forward_scores = []

    # Processing models
    for model in args.models:
        # Constructing file paths
        score_file_forward = (
            f"datasets/{args.src_lan}-{args.tgt_lan}-new"
            f"/{model_dict[model]}_{args.forward}.{args.metric}"
        )

        # Read the file and extract the value
        with open(score_file_forward, 'r') as file:
            values = [float(line.split()[0]) for line in file]

        # Print model information and averages
        print(model_dict[model], mean(values))

        # Add to forward_scores
        forward_scores.append(values)

    final_scores = []
    for i in range(len(forward_scores[0])):
        final_scores.append(forward_scores[max_backward_index[i]][i])
    #print(f"Shape of final_scores: ({len(final_scores)},)")
    print(f"Merge:", round((sum(final_scores)/len(final_scores))*100, 2))





    