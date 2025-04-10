import argparse
from collections import Counter
from numpy import *

model_dict = {
    "gpt-3.5-turbo": "G35",
    "gpt-4o": "G4o",
    "claude-3-opus": "C3",
    "claude-3.5-sonnet": "C35",
    "gemini-pro": "GP"
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

    # Iterate over models and calculate the cumulative performance
    for num_models in range(1, len(args.models) + 1):
        selected_models = args.models[:num_models]  # Select the first num_models

        backward_scores = []  # Clear backward_scores for each iteration
        forward_scores = []   # Clear forward_scores for each iteration

        # Processing backward translation scores
        for model in selected_models:
            # Constructing file paths for backward translation
            score_file_backward = (
                f"datasets/{args.src_lan}-{args.tgt_lan}-new/"
                f"{model_dict[model]}_{args.forward}_{args.backward}.{args.metric}"
            )

            # Reading values from a file for backward translation
            with open(score_file_backward, 'r') as file:
                values = [float(line.split()[0]) for line in file]
                backward_scores.append(values)

        max_backward_index = []

        # Iterate through the index of each column for backward translation scores
        for i in range(len(backward_scores[0])):
            # Extract the value of each column
            column_values = [item[i] for item in backward_scores]
            # Find the index of the maximum value in a column
            max_index = column_values.index(max(column_values))
            max_backward_index.append(max_index)

        # Processing forward translation scores
        for model in selected_models:
            # Constructing file paths for forward translation
            score_file_forward = (
                f"datasets/{args.src_lan}-{args.tgt_lan}-new"
                f"/{model_dict[model]}_{args.forward}.{args.metric}"
            )

            # Read the file and extract the values for forward translation
            with open(score_file_forward, 'r') as file:
                values = [float(line.split()[0]) for line in file]

            # Print model information and averages
            print(model_dict[model], mean(values))

            # Add to forward_scores
            forward_scores.append(values)

        # Ensure forward_scores and max_backward_index have matching length
        if len(forward_scores[0]) != len(max_backward_index):
            print("Error: Mismatch between forward_scores length and max_backward_index.")
            continue

        final_scores = []
        for i in range(len(forward_scores[0])):
            final_scores.append(forward_scores[max_backward_index[i]][i])

        # Print the cumulative score for this iteration
        cumulative_score = round((sum(final_scores) / len(final_scores)) * 100, 2)
        print(f"Cumulative score with {num_models} models: {cumulative_score}")
