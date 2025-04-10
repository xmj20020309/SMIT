import argparse
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
        {"flags": ["--iterations"], "kwargs": {"type": int, "default": 5, "help": "Number of iterations for each model"}},  # 5 iterations
        {"flags": ["--models"], "kwargs": {"nargs": '+', "help": "List of models to test"}},  # Multiple models
    ]

    # Initialize the parser
    parser = argparse.ArgumentParser('Command-line script to use')

    # adding parameters
    for arg in arguments:
        parser.add_argument(*arg["flags"], **arg["kwargs"])

    # Parsing parameters
    args = parser.parse_args()

    # print parameters
    print(f"Source language: {args.src_lan}, Target language: {args.tgt_lan}")

    # Iterate over each model specified in the command line arguments
    for model in args.models:
        print(f"Testing model: {model}")

        # Process each model for 5 iterations
        for t in range(1, args.iterations+1):  # Loop over the iterations (0 to 4)
            backward_scores = []  # Store backward translation scores
            forward_scores = []   # Store forward translation scores
            for count in range(t):
                # Construct file paths for forward and backward translations
                score_file_backward = (
                    f"datasets/{args.src_lan}-{args.tgt_lan}-new/"
                    f"{model_dict[model]}_{args.forward}_{args.backward}_{count}.{args.metric}"
                )
                # Read the backward translation results
                try:
                    with open(score_file_backward, 'r') as file:
                        values = [float(line.split()[0]) for line in file]
                        backward_scores.append(values)
                except FileNotFoundError:
                    print(f"File not found: {score_file_backward}")
                    continue


            # Calculate the maximum index of backward scores
            max_backward_index = []
            for i in range(len(backward_scores[0])):
                column_values = [item[i] for item in backward_scores]
                max_index = column_values.index(max(column_values))
                max_backward_index.append(max_index)

            for count in range(t):
                score_file_forward = (
                    f"datasets/{args.src_lan}-{args.tgt_lan}-new/"
                    f"{model_dict[model]}_{args.forward}_{count}.{args.metric}"
                )

                # Read the forward translation results
                try:
                    with open(score_file_forward, 'r') as file:
                        values = [float(line.split()[0]) for line in file]
                        forward_scores.append(values)
                except FileNotFoundError:
                    print(f"File not found: {score_file_forward}")
                    continue


            # Final scores based on forward translation and max backward index
            final_scores = []
            for i in range(len(forward_scores[0])):
                final_scores.append(forward_scores[max_backward_index[i]][i])

            # Print the result for each iteration
            print(f"{model} {t}: {round((sum(final_scores) / len(final_scores)) * 100, 2)}")