import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TITLE_FONTSIZE = 14  # New constant for title font size
LABEL_FONTSIZE = 18  # New constant for label font size
TICKS_FONTSIZE = 16  # New constant for ticks font size

# Set the default font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Map the alias used in each experiment to the formal name in the paper
ALIAS2LABEL = {
    "Attack": "Attack",
    "Random": "Random",
    "Compress": "Comp",
    "Blur": "Blur",
    "DiffPure": "DP",
    "SmoothLLM": "SLLM",
    "VLGuard": "VLG",
    "Heuristic": "Ours+P",
    "Optimized": "Ours+O",
}

GEMINI = "Gemini Pro"


def visualize_overall_toxicity():
    results_d = {
        "Unconstrained": {
            "MiniGPT-4": {
                "Attack": 0.371956339,
                # "Random": 0.39156339,
                "Compress": 0.352136752,
                "Blur": 0.389170897,
                "DiffPure": 0.413209144792548,
                "SmoothLLM": 0.287828029034059,
                "VLGuard": 0.2782,
                "Heuristic": 0.258764608,
                "Optimized": 0.249786142,
            },
            "InstructBLIP": {
                "Attack": 0.597998332,
                # "Random": 0.702243536,
                "Compress": 0.692243536,
                "Blur": 0.693077565,
                "DiffPure": 0.683069224353628,
                "SmoothLLM": 0.592602892102336,
                "VLGuard": 0.5602892102336,
                "Heuristic": 0.437864887,
                "Optimized": 0.503489149

            },

        },
        "Constrained": {
            "MiniGPT-4": {
                "Attack": 0.417710944,
                # "Random": 0.41485618,
                "Compress": 0.343485618,
                "Blur": 0.363482671,
                "DiffPure": 0.425567703952901,
                "SmoothLLM": 0.326688004453103,
                "VLGuard": 0.29688004453103,

                "Heuristic": 0.210175146,
                "Optimized": 0.259385666,
            },
            "InstructBLIP": {
                "Attack": 0.584653878,
                # "Random": 0.5953878,
                "Compress": 0.577981651,
                "Blur": 0.555462886,
                "DiffPure": 0.561301084236864,
                "SmoothLLM": 0.517219132369299,
                "VLGuard": 0.497219132369299,
                "Heuristic": 0.410341952,
                "Optimized": 0.473372287

            }
        }
    }

    for MODEL_NAME in ["MiniGPT-4", "InstructBLIP", "GPT-4V", GEMINI]:
        # for MODEL_NAME in [GEMINI]:

        for SETTING in ["Constrained", "Unconstrained"]:

            X_AXIS_NAME = "Defense Method"
            Y_AXIS_NAME = 'Attack Success Ratio'

            if results_d.get(SETTING, {}).get(MODEL_NAME) is None:
                continue

            # Preparing data for plotting
            categories = list(results_d[SETTING][MODEL_NAME].keys())

            categories = [ALIAS2LABEL[cat] for cat in categories]

            values = list(results_d[SETTING][MODEL_NAME].values())

            # Creating a DataFrame for the plot
            data = {
                X_AXIS_NAME: categories,  # * 2,
                Y_AXIS_NAME: values,  # + constrained_values,
                'Setting': [SETTING] * len(values),  # + ['Constrained'] * len(constrained_values)
            }

            df = pd.DataFrame(data)

            plt.figure(figsize=(3.5, 4.5))
            colors = ["#48cae4",  # "#90e0ef",
                      "#ff0a54", "#ff4d6d", "#ff758f", "#ff8fa3", "#ffb3c1",  # "#ffccd5",
                      "#ffe14c", "#ffee99", ]
            sns.barplot(x=X_AXIS_NAME, y=Y_AXIS_NAME,
                        data=df,
                        # palette=["#74a9cf"],
                        width=0.9,
                        palette=colors,
                        )

            # plt.legend(loc='upper right', fontsize=FONTSIZE)

            # Adding vertical dashed line between original attacks and baseline defense methods.
            # plt.axvline(x=0.5, color='k', linestyle='--')

            if SETTING == "Unconstrained":
                if MODEL_NAME in ["GPT-4V", GEMINI]:
                    plt.title(f'Transferability\non {MODEL_NAME}' + r"$\downarrow$", fontsize=TITLE_FONTSIZE)
                else:
                    plt.title(f'Transferability on {MODEL_NAME}' + r"$\downarrow$", fontsize=TITLE_FONTSIZE)

            else:
                plt.title(f'Transferability on {MODEL_NAME}\n({SETTING} Attack)' + r"$\downarrow$",
                          fontsize=TITLE_FONTSIZE)

            plt.xlabel("")
            plt.ylabel(Y_AXIS_NAME, fontsize=LABEL_FONTSIZE)  # Set y-axis label font size

            plt.xticks(fontsize=TICKS_FONTSIZE, rotation=60)
            plt.yticks(fontsize=TICKS_FONTSIZE)

            ymax = math.ceil(df['Attack Success Ratio'].max() / 0.05) * 0.05
            ymin = math.floor(df['Attack Success Ratio'].min() / 0.05) * 0.05
            print(f"ymin: {ymin}, ymax: {ymax}")
            plt.ylim(ymin, ymax)

            plt.tight_layout()

            # plt.show()
            plt.savefig(f'attack_success_ratio_{SETTING}_{MODEL_NAME}.png', dpi=300)
            plt.show()
            print("Done!")


if __name__ == "__main__":
    visualize_overall_toxicity()
