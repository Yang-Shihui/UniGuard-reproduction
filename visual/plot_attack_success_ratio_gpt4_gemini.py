import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TITLE_FONTSIZE = 16  # New constant for title font size
LABEL_FONTSIZE = 18  # New constant for label font size
TICKS_FONTSIZE = 18  # New constant for ticks font size

# Set the default font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(3, 4.8))

# Map the alias used in each experiment to the formal name in the paper
ALIAS2LABEL = {
    "Attack": "Attack",
    "Compress": "Comp",
    "Blur": "Blur",
    "DiffPure": "DP",
    "SmoothLLM": "SLLM",
    "VLGuard": "VLG",
    "Heuristic": "OursOurs+P",
    "Optimized": "OursOurs+O",
}

GEMINI = "Gemini"
GPT4O = "GPT-4o"

results = {
    GPT4O: {
        f"{GPT4O}": 0.151010101010101,
        f"{GPT4O}Ours+P": 0.03,
        f"{GPT4O}Ours+O": 0.053092783505154,
    },

    GEMINI: {
        f"{GEMINI}": 0.744718309859154,
        f"{GEMINI}Ours+P": 0.565293602103418,
        f"{GEMINI}Ours+O": 0.6039783001808,
    }
}

df = pd.concat([pd.Series(results[GPT4O]), pd.Series(results[GEMINI])]).reset_index()


X_AXIS_NAME = "Defense Method"
Y_AXIS_NAME = 'Attack Success Ratio'

df.columns = [X_AXIS_NAME, Y_AXIS_NAME]
df['Internal Label'] = ["GPT-4o", "Ours+P1", "Ours+O1", "Gemini Pro", "Ours+P2", "Ours+O2"]
df[X_AXIS_NAME] = [f"{GPT4O}", f"Ours+P", f"Ours+O", f"{GEMINI}", f"Ours+P", f"Ours+O"]
colors = ["#48cae4", "#ffe14c", "#ffee99",]

sns.barplot(x='Internal Label', y=Y_AXIS_NAME,
                        data=df,
                        # palette=["#74a9cf"],
                        width=0.9,
                        palette=colors * 2,
                        )

# Replace internal labels with original displayed labels
plt.xticks(
    ticks=range(len(df['Internal Label'])),
    labels=df['Defense Method']
)

plt.xlabel("")  # Set x-axis label font size
plt.ylabel(Y_AXIS_NAME, fontsize=LABEL_FONTSIZE)  # Set y-axis label font size

ymax = math.ceil(df['Attack Success Ratio'].max() / 0.05) * 0.05 + 0.05
ymin = math.floor(df['Attack Success Ratio'].min() / 0.05) * 0.05
print(f"ymin: {ymin}, ymax: {ymax}")
plt.ylim(ymin, ymax)
plt.title("Transferability on \nGPT-4o and Gemini" + r"$\downarrow$", fontsize=TITLE_FONTSIZE)
plt.xticks(fontsize=TICKS_FONTSIZE, rotation=60)
plt.yticks(fontsize=TICKS_FONTSIZE)
plt.tight_layout()
plt.savefig("attack_success_ratio_gpt4o_gemini.pdf", dpi=300)
plt.show()

