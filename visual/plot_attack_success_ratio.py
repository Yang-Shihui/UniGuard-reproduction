import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# 强制关闭 LaTeX 文本渲染
matplotlib.rcParams['text.usetex'] = False 
# 强制关闭 LaTeX 数学公式渲染 (关键！)
matplotlib.rcParams['mathtext.fontset'] = 'stix' 
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import matplotlib.pyplot as plt



# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'Times New Roman'


X_LABEL = "Distance Constraint"
Y_LABEL = "Attack Success Ratio"
VAR_NAME = "Method"


attack_success_ratio = {
    "Predefined": [0.5526755853, 0.3662691652, 0.2568807339],
    "Optimized": [0.4940374787, 0.311965812, 0.2517182131]
}

# Create DataFrame
attack_success_ratio_df = pd.DataFrame(attack_success_ratio)

# Reshape the DataFrame to long format for seaborn
attack_success_ratio_df[X_LABEL] = ["16/255", "32/255", "64/255"]  # Add category labels for x-axis
attack_success_ratio_melted = attack_success_ratio_df.melt(id_vars=X_LABEL, var_name=VAR_NAME, value_name=Y_LABEL)

# Set up the figure size
plt.figure(figsize=(3, 5))

# colors = ["#023e8a", "#48cae4"]
colors = ["#38b000", "#9ef01a"]

# Create the barplot
sns.barplot(data=attack_success_ratio_melted, x=X_LABEL, y=Y_LABEL, hue=VAR_NAME, palette=colors)


plt.title("")

FONT_SIZE = 16
plt.xlabel(X_LABEL + " " + r"$\epsilon$", fontsize=FONT_SIZE)
plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)

plt.ylim(0, 0.6)

plt.legend(title="Method", title_fontsize=FONT_SIZE, fontsize=12, frameon=False)

plt.tick_params(axis='x', labelsize=FONT_SIZE)
plt.tick_params(axis='y', labelsize=FONT_SIZE)


# Show plot
plt.tight_layout()
plt.savefig("vary_eps.png", dpi=300)
plt.show()


X_LABEL = "Text Guardrail Length $L$"
Y_LABEL = "Attack Success Ratio"


attack_success_ratio_image = {
    "Image": [0.4095477387, 0.336432798, 0.2671755725]
}

# Create DataFrame
attack_success_ratio_image_df = pd.DataFrame(attack_success_ratio_image)

plt.figure(figsize=(3, 5))
# Add the category labels for the x-axis
attack_success_ratio_image_df[X_LABEL] = ["8", "16", "32"]

# Set up the figure size
plt.figure(figsize=(3, 5))

# Create the barplot
sns.barplot(data=attack_success_ratio_image_df, x=X_LABEL, y="Image", palette=["#38b000", "#70e000", "#9ef01a"])


plt.title("")
plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)

plt.tick_params(axis='x', labelsize=FONT_SIZE)
plt.tick_params(axis='y', labelsize=FONT_SIZE)


plt.ylim(0, 0.6)

# Show plot
plt.tight_layout()

plt.savefig("vary_text_guardrail_length.png", dpi=300)
plt.show()





