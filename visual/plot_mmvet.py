import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams['text.usetex'] = True
# Set the default font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


FONT_SIZE = 12

# Data for LLaVA-v1.5-7B and 13B
df = pd.DataFrame({
    "Metric": ["Rec", "Know", "Gen", "Spat", "OCR", "Math"],
    "7B": [32.9, 19.0, 20.1, 25.7, 20.1, 5.2],
    "7B+UniGuard": [32.7, 19.0, 19.9, 25.6, 20.1, 5.2],
    "13B": [39.2, 26.5, 29.3, 29.6, 22.7, 7.7],
    "13B+UniGuard": [39.0, 26.3, 29.3, 29.4, 22.8, 7.8]
})

# Melt the DataFrame
df_melted = df.melt(id_vars="Metric", var_name="Model", value_name="Score")


# colors = ["#7bdff2", "#b2f7ef", "#f2b5d4", "#f7d6e0"]
colors = ["#68d8d6", "#9ceaef", "#ff758f", "#ffb3c1"]
# Create the barplot
plt.figure(figsize=(5, 4))
sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette = colors)
plt.title("Performance before / after applying UniGuard" + r"$\downarrow$", fontsize=FONT_SIZE)

X_LABEL = "MM-Vet Tasks"
Y_LABEL = "Performance"

plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
plt.ylabel("Score", fontsize=FONT_SIZE)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=FONT_SIZE)
plt.tick_params(axis='y', labelsize=FONT_SIZE)

LEGEND_FONT_SIZE = 9
legend = plt.legend(title="Model", title_fontsize=FONT_SIZE, fontsize=LEGEND_FONT_SIZE, frameon=False)

legend.get_frame().set_alpha(0.5)

plt.tight_layout()

plt.savefig("visual/MM-Vet_LLaVA.pdf", dpi=300)
plt.show()
