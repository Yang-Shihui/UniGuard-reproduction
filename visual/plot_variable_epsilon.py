import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# Data from the table
categories = ['Rec', 'OCR', 'Know', 'Gen', 'Spat', 'Math']
models = ['LLaVA', '16/255', '32/255', '64/255']
values = [
    [39.2, 22.7, 26.5, 29.3, 29.6, 7.7],
    [39.1, 22.7, 26.4, 29.3, 29.6, 7.7],
    [39.0, 22.8, 26.3, 29.3, 29.4, 7.8],
    # [38.8, 22.5, 26.2, 29.1, 29.0, 7.7],
    [38.3, 22.0, 26.2, 28.5, 28.9, 7.6]
]

TITLE_FONTSIZE = 15  # New constant for title font size
LABEL_FONTSIZE = 15  # New constant for label font size
TICKS_FONTSIZE = 15  # New constant for ticks font size

# Set the default font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# Plot setup
x = np.arange(len(categories))  # Label locations
width = 0.15  # Width of bars

# Create the plot
fig, ax = plt.subplots(figsize=(4, 4))


# Prepare data for seaborn
data = []
for model, row in zip(models, values):
    for category, value in zip(categories, row):
        data.append({'Model': model, 'Task': category, 'Value': value})

df = pd.DataFrame(data)

# Plot using seaborn
plt.figure(figsize=(5, 4))
sns.lineplot(data=df, x='Model', y='Value', hue='Task', marker='o', markersize=10, linewidth=3)



# Formatting the plot
plt.title(r'MM-Vet with varying $\epsilon$' + r"$\uparrow$", fontsize=TITLE_FONTSIZE, )
plt.xlabel(r'Distance Constraint $\epsilon$', fontsize=LABEL_FONTSIZE)
# plt.ylabel('Performance', fontsize=LABEL_FONTSIZE)
plt.ylabel('')


plt.tick_params(axis='x', labelsize=TICKS_FONTSIZE)
plt.tick_params(axis='y', labelsize=TICKS_FONTSIZE)

plt.legend(title='Tasks', fontsize=LABEL_FONTSIZE, title_fontsize=TITLE_FONTSIZE, ncol=3, loc='lower center',
           bbox_to_anchor=(0.5, 0.1), frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("visual/fig_MM-Vet_varying_eps.pdf", dpi=300)
plt.show()
