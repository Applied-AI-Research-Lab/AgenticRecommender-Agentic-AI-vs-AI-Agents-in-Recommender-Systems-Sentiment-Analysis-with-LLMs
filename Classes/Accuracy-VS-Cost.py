import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

models = [
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gpt-4.1",
    "claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1",
    "deepseek-chat", "deepseek-reasoner",
    "GPT-5 as an Orchestrator", "GPT-5-mini as an Orchestrator"
]

accuracy = [
    0.5906, 0.5686, 0.5844,
    0.6240, 0.6234, 0.5984,
    0.6354,
    0.6502, 0.5950, 0.6448,
    0.6334, 0.6022,
    0.7140, 0.7032
]

cost = [
    43.97, 7.36, 0.38,
    11.33, 1.69, 0.67,
    3.09,
    5.76, 1.92, 27.33,
    0.24, 1.98,
    24.73, 6.14
]


# Convert to DataFrame
df = pd.DataFrame({"Model": models, "Accuracy": accuracy, "Cost": cost})

# Remove models with missing cost values
df = df.dropna()

# Scatter plot settings
plt.figure(figsize=(10, 6))
palette = sns.color_palette("pastel", len(df))
sns.scatterplot(data=df, x="Cost", y="Accuracy", hue="Model", palette=palette, s=100, edgecolor="black")

# Annotate each point slightly above and to the right
for i, row in df.iterrows():
    plt.text(row["Cost"] * 1.05, row["Accuracy"] + 0.002, row["Model"], fontsize=9, ha='left', va='bottom')

# Labels and title
plt.xscale("log")  # Log scale to handle large differences in cost
plt.xlabel("Cost ($, Log Scale)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Cost")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Hide legend for clarity
plt.legend([],[], frameon=False)

# Show plot
plt.show()