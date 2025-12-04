# run this script, plot figures being saved to this same folder

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Data Table (based on your results)
# -----------------------------
data = [
    # Chinese
    ["Chinese", "bert-base-chinese", 0.6596, 0.6087],
    ["Chinese", "hfl/chinese-macbert-base", 0.6810, 0.6250],
    ["Chinese", "hfl/chinese-roberta-wwm-ext", 0.7043, 0.6502],

    # English
    ["English", "bert-base-uncased", 0.7313, 0.6777],
    ["English", "distilbert-base-uncased", 0.7474, 0.6821],
    ["English", "roberta-base", 0.7593, 0.5942],

    # Hindi
    ["Hindi", "ai4bharat/IndicBERTv2-MLM-only", 0.7932, 0.7332],
    ["Hindi", "bert-base-multilingual-cased", 0.5952, 0.5123],
    ["Hindi", "xlm-roberta-base", 0.6038, 0.5459],
]

df = pd.DataFrame(data, columns=["Language", "Model", "Supervised", "Unsupervised"])

# seaborn style
sns.set(style="whitegrid", font_scale=1.2)


# ==========================================================
# 1. Supervised Spearman vs Language (Grouped Bar Chart)
# ==========================================================
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Language", y="Supervised", hue="Model")
plt.title("Supervised Spearman by Language")
plt.ylabel("Spearman Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supervised_by_language.png"), dpi=300)
plt.close()


# ==========================================================
# 2. Unsupervised Spearman vs Language (Grouped Bar Chart)
# ==========================================================
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Language", y="Unsupervised", hue="Model")
plt.title("Unsupervised Spearman by Language")
plt.ylabel("Spearman Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "unsupervised_by_language.png"), dpi=300)
plt.close()


# ==========================================================
# 3. Chinese - Sup vs Unsup
# ==========================================================
df_ch = df[df["Language"] == "Chinese"].melt(
    id_vars=["Model"],
    value_vars=["Supervised", "Unsupervised"],
    var_name="Setting",
    value_name="Score",
)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_ch, x="Model", y="Score", hue="Setting")
plt.title("Chinese Models: Supervised vs Unsupervised")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chinese_models_sup_unsup.png"), dpi=300)
plt.close()


# ==========================================================
# 4. English - Sup vs Unsup
# ==========================================================
df_en = df[df["Language"] == "English"].melt(
    id_vars=["Model"],
    value_vars=["Supervised", "Unsupervised"],
    var_name="Setting",
    value_name="Score",
)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_en, x="Model", y="Score", hue="Setting")
plt.title("English Models: Supervised vs Unsupervised")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "english_models_sup_unsup.png"), dpi=300)
plt.close()


# ==========================================================
# 5. Hindi - Sup vs Unsup
# ==========================================================
df_hi = df[df["Language"] == "Hindi"].melt(
    id_vars=["Model"],
    value_vars=["Supervised", "Unsupervised"],
    var_name="Setting",
    value_name="Score",
)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_hi, x="Model", y="Score", hue="Setting")
plt.title("Hindi Models: Supervised vs Unsupervised")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hindi_models_sup_unsup.png"), dpi=300)
plt.close()


# ==========================================================
# 6. All Models - Supervised vs Unsupervised (Global Comparison)
#    With language grouping (English → Chinese → Hindi)
# ==========================================================

# Reorder languages and filter df in that order
language_order = ["English", "Chinese", "Hindi"]
df_sorted = pd.concat([df[df["Language"] == lang] for lang in language_order])

# Melt for plotting (sup vs unsup)
df_all = df_sorted.melt(
    id_vars=["Language", "Model"],
    value_vars=["Supervised", "Unsupervised"],
    var_name="Setting",
    value_name="Score",
)

# We want x-axis order: English models, gap, Chinese models, gap, Hindi models
english_models = df_sorted[df_sorted["Language"] == "English"]["Model"].tolist()
chinese_models = df_sorted[df_sorted["Language"] == "Chinese"]["Model"].tolist()
hindi_models = df_sorted[df_sorted["Language"] == "Hindi"]["Model"].tolist()

gap = " "  # a visible empty tick for spacing
x_order = english_models + [gap] + chinese_models + [gap] + hindi_models

# Add a column with this order; entries with separator will be blank and not used
df_all["Model_with_gap"] = df_all["Model"]

# Keep only real models (we won't plot the separator as data; it's just for ticks layout)
df_all = df_all[df_all["Model_with_gap"] != ""]

plt.figure(figsize=(14, 6))
plt.ylim(0, 1.0)
ax = sns.barplot(
    data=df_all,
    x="Model_with_gap",
    y="Score",
    hue="Setting",
    order=[m for m in x_order if m != gap],
)

plt.xticks(rotation=20, ha="right")
plt.title("All Models: Supervised vs Unsupervised (Grouped by Language)")
plt.ylabel("Spearman Score")

# Compute x-coordinates for each real tick
xticks = ax.get_xticks()

# English: indices 0..len(english_models)-1
english_start = 0
english_end = len(english_models) - 1

# Chinese: next len(chinese_models) positions
chinese_start = english_end + 1
chinese_end = chinese_start + len(chinese_models) - 1

# Hindi: next len(hindi_models) positions
hindi_start = chinese_end + 1
hindi_end = hindi_start + len(hindi_models) - 1

y_top = 0.92  # safely below title but above bars

def draw_group_line(x_start_idx, x_end_idx, label):
    x_start = xticks[x_start_idx]
    x_end = xticks[x_end_idx]
    plt.plot([x_start, x_end], [y_top - 0.01, y_top - 0.01], color="black", linewidth=1.5)
    plt.text(
        (x_start + x_end) / 2,
        y_top - 0.02,   # lowered label for cleaner spacing
        label,
        ha="center",
        va="bottom",
        fontsize=12,
    )

draw_group_line(english_start, english_end, "English")
draw_group_line(chinese_start, chinese_end, "Chinese")
draw_group_line(hindi_start, hindi_end, "Hindi")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_models_sup_unsup_grouped.png"), dpi=300)
plt.close()

print("All plots saved successfully!")