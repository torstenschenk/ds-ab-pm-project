import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import base64
from io import BytesIO
import seaborn as sns  # Added for richer visualisations

# Style configuration for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 160  # Increase image quality

INPUT_PATH = "data/ab_test_clean.csv"
SUMMARY_OUTPUT = "data/ab_test_summary.csv"
HTML_REPORT_OUTPUT = "data/ab_test_report.html"  # New output path

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def plot_to_base64(fig):
    """Save Matplotlib/Seaborn figure to in-memory PNG and return Base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

def z_test_proportions(convA, totA, convB, totB):
    p1 = convA / totA
    p2 = convB / totB
    pooled = (convA + convB) / (totA + totB)
    se = np.sqrt(pooled * (1 - pooled) * (1/totA + 1/totB))
    if se == 0:
        return 0, 1
    z = (p2 - p1) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value

def confidence_interval(p_hat, n, alpha=0.05):
    """Compute Z Confidence Interval (proportions)."""
    if n == 0:
        return 0, 0
    z_alpha = norm.ppf(1 - alpha / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margin_of_error = z_alpha * se
    # Ensure limits are not negative
    lower = max(0, p_hat - margin_of_error)
    upper = p_hat + margin_of_error
    return lower, upper

# ===================================================================
# 1. LOAD DATA
# ===================================================================
df = pd.read_csv(INPUT_PATH)
print("Loaded dataset:", df.shape)
print(df.head())

# ===================================================================
# 2. KPI CALCULATIONS (INCL. CIs)
# ===================================================================
summary = df.groupby("variant").agg(
    visitors=("user_id", "count"),
    add_to_cart_sum=("add_to_cart", "sum"),
    conversion_rate=("add_to_cart", "mean"),
    avg_scroll=("scroll_depth", "mean"),
    avg_image_int=("image_interactions", "mean"),
    avg_session=("session_duration_sec", "mean"),
    pct_mobile=("is_mobile", "mean"),
).reset_index()

# Apply 95 % Confidence Interval
summary["lower_ci"], summary["upper_ci"] = zip(*summary.apply(
    lambda row: confidence_interval(
        row["conversion_rate"],
        row["visitors"]
    ), axis=1
))

summary["conversion_rate_pct"] = summary["conversion_rate"] * 100
summary["pct_mobile"] = summary["pct_mobile"] * 100
summary["ci_error"] = summary["upper_ci"] - summary["conversion_rate"]

print("\n===== A/B/C TEST KPI SUMMARY =====")
print(summary.head())

# ===================================================================
# 3. STATISTICAL SIGNIFICANCE (Z-Test for proportions)
# ===================================================================
def compare_to_control(treat_variant):
    A = summary[summary["variant"] == "A"].iloc[0]
    B = summary[summary["variant"] == treat_variant].iloc[0]
    z, p = z_test_proportions(A.add_to_cart_sum, A.visitors, B.add_to_cart_sum, B.visitors)
    lift = (B.conversion_rate - A.conversion_rate) / A.conversion_rate
    return round(lift*100, 2), round(z, 4), p

stats_results = []
for variant in ["B", "C"]:
    if variant in summary["variant"].values:
        lift, z, p = compare_to_control(variant)
        stats_results.append({
            "variant": variant,
            "lift_vs_A_%": lift,
            "z_score": z,
            "p_value": p
        })

stats_df = pd.DataFrame(stats_results)
print("\n===== STATISTICAL SIGNIFICANCE =====")
print(stats_df)

# ===================================================================
# 4. PREPARE FULL REPORT & SAVE CSV
# ===================================================================
full_report = summary.merge(stats_df, how="left", on="variant")
full_report.to_csv(SUMMARY_OUTPUT, index=False)
print("\nFull summary report saved:", SUMMARY_OUTPUT)

# ===================================================================
# 5. VISUALISATIONS (7 PLOTS TOTAL)
# ===================================================================

# --- PLOT 1: Conversion rate (Original) ---
fig1 = plt.figure(figsize=(8,5))
plt.bar(summary["variant"], summary["conversion_rate_pct"])
plt.xlabel("Variant")
plt.ylabel("Conversion Rate (%)")
plt.title("Conversion Rate by Variant")
plt.grid(axis="y", alpha=0.3)
img_conversion_base64 = plot_to_base64(fig1)

# --- PLOT 2: Engagement (Original 3-in-1) ---
fig2, ax = plt.subplots(1, 3, figsize=(14,4))
ax[0].bar(summary["variant"], summary["avg_scroll"])
ax[0].set_title("Average Scroll Depth")
ax[0].set_ylabel("0–1 (Normalised)")
ax[0].grid(axis="y", alpha=0.3)
ax[1].bar(summary["variant"], summary["avg_image_int"])
ax[1].set_title("Average Image Interactions")
ax[1].grid(axis="y", alpha=0.3)
ax[2].bar(summary["variant"], summary["avg_session"])
ax[2].set_title("Average Session Duration (s)")
ax[2].grid(axis="y", alpha=0.3)
plt.tight_layout()
img_engagement_base64 = plot_to_base64(fig2)


# --- NEW PLOT 3: Conversion Rate with Confidence Interval (CI) ---
fig3 = plt.figure(figsize=(8, 5))
colors = sns.color_palette("viridis", 3)
plt.bar(summary["variant"], summary["conversion_rate_pct"],
        yerr=summary["ci_error"] * 100,  # Add Y error
        capsize=10,
        color=colors)
plt.xlabel("Variant")
plt.ylabel("Conversion Rate (%)")
plt.title("Conversion Rate with 95 % Confidence Interval")
plt.grid(axis="y", alpha=0.3)
img_ci_base64 = plot_to_base64(fig3)


# --- NEW PLOT 4: Box Plot of Session Duration (Distribution) ---
fig4 = plt.figure(figsize=(8, 5))
# Show median, quartiles and outliers of session duration per variant
sns.boxplot(x='variant', y='session_duration_sec', data=df, palette="pastel")
plt.title("Session Duration Distribution (Sec)")
plt.ylabel("Session Duration (seconds)")
plt.xlabel("Variant")
img_boxplot_base64 = plot_to_base64(fig4)


# --- NEW PLOT 5: Lift vs Control with P-value ---
fig5 = plt.figure(figsize=(7, 5))
# Filter only B and C
lift_variants = stats_df[stats_df["variant"].isin(["B", "C"])]
plt.bar(lift_variants["variant"], lift_variants["lift_vs_A_%"], color=['#ff7f0e', '#2ca02c'])
plt.axhline(0, color='red', linestyle='--', alpha=0.6)
plt.ylabel("Lift vs. Current PDP -> A (%)")
plt.title("Conversion Rate Lift vs. A (current PDP)")

for i, row in lift_variants.iterrows():
    p_val = row['p_value']
    lift_val = row['lift_vs_A_%']

    # P-value annotation
    p_label = f"p = {p_val:.4f}"
    if p_val < 0.05:
        p_label += " (Sig. α=0.05)"

    # Annotation position
    y_pos = lift_val if lift_val > 0 else 0
    va_align = 'bottom' if lift_val >= 0 else 'top'
    plt.text(i, y_pos + np.sign(lift_val or 1) * 2, p_label, ha='center', va=va_align, fontsize=8, color='black')

plt.grid(axis="y", alpha=0.3)
img_lift_base64 = plot_to_base64(fig5)


# --- NEW PLOT 6: Conversion Rate Segmented by Device ---
device_summary = df.groupby(["variant", "is_mobile"]).agg(
    conversion_rate=("add_to_cart", "mean")
).reset_index()
device_summary["conversion_rate_pct"] = device_summary["conversion_rate"] * 100

fig6 = plt.figure(figsize=(9, 6))
# Use is_mobile as 'hue' to segment
sns.barplot(x='variant', y='conversion_rate_pct', hue='is_mobile', data=device_summary, palette="Set2")
plt.title("Conversion Rate by Variant and Device")
plt.ylabel("Conversion Rate (%)")
plt.xlabel("Variant")
plt.legend(title='Device', labels=['Desktop', 'Mobile'])
img_device_base64 = plot_to_base64(fig6)


# --- NEW PLOT 7: Scatter Plot Conversion vs. Engagement (Scroll) ---
fig7 = plt.figure(figsize=(8, 5))
# Create scatter plot using average scroll vs. conversion rate
plt.scatter(summary["avg_scroll"], summary["conversion_rate_pct"],
            c=colors, s=150, alpha=0.8, edgecolors='k')
plt.xlabel("Average Scroll Depth (Engagement)")
plt.ylabel("Conversion Rate (%)")
plt.title("Conversion vs. Engagement by Variant")
for i, row in summary.iterrows():
    # Annotate A, B, C on the plot
    plt.annotate(row['variant'],
                 (row['avg_scroll'], row['conversion_rate_pct']),
                 xytext=(5, 5), textcoords='offset points',
                 fontweight='bold')
plt.grid(axis="both", alpha=0.3)
img_scatter_base64 = plot_to_base64(fig7)


# ===================================================================
# 6. GENERATE HTML REPORT
# ===================================================================

# Convert DataFrames to HTML
summary_html = summary.to_html(index=False, float_format="%.2f")
stats_html = stats_df.to_html(index=False, float_format="%.4f")
full_report_html = full_report.to_html(index=False, float_format="%.4f")

# Define HTML content
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>A/B/C Test Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #1F3F60; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #f0f8ff; font-weight: bold; }}
        .plot-group {{ margin-bottom: 40px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
        .plot-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; gap: 20px; }}
        .plot-item {{ flex: 1 1 45%; min-width: 300px; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <h1>📈 A/B/C Test Analysis Report</h1>
    <p>Performance analysis of variants A (Current PDP), B (Social) and C (AI) on the primary KPI: <strong>Add to Cart</strong>.</p>

    <h2>1. KPI Summary & Confidence Intervals</h2>
    {summary_html}

    <h2>2. Statistical Significance (Z-Test)</h2>
    <p>Comparison of variants B and C against A (Current PDP) regarding Conversion Rate.</p>
    {stats_html}

    <h2>3. Visualisations</h2>

    <div class="plot-group">
        <h3>3.1 Statistical & Distribution Analysis</h3>
        <p>These charts assess conversion-rate certainty and engagement-data distribution.</p>
        <div class="plot-container">
            <div class="plot-item">
                <p><strong>A. Conversion Rate with 95 % CI</strong></p>
                <img src="{img_ci_base64}" alt="Conversion Rate with CI Plot">
            </div>
            <div class="plot-item">
                <p><strong>B. Lift vs. Curr PDP A</strong></p>
                <img src="{img_lift_base64}" alt="Lift vs Control Plot">
            </div>
             <div class="plot-item">
                <p><strong>C. Session Duration Distribution (Box Plot)</strong></p>
                <img src="{img_boxplot_base64}" alt="Session Duration Box Plot">
            </div>
            <div class="plot-item">
                <p><strong>D. Conversion Rate (Original)</strong></p>
                <img src="{img_conversion_base64}" alt="Conversion Rate Plot">
            </div>
        </div>
    </div>

    <div class="plot-group">
        <h3>3.2 Segmented & Correlation Analysis</h3>
        <p>Exploring performance by device type and the relationship between engagement and conversion.</p>
        <div class="plot-container">
            <div class="plot-item">
                <p><strong>E. Conversion Segmented by Device</strong></p>
                <img src="{img_device_base64}" alt="Conversion by Device Plot">
            </div>
             <div class="plot-item">
                <p><strong>F. Conversion vs. Scroll Depth (Scatter)</strong></p>
                <img src="{img_scatter_base64}" alt="Conversion vs Scroll Scatter Plot">
            </div>
            <div style="flex: 1 1 100%;">
                <p><strong>G. Engagement Metrics (Original 3-in-1)</strong></p>
                <img src="{img_engagement_base64}" alt="Engagement Metrics Plot">
            </div>
        </div>
    </div>

    <div style="clear: both;"></div>

    <h2>4. Full Report</h2>
    {full_report_html}

</body>
</html>
"""

# Save HTML file using UTF-8 to ensure emojis and special characters are displayed
with open(HTML_REPORT_OUTPUT, "w", encoding="utf-8") as f:
    f.write(html_content)

print("\nHTML report saved:", HTML_REPORT_OUTPUT)

# ===================================================================
# DONE
# ===================================================================
print("\n✓ A/B/C Test Analysis Completed Successfully!")