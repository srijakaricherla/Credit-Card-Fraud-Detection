"""
Script to generate architecture diagram for Credit Card Fraud Detection project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_data = '#3498db'
color_process = '#2ecc71'
color_model = '#e74c3c'
color_result = '#f39c12'
color_best = '#9b59b6'

# Define box style
def create_box(x, y, width, height, text, color, text_color='white', fontsize=11):
    """Create a styled box with text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=text_color, wrap=True)

# Define arrow style
def create_arrow(x1, y1, x2, y2, color='black', width=2):
    """Create an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->',
                           mutation_scale=20,
                           color=color,
                           linewidth=width,
                           zorder=1)
    ax.add_patch(arrow)

# Title
ax.text(5, 9.5, 'Credit Card Fraud Detection - Architecture', 
        ha='center', va='center', fontsize=20, fontweight='bold')

# Step 1: Raw Data
create_box(0.5, 7.5, 1.5, 1, 'Raw Kaggle\nData', color_data)

# Step 2: Preprocessing
create_box(2.5, 7.5, 1.5, 1, 'Preprocessing\n& Cleaning', color_process)
create_arrow(2, 7.5, 2.5, 7.5, color='black', width=2)

# Step 3: Feature Engineering
create_box(4.5, 7.5, 1.5, 1, 'Feature\nEngineering', color_process)
create_arrow(4, 7.5, 4.5, 7.5, color='black', width=2)

# Step 4: Model Training (3 models side by side)
create_box(1, 5, 1.2, 1, 'Logistic\nRegression', color_model)
create_box(2.5, 5, 1.2, 1, 'Random\nForest', color_model)
create_box(4, 5, 1.2, 1, 'Gradient\nBoosting', color_model)

# Arrow from Feature Engineering to Models
create_arrow(5.25, 7.5, 3.1, 6, color='black', width=2)
create_arrow(5.25, 7.5, 2.5, 6, color='black', width=2)
create_arrow(5.25, 7.5, 1.6, 6, color='black', width=2)

# Step 5: Evaluation
create_box(6, 5, 1.5, 1, 'Evaluation\n(Metrics)', color_result)
create_arrow(5.2, 5.5, 6, 5.5, color='black', width=2)
create_arrow(3.7, 5.5, 6, 5.5, color='black', width=2)
create_arrow(2.2, 5.5, 6, 5.5, color='black', width=2)

# Step 6: Best Model
create_box(2.5, 3, 2, 1, 'Best Model:\nGradient Boosting', color_best, fontsize=12)
create_arrow(6.75, 5, 3.5, 4, color='black', width=3)

# Step 7: Reports
create_box(6, 3, 1.5, 1, 'Reports\n& Insights', color_result)
create_arrow(4.5, 3.5, 6, 3.5, color='black', width=2)

# Add metrics text
metrics_text = """Metrics:
• Accuracy: ~99.8%
• F1-Score: ~0.85
• Precision, Recall
• ROC-AUC
• Confusion Matrix"""

ax.text(1, 1.5, metrics_text,
        ha='left', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=color_data, label='Data'),
    mpatches.Patch(facecolor=color_process, label='Processing'),
    mpatches.Patch(facecolor=color_model, label='Models'),
    mpatches.Patch(facecolor=color_result, label='Results'),
    mpatches.Patch(facecolor=color_best, label='Best Model')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('reports/architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to reports/architecture_diagram.png")
plt.close()

