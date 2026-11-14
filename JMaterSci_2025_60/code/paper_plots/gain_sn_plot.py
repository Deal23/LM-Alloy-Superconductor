# Adapted from Code/5_GaInSn_best_predict/best-Alloy.ipynb at commit 82898a6.

# %% Original notebook cell 3
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/ga_in_sn'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Set matplotlib options for font handling and minus signs
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial

file_path = str(PROJECT_ROOT / 'data/research/ga_in_sn/ga_in_sn_rf.csv')  # file path

# Load data
data = pd.read_csv(file_path)  # read the file

# Extract Ga, In, and Sn contents and corresponding Tc values
ga = data['formula'].str.extract(r'Ga(\d+\.\d+)').astype(float).values.flatten()
in_ = data['formula'].str.extract(r'In(\d+\.\d+)').astype(float).values.flatten()
sn = data['formula'].str.extract(r'Sn(\d+\.\d+)').astype(float).values.flatten()
tc = data['Predicted_Tc'].values

fig_height = 6.3  # cm
fig_width = 6  # cm

def ternary_xy(a, b, c):
    total = a + b + c
    a, b, c = a / total, b / total, c / total
    return c + 0.5 * a, np.sqrt(3) / 2 * a


# Initialize the plot
fig = plt.figure(figsize=(fig_width/2.54, fig_height/2.54))
ax = fig.add_subplot()

# Plot the ternary diagram
x, y = ternary_xy(ga, in_, sn)
scatter = ax.scatter(x, y, c=tc, cmap='cividis', marker='o', alpha=1, s=15, edgecolor='none')

top_idx = np.argsort(-tc)[:10]
# Highlight the ten points with the highest Tc using larger markers and a distinct symbol
top_x, top_y = ternary_xy(ga[top_idx], in_[top_idx], sn[top_idx])
top_points = ax.scatter(top_x, top_y,
                        c='red', marker='*', s=30, label='Top 10 $T_c$',edgecolor='none')

# Add the legend and adjust its position and size
ax.legend(loc='upper left', bbox_to_anchor=(0.68, 1.07), borderaxespad=0,
          fontsize='6', framealpha=0)

# Set axis labels
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
clip = Polygon(triangle, closed=True, transform=ax.transData)
scatter.set_clip_path(clip)
top_points.set_clip_path(clip)
ax.add_patch(Polygon(triangle, closed=True, fill=False, edgecolor='black', linewidth=0.8))
for fraction in np.linspace(0.1, 0.9, 9):
    ax.plot(*zip(ternary_xy(np.full(2, fraction), np.array([0, 1 - fraction]),
                            np.array([1 - fraction, 0]))), color='0.8', linewidth=0.35)
    ax.plot(*zip(ternary_xy(np.array([0, 1 - fraction]), np.full(2, fraction),
                            np.array([1 - fraction, 0]))), color='0.8', linewidth=0.35)
    ax.plot(*zip(ternary_xy(np.array([0, 1 - fraction]), np.array([1 - fraction, 0]),
                            np.full(2, fraction))), color='0.8', linewidth=0.35)
ax.text(0.5, np.sqrt(3) / 2 + 0.04, 'Ga', ha='center', fontsize=8)
ax.text(-0.03, -0.04, 'In', ha='right', va='top', fontsize=8)
ax.text(1.03, -0.04, 'Sn', ha='left', va='top', fontsize=8)
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.12)
ax.set_aspect('equal')
ax.axis('off')

# Add a colorbar and adjust its position and size
cbar = fig.colorbar(scatter, ax=ax, pad=0.2, shrink=0.6)
cbar.set_label('Predicted T$_c$ (K)', fontsize='8')
cbar.set_ticks(np.arange(3, 9, 1))
cbar.ax.tick_params(axis='both', which='major', labelsize=6)

# Add grid lines
ax.grid(True)

plt.tight_layout()
plt.savefig(PLOT_OUTPUT / 'fig4d_1.png', dpi=300,  bbox_inches='tight', transparent=True)
plt.show()
