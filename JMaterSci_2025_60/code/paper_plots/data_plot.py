# Adapted from Pic/data_plot.ipynb at commit 82898a6.

# %% Original notebook cell 1
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/data_analysis'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

color2 = '#858F03'   #density
color1 = '#115F38'   #histogram
# Read data
file_path = str(PROJECT_ROOT / 'data/processed/mdr_clean.csv')
df = pd.read_csv(file_path)

# Replace infinities with NaN and remove NaN values
data = df['Tc'].replace([np.inf, -np.inf], np.nan).dropna().tolist()

# Create the plot
fig_height = 7  #cm
fig_width = 9 #cm
fig, ax1 = plt.subplots(figsize=(fig_width/2.54, fig_height/2.54))
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
# Plot the histogram
ax1.hist(data, bins=30, color=color1, alpha=0.6, edgecolor='black')
ax1.set_xlabel(r'$\mathit{T}_{\mathit{c}}$ (K)', fontsize=10)
ax1.set_ylabel('Number of Formulas', color='black', size=10)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=8)

# Create a second axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot the KDE curve
sns.kdeplot(data, ax=ax2, color=color2, fill=False)
ax2.set_ylabel('Density', color='black', size=10)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)

# Set x-axis parameters
ax1.tick_params(axis='x', labelsize=8)
ax1.set_xlim(0, 150)

# Save and show the plot
plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig2a.png'), dpi=300, bbox_inches='tight',transparent=True)
plt.show()

# %% Original notebook cell 3
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/data_analysis'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Read the selected column from the file
file_path = str(PROJECT_ROOT / 'data/processed/mdr_clean.csv')
column = 'formula'

# Use pandas to read the selected column from the file
df = pd.read_csv(file_path)

# Define a function to count elements in a formula
def count_elements(formula):
    # Use a regular expression to match all elements followed by optional numbers
    elements = re.findall(r'[A-Z][a-z]*', formula)
    return len(elements)

# Apply the function and create a new column for element counts
df['element_count'] = df[column].apply(lambda x: count_elements(str(x)) if pd.notnull(x) else 0)

# Count formulas by number of elements
element_counts = df['element_count'].value_counts()

# Plot the data distribution
fig_height = 7  #cm
fig_width = 8 #cm
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
plt.figure(figsize=(fig_width/2.54, fig_height/2.54))
barplot = sns.barplot(x=element_counts.index, y=element_counts.values, hue=element_counts.index, palette="YlGn", legend=False)
# Add text annotations
for i, (element_count, value) in enumerate(element_counts.items()):
    plt.text(element_count-1,value, str(value), ha='center', va='bottom',fontsize=7)

plt.xlabel('Number of Elements', fontsize=10)
plt.ylabel('Number of Formulas', fontsize=10)
plt.xticks(fontsize=8)  # ensure that all x-axis labels are visible
plt.yticks(fontsize=8)  # set the y-axis tick-label font size
# Save and show the plot
plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig2b.png'), dpi=300,bbox_inches='tight',transparent=True)
plt.show()

# Filter rows with a specified number of elements
formulas_with_ten_elements = df[df['element_count'] == 8][column]

# Output these formulas
print("Formulas with 8 elements:")
print(formulas_with_ten_elements)

# %% Original notebook cell 5
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/data_analysis'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import pandas as pd
import re
from types import SimpleNamespace
import numpy as np
import csv

def get_all_elements():
    """Periodic-table positions needed by the original frequency plot."""
    rows = [
        (1, 'H He', [1, 18]),
        (2, 'Li Be B C N O F Ne', [1, 2, 13, 14, 15, 16, 17, 18]),
        (3, 'Na Mg Al Si P S Cl Ar', [1, 2, 13, 14, 15, 16, 17, 18]),
        (4, 'K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr', list(range(1, 19))),
        (5, 'Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe', list(range(1, 19))),
        (6, 'Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn',
         [1, 2] + [None] * 15 + list(range(4, 19))),
        (7, 'Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og',
         [1, 2] + [None] * 15 + list(range(4, 19))),
    ]
    elements = []
    for period, symbols, groups in rows:
        for symbol, group in zip(symbols.split(), groups):
            elements.append(SimpleNamespace(symbol=symbol, group_id=group,
                                            period=period, atomic_number=len(elements) + 1))
    return elements

# Parse chemical formulas into dictionaries
def parse_formula(formula):
    formula = str(formula)  # ensure that formula is a string
    elements_ratios = {}
    for element, ratio in re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula):
        if element in elements_ratios:
            elements_ratios[element] += float(ratio) if ratio else 1
        else:
            elements_ratios[element] = float(ratio) if ratio else 1
    return elements_ratios

# Data preprocessing: count element frequencies
def preprocess_molecular_data(data, Tc_line=5):
    element_counts = {}
    for index, row in data.iterrows():  # iterate over the DataFrame with iterrows()
        if row['Tc'] > Tc_line:  # access Tc directly from the row
            for element in parse_formula(row['formula']):
                element_counts[element] = element_counts.get(element, 0) + 1
    return element_counts

# Draw a single element cell
def plot_element(ax, x, y, symbol, atomic_num, frequency, color_map, normalize):
    symb_fs = 9; # element-symbol font size
    num_fs = 6 # atomic-number font size
    Fre_fs = 6 # frequency-count font size
    color = color_map(normalize(frequency)) if frequency else 'lightgrey'
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor='black'))#set the border
    ax.text(x, y + 0.03, symbol, fontsize=symb_fs, ha='center', va='center', color='black')
    ax.text(x - 0.18, y - 0.3, f'{atomic_num}', fontsize=num_fs, ha='center', va='center', color='black')

    if frequency:
        ax.text(x, y + 0.37, f'{frequency}', fontsize=Fre_fs, ha='center', va='center', color='black')

# Draw periodic-table elements
def plot_elements(ax, elements, element_counts, lanthanoid, actinoid, color_map, normalize):
    for el in elements:
        if el.symbol in lanthanoid:
            x, y = 3 + lanthanoid.index(el.symbol), 9
        elif el.symbol in actinoid:
            x, y = 3 + actinoid.index(el.symbol), 10
        else:
            x, y = el.group_id, el.period
        plot_element(ax, x, y, el.symbol, el.atomic_number, element_counts.get(el.symbol, 0), color_map, normalize)

# Draw lanthanide and actinide ranges
def plot_lanthanoid_actinoid_labels(ax):
    """Draw the lanthanide and actinide ranges"""
    # Special handling for lanthanides and actinides
    ax.text(2, 9, "Lanthanids", fontsize=8, ha='right', va='center', color='black')
    ax.text(2, 10, "Actinids", fontsize=8, ha='right', va='center', color='black')
    bg_color = 'lightgrey'
    # # lanthanides
    ax.text(3, 6+0.1, 'La~\nLu', fontsize=8, ha='center', va='center', color='black')
    # ax.text(3 - 0.05, 6 - 0.3, '57~71', fontsize=8, ha='center', va='center', color='black')
    ax.add_patch(plt.Rectangle((3 - 0.5, 6 - 0.5), 1, 1, facecolor=bg_color, edgecolor='black'))

    # # actinides
    ax.text(3, 7+0.1, 'Ac~\nLr', fontsize=8, ha='center', va='center', color='black')
    # ax.text(3 - 0.05, 7 - 0.3, '89~103', fontsize=8, ha='center', va='center', color='black')
    ax.add_patch(plt.Rectangle((3 - 0.5, 7 - 0.5), 1, 1, facecolor=bg_color, edgecolor='black'))

# Main function: plot the periodic table
def main(Tc_line=5):
    # Read data and count frequencies
    df = pd.read_csv(DATA_PATH, low_memory=False)
    element_counts = preprocess_molecular_data(df,Tc_line)

    lanthanoid = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    actinoid = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

    max_freq = max(element_counts.values(), default=1)
    normalize = Normalize(vmin=0, vmax=max_freq)

    # Initialize the plot
    fig_height = 11  # cm
    fig_width = 16  # cm
    plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
    fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))
    # Draw elements and lanthanide/actinide labels
    plot_lanthanoid_actinoid_labels(ax)
    plot_elements(ax, elements, element_counts, lanthanoid, actinoid, color_map, normalize)
    # Plot settings
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 11)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')  # hide the full axis frame

    # Add a colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=normalize, cmap=color_map), ax=ax, orientation='vertical',pad=-0.01,shrink=0.9)
    cbar.set_label('Frequency', fontsize=8)

    cbar.ax.tick_params(labelsize=8)  # set tick-label font size
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.savefig(str(PLOT_OUTPUT / 'fig2c.png'), dpi=300, bbox_inches='tight',transparent=True)
    plt.show()


# Run the main function
if __name__ == "__main__":
    # data path
    DATA_PATH = str(PROJECT_ROOT / 'data/processed/mdr_clean.csv')
    # get periodic-table elements
    elements = get_all_elements()
    # adjust RGB values of the colormap
    original_cmap = cm.YlGn  # predefined
    colors = original_cmap(np.linspace(0, 1, original_cmap.N))  # get colors
    colors[:, -1] = colors[:, -1] * 0.7  # -1 is the alpha channel; previous columns are RGB values
    color_map = LinearSegmentedColormap.from_list("lighter_YlOrBr", colors)
    main(Tc_line=5)

# %% Original notebook cell 7
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/alloy_screening'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt

data_num = ['5 k', '2k', '1 k', '500', '200', '50']
alloys_data = {
    'Alloy-1': [
        'In0.500Sn0.500', 'In0.487Sn0.494Zn0.019', 'In0.500Sn0.500',
        'In0.500Sn0.500', 'In0.500Sn0.500', 'In0.500Sn0.500', 'In0.506Sn0.494',
        'In0.502Sn0.498'
    ],
    'Alloy-2': [
        'Ga0.002In0.596Sn0.402', 'In0.500Sn0.500', 'In0.501Sn0.499',
        'In0.501Sn0.499', 'In0.501Sn0.499', 'In0.502Sn0.498', 'In0.498Sn0.502',
        'In0.508Sn0.492'
    ],
    'Alloy-3': [
        'Bi0.005In0.592Sn0.403', 'In0.499Sn0.497Ag0.004', 'In0.486Sn0.494Zn0.020',
        'In0.502Sn0.498', 'In0.502Sn0.498', 'In0.499Sn0.501', 'In0.496Sn0.504',
        'In0.490Sn0.502Zn0.008'
    ],
}
tc_values = {
    'Alloy-1': [7.01, 7.01, 7.01, 7.01, 6.99, 7.00],
    'Alloy-2': [7.00, 7.00, 7.00, 7.00, 6.98, 6.97],
    'Alloy-3': [7.00, 7.00, 7.00, 6.99, 6.97, 6.96]
}
# Specify colors
colors = [ '#BBA315','#6C6E72','#143670']

# Create a line plot
fig_height = 6 #cm
fig_width = 10 #cm
# Initialize the plot
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
plt.figure(figsize=(fig_width/2.54, fig_height/2.54))

# Reverse the data order
data_num_reversed = data_num[::-1]
tc_values_reversed = {alloy: tc_values[alloy][::-1] for alloy in tc_values}

# Plot each line and specify colors
for alloy, color in zip(alloys_data.keys(), colors):
    plt.plot(data_num_reversed, tc_values_reversed[alloy], label=alloy, marker='o', color=color,markersize=3)

# Add title and labels
plt.xlabel('Number of Data', color='black', fontsize=8, labelpad=0)
plt.ylabel('Predicted $\mathit{T}_{\mathit{c}}$ (K)', color='black', fontsize=8, labelpad=0)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Add a legend
plt.legend(fontsize=8, framealpha=0)
# Show the plot
plt.savefig(str(PLOT_OUTPUT / 'fig4b.png'), dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# %% Original notebook cell 9
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/alloy_screening'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

DATA_PATH = str(PROJECT_ROOT / 'data/research/screening/alloy_2_et.csv')
# Read data from the CSV file
df = pd.read_csv(DATA_PATH, low_memory=False)

# Extract element names as x- and y-axis labels
x_labels = []
y_labels = []
for _, row in df.iterrows():
    x, y = row['elements'].split('-')
    x_labels.append(x)
    y_labels.append(y)

# Store Tc values
tc_values = defaultdict(lambda: defaultdict(float))
for i, (x, y) in enumerate(zip(x_labels, y_labels)):
    tc_values[x][y] = round(df.loc[i, 'Predicted_Tc'], 2)  # round to two decimal places
    tc_values[y][x] = round(df.loc[i, 'Predicted_Tc'], 2)  # round to two decimal places

# Convert dictionaries to NumPy arrays for plotting
element_unique = sorted(list(tc_values.keys()))
n_ele = len(element_unique)

# Create an empty matrix for Tc values
matrix = np.full((n_ele, n_ele), np.nan)

# Fill off-diagonal matrix positions
for i, x in enumerate(element_unique):
    for j, y in enumerate(element_unique):
        if i!=j:
            matrix[i, j] = tc_values[x][y]


# Create a heatmap
fig_height = 6 #cm
fig_width = 6 #cm
# Initialize the plot
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
plt.figure(figsize=(fig_width/2.54, fig_height/2.54))
heatmap = plt.imshow(matrix, cmap='cividis', interpolation='none')

f_size = 8
# Add a colorbar
cbar = plt.colorbar(heatmap,orientation='vertical',pad=0.01,shrink=0.75)  # adjust colorbar spacing
cbar.set_ticks(np.arange(0, 41, 10))
cbar.set_label('Predicted $\mathit{T}_{\mathit{c}}$ (K)', fontsize=f_size)
cbar.ax.tick_params(labelsize=f_size)  # set tick-label font size

# # Set plot title and axis labels
# plt.xlabel('Elements', fontsize=f_size)
# plt.ylabel('Elements', fontsize=f_size)

# # Set x- and y-axis ticks
# plt.xticks(range(n_ele), element_unique, rotation=89, fontsize=12)
# plt.yticks(range(n_ele), element_unique, fontsize=12)
plt.axis('off')


# Adjust plot margins
plt.tight_layout()

# Show the plot
plt.savefig(str(PLOT_OUTPUT / 'fig4c_2.png'), dpi=300, bbox_inches='tight',transparent=True)
plt.show()

# %% Original notebook cell 11
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/alloy_screening'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from collections import defaultdict
import matplotlib.cm as cm

# Read the CSV file
DATA_PATH = str(PROJECT_ROOT / 'data/research/screening/alloy_3_rf.csv')
df = pd.read_csv(DATA_PATH, low_memory=False)

# Extract all element names
e1_labels = []
e2_labels = []
e3_labels = []
for _, row in df.iterrows():
    e1, e2, e3 = row['elements'].split('-')
    e1_labels.append(e1)
    e2_labels.append(e2)
    e3_labels.append(e3)

# Store Tc values
tc_values = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
for i, (e1, e2, e3) in enumerate(zip(e1_labels, e2_labels, e3_labels)):
    tc_value = round(df.loc[i, 'Predicted_Tc'], 2)
    tc_values[e1][e2][e3] = tc_value
    tc_values[e1][e3][e2] = tc_value
    tc_values[e2][e1][e3] = tc_value
    tc_values[e2][e3][e1] = tc_value
    tc_values[e3][e1][e2] = tc_value
    tc_values[e3][e2][e1] = tc_value

# Deduplicate and sort element names
unique_elements = sorted(set(e1_labels + e2_labels + e3_labels))
element_to_index = {e: i for i, e in enumerate(unique_elements)}
n_ele = len(unique_elements)

# Build coordinates
x, y, z, tc_vals = [], [], [], []
for e1, nested1 in tc_values.items():
    for e2, nested2 in nested1.items():
        for e3, tc in nested2.items():
            if e1 in element_to_index and e2 in element_to_index and e3 in element_to_index:
                # Map elements to coordinate axes (0-65)
                x.append(element_to_index[e1])
                y.append(element_to_index[e2])
                z.append(element_to_index[e3])
                tc_vals.append(tc)

x = np.array(x)
y = np.array(y)
z = np.array(z)
tc_vals = np.array(tc_vals)

mask = (x <= 30) | (y <= 30) | (z <= 20)
x_masked = x[mask]
y_masked = y[mask]
z_masked = z[mask]
tc_vals_masked = tc_vals[mask]

# Create a 3D scatter plot
fig_height = 7 #cm
fig_width = 7 #cm
# Initialize the plot
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
fig = plt.figure(figsize=(fig_width/2.54, fig_height/2.54))
ax = fig.add_subplot(111, projection='3d')
# original_cmap = cm.turbo  # predefined
# colors = original_cmap(np.linspace(0, 1, original_cmap.N))  # get colors
# colors[:, -1] = colors[:, -1] * 0.95  # -1 is the alpha channel; previous columns are RGB values
# colors[:, :3] *= 0.9 # reduce brightness
# color_map = LinearSegmentedColormap.from_list("user", colors)
# color_map.set_bad(color='none')  # set NaN values to transparent
# Plot the sliced scatter points
sc = ax.scatter(x_masked, y_masked, z_masked, c=tc_vals_masked, cmap='cividis', s=5)

# Add a colorbar
f_size = 8
cbar = plt.colorbar(sc, orientation='vertical', pad=-0.01, shrink=0.6)
cbar.set_ticks(np.arange(0, 41, 10))
cbar.set_label('Predicted $\mathit{T}_{\mathit{c}}$ (K)', fontsize=f_size)
cbar.ax.tick_params(labelsize=f_size)  # set tick-label font size

# Set view angle and labels
ax.view_init(elev=30, azim=60)  # view rotation angle
# ax.set_xticks(range(n_ele))
# ax.set_xticklabels(unique_elements, fontsize=12)
# ax.set_yticks(range(n_ele))
# ax.set_yticklabels(unique_elements, fontsize=12)
# ax.set_zticks(range(n_ele))
# ax.set_zticklabels(unique_elements, fontsize=12)
# ax.set_xlabel('Element 1')
# ax.set_ylabel('Element 2')
# ax.set_zlabel('Element 3',rotation=90)
plt.axis('off')
# Adjust plot margins
plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig4d_3.png'), dpi=300, bbox_inches='tight',transparent=True)
plt.show()

# %% Original notebook cell 13
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
# Describe prediction accuracy and residuals
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def right_count(y_pred, y_true):
    relative_error = abs(y_pred - y_true)
    return (relative_error < 5).sum().item()

def mare_c(y_true, y_pred):
    # Use only samples with true values greater than 10
    mask = (y_true > 5)
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]

    if len(y_true_nonzero) == 0:
        return float('nan')  # Return NaN if there are no nonzero samples

    return np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero))

def smape_c(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return np.mean(diff / denominator) * 100  # return as a percentage

# Plot the scatter plot
def performancecalculator(trueVals, predVals, name):
    fig_height = 5  #cm
    fig_width = 8.5 #cm
    plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
    plt.figure(figsize=(fig_width/2.54, fig_height/2.54))
    # mt = max(trueVals)
    # mp = max(predVals)
    mt = 150
    mp = 150
    plt.subplot(1, 2, 1)
    plt.scatter(trueVals, predVals, color='#1f77b4', label='Predictions',s=3,marker='o',edgecolor='none')
    plt.plot([0, mt], [0, mp], 'k--', lw=0.5, label='Ideal fit')
    accuracy = right_count(predVals, trueVals) / len(predVals) * 100
    plt.title(
        f'Model: {name}\n'
        f'Test Accuracy: {accuracy:.2f}%',
        fontsize=8)
    plt.legend(fontsize=5)
    plt.xlabel('Experiment Value (K)',fontsize=8)
    plt.ylabel('Predicted Value (K)',fontsize=8,labelpad=-0.5)
    plt.xticks(fontsize=6)  # ensure that all x-axis labels are visible
    plt.yticks(fontsize=6)  # set the y-axis tick-label font size
    plt.xticks(range(0, mt+1, 30))
    plt.yticks(range(0, mt+1, 30))


    plt.subplot(1, 2, 2)
    residuals = predVals - trueVals
    mare = mare_c(trueVals, predVals)
    smape = smape_c(trueVals, predVals)
    rmse = np.sqrt(mean_squared_error(trueVals, predVals))
    r2 = r2_score(trueVals, predVals)
    plt.title(
        f'RMSE: {rmse:.4f}  $R^2$: {r2:.4f}\n'
        f'MARE: {mare:.4f}  SMAPE: {smape:.2f}%',
        fontsize=8, loc='left', x = -3e-1)
    plt.scatter(trueVals, residuals, color='#ff7f0e', label='Residuals',s=3,marker='o',edgecolor='none')
    plt.hlines(y=0, xmin=0, xmax=mt, colors='black', linestyles='dashed',lw=0.5)
    plt.hlines(y=5, xmin=0, xmax=mt, colors='black', linestyles='solid',lw=0.5)
    plt.hlines(y=-5, xmin=0, xmax=mt, colors='black', linestyles='solid',lw=0.5)
    plt.xlabel('Experiment Value (K)',fontsize=8)
    plt.ylabel('Residual (K)',fontsize=8,labelpad=-0.5)
    plt.xticks(fontsize=6)  # ensure that all x-axis labels are visible
    plt.yticks(fontsize=6)  # set the y-axis tick-label font size
    plt.xticks(range(0, mt+1, 30))
    plt.yticks(range(-90, 91, 30))
    plt.legend(fontsize=5)
    # Adjust layout
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT / f'fig5_{name.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight', pad_inches=0.05,transparent=True)
    plt.show()

# file_path = str(PROJECT_ROOT / 'data/research/prediction/updated_mdr_rf.csv')
# name = 'Random Forest'
file_path = str(PROJECT_ROOT / 'data/research/prediction/updated_mdr_et.csv')
name = 'Extra-Trees'

data = pd.read_csv(file_path)
performancecalculator(data['Tc'], data['Predicted'], name)

# %% Original notebook cell 15
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    'LM-based material': ['GaIn$_{20.5}$Sn$_{12.5}$', 'GaIn$_{15}$Sn$_7$', 'GaIn$_{47}$Sn$_{23}$'],
    'RF-Tc (K)': [4.44, 4.48, 5.07],
    'ET-Tc (K)': [5.66, 5.39, 6.37],
    'Tc (K)': [6.6, 6.0, 6.6]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set bar positions and width
n = len(df)
bar_width = 0.15
index = np.arange(n)

# Plot the bar chart
fig_height = 4.5  #cm
fig_width = 8 #cm
plt.rcParams['font.family'] = 'Arial'  # set the font to Arial
fig, ax = plt.subplots(figsize=(fig_width/2.54, fig_height/2.54))

# Draw each bar
bar1 = ax.bar(index, df['RF-Tc (K)'], bar_width, label='Random Forest')
bar2 = ax.bar(index + bar_width, df['ET-Tc (K)'], bar_width, label='Extra-Trees')
bar3 = ax.bar(index + 2 * bar_width, df['Tc (K)'], bar_width, label='Experiment')

# Set x-axis labels
ax.set_xticks(index + bar_width)
ax.set_xticklabels(df['LM-based material'], rotation=0, ha='center',fontsize=8)

# Set y-axis labels
ax.set_ylabel('$\mathit{T}_{\mathit{c}}$(K)', fontsize=8)
ax.set_ylim(0, 7)  # set the y-axis range from 0 to 7
plt.yticks(np.arange(0, 8, 1), fontsize=8)  # set y-axis ticks from 0 to 7

# Add and position the legend
# ax.legend(loc='upper right', bbox_to_anchor=(0.548, 1.02),fontsize=6,framealpha=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=6, framealpha=0, ncol=3)

# Show the plot
plt.tight_layout()
# Save the plot
plt.savefig(str(PLOT_OUTPUT / 'fig5d.png'), dpi=300,  bbox_inches='tight', transparent=True)
plt.show()

# %% Original notebook cell 17
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style
sns.set_style("white")
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 6,
    'font.family': 'Arial'
})

# Data
data = {
    'formula': ['Ga0.772In0.143Sn0.085', 'Ga0.855In0.100Sn0.045', 'Ga0.416In0.396Sn0.187'],
    'alloy%': ['GaIn$_{20.5}$Sn$_{12.5}$', 'GaIn$_{15}$Sn$_7$', 'GaIn$_{47}$Sn$_{23}$'],
    'Tc': [6.6, 6, 6.6],
    'ET_element': [5.4375, 5.4557, 6.1999],
    'ET_alloy%': [4.6849, 4.6920, 4.4079],
    'ET_element_wt': [4.9588, 5.1160, 5.6945],
    'ET_alloy%_wt': [7.7406, 7.7715, 7.0238]
}

df = pd.DataFrame(data)

# Calculate deviations from Tc
methods = ['ET_element', 'ET_alloy%', 'ET_element_wt', 'ET_alloy%_wt']
for method in methods:
    df[f'dev_{method}'] = df[method] - df['Tc']

# Prepare data
dev_columns = [f'dev_{m}' for m in methods]
labels = ['at.\nData-p', 'wt%\nData-p', 'at.\nData-n', 'wt%\nData-n']
colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78']
bar_width = 0.22
index = range(len(df))

# Create the canvas
fig_height = 5.5  # cm
fig_width = 8    # cm
fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))

# Plot deviation bars for each method
for i, dev_col in enumerate(dev_columns):
    bar_positions = [x + i * bar_width for x in index]
    ax.bar(bar_positions, df[dev_col], width=bar_width, color=colors[i],
           edgecolor='none',label=labels[i])

# Plot the zero baseline (Tc)
ax.axhline(0, color='black', linewidth=1, linestyle='--')

# Labels and title
ax.set_ylabel('$\Delta$ $\\mathit{T}_\\mathit{c}$ (K)',labelpad=0)
ax.set_xticks([r + bar_width for r in index])
ax.set_xticklabels(df['alloy%'], rotation=0, ha='center')
ax.set_yticks(np.arange(-2, 3, 1))
ax.tick_params(axis='y', which='major', length=3, color='black',direction='out',left=True)
ax.tick_params(axis='x', which='major', length=3, color='black',direction='out',bottom=True)
# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = ax.legend(by_label.values(), by_label.keys(), loc='upper center',
          bbox_to_anchor=(0.47, 1.2), ncol=4, frameon=False, fontsize=6)
for txt in legend.get_texts():
    txt.set_ha('center')  # horizontally centered
    txt.set_va('bottom')  # vertically aligned to the bottom

# Polish layout
plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig7a.png'), dpi=300,  bbox_inches='tight', transparent=True)
plt.show()

# %% Original notebook cell 19
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("white")
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 6,
    'font.family': 'Arial'
})

# Data
data = {
    'formula': ['Ga0.772In0.143Sn0.085', 'Ga0.855In0.100Sn0.045', 'Ga0.416In0.396Sn0.187'],
    'alloy%': ['GaIn$_{20.5}$Sn$_{12.5}$', 'GaIn$_{15}$Sn$_7$', 'GaIn$_{47}$Sn$_{23}$'],
    'Tc': [6.6, 6, 6.6],
    'ET_element': [5.4375, 5.4557, 6.1999],
    'ET_element_os': [6.8664,8.2514,6.0858],
    'ET_element_od': [21.8546,25.0627,23.7313]
}

df = pd.DataFrame(data)

# Calculate deviations from Tc
methods = ['ET_element', 'ET_element_os', 'ET_element_od']
for method in methods:
    df[f'dev_{method}'] = df[method] - df['Tc']

# Prepare data
dev_columns = [f'dev_{m}' for m in methods]
labels = ['Original', 'Duplicated', 'SMOTE']
bar_width = 0.28
index = range(len(df))

# Create the canvas
fig_height = 5.5  # cm
fig_width = 8    # cm
fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))

# Plot deviation bars for each method
for i, dev_col in enumerate(dev_columns):
    bar_positions = [x + i * bar_width for x in index]
    ax.bar(bar_positions, df[dev_col], width=bar_width,
           edgecolor='none',label=labels[i])

# Plot the zero baseline (Tc)
ax.axhline(0, color='black', linewidth=1, linestyle='--')

# Labels and title
ax.set_ylabel('$\Delta$ $\\mathit{T}_\\mathit{c}$ (K)',labelpad=0)
ax.set_xticks([r + bar_width for r in index])
ax.set_xticklabels(df['alloy%'], rotation=0, ha='center')
ax.tick_params(axis='y', which='major', length=3, color='black',direction='out',left=True)
ax.tick_params(axis='x', which='major', length=3, color='black',direction='out',bottom=True)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = ax.legend(by_label.values(), by_label.keys(), loc='upper center',
          bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=6)

# Polish layout
plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig7b.png'), dpi=300,  bbox_inches='tight', transparent=True)
plt.show()

# %% Original notebook cell 21
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("white")
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 6,
    'font.family': 'Arial'
})

# Data
data = {
    'formula': ['La2.85Pr0.15Ni2O7', 'Sm0.95La0.05O0.85F0.15Fe1As1', 'Li0.9N0.5H1.5Fe2Se2', 'Sr2Cu1O3.3', 'Y1Sr2Cu2.75Mo0.25O7.54'],
    'alloy%': ['LaPr$_{5.26}$Ni$_{6.99}$O$_{24.55}$', 'Sm$_{95}$La$_{5}$O$_{85}$F$_{15}$Fe$_{100}$As$_{100}$', 'Li$_{60}$N$_{33.3}$H$_{100}$Fe$_{133}$Se$_{133}$', 'Sr$_{66.7}$Cu$_{33.3}$O$_{100}$', 'Y$_{33.3}$Sr$_{66.7}$Cu$_{92}$Mo$_{8.7}$O$_{251}$'],
    'Tc': [45, 57.3, 44, 95, 84],
    'ET_element': [2.023323333, 45.10094444, 41.6175787, 77.88297222, 70.88338889],
    'ET_element_os': [10.59883333, 10.76864141, 33.02027778, 28.3875, 47.09933333],
    'ET_element_od': [1.507652778, 23.85652611, 17.60869093, 74.24161111, 55.46295278]

}

df = pd.DataFrame(data)

# Calculate deviations from Tc
methods = ['ET_element', 'ET_element_os', 'ET_element_od']
for method in methods:
    df[f'dev_{method}'] = df[method] - df['Tc']

# Prepare data
dev_columns = [f'dev_{m}' for m in methods]
labels = ['Original', 'Duplicated', 'SMOTE']
bar_width = 0.3
index = range(len(df))

# Create the canvas
fig_height = 5.7  # cm
fig_width = 8    # cm
fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))

# Plot deviation bars for each method
for i, dev_col in enumerate(dev_columns):
    bar_positions = [x + i * bar_width for x in index]
    ax.bar(bar_positions, df[dev_col], width=bar_width,
           edgecolor='none',label=labels[i])

# Plot the zero baseline (Tc)
ax.axhline(0, color='black', linewidth=1, linestyle='--')

# Labels and title
ax.set_ylabel('$\Delta$ $\\mathit{T}_\\mathit{c}$ (K)',labelpad=0)
ax.set_xticks([r + bar_width for r in index])
ax.set_xticklabels(['Ni-based','Fe-based','Fe-based','Cu-based','Cu-based'], rotation=-15, ha='center')
ax.tick_params(axis='y', which='major', length=3, color='black',direction='out',left=True)
ax.tick_params(axis='x', which='major', length=3, color='black',direction='out',bottom=True)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = ax.legend(by_label.values(), by_label.keys(), loc='upper center',
          bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=6)

# Polish layout
plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig7c.png'), dpi=300,  bbox_inches='tight', transparent=True)
plt.show()

# %% Original notebook cell 23
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT = PROJECT_ROOT / 'paper/figures/model_comparison'
PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("white")
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 6,
    'font.family': 'Arial'
})

# Data
data = {
    'formula': ['Al$_2$O$_3$', 'Co', 'Ni', 'Cu$_{0.9}$Sn$_{0.1}$', 'Cu$_{0.7}$Zn$_{0.3}$'],
    'Tc': [0, 0, 0, 0, 0],
    'ET_element': [1.183454192, 0.683332143, 0.917555833, 0.99501, 1.234167556],
    'ET_element_os': [1.446222222, 3.209563492, 3.564603175,   2.154444444, 1.349444444],
    'ET_element_od': [2.150907146,  1.153587302, 0.989859028,  1.530853333, 9.318015833]
}

df = pd.DataFrame(data)

# Calculate deviations from Tc
methods = ['ET_element', 'ET_element_os', 'ET_element_od']
for method in methods:
    df[f'dev_{method}'] = df[method] - df['Tc']

# Prepare data
dev_columns = [f'dev_{m}' for m in methods]
labels = ['Original', 'Duplicated', 'SMOTE']
bar_width = 0.3
index = range(len(df))

# Create the canvas
fig_height = 4.5  # cm
fig_width = 8    # cm
fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))

# Plot deviation bars for each method
for i, dev_col in enumerate(dev_columns):
    bar_positions = [x + i * bar_width for x in index]
    ax.bar(bar_positions, df[dev_col], width=bar_width,
           edgecolor='none',label=labels[i])

# Plot the zero baseline (Tc)
ax.axhline(0, color='black', linewidth=1, linestyle='--')

# Labels and title
ax.set_ylabel('$\Delta$ $\\mathit{T}_\\mathit{c}$ (K)',labelpad=0)
ax.set_xticks([r + bar_width for r in index])
ax.set_xticklabels(df['formula'], rotation=-15, ha='center')
ax.tick_params(axis='y', which='major', length=3, color='black',direction='out',left=True)
ax.tick_params(axis='x', which='major', length=3, color='black',direction='out',bottom=True)


# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = ax.legend(by_label.values(), by_label.keys(), loc='upper center',
          bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=6)

# Polish layout
# plt.tight_layout()
plt.savefig(str(PLOT_OUTPUT / 'fig7d.png'), dpi=300,  bbox_inches='tight', transparent=True)
plt.show()
