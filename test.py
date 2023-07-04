import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a grid of subplots
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 3)

# Create subplots in the grid
subplots = []
for i in range(3):
    for j in range(3):
        subplots.append(fig.add_subplot(gs[i-1, j-1]))

# Create subplots for row labels
row_labels = [fig.add_subplot(gs[i, 0]) for i in range(3)]

# Create subplots for column labels
col_labels = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Set row and column labels
row_labels[0].set_ylabel('Row 1')
row_labels[1].set_ylabel('Row 2')
row_labels[2].set_ylabel('Row 3')

col_labels[0].set_xlabel('Column 1')
col_labels[1].set_xlabel('Column 2')
col_labels[2].set_xlabel('Column 3')

# Set titles for subplots
for i, ax in enumerate(subplots):
    ax.set_title(f'Plot {i+1}')

# Add plot content
for ax in subplots:
    ax.plot([1, 2, 3], [4, 5, 6])

# Adjust spacing
plt.tight_layout()

# Show the plot
plt.show()
