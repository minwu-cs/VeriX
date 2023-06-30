import matplotlib.pyplot as plt

# Define the number of rows and columns in the grid
num_rows = 3
num_cols = 4

# Create the grid of subplots
fig, axes = plt.subplots(num_rows, num_cols)

# Add content to the subplots (e.g., plot data)
for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        ax.plot([1, 2, 3], [4, 5, 6])  # Example plot data

        # Add row and column labels
        ax.text(0.5, 0.5, f'Row {i+1}\nCol {j+1}', ha='center', va='center', transform=ax.transAxes)

# Add overall row and column labels
for i in range(num_rows):
    fig.text(0.06, (i+0.5) / num_rows, f'Row {i+1}', ha='center', va='center', rotation='vertical')
for j in range(num_cols):
    fig.text((j+0.5) / num_cols, 0.94, f'Col {j+1}', ha='center', va='center')

# Adjust the layout
# fig.tight_layout()

# Show the plot
plt.show()
