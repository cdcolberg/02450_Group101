from load_data import load_data
import importlib_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd


data, target = load_data() # returns a pandas dataframe

n_rows, n_cols = data.shape
print(n_rows, n_cols)

data_values = data.values

# Making the data matrix X by indexing into data.
cols = range(0, 29)

X = data_values[:, cols]

# Extracting the attribute names from the header
attributeNames = np.asarray(data.columns[cols])

#Finding and determining the unique class names in the target data
classNames = np.unique(target)

#Assigning each class with a number by making a Python dictionary
classDict = dict(zip(classNames, range(len(classNames))))

#Making class index vector y:
y = np.array([classDict[cl] for cl in target])

# Finding the number of data objects and number of attributes using the shape of X
N, M = X.shape

#Finding number of classes, C:
C = len(classNames)

# missing values:
missing_idx = np.isnan(data)
obs_w_missing = np.sum(missing_idx, 1) > 0

#Plot over missing values
plt.title("Visual inspection of missing values")
plt.imshow(missing_idx)
plt.ylabel("Observations")
plt.xlabel("Attributes")
plt.show()

# Compute values for every attribute
for i, attribute in enumerate(attributeNames):
    x = X[:, i]  # Extracting individual attribute column
    
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1)  # ddof: Delta Degrees of Freedom
    std_b_x = np.std(x, ddof=0)
    median_x = np.median(x)
    range_x = np.max(x) - np.min(x)
    
    # Display results
    print(f"Attribute: {attribute}")
    print(f"  Mean: {mean_x}")
    print(f"  Standard Deviation: {std_x}")
    print(f"  Standard Deviation (biased): {std_b_x}")
    print(f"  Median: {median_x}")
    print(f"  Range: {range_x}")

u = int(np.floor(np.sqrt(M)))
v = int(np.ceil(M / u))

# Use a colormap to assign different colors as we are dealing with 29 attributes
colors = cm.viridis(np.linspace(0, 1, M))  # Generates M unique colors

for i in range(M):
    plt.subplot(u, v, i + 1)
    plt.hist(X[:, i], color=colors[i], edgecolor="black", alpha=0.7)  # Use different colors
    plt.xlabel(attributeNames[i], fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, N / 2)  # Limit y-axis
    
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Bar chart for target distribution
plt.figure(figsize=(6, 5))  

# Class count
_, class_counts = np.unique(target, return_counts=True)

# Plot bar chart
plt.bar(classNames, class_counts, color='skyblue', edgecolor="black", alpha=0.7)
plt.xlabel("Target Categories", fontsize=10)
plt.ylabel("Frequency", fontsize=10)
plt.title("Distribution of Target Categories")
plt.xticks(classNames, fontsize=9)
plt.yticks(fontsize=9)

plt.show()


#Boxplot for all attributes in X
plt.figure(figsize=(10, 6))  
plt.boxplot(X) 
# adjusting x-axis labels to match dataset attributes
plt.xticks(range(1, X.shape[1] + 1), attributeNames, rotation=45)  # Rotate labels if long

plt.ylabel("Value") 
plt.title("Boxplot of Dataset Attributes") 
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add a light grid for readability
plt.show()

# Standardize X
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)  # Transformed dataset (mean=0, std=1)
# Can also be done by useing 
# from scipy.stats import zscore
#_standarized = zscore(X, ddof=1)  

# Boxplot for standardized attributes
plt.figure(figsize=(10, 6))
plt.boxplot(X_standardized)  # Now using standardized data
# Adjust x-axis labels
plt.xticks(range(1, X.shape[1] + 1), attributeNames, rotation=45)  

plt.ylabel("Standardized Value (Z-score)")  # Indicate standardization
plt.title("Standardized Boxplot of Dataset Attributes")  
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add a light grid

plt.show()


#Class boxplot
for c in range(C):
    plt.subplot(1, C, c + 1)  # Create subplot for each class
    class_mask = y == c  # Filter rows belonging to class c

    plt.boxplot(X[class_mask, :])  # Boxplot only for class c
    plt.title("Class: " + str(classNames[c]))  # Add title with class name

    # Set x-axis labels, truncate long names for readability
    plt.xticks(range(1, X.shape[1] + 1), [a[:7] for a in attributeNames], rotation=45)

    # Set consistent y-axis limits for comparison
    y_up = X.max() + (X.max() - X.min()) * 0.1
    y_down = X.min() - (X.max() - X.min()) * 0.1
    plt.ylim(y_down, y_up)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#BStadardized class boxplot
plt.figure(figsize=(15, 10))  # Adjust size

# Loop over each attribute to create a separate boxplot
for i in range(M):
    plt.subplot(int(np.ceil(M / 5)), 5, i + 1)  # Arrange in a grid
    for c in range(C):
        class_mask = y == c  # Filter by class
        plt.boxplot(X_standardized[class_mask, i], positions=[c + 1], widths=0.6)

    plt.title(attributeNames[i])  # Attribute name as title
    plt.xticks([1, 2], classNames, fontsize=8)  # Class labels
    plt.yticks(fontsize=8)
    plt.grid(axis='y', linestyle="--", alpha=0.5)  # Light grid

plt.tight_layout()
plt.show()

#Matrix of scatter plots, showing the relationship between every pair of attributes in X
for m1 in range(M):  
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        
        for c, color in zip(np.unique(y), colors):  #Loop through the unique class labels
            class_mask = y == c
            plt.scatter(X[class_mask, m2], X[class_mask, m1], color=color, s=5, alpha=0.6, label=classNames[c])
        
        if m1 == M - 1:
            plt.xlabel(attributeNames[m2], fontsize=8)
        else:
            plt.xticks([])

        if m2 == 0:
            plt.ylabel(attributeNames[m1], fontsize=8)
        else:
            plt.yticks([])
        
plt.legend(classNames, loc="upper right", bbox_to_anchor=(1.5, 1.0))
plt.tight_layout()
plt.show()

#Scatter plot of selected attribute relationships
# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)

# Find strongest correlations (absolute > 0.7)
strong_pairs = np.argwhere(np.abs(corr_matrix) > 0.7)

# Remove duplicate pairs and diagonal (self-correlation)
strong_pairs = [(i, j) for i, j in strong_pairs if i < j]

# Plot only the selected scatter plots
plt.figure(figsize=(12, 8))
for idx, (m1, m2) in enumerate(strong_pairs[:9]):  # Show top 9 pairs
    plt.subplot(3, 3, idx + 1)
    plt.scatter(X[:, m2], X[:, m1], alpha=0.5, s=5)
    plt.xlabel(attributeNames[m2])
    plt.ylabel(attributeNames[m1])
    plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(np.corrcoef(X.T), cmap="coolwarm", annot=False, xticklabels=attributeNames, yticklabels=attributeNames)
plt.title("Attribute Correlation Heatmap")
plt.show()


#3D scatter plot
# Extract unique feature indices from the strongest correlations
top_features = list(set(i for pair in strong_pairs for i in pair))[:3]

ind = top_features  # Set indices for 3D plot

# Use a colormap for class colors
colors = cm.viridis(np.linspace(0, 1, C))

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Loop through each class and plot
for c, color in zip(np.unique(y), colors):
    class_mask = y == c
    ax.scatter(
        X[class_mask, ind[0]], X[class_mask, ind[1]], X[class_mask, ind[2]], 
        color=color, label=classNames[c], alpha=0.7
    )

# Adjust viewing angle
ax.view_init(30, 220)

# Set axis labels dynamically based on selected features
ax.set_xlabel(attributeNames[ind[0]], fontsize=10)
ax.set_ylabel(attributeNames[ind[1]], fontsize=10)
ax.set_zlabel(attributeNames[ind[2]], fontsize=10)

ax.legend(loc="best")
plt.show()

#Another heatmap
plt.figure(figsize=(12, 6))
plt.imshow(X_standardized, interpolation="none", aspect="auto", cmap=plt.cm.viridis)  # Use 'viridis' for better contrast

# Adjust x-axis labels dynamically
plt.xticks(range(X.shape[1]), attributeNames, rotation=90)  # Rotate labels for readability

plt.xlabel("Attributes")
plt.ylabel("Data Objects")
plt.title("Standardized Data Matrix Heatmap")

# Add color scale
plt.colorbar(label="Z-score")
plt.show()


# Plotting the 5 strongest attribute pairs against eachother in a scatterplot
num_pairs_to_plot = min(5, len(strong_pairs))  

# Create subplots for multiple attribute pair scatter plots
fig, axes = plt.subplots(1, num_pairs_to_plot, figsize=(15, 5))  

# Loop through selected pairs and plot them
for idx, (i, j) in enumerate(strong_pairs[:num_pairs_to_plot]):
    ax = axes[idx] if num_pairs_to_plot > 1 else axes  # Handle single vs multiple subplots
    
    for c in range(C):
        class_mask = y == c  # Select data points belonging to class c
        ax.scatter(X[class_mask, i], X[class_mask, j], alpha=0.5, label=classNames[c])
    
    ax.set_xlabel(attributeNames[i])
    ax.set_ylabel(attributeNames[j])
    ax.set_title(f"{attributeNames[i]} vs {attributeNames[j]}")
    ax.legend()

plt.suptitle("Scatter Plots of Highly Correlated Feature Pairs")
plt.tight_layout()
plt.show()


#PCA on data

# Standardize X by centering around zero mean
Y = X - np.mean(X, axis=0)

# Perform PCA using SVD
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T  # Transpose to get correct PCA component orientations

# Compute variance explained by each principal component
rho = (S ** 2) / np.sum(S ** 2)

# Threshold for variance explained (e.g., 90%)
threshold = 0.9

# Plot variance explained
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual Variance")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative Variance")
plt.axhline(y=threshold, color="k", linestyle="--", label="90% Threshold")
plt.title("Variance Explained by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.legend()
plt.grid()
plt.show()

# Print how many components explain at least 90% of variance
num_components = np.argmax(np.cumsum(rho) >= threshold) + 1
print(f"Minimum number of principal components to reach {threshold * 100:.0f}% variance: {num_components}")


#Looking at the dfirst PCA's

Z = Y @ V  # Transformed data in PCA space
pc1, pc2 = 0, 1  # First two principal components

plt.figure(figsize=(8, 6))
plt.title("PCA Projection of Dataset")

# Scatter plot for each class
for c in range(C):
    class_mask = y == c
    plt.scatter(Z[class_mask, pc1], Z[class_mask, pc2], alpha=0.5, label=classNames[c])

# Labels, legend, and grid
plt.xlabel(f"Principal Component {pc1 + 1}")
plt.ylabel(f"Principal Component {pc2 + 1}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()

# Print variance explained
var_explained = (S ** 2) / np.sum(S ** 2)
print(f"PC1 explains {var_explained[0] * 100:.2f}% of variance")
print(f"PC2 explains {var_explained[1] * 100:.2f}% of variance")


#Looking further at the first three PCA's
# Selecting principal components to analyze
pcs = [0, 1, 2]  # First three PCs
legendStrs = [f"PC{e + 1}" for e in pcs]
bw = 0.2  # Bar width
r = np.arange(1, M + 1)  # Attribute indices

# Plot PCA Component Coefficients
plt.figure(figsize=(12, 6))
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw, alpha=0.7)

plt.xticks(r + bw, attributeNames, rotation=90)  # Rotate labels for clarity
plt.xlabel("Attributes")
plt.ylabel("Component Coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("PCA Component Coefficients")
plt.show()

# Print PC2 Coefficients
print("PC2 Coefficients:")
print(V[:, 1].T)

# Project a selected class (e.g., first class) onto PC2
selected_class = 0  # Change this to analyze other classes
selected_class_data = Y[y == selected_class, :]

print(f"First observation of class {classNames[selected_class]}")
print(selected_class_data[0, :])

print(f"...and its projection onto PC2")
print(selected_class_data[0, :] @ V[:, 1])


# Comparing PCA with and without standarization

# Compute standard deviations for each feature
plt.figure(figsize=(10, 5))
plt.bar(np.arange(1, X.shape[1] + 1), np.std(X, axis=0))
plt.xticks(np.arange(1, X.shape[1] + 1), attributeNames, rotation=90)
plt.ylabel("Standard Deviation")
plt.xlabel("Attributes")
plt.title("Feature Standard Deviations")
plt.grid()
plt.show()

# Zero-mean data (not standardized)
Y1 = X - np.mean(X, axis=0)

# Standardized data (zero-mean, unit variance)
Y2 = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Store both versions
Ys = [Y1, Y2]
titles = ["Zero-mean", "Zero-mean and unit variance"]
threshold = 0.9
pc1, pc2 = 0, 1  # Principal components to plot

# Create figure
plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.4)
nrows, ncols = 3, 2  # Grid layout

for k in range(2):
    # PCA
    U, S, Vh = svd(Ys[k], full_matrices=False)
    V = Vh.T

    if k == 1:
        V = -V  # Flip to align with non-standardized PCA

    # Compute variance explained
    rho = (S ** 2) / np.sum(S ** 2)

    # Compute projection onto principal components
    Z = U * S

    # Plot PCA projections
    plt.subplot(nrows, ncols, 1 + k)
    for c in np.unique(y):
        plt.scatter(Z[y == c, pc1], Z[y == c, pc2], alpha=0.5, label=classNames[c])
    plt.xlabel(f"PC{pc1 + 1}")
    plt.ylabel(f"PC{pc2 + 1}")
    plt.legend()
    plt.title(f"{titles[k]}: PCA Projection")
    plt.axis("equal")

    # Plot attribute coefficients in PCA space
    plt.subplot(nrows, ncols, 3 + k)
    for att in range(V.shape[1]):
        plt.arrow(0, 0, V[att, pc1], V[att, pc2])
        plt.text(V[att, pc1], V[att, pc2], attributeNames[att])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel(f"PC{pc1 + 1}")
    plt.ylabel(f"PC{pc2 + 1}")
    plt.grid()
    plt.title(f"{titles[k]}: Attribute Coefficients")
    plt.axis("equal")

    # Plot cumulative variance explained
    plt.subplot(nrows, ncols, 5 + k)
    plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual Variance")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative Variance")
    plt.axhline(y=threshold, color="k", linestyle="--", label="90% Threshold")
    plt.title(f"{titles[k]}: Variance Explained")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.legend()
    plt.grid()

plt.show()

##
#PCA but on standarized data

# Standardize X using StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA using SVD
U, S, Vh = svd(X_std, full_matrices=False)
V = Vh.T  # Transpose to get correct PCA component orientations

# Compute variance explained by each principal component
rho = (S ** 2) / np.sum(S ** 2)

# Threshold for variance explained (e.g., 90%)
threshold = 0.9

# Plot variance explained
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual Variance")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative Variance")
plt.axhline(y=threshold, color="k", linestyle="--", label="90% Threshold")
plt.title("Variance Explained by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.legend()
plt.grid()
plt.show()

# Print how many components explain at least 90% of variance
num_components = np.argmax(np.cumsum(rho) >= threshold) + 1
print(f"Minimum number of principal components to reach {threshold * 100:.0f}% variance: {num_components}")

# Transform data to PCA space
Z = X_std @ V
pc1, pc2 = 0, 1  # First two principal components

plt.figure(figsize=(8, 6))
plt.title("PCA Projection of Dataset")

# Scatter plot for each class
for c in range(C):
    class_mask = y == c
    plt.scatter(Z[class_mask, pc1], Z[class_mask, pc2], alpha=0.5, label=classNames[c])

# Labels, legend, and grid
plt.xlabel(f"Principal Component {pc1 + 1}")
plt.ylabel(f"Principal Component {pc2 + 1}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Print variance explained
print(f"PC1 explains {rho[0] * 100:.2f}% of variance")
print(f"PC2 explains {rho[1] * 100:.2f}% of variance")

# Analyzing the first three PCs
pcs = [0, 1, 2]  # First three PCs
legendStrs = [f"PC{e + 1}" for e in pcs]
bw = 0.2  # Bar width
r = np.arange(1, M + 1)  # Attribute indices

# Plot PCA Component Coefficients
plt.figure(figsize=(12, 6))
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw, alpha=0.7)

plt.xticks(r + bw, attributeNames, rotation=90)
plt.xlabel("Attributes")
plt.ylabel("Component Coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("PCA Component Coefficients")
plt.show()

# Print PC2 Coefficients
print("PC2 Coefficients:")
print(V[:, 1].T)

# Project a selected class (e.g., first class) onto PC2
selected_class = 0  # Change this to analyze other classes
selected_class_data = X_std[y == selected_class, :]

print(f"First observation of class {classNames[selected_class]}")
print(selected_class_data[0, :])

print(f"...and its projection onto PC2")
print(selected_class_data[0, :] @ V[:, 1])
