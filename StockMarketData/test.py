# Re-import necessary libraries since the previous code execution environment is lost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# Recreate the initial dataset
data = {
    "income": [50000, 60000, 25000, 80000, 120000, 30000, 45000, 70000, 90000, 40000],
    "credit_score": [700, 650, 600, 750, 800, 580, 630, 720, 770, 590],
    "loan_amount": [20000, 15000, 25000, 30000, 40000, 10000, 18000, 22000, 35000, 12000],
    "loan_term": [5, 3, 7, 5, 10, 4, 6, 5, 7, 3],
    "approved": [1, 1, 0, 1, 1, 0, 0, 1, 1, 0]  # 1 = Approved, 0 = Denied
}

df = pd.DataFrame(data)

# Split dataset
X = df.drop("approved", axis=1)
y = df["approved"]

# Define entropy function
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Define information gain function
def information_gain(y, y_left, y_right):
    parent_entropy = entropy(y)
    left_entropy = entropy(y_left)
    right_entropy = entropy(y_right)

    left_weight = len(y_left) / len(y)
    right_weight = len(y_right) / len(y)

    return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

# Define decision tree node class
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Define best split function
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    best_left = None
    best_right = None

    for feature in X.columns:
        thresholds = np.unique(X[feature])
        for threshold in thresholds:
            left_mask = X[feature] <= threshold
            right_mask = ~left_mask

            y_left, y_right = y[left_mask], y[right_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gain = information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left, best_right = (X[left_mask], y_left), (X[right_mask], y_right)

    return best_feature, best_threshold, best_gain, best_left, best_right

# Define recursive tree building function
def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1:
        return DecisionTreeNode(value=y.iloc[0])

    if depth >= max_depth:
        most_common_label = y.value_counts().idxmax()
        return DecisionTreeNode(value=most_common_label)

    best_feature, best_threshold, best_gain, best_left, best_right = best_split(X, y)

    if best_gain == 0:
        most_common_label = y.value_counts().idxmax()
        return DecisionTreeNode(value=most_common_label)

    left_subtree = build_tree(best_left[0], best_left[1], depth + 1, max_depth)
    right_subtree = build_tree(best_right[0], best_right[1], depth + 1, max_depth)

    return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Define prediction function
def predict(tree, x):
    if tree.value is not None:
        return tree.value

    if x[tree.feature] <= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)

def predict_batch(tree, X):
    return X.apply(lambda row: predict(tree, row), axis=1)

# Creating a new test dataset with challenging cases
test_data = {
    "income": [35000, 110000, 48000, 75000, 26000, 95000, 55000, 40000, 87000, 62000],
    "credit_score": [620, 790, 640, 710, 590, 780, 670, 600, 750, 630],
    "loan_amount": [15000, 38000, 22000, 29000, 13000, 40000, 21000, 17000, 33000, 14000],
    "loan_term": [3, 10, 5, 6, 4, 8, 5, 3, 7, 4],
    "approved": [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # 1 = Approved, 0 = Denied
}

# Convert to DataFrame
df_test = pd.DataFrame(test_data)

# Split test dataset
X_test = df_test.drop("approved", axis=1)
y_test = df_test["approved"]

# Build the decision tree model
tree = build_tree(X, y, max_depth=3)

# Make predictions on the test dataset
y_pred = predict_batch(tree, X_test)

# Compare predictions with actual values
df_test["predicted"] = y_pred


accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Plot actual vs predicted results for visual comparison
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), y_test, label="Actual", marker='o', color='blue')
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", marker='x', color='red')
plt.xlabel("Test Sample Index")
plt.ylabel("Approval (1 = Approved, 0 = Denied)")
plt.title("Loan Approval: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()


import graphviz

def visualize_tree(tree, feature_names, dot=None, depth=0, parent_name="Root", label=""):
    """Recursively visualize the decision tree using Graphviz."""
    if dot is None:
        dot = graphviz.Digraph(format="png")
        dot.node(name="Root", label="Start", shape="ellipse")

    if tree.value is not None:  # Leaf node
        dot.node(name=str(id(tree)), label=f"Class: {tree.value}", shape="box", style="filled", fillcolor="lightgrey")
        dot.edge(parent_name, str(id(tree)), label=label)
    else:
        dot.node(name=str(id(tree)), label=f"{tree.feature} ≤ {tree.threshold:.2f}", shape="ellipse")
        if depth > 0:
            dot.edge(parent_name, str(id(tree)), label=label)

        visualize_tree(tree.left, feature_names, dot, depth + 1, str(id(tree)), "Yes")
        visualize_tree(tree.right, feature_names, dot, depth + 1, str(id(tree)), "No")

    return dot

# Visualizing the decision tree
tree_visual = visualize_tree(tree, X.columns)
tree_visual_data = tree_visual.pipe(format='png')


tree_visual.render(filename="decision_tree", format="png", cleanup=True)



