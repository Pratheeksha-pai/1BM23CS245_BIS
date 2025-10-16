import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Step 1: Create dataset
np.random.seed(42)
X = np.linspace(-10, 10, 200).reshape(-1, 1)
y = X[:, 0]**2 + 2 * X[:, 0] + 1 + np.random.normal(0, 5, X.shape[0])  # add noise

# Step 2: Initialize the symbolic regressor
model = SymbolicRegressor(
    population_size=1000,
    generations=10,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.001,
    random_state=42
)

# Step 3: Train the model
model.fit(X, y)

# Step 4: Predict
y_pred = model.predict(X)

# Step 5: Display the evolved formula
print("\n Discovered Expression:")
print(model._program)

# Step 6: Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual Data", alpha=0.5)
plt.plot(X, y_pred, color='red', label="GEP Predicted")
plt.title("Symbolic Regression via Genetic Programming")
plt.legend()
plt.grid(True)
plt.show()
