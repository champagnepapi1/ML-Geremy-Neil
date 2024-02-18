import numpy as np
import matplotlib.pyplot as plt

mean_X = 30
std_X = 10
mean_Y = 170
std_Y = 15

def empirical_average(samples_X, samples_Y, n):
    return np.array([np.mean(samples_X[:n]), np.mean(samples_Y[:n])])

print("1. Defining Random Variables")
print(f"X (Age): Mean = {mean_X}, Std Dev = {std_X}")
print(f"Y (Height): Mean = {mean_Y}, Std Dev = {std_Y}\n")

n = 1000
X_samples = np.random.normal(mean_X, std_X, n)
Y_samples = np.random.normal(mean_Y, std_Y, n)

print(f"2. Sampling {n} Points from the Distribution of Z")
print(np.column_stack((X_samples[:5], Y_samples[:5])), "\n")

expected_value = np.array([mean_X, mean_Y])
print("3. Value of Z")
print(f"Z = {expected_value}\n")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_samples, Y_samples, alpha=0.6)
plt.title('Sampled Points from the Distribution of Z')
plt.xlabel('Age (years)')
plt.ylabel('Height (cm)')
plt.grid(True)

print("4. Empirical Averages and Euclidean Distances for Increasing n")
n_values = range(1, n+1)
distances = []

for n_value in n_values:
    emp_avg = empirical_average(X_samples, Y_samples, n_value)
    distance = np.linalg.norm(emp_avg - expected_value)
    distances.append(distance)
    if n_value in [1, 10, 100, 500, 1000]:
        print(f"n = {n_value}, Empirical Average = {emp_avg}, Euclidean Distance = {distance}")

plt.subplot(1, 2, 2)
plt.plot(n_values, distances)
plt.title('Euclidean Distance vs Sample Size')
plt.xlabel('Sample Size (n)')
plt.ylabel('Euclidean Distance')
plt.grid(True)

plt.tight_layout()
plt.show()
