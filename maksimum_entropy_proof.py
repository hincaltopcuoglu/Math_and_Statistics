"""
Problem Statement:
------------------

We aim to find the discrete probability distribution {p_i} that maximizes the Shannon entropy:

    H(p) = -∑ p_i * log(p_i)

subject to the following constraints:

1. Normalization constraint:
    ∑ p_i = 1

2. Moment (expectation) constraint:
    ∑ p_i * g_j(x_i) = M_j     for j = 1, 2, ..., m

Where:
- p_i are the unknown probabilities (p1, p2, ..., pn)
- g_j(x_i) is a known feature function applied to each outcome x_i
- M_j is the expected value of the j-th feature function under distribution {p_i}

This optimization problem will be solved using the method of Lagrange multipliers.
"""

import numpy as np
import sympy as sp
from scipy.optimize import root_scalar
sp.init_printing()


# Multipliers
lambda_ = sp.Symbol('lambda')
theta = sp.Symbol('theta')

# Feature function values and expected value
g_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
M = sp.Symbol('M')


def calculate_probs_and_entropy(n, p_random=None):
    # step 1: Define symbols
    # suppose we are working with discrete distribution over n outcomes, for example n = 10, ie. p1, p2, p3, ..., p10

    def entropy(probs):
        return -np.sum(probs * np.log(probs))


    # probabilities
    p = sp.symbols(f'p1:{n+1}')

    # step 2: generate random probabilities that sum to 1
    if p_random is None:
        rand_vals = np.random.rand(n)
        rand_probs = rand_vals / rand_vals.sum()
    else:
        rand_probs = p_random


    # step 3: Map to dictionary for substition
    p_values = {p[i] : rand_probs[i] for i in range(n)}
    probs = np.array([p_values[sym] for sym in p])
    entropy_val = entropy(probs)

    return p, p_values, entropy_val



# Create random distribution once
rand_vals = np.random.rand(10)
p_random = rand_vals / np.sum(rand_vals)


p, p_values, entropy_val = calculate_probs_and_entropy(10,p_random=p_random)

# Pretty print the probabilities
print("📊 Random Distribution:")
for sym in p:
    print(f"  {str(sym)} = {p_values[sym]:.4f}")
print(f"\n🧠 Entropy: {entropy_val:.4f}")



def lagrangian(n, x, lambda_, theta, M, p_random):
    probs = np.array(p_random)
    entropy_val = -np.sum(probs * np.log(probs))
    normalization_eq = -lambda_ * (np.sum(probs)-1)
    moment_eq = theta * (np.sum(probs * x) - M)
    return entropy_val + normalization_eq + moment_eq

L_val = lagrangian(10, np.arange(1, 11), 0.5, 0.1, 5.5, p_random)
print(f"\n📐 Lagrangian (using same random p): {L_val:.4f}")




# The Lagrangian value computed above is not the optimal solution.
# It is simply the value at randomly generated probabilities.
# 
# To find the probabilities that maximize entropy under constraints,
# we must take the partial derivatives of the Lagrangian with respect to each p_i,
# and solve the resulting system of equations:
#
#     ∂𝓛 / ∂p_i = -log(p_i) - 1 - λ - θ * x_i = 0     for i = 1 to n
#
# Solving this equation yields the maximum entropy distribution:
#
#     p_i = (1/Z) * exp(-θ * x_i)
#
# where Z is the normalization constant ensuring that ∑ p_i = 1:
#
#     Z = ∑ exp(-θ * x_i)


def lagrangian_symbolic(n):
    # step1: symbols
    p = sp.symbols(f'p1:{n+1}') # p1 to p10
    x = list(range(1, n+1))
    lambda_ = sp.Symbol('lambda')
    theta = sp.Symbol('theta')
    M = sp.Symbol('M')

    # step 2: lagrangian expression
    entropy_term = -sum(p[i] * sp.log(p[i]) for i in range(n))
    normaalization_term = -lambda_ * (sum(p) - 1)
    moment_term = -theta * (sum(p[i] * x[i] for i in range(n)) - M)
    L = entropy_term + normaalization_term + moment_term

    # Step 5: Derivatives
    dL_dp = [sp.diff(L, p[i]) for i in range(n)]        # ∂L/∂p_i
    dL_dlambda = sp.diff(L, lambda_)                    # ∂L/∂λ
    dL_dtheta = sp.diff(L, theta)                       # ∂L/∂θ

    # Display results
    print("🧮 Derivatives with respect to p_i:")
    for i in range(n):
        print(f"∂L/∂p{i+1} =", dL_dp[i])

    print("\n📏 Derivative with respect to λ (normalization):")
    print("∂L/∂λ =", dL_dlambda)

    print("\n📈 Derivative with respect to θ (moment constraint):")
    print("∂L/∂θ =", dL_dtheta)

    return {
        'L': L,
        'dL_dp': dL_dp,
        'dL_dlambda': dL_dlambda,
        'dL_dtheta': dL_dtheta
    }


derivatives = lagrangian_symbolic(10)


import numpy as np
from scipy.optimize import root_scalar

def solve_max_entropy(n=10, M_target=5.5):
    x = np.arange(1, n + 1)

    # ----------------------------
    # STEP 1: Random (original) distribution
    # ----------------------------
    rand_vals = np.random.rand(n)
    p_random = rand_vals / np.sum(rand_vals)
    entropy_random = -np.sum(p_random * np.log(p_random))
    expected_random = np.sum(p_random * x)

    print("🎲 Original Random Distribution (non-optimal):")
    for i, pi in enumerate(p_random, start=1):
        print(f"  p{i} = {pi:.4f}")
    print(f"  ∑p_i        = {np.sum(p_random):.6f}")
    print(f"  ∑p_i * x_i  = {expected_random:.6f}")
    print(f"  Entropy H(p_random) = {entropy_random:.6f}\n")

    # ----------------------------
    # STEP 2: Solve for θ to find max entropy distribution
    # ----------------------------
    def moment_constraint(theta):
        exp_terms = np.exp(-theta * x)
        Z = np.sum(exp_terms)
        p = exp_terms / Z
        expected_value = np.sum(p * x)
        return expected_value - M_target

    sol = root_scalar(moment_constraint, bracket=[-10, 10], method='brentq')
    if not sol.converged:
        raise RuntimeError("Failed to solve for theta.")
    theta_star = sol.root

    # ----------------------------
    # STEP 3: Compute optimal (max entropy) distribution
    # ----------------------------
    exp_terms = np.exp(-theta_star * x)
    Z = np.sum(exp_terms)
    p_opt = exp_terms / Z
    entropy_opt = -np.sum(p_opt * np.log(p_opt))
    expected_opt = np.sum(p_opt * x)

    # ----------------------------
    # STEP 4: Show optimal results
    # ----------------------------
    print(f"✅ Solved θ = {theta_star:.4f}")
    print("\n📊 Maximum Entropy Probabilities:")
    for i, pi in enumerate(p_opt, start=1):
        print(f"  p{i} = {pi:.4f}")
    print(f"\n📏 Check Constraints:")
    print(f"  ∑p_i        = {np.sum(p_opt):.6f} (should be 1)")
    print(f"  ∑p_i * x_i  = {expected_opt:.6f} (target M = {M_target})")
    print(f"🧠 Maximum Entropy H(p_opt) = {entropy_opt:.6f}")

    # ----------------------------
    # STEP 5: Comparison Summary
    # ----------------------------
    print("\n📈 Comparison Summary:")
    print(f"  Random Entropy       = {entropy_random:.6f}")
    print(f"  Maximum Entropy      = {entropy_opt:.6f}")
    print(f"  Difference           = {entropy_opt - entropy_random:.6f}")

    return p_opt, p_random, theta_star, entropy_opt, entropy_random


solve_max_entropy(n=10, M_target=5.5)
