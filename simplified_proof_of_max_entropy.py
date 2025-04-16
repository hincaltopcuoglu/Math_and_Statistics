import numpy as np
from scipy.optimize import root_scalar

def max_entropy_vs_random(n=10, M_target=5.5):
    """
    Shows maximum entropy (using Lagrange multipliers) vs
    random entropy of a variable under the same support.
    """
    x = np.arange(1, n + 1)

    # ----------------------------
    # 1. Random Distribution (non-optimal)
    # ----------------------------
    rand_vals = np.random.rand(n)
    p_random = rand_vals / np.sum(rand_vals)
    entropy_random = -np.sum(p_random * np.log(p_random))
    expected_random = np.sum(p_random * x)

    print("🎲 Random Distribution (not optimized):")
    for i, pi in enumerate(p_random, start=1):
        print(f"  p{i} = {pi:.6f}")
    print(f"  ∑p_i = {np.sum(p_random):.6f}")
    print(f"  ∑p_i * x_i = {expected_random:.6f}")
    print(f"  Entropy H(p_random) = {entropy_random:.6f}\n")

    # ----------------------------
    # 2. Maximum Entropy via Lagrange Multipliers
    # ----------------------------
    def moment_constraint(theta):
        exp_terms = np.exp(-theta * x)
        Z = np.sum(exp_terms)
        p = exp_terms / Z
        return np.sum(p * x) - M_target

    sol = root_scalar(moment_constraint, bracket=[-10, 10], method='brentq')
    if not sol.converged:
        raise RuntimeError("Could not solve for θ.")
    theta = sol.root

    exp_terms = np.exp(-theta * x)
    Z = np.sum(exp_terms)
    p_opt = exp_terms / Z
    entropy_opt = -np.sum(p_opt * np.log(p_opt))
    expected_opt = np.sum(p_opt * x)
    C = 1 / Z
    lambda_value = -1 - np.log(C)

    print("📈 Maximum Entropy Distribution (solved):")
    for i, pi in enumerate(p_opt, start=1):
        print(f"  p{i} = {pi:.6f}")
    print(f"  θ = {theta:.6f}")
    print(f"  λ = {lambda_value:.6f}")
    print(f"  ∑p_i = {np.sum(p_opt):.6f}")
    print(f"  ∑p_i * x_i = {expected_opt:.6f} (target M = {M_target})")
    print(f"  Entropy H(p_opt) = {entropy_opt:.6f}")

    # ----------------------------
    # 3. Comparison Summary
    # ----------------------------
    print("\n📊 Entropy Comparison:")
    print(f"  Random Entropy       = {entropy_random:.6f}")
    print(f"  Maximum Entropy      = {entropy_opt:.6f}")
    print(f"  Difference           = {entropy_opt - entropy_random:.6f}")

    return {
        "p_random": p_random,
        "H_random": entropy_random,
        "expected_random": expected_random,
        "p_opt": p_opt,
        "H_opt": entropy_opt,
        "expected_opt": expected_opt,
        "theta": theta,
        "lambda": lambda_value,
    }

# Run it
max_entropy_vs_random(n=10, M_target=5.5)
