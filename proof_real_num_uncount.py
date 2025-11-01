import numpy as np
from decimal import Decimal, getcontext

def proof(x: float, N: int = 10):
    """Compute decimal digits a_n and remainders b_n for x in [0,1).

    Implements the pseudocode:
      b_prev = x
      for n = 1..N:
          temp = 10 * b_prev
          a_n = floor(temp)
          b_n = temp - a_n
          b_prev = b_n

    Returns two dicts (a, b) where a[n] and b[n] are 1-indexed.
    """
    if not (0 <= x < 1):
        raise ValueError("x must be in [0,1)")

    a = {}
    b = {}

    b_prev = x
    for n in range(1, N + 1):
        temp = 10 * b_prev
        a[n] = int(np.floor(temp))
        b[n] = temp - a[n]
        b_prev = b[n]

    return a, b


def proof_decimal(x_str: str, N: int = 10):
    getcontext().prec = N + 10
    x = Decimal(x_str)
    if not (Decimal('0') <= x < Decimal('1')):
        raise ValueError("x must be in [0,1)")
    a = {}
    b = {}
    b_prev = x
    for n in range(1, N+1):
        temp = b_prev * Decimal(10)
        a[n] = int(temp // 1)          # integer part
        b_prev = temp - Decimal(a[n])  # fractional remainder
        b[n] = b_prev                  # save remainder (Decimal)
    return a, b

def proof_sum_decimal(x_str: str, N: int = 5):
    """Reconstruct the original number from its decimal digits.
    For 0.876:
        8 × 0.1    = 0.8
        7 × 0.01   = 0.07
        6 × 0.001  = 0.006
        sum = 0.876
    """
    getcontext().prec = N + 10
    x = Decimal(x_str)
    if not (Decimal('0') <= x < Decimal('1')):
        raise ValueError("x must be in [0,1)")
    
    result = Decimal('0')
    b_prev = x
    power = Decimal('0.1')
    
    for n in range(1, N+1):
        temp = b_prev * Decimal(10)
        digit = int(temp // 1)  # Get the nth digit
        b_prev = temp - Decimal(digit)  # Remainder for next iteration
        
        # Add digit × power to result
        result += Decimal(digit) * power
        power *= Decimal('0.1')  # Next power of 1/10
    
    return result


if __name__ == "__main__":
    # Example usage and small demonstration (prints first 6 digits)
    a, b = proof(0.876, N=4)
    print("x=0.876 -> a:", a)
    print("x=0.876 -> b:", b)

    a2, b2 = proof(0.01, N=4)
    print("x=0.01  -> a:", a2)
    print("x=0.01  -> b:", b2)

    a3, b3 = proof_decimal("0.876", 4)
    print("x=0.876 -> a:", a3)
    print("x=0.876 -> b:",b3)

    # Test reconstruction of 0.876
    x = "0.876"
    reconstructed = proof_sum_decimal(x, 4)
    print(f"Original number: {x}")
    print(f"Reconstructed:   {reconstructed}")
    print(f"Are they equal? {Decimal(x) == reconstructed}")

    