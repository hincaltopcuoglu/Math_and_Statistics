import numpy as np

def proof(x:float):
    a = {}
    b = {}
    n = 1  # Start with n = 1
    
    # Initialize first values
    b[n] = x  # Start with the original number
    
    while n <= 4:  # Limit iterations to prevent infinite loop
        a[n] = int(np.floor(10*b[n]))  # Get the first digit after decimal
        b[n+1] = 10*b[n] - a[n]  # Get the remaining decimal part
        n += 1
    
    return a, b  # Return both sequences
    
    
print(proof(0.874))
print(proof(0.01))