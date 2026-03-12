# Option 1: Single-source file
# To run: python3 strassen.py <args>

# Assume that the cost of any single arithmetic operation  is 1, and that
# all other operations are free.

# Goal: to multiply two n x n matrices, we start with Strassen's,
# but at some value n_0, we switch to conventional alg

# First, we want to analyitically try to find the cross-over point

# Then, we try to find it experimentally using our code! We want to try and find
# as small as possible point, which depends on how efficiently we can implement the alg

# Finally, we do the "triangle in random graphs" part

import numpy as np

def create_test_matrix(n):
    M = [[0]*n for i in range(n)] # Initialize M to just 0's
    # Add the random values

    return M

def subtract_matrices(A,B,n):
    C = [[0]*n for i in range(n)] 
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C

def add_matrices(A,B,n):
    C = [[0]*n for i in range(n)] 
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C    
    


def conventional_mm(A, B, n):
    # Assume A and B are both nxn matrices
    # Let C be the nxn matrix representing AB
    C = [[0]*n for i in range(n)] # Initialize C to just 0's
    for i in range(n): # Row
        for j in range(n): # Column
            dot_prod = 0 # Dot product of row i of A with column j of B
            for k in range(n):
                dot_prod += (A[i][k] * B[k][j])
            C[i][j] = dot_prod

    return C

def strassen(X,Y,n):
    # Let's first assume that n is even:
    if n % 2 == 0:
        #divide up X:
        A = X[:n/2][:n/2]
        B = X[:n/2][n/2:]
        C = X[n/2:][:n/2]
        D = X[n/2:][n/2:]
        #divide up Y:
        E = Y[:n/2][:n/2]
        F = Y[:n/2][n/2:]
        G = Y[n/2:][:n/2]
        H = Y[n/2:][n/2:]
        #products:
        P_1 = strassen(A, subtract_matrices(F,H,n/2))
        P_2 = strassen(add_matrices(A,B, n/2), H)
        P_3 = strassen(add_matrices(C,D,n/2), E)
        P_4 = strassen(D, subtract_matrices(G-H,n/2))
        P_5 = strassen(add_matrices(A,D,n/2), add_matrices(E,H,n/2))
        P_6=  strassen(subtract_matrices(B,D,n/2), add_matrices(G,H,n/2))
        P_7 = strassen(subtract_matrices(C,A,n/2), add_matrices(E,F,n/2))

    return None