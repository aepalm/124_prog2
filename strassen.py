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


def conventional_mm(A, B, n):
    # Not using any kind of numpy function for this
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
    New = [[0]*n for i in range(n)]
    
    if n % 2 == 0:
        #divide up X and Y: 
        new_size =  int(n/2)
        A = [[0 for _ in range( new_size)] for _ in range( new_size)]
        B = [[0 for _ in range( new_size)] for _ in range( new_size)]
        C = [[0 for _ in range( new_size)] for _ in range( new_size)]
        D = [[0 for _ in range( new_size)] for _ in range( new_size)]
        E = [[0 for _ in range( new_size)] for _ in range( new_size)]
        F = [[0 for _ in range( new_size)] for _ in range( new_size)]
        G = [[0 for _ in range( new_size)] for _ in range( new_size)]
        H = [[0 for _ in range( new_size)] for _ in range( new_size)]
         
        for i in range( new_size):
            for j in range( new_size):
                A[i][j] = X[i][j]
                B[i][j] = X[i][j +  new_size]
                C[i][j] = X[i +  new_size][j]
                D[i][j] = X[i +  new_size][j +  new_size]
                E[i][j] = Y[i][j]
                F[i][j] = Y[i][j +  new_size]
                G[i][j] = Y[i +  new_size][j]
                H[i][j] = Y[i +  new_size][j +  new_size]

        #products:
        P1 = strassen(A, np.subtract(F,H),  new_size)
        P2 = strassen(np.add(A,B), H,  new_size)
        P3 = strassen(np.add(C,D), E,  new_size)
        P4 = strassen(D, np.subtract(G,H),  new_size)
        P5 = strassen(np.add(A,D), np.add(E,H),  new_size)
        P6=  strassen(np.subtract(B,D), np.add(G,H),  new_size)
        P7 = strassen(np.subtract(C,A), np.add(E,F),  new_size)

        upper_left = np.add(np.subtract(P4,P2), np.add(P5, P6))
        upper_right = np.add(P1, P2)
        lower_left = np.add(P3, P4)
        lower_right = np.add(np.subtract(P1,P3), np.add(P5, P7))

        #combine:

        for i in range( new_size):
            for j in range( new_size):
                New[i][j]= upper_left[i][j]
                New[i][j +  new_size] = upper_right[i][j]
                New[i +  new_size][j] = lower_right[i][j]
                New[i +  new_size][j +  new_size] = lower_left[i][j]
        

    return New


X = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
stras = strassen(X, X, 4)
reg = conventional_mm(X, X, 4)
for i in range(4):
    for j in range(4):
        print("i,j: ", i,j, "stras val:", stras[i][j], "reg val: ", reg[i][j])