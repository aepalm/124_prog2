'''
Option 1: Single-source file
To run: python3 strassen.py <flag> <dimension> <inputfile>
'''

import numpy as np
import time
import sys

def add_matrix(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def sub_matrix(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def equal_matrix(A, B):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != B[i][j]:
                return False
    return True

def create_test_matrix(n):
    return [[np.random.randint(-1, 2) for _ in range(n)] for _ in range(n)]

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

def strassen(X,Y,n, n_0):
    if n == 1:
        return [[X[0][0] * Y[0][0]]]
    
    if n <= n_0: #cross-over point
        return conventional_mm(X,Y,n)
    
    # Let's first assume that n is even:
    if n % 2 == 0:
        #divide up X and Y: 
        New = [[0]*n for i in range(n)]
        new_size =  n//2
        A = [[0]*new_size for _ in range(new_size)]
        B = [[0]*new_size for _ in range(new_size)]
        C = [[0]*new_size for _ in range(new_size)]
        D = [[0]*new_size for _ in range(new_size)]
        E = [[0]*new_size for _ in range(new_size)]
        F = [[0]*new_size for _ in range(new_size)]
        G = [[0]*new_size for _ in range(new_size)]
        H = [[0]*new_size for _ in range(new_size)]
         
        for i in range( new_size):
            for j in range( new_size):
                A[i][j] = X[i][j]
                B[i][j] = X[i][j + new_size]
                C[i][j] = X[i + new_size][j]
                D[i][j] = X[i + new_size][j + new_size]
                E[i][j] = Y[i][j]
                F[i][j] = Y[i][j + new_size]
                G[i][j] = Y[i + new_size][j]
                H[i][j] = Y[i + new_size][j + new_size]

        #products:
        P1 = strassen(A, sub_matrix(F,H),  new_size, n_0)
        P2 = strassen(add_matrix(A,B), H,  new_size, n_0)
        P3 = strassen(add_matrix(C,D), E,  new_size, n_0)
        P4 = strassen(D, sub_matrix(G,E),  new_size, n_0)
        P5 = strassen(add_matrix(A,D), add_matrix(E,H),  new_size, n_0)
        P6=  strassen(sub_matrix(B,D), add_matrix(G,H),  new_size, n_0)
        P7 = strassen(sub_matrix(C,A), add_matrix(E,F),  new_size, n_0)

        upper_left = add_matrix(add_matrix(sub_matrix(P4, P2), P5), P6)
        upper_right = add_matrix(P1, P2)
        lower_left = add_matrix(P3, P4)
        lower_right = add_matrix(add_matrix(sub_matrix(P1, P3), P5), P7)

        for i in range(new_size):
            for j in range(new_size):
                New[i][j] = upper_left[i][j]
                New[i][j + new_size] = upper_right[i][j]
                New[i + new_size][j] = lower_left[i][j]
                New[i + new_size][j + new_size] = lower_right[i][j]
        
        return New
    
    else: #n is an odd number
        #pad the matrices with 0's to make them even, call strassen on the new matrices, 
        #then remove the padding from the result
        X_padded = [[0]*(n+1) for i in range(n+1)]
        Y_padded = [[0]*(n+1) for i in range(n+1)]
        for i in range(n):
            for j in range(n):
                X_padded[i][j] = X[i][j]
                Y_padded[i][j] = Y[i][j]
                
        result_padded = strassen(X_padded, Y_padded, n+1, n_0)
        result = [[0]*n for i in range(n)]
        for i in range(n):
            for j in range(n):  
                result[i][j] = result_padded[i][j]
        return result




def main():
    #experimentally optimize the cross-over point n_0
    #want to find the smallest n_0 possible
    #analytically we found n_0 = 15
    '''
    # testing values 
    sizes = [8, 9, 16, 17, 32, 33, 64, 65]
    for size in sizes:
        print("Matrix size: ", size)
        for n_0 in range(10,30):
            print("\t Testing n_0: ", n_0)
            total_time = 0
            for _ in range(5): #run each n_0 value 5 times to get an average time
                X = create_test_matrix(size)
                Y = create_test_matrix(size)
                start_time = time.perf_counter()
                A = strassen(X, Y, size, n_0)
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                #quick check correctness:
                if not equal_matrix(A, conventional_mm(X, Y, size)):
                    print("Error: strassen and conventional mm do not match")
                    sys.exit(1)
            print("\t \t Average Time taken: ", total_time / 5)
    '''
    #get input arguments: flag, dimension, inputfile
    flag = sys.argv[1]
    dimension = int(sys.argv[2])
    inputfile = sys.argv[3]

    # input file is an ASCII file with 2d^2 integers, one per line, representing A and B
    # first d^2 integers are A, next d^2 integers are B
    # in row-major order: a_01, a_02, ...
    with open(inputfile, 'r') as f:
        data = f.read().splitlines()
    A = [[0]*dimension for i in range(dimension)]
    B = [[0]*dimension for i in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            A[i][j] = int(data[i*dimension + j])
            B[i][j] = int(data[dimension*dimension + i*dimension + j])
            
    #call strassen on A and B, print the result
    for n_0 in range(10,30):
            print("\t Testing n_0: ", n_0)
            total_time = 0
            for _ in range(5): #run each n_0 value 5 times to get an average time
                X = create_test_matrix(dimension)
                Y = create_test_matrix(dimension)
                start_time = time.perf_counter()
                A = strassen(X, Y, dimension, n_0)
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                #quick check correctness:
                if not equal_matrix(A, conventional_mm(X, Y, dimension)):
                    print("Error: strassen and conventional mm do not match")
                    sys.exit(1)
            print("\t \t Average Time taken: ", total_time / 5)

if __name__ == "__main__":    
    main()
