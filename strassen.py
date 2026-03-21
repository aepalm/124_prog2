'''
Option 1: Single-source file
To run: python3 strassen.py <flag> <dimension> <inputfile>
'''

import numpy as np
import time
import sys

def add_into_quadrant(New, P, row_off, col_off, sign):
    m = len(P)
    for i in range(m):
        New_i = New[row_off + i]
        P_i = P[i]
        for j in range(m):
            New_i[col_off + j] += sign * P_i[j]


def add_block(X, x_row, x_col, Y, y_row, y_col, n):
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = X[x_row + i][x_col + j] + Y[y_row + i][y_col + j]
    return out

def sub_block(X, x_row, x_col, Y, y_row, y_col, n):
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = X[x_row + i][x_col + j] - Y[y_row + i][y_col + j]
    return out

def equal_matrix(A, B):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != B[i][j]:
                return False
    return True

def create_test_matrix(n):
    return [[np.random.randint(-1, 2) for _ in range(n)] for _ in range(n)]

def conventional_mm(A, a_row, a_col, B, b_row, b_col, n):
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            a = A[a_row + i][a_col + k]
            for j in range(n):
                C[i][j] += a * B[b_row + k][b_col + j]
    return C

def strassen(X, x_row, x_col, Y, y_row, y_col, n, n_0):
    if n == 1:
        return [[X[x_row][x_col] * Y[y_row][y_col]]]

    if n <= n_0: #cross-over point
        return conventional_mm(X,x_row, x_col, Y,y_row, y_col, n)
    
    if n % 2 != 0:
        X_padded = [[0] * (n + 1) for _ in range(n + 1)]
        Y_padded = [[0] * (n + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(n):
                X_padded[i][j] = X[x_row + i][x_col + j]
                Y_padded[i][j] = Y[y_row + i][y_col + j]

        result_padded = strassen(X_padded, 0, 0, Y_padded, 0, 0, n + 1, n_0)

        result = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[i][j] = result_padded[i][j]
        return result
    
    # Let's first assume that n is even:
    #divide up X and Y: 
    new_size =  n//2
    
    #instead of creating new matrices for the submatrices we just pass in the appropriate indices
    A = (x_row, x_col)
    B = (x_row, x_col + new_size)
    C = (x_row + new_size, x_col)
    D = (x_row + new_size, x_col + new_size)
    E = (y_row, y_col)
    F = (y_row, y_col + new_size)
    G = (y_row + new_size, y_col)
    H = (y_row + new_size, y_col + new_size)

    # products:

    New = [[0] * n for _ in range(n)]

    # P1 = A(F - H)
    T1 = sub_block(Y, F[0], F[1], Y, H[0], H[1], new_size)
    P = strassen(X, A[0], A[1], T1, 0, 0, new_size, n_0)
    add_into_quadrant(New, P, 0, new_size, 1)               
    add_into_quadrant(New, P, new_size, new_size, 1)         

    # P2 = (A + B)H
    T1 = add_block(X, A[0], A[1], X, B[0], B[1], new_size)
    P = strassen(T1, 0, 0, Y, H[0], H[1], new_size, n_0)
    add_into_quadrant(New, P, 0, 0, -1)                     
    add_into_quadrant(New, P, 0, new_size, 1)               

    # P3 = (C + D)E
    T1 = add_block(X, C[0], C[1], X, D[0], D[1], new_size)
    P = strassen(T1, 0, 0, Y, E[0], E[1], new_size, n_0)
    add_into_quadrant(New, P, new_size, 0, 1)              
    add_into_quadrant(New, P, new_size, new_size, -1)       

    # P4 = D(G - E)
    T1 = sub_block(Y, G[0], G[1], Y, E[0], E[1], new_size)
    P = strassen(X, D[0], D[1], T1, 0, 0, new_size, n_0)
    add_into_quadrant(New, P, 0, 0, 1)                      
    add_into_quadrant(New, P, new_size, 0, 1)                

    # P5 = (A + D)(E + H)
    T1 = add_block(X, A[0], A[1], X, D[0], D[1], new_size)
    T2 = add_block(Y, E[0], E[1], Y, H[0], H[1], new_size)
    P = strassen(T1, 0, 0, T2, 0, 0, new_size, n_0)
    add_into_quadrant(New, P, 0, 0, 1)                       
    add_into_quadrant(New, P, new_size, new_size, 1)        

    # P6 = (B - D)(G + H)
    T1 = sub_block(X, B[0], B[1], X, D[0], D[1], new_size)
    T2 = add_block(Y, G[0], G[1], Y, H[0], H[1], new_size)
    P = strassen(T1, 0, 0, T2, 0, 0, new_size, n_0)
    add_into_quadrant(New, P, 0, 0, 1)                      

    # P7 = (C - A)(E + F)
    T1 = sub_block(X, C[0], C[1], X, A[0], A[1], new_size)
    T2 = add_block(Y, E[0], E[1], Y, F[0], F[1], new_size)
    P = strassen(T1, 0, 0, T2, 0, 0, new_size, n_0)
    add_into_quadrant(New, P, new_size, new_size, 1)        
    del P

    return New
    


def find_triangles(p): # this is for task 3
    # A represents original adjacency matrix:
    # 1024x1024 of 0's and 1's with probability p of being 1
    A = [[0]*1024 for i in range(1024)]
    for i in range(1024):
        for j in range(i+1, 1024):
            if np.random.rand() < p:
                A[i][j] = 1
                A[j][i] = 1
    print("running squaring")
    # count triangles
    A_squared = strassen(A, 0,0, A, 0,0, 1024, 512) # n_0 = 128 is the best value we found for various sizes
    print("running cubing")
    A_cubed = strassen(A_squared, 0,0, A, 0,0, 1024, 512)
    triangle_count = 0
    for i in range(1024):
        triangle_count += A_cubed[i][i]

    return triangle_count/6

def main():
    flag = int(sys.argv[1])

    #experimentally optimize the cross-over point n_0
    #want to find the smallest n_0 possible
    #analytically we found n_0 = 15
    if flag == 1:
        # testing values 
        sizes = [1024, 2048, 4096]
        n_0s = [256, 512]
        for size in sizes:
            print("Matrix size: ", size)
            for n_0 in n_0s:
                print("\t Testing n_0: ", n_0)
                total_time = 0
                X = create_test_matrix(size)
                Y = create_test_matrix(size)
                for _ in range(5): #run each n_0 value 20 times to get an average time
                    start_time = time.perf_counter()
                    A = strassen(X, 0,0, Y, 0,0, size, n_0)
                    end_time = time.perf_counter()
                    total_time += (end_time - start_time)
                    #quick check correctness:
                    #if not equal_matrix(A, conventional_mm(X, 0, 0, Y, 0, 0, size)):
                    #    print("Error: strassen and conventional mm do not match")
                    #    sys.exit(1)
                print("\t \t Average Time taken: ", total_time / 5)
    elif flag == 2: #test triangles
        ps = [0.01, 0.02, 0.03, 0.04, 0.05]
        for p in ps:
            print("p: ", p)
            triangle_count = find_triangles(p)
            expected_count = ((1024*1023*1022)/6) * (p**3) #1024 choose 3
            print("\t Triangle count: ", triangle_count)
            print("\t Expected count: ", expected_count)
    
    else:
        #after testing, n_0 = 21 is seemingly the best value for various sizes
        #get other input arguments: dimension, inputfile
        dimension = int(sys.argv[2])
        inputfile = sys.argv[3]

        # input file is an ASCII file with 2d^2 integers, one per line, representing A and B
        # first d^2 integers are A, next d^2 integers are B
        # in row-major order: a_01, a_02, ...
        n_0 = 21
        with open(inputfile, 'r') as f:
            data = f.read().splitlines()
        A = [[0]*dimension for i in range(dimension)]
        B = [[0]*dimension for i in range(dimension)]

        for i in range(dimension):
            for j in range(dimension):
                A[i][j] = int(data[i*dimension + j])
                B[i][j] = int(data[dimension*dimension + i*dimension + j])

        #call strassen on A and B, print the result       
        C = strassen(A, B, dimension, n_0)

        for i in range(dimension):
            print(C[i][i])
    

if __name__ == "__main__":    
    main()
