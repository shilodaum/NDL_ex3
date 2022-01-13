import numpy as np
from nltk.stem import PorterStemmer


def main():
    for b in np.arange(0.1, 0.9, 0.1):
        print(f"beta is {b}:")
        print(page_rank(b))


def page_rank(b):
    M = np.zeros((6, 6))
    M[0, 3] = 1
    M[0, 4] = 1
    M[0, 5] = 1.0 / 2
    M[1, 0] = 1
    M[1, 5] = 1.0 / 2
    M[2, 1] = 1.0
    # # S set
    # M[0, 1] = 1.0
    # M[1, 2] = 1.0
    #
    # M[1, 3] = 1.0 / 2
    # M[2, 3] = 1.0 / 2
    #
    # M[2, 4] = 1.0
    # M[2, 5] = 1.0

    # B = np.zeros((6, 6))
    # B += 1.0 / 6
    # print(B)
    # b = 0.7

    # A = b * M + (1 - b) * B
    # print(A)

    r = np.zeros(6)
    r += 1.0 / 6

    for i in range(10):
        # print(f'iteration {i}')
        # print(r)
        # print('-----------------')
        r = np.matmul(b * M, r) + (1 - b) / 6  # np.matmul(A, r)
        # print(f'here {r}')
    # print(r)
    return r


def q1():
    M = np.zeros((3, 3))
    M[:, 0] = 1.0 / 3
    M[0, 1] = 1.0 / 2
    M[2, 1] = 1.0 / 2

    M[1, 2] = 1.0 / 2
    M[2, 2] = 1.0 / 2

    b = 0.7

    # A = b * M + (1 - b) * B
    # print(A)

    r = np.zeros(3)
    r += 1.0 / 3

    A = np.zeros(3)
    A[0:2] = 1.0 / 2

    for i in range(10):
        print(f'iteration {i}')
        print(r)
        print('-----------------')
        r = np.matmul(b * M, r) + (1 - b) * A  # np.matmul(A, r)
        # print(f'here {r}')
    # print(r)
    return r


def stem_root():
    ps = PorterStemmer()
    a = ps.stem('alumnus')
    b = ps.stem('alumni')
    print(a, b)


"""
Me: How much water can i drink in an hour? 
Clev: 2 times minimum.
"""

if __name__ == '__main__':
    q1()
