#compute the frechet distance between two curves, by cjzhang 20210317

import numpy as np

def calculateFrechetDistance(A, i, j, P, Q):

    if A[i, j] > -1:
        return A[i, j]
    elif i == 0 and j == 0:
        A[i, j] = np.linalg.norm(P[i] - Q[j])
    elif i > 0 and j == 0:
        A[i, j] = max(calculateFrechetDistance(A, i-1, 0, P, Q), np.linalg.norm(P[i] - Q[j]))
    elif i == 0 and j > 0:
        A[i, j] = max(calculateFrechetDistance(A, 0, j-1, P, Q), np.linalg.norm(P[i] - Q[j]))
    elif i > 0 and j > 0:
        A[i, j] = max(
            min(
                calculateFrechetDistance(A, i-1, j, P, Q),
                calculateFrechetDistance(A, i-1, j-1, P, Q),
                calculateFrechetDistance(A, i, j-1, P, Q)
            ),
            np.linalg.norm(P[i] - Q[j])
            )
    else:
        A[i, j] = float('inf')

    return A[i, j]


def frenetDist(P, Q):
    
    P = np.array(P, np.float64)
    Q = np.array(Q, np.float64)

    LP = len(P)
    LQ = len(Q)

    if LP == 0 or LQ == 0:
        raise ValueError('Empty curve!')

    if len(P[0]) != len(Q[0]):
        raise ValueError('Different dimensions!')
        
    if LP != LQ:
        raise ValueError('Different length!')
        

    A = (np.ones((LP, LQ), dtype=np.float64) * -1)

    frechetDistance = calculateFrechetDistance(A, LP-1, LQ-1, P, Q)
    
    return frechetDistance

