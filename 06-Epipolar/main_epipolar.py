import numpy as np
import matplotlib.pyplot as plt
import itertools

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    ### YOUR CODE BEGINS HERE
    
    # build matrix for equations in Page 51
    A = None
    for x, xdot in itertools.zip_longest(np.transpose(x1), np.transpose(x2)):
        row = np.array([x[0]*xdot[0], x[1]*xdot[0], xdot[0], x[0]*xdot[1], x[1]*xdot[1], xdot[1], x[0], x[1], 1])
        if A is None:
            A = row
        else:
            A = np.vstack((A, row))
    
    # compute the solution in Page 51
    
    # Calculate eigen values and eigen vectors
    w, v = np.linalg.eig(np.matmul(A.T, A))
    # Get eigen vector with minimum eigen value
    eig_vector = v[:, np.argmin(w)]
    # Get the fundamental matrix
    h = eig_vector.reshape(3, 3)

    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    U, S, V = np.linalg.svd(h)
    S[2] = 0
    F = np.matmul(U, np.matmul(np.diag(S), V))

    ### YOUR CODE ENDS HERE
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None

    ### YOUR CODE BEGINS HERE
    # Compute epipole e1 using fundamental matrix
    w, v = np.linalg.eig(np.matmul(F.T, F))
    e1 = v[:, np.argmin(w)]
    # Compute epipole e2 using fundamental matrix
    w, v = np.linalg.eig(np.matmul(F, F.T))
    e2 = v[:, np.argmin(w)]
    # Normalize e1 and e2 with homogeneous coordinates
    e1 = e1 / e1[2]
    e2 = e2 / e2[2]
    ### YOUR CODE ENDS HERE

    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)
    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE
    plt.figure(figsize=(10,10))

    plt.subplot(1,2,1)
    plt.imshow(img1)
    # Draw epipolar lines for img1
    for x, y, _ in cor1.T:
        m = (e1[1] - y) / (e1[0] - x)
        b = y - m * x
        xspace = np.linspace(0, img1.shape[1])
        plt.plot(xspace, m*xspace + b, color=np.random.rand(3))
        plt.scatter(x ,y)
    plt.xlim([0, img1.shape[1]])
    plt.ylim([0, img1.shape[0]])
    plt.gca().invert_yaxis()

    plt.subplot(1,2,2)
    plt.imshow(img2)
    # Draw epipolar lines for img2
    for x, y, _ in cor2.T:
        m = (e2[1] - y) / (e2[0] - x)
        b = y - m * x
        xspace = np.linspace(0, img2.shape[1])
        plt.plot(xspace, m*xspace + b, color=np.random.rand(3))
        plt.scatter(x ,y)
    plt.xlim([0, img2.shape[1]])
    plt.ylim([0, img2.shape[0]])
    plt.gca().invert_yaxis()

    plt.show()
    
    ### YOUR CODE ENDS HERE


draw_epipolar_lines(img1, img2, cor1, cor2)