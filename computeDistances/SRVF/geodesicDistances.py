import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from copy import deepcopy
import fdasrsf.geodesic as gd
import fdasrsf.curve_stats as cs
import time
import fdasrsf.curve_functions as cf
from scipy.linalg import norm,solve
import tqdm
import os

def init_path_geod(beta1, beta2, T=100, k=5):
    r"""
    Initializes a path in :math:`\cal{C}`. beta1, beta2 are already
    standardized curves. Creates a path from beta1 to beta2 in
    shape space, then projects to the closed shape manifold.

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return alpha: a path between two q-functions
    :return beta:  a path between two curves
    :return O: rotation matrix

    """
    alpha = np.zeros((2, T, k))
    beta = np.zeros((2, T, k))

    dist, pathq, O = gd.geod_sphere(beta1, beta2, k)

    for tau in range(0, k):
        alpha[:, :, tau] = project_curve(pathq[:, :, tau])
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + np.tile(a, [T, 1]).T

    return(alpha, beta, O)

def update_path(alpha, beta, gradE, delta, T=100, k=5):
    """
    Update the path along the direction -gradE

    :param alpha: numpy ndarray of shape (2,M) of M samples
    :param beta: numpy ndarray of shape (2,M) of M samples
    :param gradE: numpy ndarray of shape (2,M) of M samples
    :param delta: gradient paramenter
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy scalar
    :return alpha: updated path of srvfs
    :return beta: updated path of curves

    """
    for tau in range(1, k-1):
        alpha_new = alpha[:, :, tau] - delta*gradE[:, :, tau]
        alpha[:, :, tau] = project_curve(alpha_new)
        x = cf.q_to_curve(alpha[:, :, tau])
        a = -1*cf.calculatecentroid(x)
        beta[:, :, tau] = x + np.tile(a, [T, 1]).T

    return(alpha, beta)

def project_curve(qvals):
    q= deepcopy(qvals)

    n,T = q.shape
    if n==2:
        dt = 0.35
    if n==3:
        dt = 0.2
    epsilon = 1e-6

    iter = 1
    res = np.ones(n)
    J = np.zeros((n,n))

    s = np.linspace(0,1,T)
    try:
      

        qnew = q.copy()
        qnew = qnew / np.sqrt(cf.innerprod_q2(qnew,qnew))

        qnorm = np.zeros(T)
        G = np.zeros(n)
        C = np.zeros(300)
        while (norm(res) > epsilon):
            if iter >= 300:
                break

            # Jacobian
            for i in range(0,n):
                for j in range(0,n):
                    J[i,j] = 3 * np.trapz(qnew[i,:]*qnew[j,:],s)

            J += np.eye(n)

            for i in range(0,T):
                qnorm[i] = norm(qnew[:,i])

            # Compute the residue
            for i in range(0,n):
                G[i] = np.trapz(qnew[i,:]*qnorm,s)

            res = -G

            if (norm(res) < epsilon):
                break

            x = solve(J,res)
            C[iter] = norm(res)

            delG = cf.Basis_Normal_A(qnew)
            temp = np.zeros((n,T))
            for i in range(0,n):
                temp += x[i]*delG[i]*dt

            qnew += temp
            iter += 1

        qnew = qnew/np.sqrt(cf.innerprod_q2(qnew,qnew))
        
    except:
        if q[0][0] == 0:
            q[0][0] = 1e-9
            
        qnew = q.copy()
        qnew = qnew / np.sqrt(cf.innerprod_q2(qnew,qnew))

        qnorm = np.zeros(T)
        G = np.zeros(n)
        C = np.zeros(300)
        while (norm(res) > epsilon):
            if iter >= 300:
                break

            # Jacobian
            for i in range(0,n):
                for j in range(0,n):
                    J[i,j] = 3 * np.trapz(qnew[i,:]*qnew[j,:],s)

            J += np.eye(n)

            for i in range(0,T):
                qnorm[i] = norm(qnew[:,i])

            # Compute the residue
            for i in range(0,n):
                G[i] = np.trapz(qnew[i,:]*qnorm,s)

            res = -G

            if (norm(res) < epsilon):
                break

            x = solve(J,res)
            C[iter] = norm(res)

            delG = cf.Basis_Normal_A(qnew)
            temp = np.zeros((n,T))
            for i in range(0,n):
                temp += x[i]*delG[i]*dt

            qnew += temp
            iter += 1

        qnew = qnew/np.sqrt(cf.innerprod_q2(qnew,qnew))
    
    return qnew

def path_straightening(beta1, beta2, betamid, init="rand", T=100, k=5):
    """
    Perform path straightening to find geodesic between two shapes in either
    the space of closed curves or the space of affine standardized curves.
    This algorithm follows the steps outlined in section 4.6 of the
    manuscript.

    :param beta1: numpy ndarray of shape (2,M) of M samples (first curve)
    :param beta2: numpy ndarray of shape (2,M) of M samples (end curve)
    :param betamid: numpy ndarray of shape (2,M) of M samples (mid curve
     Default = NULL, only needed for init "rand")
    :param init: initialize path geodesic or random (Default = "rand")
    :param T: Number of samples of curve (Default = 100)
    :param k: number of samples along path (Default = 5)

    :rtype: numpy ndarray
    :return dist: geodesic distance
    :return path: geodesic path
    :return pathsqnc: geodesic path sequence
    :return E: energy

    """
    inits = ["rand", "geod"]
    init = [i for i, x in enumerate(inits) if x == init]
    if len(init) == 0:
        init = 0
    else:
        init = init[0]

    betanew1, qnew1, A1 = pre_proc_curve(beta1)
    betanew2, qnew2, A2 = pre_proc_curve(beta2)

    if init == 0:
        betanewmid, qnewmid, Amid = pre_proc_curve(beta2)

    if init == 0:
        alpha, beta, O = gd.init_path_rand(betanew1, betanewmid,
                                        betanew2, T, k)
    elif init == 1:
        alpha, beta, O = init_path_geod(betanew1, betanew2, T, k)

    # path straightening
    tol = 1e-2
    n = beta.shape[0]
    T = beta.shape[1]
    maxit = 20
    i = 0
    g = 1
    delta = 0.5
    E = np.zeros(maxit+1)
    gradEnorm = np.zeros(maxit+1)
    pathsqnc = np.zeros((n, T, k, maxit+1))

    pathsqnc[:, :, :, 0] = beta

    while i < maxit:
        # algorithm 8:
        # compute dalpha/dt along alpha using finite difference approx
        # First calculate basis for normal sapce at each point in alpha
        basis = gd.find_basis_normal_path(alpha, k)
        alphadot = gd.calc_alphadot(alpha, basis, T, k)
        E[i] = gd.calculate_energy(alphadot, T, k)

        # algorithm 9:
        # compute covariant integral of alphadot along alpha. This is
        # the gradient
        # of E in \cal{H}. Later we will project it to the space \cal{H}_{O}
        u1 = gd.cov_integral(alpha, alphadot, basis, T, k)

        # algorithm 10:
        # backward parallel transport of u(1)
        utilde = gd.back_parallel_transport(u1[:, :, -1], alpha, basis, T, k)

        # algorithm 11:
        # compute gradient vector field of E in \cal{H}_{O}
        gradE, normgradE = gd.calculate_gradE(u1, utilde, T, k)
        gradEnorm[i] = norm(normgradE)
        g = gradEnorm[i]

        # algorithm 12:
        # update the path along the direction -gradE
        alpha, beta = update_path(alpha, beta, gradE, delta, T, k)

        # path evolution
        pathsqnc[:, :, :, i+1] = beta

        if g < tol:
            break

        i += 1

    if i > 0:
        E = E[0:i]
        gradEnorm = gradEnorm[0:i]
        pathsqnc = pathsqnc[:, :, :, 0:(i+2)]
    else:
        E = E[0]
        gradEnorm = gradEnorm[0]
        pathsqnc = pathsqnc[:, :, :, 0:(i+2)]

    path = beta
    dist = gd.geod_dist_path_strt(beta, k)

    return(dist, path, pathsqnc, E)

def pre_proc_curve(beta):
    """
    This function prepcoessed a curve beta to set of closed curves

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param T: number of samples (default = 100)

    :rtype: numpy ndarray
    :return betanew: projected beta
    :return qnew: projected srvf
    :return A: alignment matrix (not used currently)

    """
    T = np.shape(beta)[1]
    try:
        q = cf.curve_to_q(beta)[0]
        qnew = project_curve(q)
    except:
        q = cf.curve_to_q(beta)
        qnew = project_curve(q)
    x = cf.q_to_curve(qnew)
    a = -1 * cf.calculatecentroid(x)
    betanew = x + np.tile(a, [T, 1]).T
    A = np.eye(2)

    return (betanew, qnew, A)


def geodDistance(X1,Y1,X2,Y2,k=5):

    F1 = np.column_stack((X1,Y1)).T
    F2 = np.column_stack((X2,Y2)).T
    T = len(X1)
    try:
        d,path,pathsq,E = path_straightening(F1,F2,'NULL',init="geod",T=T,k=int(k))
    except:
        d,path,pathsq,E = path_straightening(F2,F1,'NULL',init="geod",T=T,k=int(k))

    return d,path,pathsq,E