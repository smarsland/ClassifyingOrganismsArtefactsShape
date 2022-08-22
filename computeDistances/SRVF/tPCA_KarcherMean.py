import numpy as np
from copy import deepcopy
import fdasrsf.curve_functions as cf
import fdasrsf.curve_stats as cs
import matplotlib.colors as plc
from numpy import zeros, sqrt, fabs, cos, sin, tile, vstack, empty, inf, mean, arange
from numpy.linalg import svd
from numpy import (
    zeros,
    sqrt,
    fabs,
    cos,
    sin,
    tile,
    vstack,
    empty,
    cov,
    inf,
    mean,
    arange,
)
from numpy.random import randn
import fdasrsf.utility_functions as uf
from joblib import Parallel, delayed
import collections
import math
import random
import pandas as pd

###################
# FDASRSF Functions
###################


class fdacurve:
    """
    This class provides alignment methods for open and closed curves using the SRVF framework

    Usage: obj = fdacurve(beta, mode, N, scale)
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')
    :param N: resample curve to N points
    :param scale: scale curve to length 1 (true/false)
    :param q:        (n,T,K) matrix defining n dimensional srvf on T samples with K srvfs
    :param betan:     aligned curves
    :param qn:        aligned srvfs
    :param basis:     calculated basis
    :param beta_mean: karcher mean curve
    :param q_mean:    karcher mean srvf
    :param gams:      warping functions
    :param v:         shooting vectors
    :param C:         karcher covariance
    :param s:         pca singular values
    :param U:         pca singular vectors
    :param coef:      pca coefficients
    :param qun:       cost function
    :param samples:   random samples
    :param gamr:      random warping functions
    :param cent:      center
    :param scale:     scale
    :param E:         energy

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  26-Aug-2020
    """

    def __init__(self, beta1, mode="O", scale=False):
        """
        fdacurve Construct an instance of this class
        :param beta: (n,T,K) matrix defining n dimensional curve on T samples with K curves
        :param mode:  Open ('O') or closed curve ('C') (default 'O')
        :param N: resample curve to N points
        :param scale: scale curve to length 1 (true/false)
        """
        self.mode = mode
        self.scale = scale

        K = beta1.shape[2]
        n = beta1.shape[0]
        N = beta1.shape[1]
        q = zeros((n, N, K))
        for ii in range(0, K):
            a = -cf.calculatecentroid(beta1[:, :, ii])
            beta1[:, :, ii] += tile(a, (N, 1)).T
            try:
                q[:, :, ii] = cf.curve_to_q(beta1[:, :, ii], self.scale, self.mode)
            except:
                q[:, :, ii] = cf.curve_to_q(beta1[:, :, ii], self.mode)[0]

        self.q = q
        self.beta = beta1

    def karcher_mean(self, parallel=False, cores=-1, method="DP"):
        """
        This calculates the mean of a set of curves
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ["O", "C"]
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # Initialize mu as one of the shapes
        mu = self.q[:, :, 0]
        betamean = self.beta[:, :, 0]
        itr = 0

        gamma = zeros((T, N))
        maxit = 20

        sumd = zeros(maxit + 1)
        v = zeros((n, T, N))
        normvbar = zeros(maxit + 1)

        delta = 0.5
        tolv = 1e-4
        told = 5 * 1e-3

        print("Computing Karcher Mean of %d curves in SRVF space.." % N)
        while itr < maxit:
            print("updating step: %d" % (itr + 1))

            if iter == maxit:
                print("maximal number of iterations reached")

            mu = mu / sqrt(cf.innerprod_q2(mu, mu))
            if mode == 1:
                self.basis = cf.find_basis_normal(mu)
            else:
                self.basis = []

            sumv = zeros((n, T))
            sumd[0] = inf
            sumd[itr + 1] = 0
            out = Parallel(n_jobs=cores)(
                delayed(karcher_calc)(
                    self.beta[:, :, n],
                    self.q[:, :, n],
                    betamean,
                    mu,
                    self.basis,
                    mode,
                    method,
                )
                for n in range(N)
            )
            v = zeros((n, T, N))
            for i in range(0, N):
                v[:, :, i] = out[i][0]
                sumd[itr + 1] = sumd[itr + 1] + out[i][1] ** 2

            sumv = v.sum(axis=2)

            # Compute average direction of tangent vectors v_i
            vbar = sumv / float(N)

            normvbar[itr] = sqrt(cf.innerprod_q2(vbar, vbar))
            normv = normvbar[itr]

            if normv > tolv and fabs(sumd[itr + 1] - sumd[itr]) > told:
                # Update mu in direction of vbar
                mu = (
                    cos(delta * normvbar[itr]) * mu
                    + sin(delta * normvbar[itr]) * vbar / normvbar[itr]
                )

                if mode == 1:
                    mu = cf.project_curve(mu)

                x = cf.q_to_curve(mu)
                a = -1 * cf.calculatecentroid(x)
                betamean = x + tile(a, [T, 1]).T
            else:
                break
            itr += 1

        self.q_mean = mu
        self.beta_mean = betamean
        self.v = v
        self.qun = sumd[0 : (itr + 1)]
        self.E = normvbar[0 : (itr + 1)]

        return

    def srvf_align(self, parallel=False, cores=-1, method="DP"):
        """
        This aligns a set of curves to the mean and computes mean if not computed
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ["O", "C"]
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # find mean
        if not hasattr(self, "beta_mean"):
            self.karcher_mean()

        self.qn = zeros((n, T, N))
        self.betan = zeros((n, T, N))
        self.gams = zeros((T, N))
        C = zeros((T, N))
        centroid2 = cf.calculatecentroid(self.beta_mean)
        self.beta_mean = self.beta_mean - tile(centroid2, [T, 1]).T
        q_mu = cf.curve_to_q(self.beta_mean)
        # align to mean

        out = Parallel(n_jobs=-1)(
            delayed(cf.find_rotation_and_seed_coord)(
                self.beta_mean, self.beta[:, :, n], mode, method
            )
            for n in range(N)
        )

        for ii in range(0, N):
            self.gams[:, ii] = out[ii][3]
            self.qn[:, :, ii] = out[ii][1]
            self.betan[:, :, ii] = out[ii][0]

        return


def karcher_calc(beta, q, betamean, mu, basis, closed, method):
    # Compute shooting vector from mu to q_i
    w, d = cf.inverse_exp_coord(betamean, beta, closed, method)

    # Project to tangent space of manifold to obtain v_i
    if closed == 0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return (v, d)


def shape_pca(v, C, beta, no=3):
    """
    Computes principal direction of variation specified by no. N is
    Number of shapes away from mean. Creates 2*N+1 shape sequence

    :param no: number of direction (default 3)
    """
    U1, s, V = svd(C)
    U = U1[:, 0:no]
    s = s[0:no]

    # express shapes as coefficients
    K = beta.shape[2]
    VM = mean(v, 2)
    x = zeros((no, K))
    for ii in range(0, K):
        tmpv = v[:, :, ii]
        Utmp = U.T
        x[:, ii] = Utmp.dot((tmpv.flatten() - VM.flatten()))

    return x


def karcher_cov(v):
    """
    This calculates the mean of a set of curves

    """
    M, N, K = v.shape
    tmpv = np.zeros((M * N, K))
    for i in range(0, K):
        tmp = v[:, :, i]
        tmpv[:, i] = tmp.flatten()

    return np.cov(tmpv)


#################
# Other Functions
#################


def align(sample, open_closed="C", PC=3, scaleCurve=False, returnDF=1):
    beta = deepcopy(sample)
    obj_c = fdacurve(beta, mode=open_closed, scale=scaleCurve)
    obj_c.srvf_align()
    obj_c.karcher_mean
    mn = obj_c.beta_mean
    aligned = obj_c.betan
    cov = karcher_cov(obj_c.v)
    coef = shape_pca(obj_c.v, cov, obj_c.beta, no=PC)
    mnx, mny = rescale(mn[0], mn[1])
    mn[0] = mnx
    mn[1] = mny
    if returnDF == 1:
        tpca = pd.DataFrame(coef).T
        if PC == 3:
            tpca = tpca.rename(columns={0: "PC1", 1: "PC2", 2: "PC3"})
        karchermean = pd.DataFrame([mn[0], mn[1]]).T
        karchermean = karchermean.rename(columns={0: "x", 1: "y"})
    else:
        tpca = deepcopy(coef)
        karchermean = deepcopy(mn)

    return aligned, karchermean, tpca


def formatData(dataFrame, varName="Name", xName="X", yName="Y"):
    # Pandas dataframe should have a column called "Name", "X","Y"

    names = np.unique(dataFrame[varName])

    cont_length = len(dataFrame[dataFrame[varName] == names[0]][xName])
    total_cont = len(names)

    beta = np.zeros((2, cont_length, total_cont))

    for i, nm in enumerate(names):
        beta[0, :, i] = list(dataFrame[dataFrame[varName] == nm][xName])
        beta[1, :, i] = list(dataFrame[dataFrame[varName] == nm][yName])

    return beta


def rescale(x, y, scaledHeightWidth=3, scaleDirection="both"):
    # This function rescales the contour so that it's between certain values (of a certain width/height)
    # and centred at (0,0). The 'scaleDirection' parameter is set to "both" as defult. This means that
    # it scales the max(height,width) to the desired width/height. The smaller side (heigh/width) will
    # then be scaled with the same scale factor, and hence lie within the desired range (and not greater).
    # If it's desired that the height will be of a certain length, or the width, then set the
    # scaleDirection paramter to "height" or "width" respectively.
    # Input: x, y coordinates, desired max height/width, and desired direction (height/width/max)
    # Output: rescaled x and y coordinates.

    if type(x) != np.ndarray:
        x = np.array(x)

    if type(y) != np.ndarray:
        y = np.array(y)

    if scaleDirection == "height":
        p = max(y) - min(y)
    elif scaleDirection == "width":
        p = max(x) - min(x)
    else:
        p = max(max(y) - min(y), max(x) - min(x))

    q = p / scaledHeightWidth

    x_ = x / q
    y_ = y / q
    mp = min(x_) + (max(x_) - min(x_)) / 2
    x_ = x_ - mp
    mp = min(y_) + (max(y_) - min(y_)) / 2
    y_ = y_ - mp

    return x_, y_
