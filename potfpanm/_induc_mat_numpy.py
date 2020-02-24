import numpy as np

def _pz(z):
    """
    Computes the panel center given panel edges
    :param z: panel edges as complex numbers (shape nz)
    :return: panel center as complex numbers (shape nz-1)
    """
    return (z[1:] + z[:-1]) / 2

def _dz(z):
    """
    Computes the panel lengths as complex numbers (abs(dz)=dp)
    :param z: panel edges as complex numbers (shape nz)
    :return: panel length as complex numbers (shape nz-1)
    """
    return np.diff(z)


def _tan(z):
    """
    Computes tangent as complex numbes
    :param z: panel edges as complex numbers (shape nz)
    :return: Computes tangent as complex numbes (shape nz-1)
    """
    dz = _dz(z)
    return dz / abs(dz)

def lin_vary_vortex_panel(pi2pj, dpj, sign):
    """
    Induction contribution from gam_j (sign=+1) or gam_j+1 (sign=-1) (related to pj) at pi with panel size dpj
    :param pi2pj: matrix with pi-pj (shape nixnj)
    :param dpj: vector of panel sizes (shape nj)
    :param sign: sign determining if contribution is from gam_j (+1) or gam_j+1 (-1)
    :return: unity induction factor at pi
    """
    factor1 = np.log((pi2pj + dpj / 2) / (pi2pj - dpj / 2))
    factor2 = 1 / 2 - sign * pi2pj / dpj
    return -(1j / (2 * np.pi) * ((factor1 * factor2) + sign)).conj()

def pi2pj_matrix(pi, pj):
    """
    Contructs the matrix pi-pj where i is the first index and j is the secound index
    :param pi: points as complex numbers (shape ni)
    :param pj: points as complex numbers (shape nj)
    :return: pi-pj matrix (shape ni,nj)
    """
    return (np.ones([len(pj), len(pi)]) * pi).T - pj

def induction_matrix(pi, z, indu_type):
    # Initilize output array
    out = np.zeros([len(pi), len(self.z)], dtype=np.complex)

    # Precomputed values
    pz = _pz(z)
    tan = _tan(z)

    # Compute pi-pj in global cordinatsystem
    pi2pj = pi2pj_matrix(pi, self.pi2pj_matrix(pi, self.pz) * self.tan.conj()
    out[:, :-1] += self._induc_coef(pi2pj, self.dp, 1) * self.tan
    out[:, 1:] += self._induc_coef(pi2pj, self.dp, -1) * self.tan
    return out

 def induction_matrix_multi_elements(pi, zs, indu_types):
    # Initlize output matrix
    out = np.zeros([len(pi), len(self.z)], dtype=np.complex)
    # compute induction from each shape
    ns = 0
    for i_shape, shape in enumerate(self._shapes):
        out[:, ns:ns + len(shape.z)] = shape.induction_matrix(pi)
        ns += len(shape.z)
    return out