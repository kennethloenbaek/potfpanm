import numpy as np

class airfoil(object):
    def __init__(self, x, y=None):
        # Convert to complex number
        if y is None:
            z = x
        else:
            z = x + 1j * y

        # Checking that coordinate direction is clock-wise (otherwise reverse z array)
        if z.imag.argmax() < z.imag.argmin(): z = z[::-1]

        # Compute center, rotation and scale (x_min at (0,0), start and end point in the middel between x-axis, scale -> x_max-x_min=1)
        # Center
        cen = z[z.real.argmin()]
        z -= cen
        # Rotation
        rot = ((z[0] + z[-1]) / 2).conj()
        rot /= abs(rot)
        z *= rot
        # Scale
        scale = z.real.max() - z.real.min()
        z /= scale

        # Assing variables
        self._z = z
        self._z_original = z.copy()
        self._center = center(cen)
        self._rot = rotation(rot)
        self.scale = scale

    @property
    def z(self):
        return self._z * self.scale * self._rot.conj() + self._center()

    @property
    def x(self):
        return self.z.real

    @property
    def y(self):
        return self.z.imag

    @property
    def xy(self):
        z = self.z
        return np.array([z.real, z.imag]).T

    @property
    def pz(self):
        z = self.z
        return (z[1:] + z[:-1]) / 2

    @property
    def px(self):
        return self.pz.real

    @property
    def py(self):
        return self.pz.imag

    @property
    def dz(self):
        return np.diff(self.z)

    @property
    def dx(self):
        return self.dz.real

    @property
    def dy(self):
        return self.dz.imag

    @property
    def dp(self):
        return abs(self.dz)

    @property
    def tan(self):
        return self.dz / self.dp

    @property
    def nor(self):
        return self.tan * 1j

    @property
    def rot(self):
        return self._rot

    @property
    def center(self):
        return self._center

    # Induction matrix methods

    @staticmethod
    def pi2pj_matrix(pi, pj):
        return (np.ones([len(pj), len(pi)]) * pi).T - pj

    def induction_matrix(self, pi):
        out = np.zeros([len(pi), len(self.z)], dtype=np.complex)
        pi2pj = self.pi2pj_matrix(pi, self.pz) * self.tan.conj()
        out[:, :-1] += self._induc_coef(pi2pj, self.dp, 1) * self.tan
        out[:, 1:] += self._induc_coef(pi2pj, self.dp, -1) * self.tan
        return out

    def induction_matrix_to_solve(self):
        out = np.zeros([len(self.z), len(self.z)], dtype=np.complex)
        mat = self.induction_matrix(self.pz)
        out[:-1, :] = (mat.T * self.tan).T
        out[-1, 0] = 1
        out[-1, -1] = -1
        return out

    @staticmethod
    def _induc_coef(pi2pj, dpj, sign):
        factor1 = np.log((pi2pj + dpj / 2) / (pi2pj - dpj / 2))
        factor2 = 1 / 2 - sign * pi2pj / dpj
        return -(1j / (2 * np.pi) * ((factor1 * factor2) + sign)).conj()

    def plot(self, ptype="shape"):
        import holoviews as hv
        if ptype == "shape":
            # Shape of airfoil (edges)
            return hv.Curve(self.xy).opts(aspect='equal', width=1000, height=500, padding=0.1)
        elif ptype == "vectors":
            # Vectors at panel centeres
            pc = hv.Scatter(np.array([self.px, self.py]).T, kdims="px", vdims="py").opts(size=5)
            vf = hv.VectorField(np.array([self.px, self.py, np.angle(self.tan), np.abs(self.tan)]).T, label="Tangent")
            vf.opts(rescale_lengths=False, magnitude=hv.dim('Magnitude') * 0.01 * self.scale, pivot="tail")
            vfnor = hv.VectorField(np.array([self.px, self.py, np.angle(self.nor), np.abs(self.nor)]).T, label="Normal")
            vfnor.opts(rescale_lengths=False, magnitude=hv.dim('Magnitude') * 0.01 * self.scale, pivot="tail",
                       color="red")  # , label="Normal")
            return self.plot("shape") * pc * vf * vfnor


class rotation(object):

    def __init__(self, rot):
        self._rot = rot

    @property
    def angle_rad(self):
        return np.angle(self._rot)

    @angle_rad.setter
    def angle_rad(self, val):
        self._rot = np.exp(1j * val).conj()

    @property
    def angle_deg(self):
        return np.angle(self._rot, deg=True)

    @angle_deg.setter
    def angle_deg(self, val):
        self._rot = np.exp(1j * np.deg2rad(val)).conj()

    def __call__(self):
        return self._rot

    def conj(self):
        return self._rot.conj()

    def __repr__(self):
        return self._rot.__repr__()


class center(object):

    def __init__(self, cen):
        self._cen = cen

    @property
    def x(self):
        return self._cen.real

    @x.setter
    def x(self, val):
        self._cen = val + 1j * self._cen.imag

    @property
    def y(self):
        return self._cen.imag

    @y.setter
    def y(self, val):
        self._cen = self._cen.real + 1j * val

    def __call__(self):
        return self._cen

    def __repr__(self):
        return self._cen.__repr__()
