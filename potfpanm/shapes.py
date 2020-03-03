import numpy as np
from copy import deepcopy

class airfoil(object):
    """
    Airfoil class
    Creates a 2D panel mesh for an airfoil from a given set of x,y values.
    It will make a transformation of the shape so it has LE at (0,0) and TE at (1,+-TE_thick), this might involve both
    translation, rotation and scaling. These values can be adjusted after the class is created.
    It will also insure that the points are order in clockwise direction.

    Parameters
    ----------
    :param x ndarray,list: x-coordinate describing the airfoil shape. If no y is passed it is assumed
    that x is a set of complex numbers of the shape.
    :param y ndarray, list: y-coordinate describing the airfoil shape.

    """
    def __init__(self, x, y=None):
        # Convert to complex number
        if y is None:
            z = np.asfarray(x, dtype=complex)
        else:
            z = np.asfarray(x) + 1j * np.asfarray(y)

        # Checking that coordinate direction is clock-wise (otherwise reverse z array)
        if z.imag.argmax() < z.imag.argmin(): z = z[::-1]

        # Compute center, rotation and scale (x_min at (0,0), start and end point in the middel between x-axis,
        # scale -> x_max-x_min=1)
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
        """Shape-edges as ndarray of complex numbers"""
        return self._z * self.scale * self._rot.conj() + self._center()

    @property
    def x(self):
        """x-component of shape-edges as ndarray of floats"""
        return self.z.real

    @property
    def y(self):
        """y-component of shape-edges as ndarray of floats"""
        return self.z.imag

    @property
    def xy(self):
        """Shape-edges as 2D ndarray of floats"""
        z = self.z
        return np.array([z.real, z.imag]).T

    @property
    def pz(self):
        """Shape-panel-center as ndarray of complex numbers"""
        z = self.z
        return (z[1:] + z[:-1]) / 2

    @property
    def px(self):
        """x-component of shape-panel-center as ndarray of floats"""
        return self.pz.real

    @property
    def py(self):
        """y-component of shape-panel-center as ndarray of floats"""
        return self.pz.imag

    @property
    def dz(self):
        """Panel-length as ndarray of complex numbers"""
        return np.diff(self.z)

    @property
    def dx(self):
        """x-component of panel-length as ndarray of floats"""
        return self.dz.real

    @property
    def dy(self):
        """y-component of panel-length as ndarray of floats"""
        return self.dz.imag

    @property
    def dp(self):
        """Panel-length as ndarray of floats (dp=abs(dz))"""
        return abs(self.dz)

    @property
    def tan(self):
        """Tangent-complex-vector as ndarray of complex numbers"""
        return self.dz / self.dp

    @property
    def nor(self):
        """Normal-vector as ndarray of complex numbers (points out of the shape)"""
        return self.tan * 1j

    @property
    def rot(self):
        """rotation instance, the angle of the whole shape can be set by either angle_deg and angle_rad
        (degrees and radians respecitivly)"""
        return self._rot

    @property
    def center(self):
        """center instance, the shape can be translated in both x and y direction though x, y as a float or z as
        complex number"""
        return self._center

    @property
    def copy(self):
        return deepcopy(self)

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

    def hvplot(self, ptype="shape"):
        """
        Plots the shape (Notices, it requries holoviews)
        Parameters
        ----------
        :param ptype str: Plot type. The following are excepted:
        "shape": Plots the shape-edges with equal aspect-ratio
        "vectors": Plots shape-edges, as well as panel-centers and tangential and normal vectors.
        """
        try:
            import holoviews as hv
        except ImportError:
            raise ImportError("To use the plot method holoviews is requried, see more here: http://holoviews.org/")

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
        """Shape rotation angle in radians, this value can be changed"""
        return np.angle(self._rot)

    @angle_rad.setter
    def angle_rad(self, val):
        self._rot = np.exp(1j * val).conj()

    @property
    def angle_deg(self):
        """Shape rotation angle in degrees, this value can be changed"""
        return np.angle(self._rot, deg=True)

    @angle_deg.setter
    def angle_deg(self, val):
        self._rot = np.exp(1j * np.deg2rad(val)).conj()

    def __call__(self):
        """Returns complex rotation vector"""
        return self._rot

    def conj(self):
        """Complex conjugate of rotation complex-vector"""
        return self._rot.conj()

    def __repr__(self):
        return self._rot.__repr__()


class center(object):

    def __init__(self, cen):
        self._cen = cen

    @property
    def x(self):
        """x-component of shape translation, this value can be changed"""
        return self._cen.real

    @x.setter
    def x(self, val):
        self._cen = val + 1j * self._cen.imag

    @property
    def y(self):
        """y-component of shape translation, this value can be changed"""
        return self._cen.imag

    @y.setter
    def y(self, val):
        self._cen = self._cen.real + 1j * val

    @property
    def z(self):
        """Shape translation as complex number, this value can be changed"""
        return self._cen

    @z.setter
    def z(self, val):
        self._cen = val

    def __call__(self):
        """Returns the translations as complex number"""
        return self._cen

    def __repr__(self):
        return self._cen.__repr__()
