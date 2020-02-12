import numpy as np
from copy import deepcopy, copy

class domain(object):
    def __init__(self, shapes=[], sings=[]):
        self._shapes = deepcopy(shapes)
        self._sings = deepcopy(sings)

    def add_airfoil(self, airfoil):
        self._shapes.append(deepcopy(airfoil))

    @property
    def z(self):
        if hasattr(self, "_sep_val"):
            return self._insert_sep_val("z")
        else:
            return np.concatenate([obj.z for obj in self._shapes])

    def _insert_sep_val(self, name):
        return np.concatenate(
            [np.concatenate([getattr(obj, name), np.array([self._sep_val])]) for obj in self._shapes])[:-1]

    def with_sep(self, val=np.nan + 1j * np.nan):
        obj = copy(self)
        obj._sep_val = val
        return obj

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
        if hasattr(self, "_sep_val"):
            return self._insert_sep_val("pz")
        else:
            return np.concatenate([obj.pz for obj in self._shapes])

    @property
    def px(self):
        return self.pz.real

    @property
    def py(self):
        return self.pz.imag

    @property
    def dz(self):
        if hasattr(self, "_sep_val"):
            return self._insert_sep_val("dz")
        else:
            return np.concatenate([obj.dz for obj in self._shapes])

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

    def plot(self, ptype="shape"):
        import holoviews as hv
        if ptype == "shape":
            # Shape of airfoil (edges)
            return hv.Curve(self.with_sep().xy).opts(aspect='equal')
        elif ptype == "vectors":
            # Vectors at panel centeres
            pc = hv.Scatter(np.array([self.px, self.py]).T, kdims="px", vdims="py").opts(size=5)
            vf = hv.VectorField(np.array([self.px, self.py, np.angle(self.tan), np.abs(self.tan)]).T, label="Tangent")
            vf.opts(rescale_lengths=False, magnitude=hv.dim('Magnitude') * 0.01, pivot="tail")
            vfnor = hv.VectorField(np.array([self.px, self.py, np.angle(self.nor), np.abs(self.nor)]).T, label="Normal")
            vfnor.opts(rescale_lengths=False, magnitude=hv.dim('Magnitude') * 0.01, pivot="tail", color="red")
            return self.plot("shape") * pc * vf * vfnor

    @staticmethod
    def pi2pj_matrix(pi, pj):
        return (np.ones([len(pj), len(pi)]) * pi).T - pj

    def induction_matrix(self, pi):
        # Initlize output matrix
        out = np.zeros([len(pi), len(self.z)], dtype=np.complex)
        # compute induction from each shape
        ns = 0
        for i_shape, shape in enumerate(self._shapes):
            out[:, ns:ns + len(shape.z)] = shape.induction_matrix(pi)
            ns += len(shape.z)
        return out

    def induction_matrix_to_solve(self):
        # Initlize output matrix
        out = np.zeros([len(self.pz), len(self.pz)], dtype=np.complex)
        # Getting number of shapes in domain
        n_shapes = len(self._shapes)
        # Setting Kutta condition
        ns = 0
        for i_shape, shape in enumerate(self._shapes):
            mat = (shape.induction_matrix(self.pz).T * self.tan.conj()).T
            out[:, ns:ns + len(shape.pz)] = mat[:, :-1].copy()
            out[:, ns] -= mat[:, -1].copy()
            ns += len(shape.pz)
        return out

    def _solution(self, Uinf, aoa_deg):
        sigma = np.linalg.solve(self.induction_matrix_to_solve().imag, -self._rhs(Uinf, aoa_deg).imag)
        sigma_out = np.empty_like(self.z)
        nz = 0
        npz = 0
        for i_shape, shape in enumerate(self._shapes):
            sigma_out[nz:nz + len(shape.z) - 1] = sigma[npz:npz + len(shape.pz)].copy()
            sigma_out[nz + len(shape.z) - 1] = sigma[npz].copy()
            nz += len(shape.z)
            npz += len(shape.pz)
        return solution(shapes=self._shapes, sings=self._sings, sigma=sigma_out, Uinf=Uinf, aoa_deg=aoa_deg)

    def polar(self, aoas_deg, Uinfs):
        if isinstance(aoas_deg, (float, int)) and isinstance(Uinfs, (float, int)):
            return self._solution(Uinf=Uinfs, aoa_deg=aoas_deg)
        elif isinstance(Uinfs, (float, int)):
            Uinfs = np.full_like(aoas_deg, Uinfs)
        elif isinstance(aoas_deg, (float, int)):
            aoas_deg = np.full_like(Uinfs, aoas_deg)

        sols = np.empty_like(aoas_deg, dtype=object)
        for i, (aoa, uinf) in enumerate(zip(aoas_deg, Uinfs)):
            sols[i] = self._solution(Uinf=uinf, aoa_deg=aoa)
        return solutions(sols)

    def _rhs(self, Uinf, aoa_deg):
        return Uinf * np.exp(1j * np.deg2rad(aoa_deg)) / self.tan


class solution(domain):
    def __init__(self, shapes, sings, sigma, Uinf, aoa_deg):
        super().__init__(shapes, sings)
        self._sigma = sigma
        self._Uinf = Uinf
        self._aoa_deg = aoa_deg

    def add_airfoil(self, air):
        raise ValueError("Airfoils can not be added to solution instance")

    def u(self, z):
        return self.u_free + self.induction_matrix(z).dot(self._sigma)

    @property
    def u_free(self):
        return self._Uinf * np.exp(1j * np.deg2rad(self._aoa_deg))

    def u_surface(self, offset=None, offset_stag=None):
        if offset_stag is not None:
            pass
        else:
            if offset is None: offset = 0.0
            pz = self.pz + self.nor * offset
            return self.u(pz)

    def pressure(self, z, rho=1.225):
        return 1/2*rho*self.u_free - 1/2*rho*np.abs(self.u(z))**2

    def Cp(self, u):
        return 1 - (np.abs(u) / np.abs(self.u_free)) ** 2

    def forces(self, rho=1.225):
        Fs = np.empty_like(self._shapes, dtype=np.complex)
        for ishape, shape in enumerate(self._shapes):
            Fs[ishape] = -np.sum(self.pressure(shape.pz, rho=rho) * shape.nor * shape.dp)
        return Fs

    def forcesld(self, rho=1.225):
        return self.forces(rho=rho)*self.u_free.conj()/self._Uinf

    @property
    def Cforces(self):
        rho = 1.0
        forceslds = self.forcesld(rho=rho)
        Cforces = np.empty_like(forceslds)
        for i_shape, (forcesld, shape) in enumerate(zip(forceslds, self._shapes)):
            Cforces[i_shape] = forcesld / (1 / 2 * rho * shape.scale * np.abs(self.u_free) ** 2)
        return Cforces

    def lift(self, rho=1.225):
        return self.forcesld(rho=rho).imag

    @property
    def Cl(self):
        return self.Cforces.imag

    def drag(self, rho=1.225):
        return self.forcesld(rho=rho).real

    @property
    def Cd(self):
        return self.Cforces.real

    def u_grid(self, x=None, y=None, return_grid=True):
        # TODO: Add cluster parameter
        if x is None or isinstance(x, (float, int)):
            x_min = self.x.min()
            x_max = self.x.max()
            x_range = x_max - x_min
            if isinstance(x, int):
                nx = x
            elif isinstance(x, list) and len(x) < 3:
                nx = x[0]
            else:
                nx = 100
            if isinstance(x, float):
                x_mult = x
            elif isinstance(x, list) and len(x) < 3:
                x_mult = x[1]
            else:
                x_mult = 1.0
            x = np.linspace(x_min - x_range * x_mult, x_max + x_range * x_mult, nx)
        if y is None or isinstance(y, (float, int)):
            y_min = self.y.min()
            y_max = self.y.max()
            y_range = y_max - y_min
            if isinstance(y, int):
                ny = y
            elif isinstance(y, list) and len(y) < 3:
                ny = y[0]
            else:
                ny = 100
            if isinstance(y, float):
                y_mult = y
            elif isinstance(y, list) and len(y) < 3:
                y_mult = y[1]
            else:
                y_mult = 1.0
            y = np.linspace(y_min - y_range * y_mult, y_max + y_range * y_mult, ny)

        xg, yg = np.meshgrid(x, y)
        zg = xg.flatten() + 1j * yg.flatten()

        u = self.u(zg).reshape(xg.shape)

        if return_grid is True:
            return xg, yg, u
        else:
            return u

    def plot_surface(self, x="x", y="Cp", offset=None, offset_stag=None):
        import holoviews as hv
        if y == "u":
            return hv.Curve(np.array([self.px, np.abs(self.u_surface(offset=offset, offset_stag=offset_stag))]).T).opts(
                width=500, height=500, padding=0.1)
        if y == "Cp":
            u = self.u_surface(offset=offset, offset_stag=offset_stag)
            return hv.Curve(np.array([self.px, self.Cp(u)]).T, kdims=["x"], vdims="Cp").opts(invert_yaxis=True)

    def plot_grid(self, x=None, y=None, vector_field=False):
        import holoviews as hv
        # Get velocity field
        xg, yg, u = self.u_grid(x=x, y=y)
        qm = hv.Image((xg[0, :], yg[:, 0], np.abs(u))).opts(aspect='equal')
        if vector_field is True:
            vf = hv.VectorField((xg, yg, np.angle(u), np.abs(u)))
            return qm * vf
        elif isinstance(vector_field, int):
            step = vector_field
            vf = hv.VectorField((xg[::step, ::step], yg[::step, ::step], np.angle(u[::step, ::step]), np.abs(u[::step, ::step])))
            return qm * vf
        else:
            return qm


class solutions(object):
    def __init__(self, solutions = []):
        self._solutions = deepcopy(solutions)

    def __getitem__(self, item):
        if isinstance(item, str):
            for i_sol, sol in enumerate(self._solutions):
                if i_sol == 0:
                    val = getattr(sol, item)
                    out = np.empty([len(self._solutions)]+list(val.shape), dtype=val.dtype)
                    out[i_sol] = val.copy()
                else:
                    out[i_sol] = getattr(sol, item)
            return out
        if isinstance(item, int):
            return self._solutions[item]


    def plot_polar(self, iair=0):
        import holoviews as hv
        # Creating aoa vs. Cl curve
        aoa = self["_aoa_deg"]
        Cl = self["Cl"][:, iair]
        curve = hv.Curve(np.array([aoa, Cl]).T, kdims="AoA", vdims="Cl")
        points = hv.Points((aoa, Cl), kdims=["AoA", "Cl"]).opts(tools=['tap', 'hover'], size=6)
        # Creating selection
        selection = hv.streams.Selection1D(source=points)
        # Callback function
        def plot_Cp(index):
            if not index:
                index = list(range(len(aoa)))
            out = dict()
            for ind in index:
                out[aoa[ind]] = self[ind].plot_surface()
            return hv.NdOverlay(out, kdims='AoA').relabel('x vs Cp (AoA-legend)')
        # Creating dynamic map
        dmap = hv.DynamicMap(plot_Cp, streams=[selection])
        return curve*points + dmap
