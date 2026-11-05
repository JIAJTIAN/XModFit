####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
from Chemical_Formula import Chemical_Formula
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
from utils import find_minmax, calc_rho, create_steps
import time

from numba import njit


@njit(parallel=False, cache=False)
def ellipsoid_ml_asaxs(q, Rx, RzRatio, rho, eirho, adensity, mu, wt):
    """
    Orientation averaged intensities of concentric ellipsoids of revolution (semi-axes: Rx, Rx, Rz).
    The last entry of rho/eirho/adensity is the surrounding medium (its Rx is not used).
    The orientation average over cos(alpha) in [0,1] is done by Gauss-Legendre quadrature (mu, wt).
    Returns <|F_rho|^2>, <F_eirho^2>, <Re(F_rho*F_adensity^*)>, <F_adensity^2> in (el*Angs^3)^2 (the cross-term is
    defined as in Sphere_Uniform)
    """
    NLayers = len(rho)
    fft = np.zeros_like(q)
    ffs = np.zeros_like(q)
    ffc = np.zeros_like(q)
    ffr = np.zeros_like(q)
    for i in range(len(q)):
        for j in range(len(mu)):
            cos2 = mu[j] * mu[j]
            sin2 = 1.0 - cos2
            tRx = 0.0
            tRz = 0.0
            tft = 0.0 + 0.0j
            tfs = 0.0
            tfr = 0.0
            for k in range(NLayers - 1):
                tRx += Rx[k]
                tRz += Rx[k] * RzRatio[k]
                V = 4.0 * np.pi * tRz * tRx ** 2 / 3.0
                rt = np.sqrt(tRx * tRx * sin2 + tRz * tRz * cos2)
                x = q[i] * rt
                if x < 1e-6:
                    fac = V
                else:
                    fac = 3.0 * V * (np.sin(x) - x * np.cos(x)) / x ** 3
                tft += (rho[k] - rho[k + 1]) * fac
                tfs += (eirho[k] - eirho[k + 1]) * fac
                tfr += (adensity[k] - adensity[k + 1]) * fac
            fft[i] += wt[j] * (tft.real ** 2 + tft.imag ** 2)
            ffs[i] += wt[j] * tfs ** 2
            ffc[i] += wt[j] * tft.real * tfr
            ffr[i] += wt[j] * tfr ** 2
    return fft, ffs, ffc, ffr


class Ellipsoid_Uniform2: #Please put the class name same as the function name
    def __init__(self, x=0, Np=20, Nalf=100, error_factor=1.0, term='Total', dist='Gaussian', Energy=None, relement='Au',
                 NrDep=False, norm=1.0, norm_err=0.01, sbkg=0.0, cbkg=0.0, abkg=0.0, D=1.0, phi=0.1, U=-1.0,
                 SF='None', mpar={'Layers':{'Material': ['Au', 'H2O'], 'Density': [19.32, 1.0], 'SolDensity': [1.0, 1.0],
                                  'Rmoles': [1.0, 1.0], 'R': [1.0, 0.0], 'RzRatio': [1.0, 1.0], 'Rsig': [0.0, 0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered ellipsoidal (prolate/oblate) nanoparticles with
        different materials. Each of the layers has its own width of the size distribution ('Rsig').

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Energy      : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np          : No. of Gauss-Hermite points with which the size distribution will be computed. Default: 20.
                      Increase it when q_max*Rsig is larger than ~2 (the intensity at the highest q needs more points)
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist        : The probability distribution function of the layer thicknesses. Default: Gaussian
        Nalf        : No. of Gauss-Legendre points used for the orientation averaging. Default: 100
        norm        : The density of the nanoparticles in nanoMolar (nanoMoles/Liter)
        norm_err    : Percentage of error on normalization to simulated energy dependent SAXS data
        sbkg        : Constant incoherent background for SAXS-term
        cbkg        : Constant incoherent background for cross-term
        abkg        : Constant incoherent background for Resonant-term
        error_factor: Error-factor to simulate the error-bars
        term        : 'SAXS-term' or 'Cross-term' or 'Resonant-term' or 'Total'
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs, the equatorial thickness of each layer),
                        Height to Radii ratio ('RzRatio', polar thickness/equatorial thickness of each layer), and
                        Width of the distribution of the thickness of each layer ('Rsig'). All the layers follow one
                        common random variable z (z~N(0,1)) so the thickness of layer k is R_k+Rsig_k*z for 'Gaussian'
                        (Rsig_k in Angs) and R_k*exp(Rsig_k*z) for 'LogNormal' (Rsig_k is dimensionless).
                        Layers with Rsig=0 keep a fixed thickness.
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        self.Nalf = Nalf
        self.norm = norm
        self.norm_err = norm_err
        self.sbkg = sbkg
        self.cbkg = cbkg
        self.abkg = abkg
        self.dist = dist
        self.Np = Np
        self.Energy = Energy
        self.relement = relement
        self.NrDep = NrDep in (True, 'True')
        self.error_factor = error_factor
        self.D = D
        self.phi = phi
        self.U = U
        self.__mpar__ = mpar  # If there is any multivalued parameter
        self.SF = SF
        self.term = term
        self.choices = {'dist': ['Gaussian', 'LogNormal'], 'NrDep': [True, False],
                        'SF': ['None', 'Hard-Sphere', 'Sticky-Sphere'],
                        'term': ['Total', 'SAXS-term', 'Cross-term', 'Resonant-term']}  # If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
        self.__cf__ = Chemical_Formula()
        self.__fit__ = False
        self.output_params = {'scaler_parameters': {}}
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params = Parameters()
        self.params.add('norm', value=self.norm, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('D', value=self.D, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg', value=self.sbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key != 'Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=-np.inf, max=np.inf, expr=None, brute_step=0.1)

    @lru_cache(maxsize=10)
    def calc_Rdist(self, R, Rsig, dist, N):
        """
        Calculates the thickness of all the layers for each of the N points (Gauss-Hermite) of the common random
        variable z~N(0,1) which drives the size distribution. Returns the thicknesses (N x Nlayers), the
        normalized weights (N) and the equatorial radius of the particle for each of the points.
        """
        R = np.array(R)
        Rsig = np.array(Rsig)
        if np.any(np.abs(Rsig[:-1]) > 1e-3):
            z, w = np.polynomial.hermite_e.hermegauss(N)
            w = w / np.sum(w)
            if dist == 'LogNormal':
                Rd = R[None, :] * np.exp(Rsig[None, :] * z[:, None])
            else:
                Rd = np.clip(R[None, :] + Rsig[None, :] * z[:, None], 0.0, None)
            Rd[:, -1] = R[-1]  # the thickness of the surrounding medium is not used
            return Rd, w, np.sum(Rd[:, :-1], axis=1)
        else:
            return R[None, :], np.ones(1), np.array([np.sum(R[:-1])])

    @lru_cache(maxsize=10)
    def new_ellipsoid(self, q, R, RzRatio, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=100):
        q = np.array(q)
        Rd, rdist, totalR = self.calc_Rdist(R, Rsig, dist, Np)
        mu, wt = np.polynomial.legendre.leggauss(Nalf)
        mu, wt = (mu + 1) / 2.0, wt / 2.0  # Gauss-Legendre points for cos(alpha) in [0,1]
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (2.818e-5 * 1.0e-8) ** 2
        for i in range(len(rdist)):
            fft, ffs, ffc, ffr = ellipsoid_ml_asaxs(q, Rd[i], np.array(RzRatio), np.array(rho), np.array(eirho),
                                                    np.array(adensity), mu, wt)
            form = form + rdist[i] * fft
            eiform = eiform + rdist[i] * ffs
            aform = aform + rdist[i] * ffr
            cform = cform + rdist[i] * ffc
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)  # in cm^2

    @lru_cache(maxsize=10)
    def new_ellipsoid_dict(self, q, R, RzRatio, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=100):
        form, eiform, aform, cform = self.new_ellipsoid(q, R, RzRatio, Rsig, rho, eirho, adensity, dist=dist, Np=Np,
                                                        Nalf=Nalf)
        return {'Total': form, 'SAXS-term': eiform, 'Resonant-term': aform, 'Cross-term': cform}

    def update_params(self):
        """Update parameters based on fitting values."""
        mkey = self.__mkeys__[0]
        Nmpar = len(self.__mpar__[mkey]['Material'])
        for key, attr in [('Density', '__density__'), ('SolDensity', '__solDensity__'), ('Rmoles', '__Rmoles__'),
                          ('R', '__R__'), ('RzRatio', '__RzRatio__'), ('Rsig', '__Rsig__')]:
            setattr(self, attr, [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
        self.__material__ = [self.__mpar__[mkey]['Material'][i] for i in range(Nmpar)]

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        self.update_params()
        rho, eirho, adensity, rhor, eirhor, adensityr, cdensityr = calc_rho(R=tuple(self.__R__), material=tuple(self.__material__),
                                                                      relement=self.relement,
                                                                      density=tuple(self.__density__),
                                                                      sol_density=tuple(self.__solDensity__),
                                                                      Energy=self.Energy, Rmoles=tuple(self.__Rmoles__),
                                                                      NrDep=self.NrDep)
        Rd, rdist, major_d = self.calc_Rdist(tuple(self.__R__), tuple(self.__Rsig__), self.dist, self.Np)
        minor_d = np.sum(Rd[:, :-1] * np.array(self.__RzRatio__)[None, :-1], axis=1)
        major = np.sum(rdist * major_d)
        minor = np.sum(rdist * minor_d)
        self.output_params['scaler_parameters']['Diameter (Angstroms)'] = 2 * major
        self.output_params['scaler_parameters']['Height (Angstroms)'] = 2 * minor
        self.output_params['scaler_parameters']['Diameter_Std (Angstroms)'] = 2 * np.sqrt(np.sum(rdist * (major_d - major) ** 2))
        self.output_params['scaler_parameters']['Height_Std (Angstroms)'] = 2 * np.sqrt(np.sum(rdist * (minor_d - minor) ** 2))
        if type(self.x) == dict:
            xkeys = list(self.x.keys())
            xref = self.x[xkeys[0]]
        else:
            xref = self.x
        sqft = self.new_ellipsoid_dict(tuple(xref), tuple(self.__R__), tuple(self.__RzRatio__), tuple(self.__Rsig__),
                                       tuple(rho), tuple(eirho), tuple(adensity), dist=self.dist, Np=self.Np, Nalf=self.Nalf)
        if self.SF == 'Hard-Sphere':
            struct = hard_sphere_sf(xref, D=self.D, phi=self.phi)
        elif self.SF == 'Sticky-Sphere':
            struct = sticky_sphere_sf(xref, D=self.D, phi=self.phi, U=self.U, delta=0.01)
        else:
            struct = np.ones_like(xref)
        pref = self.norm * 1e-9 * 6.022e20  # nanoMolar to particles per cubic-cm
        bkg = {'Total': self.sbkg, 'SAXS-term': self.sbkg, 'Cross-term': self.cbkg, 'Resonant-term': self.abkg}
        sqfs = {key: pref * sqft[key] * struct + bkg[key] for key in sqft.keys()}  # in cm^-1
        if type(self.x) == dict:
            sqf = {key: sqfs[key] for key in self.x.keys()}
        else:
            sqf = sqfs[self.term]
        if not self.__fit__:  # Generate all the quantities below while not fitting
            self.output_params['Distribution'] = {'x': major_d, 'y': rdist,
                                                  'names': ['Equatorial radius (Angs)', 'Probability']}
            signal = sqfs['Total']
            minsignal = np.min(signal)
            normsignal = signal / minsignal
            norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
            sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
            meta = {'Energy': self.Energy}
            if self.Energy is not None:
                self.output_params['simulated_w_err_%.4fkeV' % self.Energy] = {'x': xref, 'y': sqerr * minsignal,
                                                                               'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                                               'meta': meta}
            else:
                self.output_params['simulated_w_err'] = {'x': xref, 'y': sqerr * minsignal,
                                                         'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                         'meta': meta}
            for key in sqfs.keys():
                self.output_params[key] = {'x': xref, 'y': sqfs[key]}
            self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1]}
            self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1]}
            self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]}
            self.output_params['Structure_Factor'] = {'x': xref, 'y': struct}
            for name, vals in [('Rmoles_radial', self.__Rmoles__), ('RzRatio_radial', self.__RzRatio__),
                               ('Density_radial', self.__density__)]:
                xtmp, ytmp = create_steps(x=list(self.__R__[:-1]), y=list(vals[:-1]))
                self.output_params[name] = {'x': xtmp, 'y': ytmp}
        return sqf


if __name__=='__main__':
    x=np.logspace(-3,0.0,200)
    fun=Ellipsoid_Uniform2(x=x)
    print(fun.y())
