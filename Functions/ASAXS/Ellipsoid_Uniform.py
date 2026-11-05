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
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
from utils import find_minmax, calc_rho, create_steps
import time

from numba import njit


@njit(parallel=False, cache=False)
def ff_ellipsoid_ml(q, Rx, RzRatio, rho, eirho, adensity, mu, wt):
    """
    Orientation averaged intensities of concentric ellipsoids of revolution (semi-axes: Rx, Rx, Rz) where, as for
    ff_sphere_ml of Sphere_Uniform, the last entry of rho/eirho/adensity is the surrounding medium.
    The orientation average over cos(alpha) in [0,1] is done by Gauss-Legendre quadrature (mu, wt).
    Returns <|F_rho|^2>, <F_eirho^2>, <F_adensity^2> and <Re(F_rho*F_adensity^*)> in (el*Angs^3)^2
    """
    NLayers = len(rho)
    fft = np.zeros_like(q)
    ffs = np.zeros_like(q)
    ffr = np.zeros_like(q)
    ffc = np.zeros_like(q)
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
            ffr[i] += wt[j] * tfr ** 2
            ffc[i] += wt[j] * tft.real * tfr
    return fft, ffs, ffr, ffc


class Ellipsoid_Uniform: #Please put the class name same as the function name
    def __init__(self, x=0, Np=20, Nalf=100, error_factor=1.0, term='Total', dist='Gaussian', Energy=None, relement='Au',
                 NrDep='True', norm=1.0, norm_err=0.01, sbkg=0.0, cbkg=0.0, abkg=0.0, D=1.0, phi=0.1, U=-1.0,
                 SF='None', Rsig=0.0,
                 mpar={'Layers':{'Material': ['Au', 'H2O'], 'Density': [19.32, 1.0], 'SolDensity': [1.0, 1.0],
                                 'Rmoles': [1.0, 1.0], 'R': [1.0, 0.0], 'RzRatio': [1.0, 1.0]}}):
        """
        Calculates the Energy dependent form factor of multilayered ellipsoidal (prolate/oblate) nanoparticles with
        different materials. For RzRatio=1 it is identical to Sphere_Uniform.

        x            : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement     : Resonant element of the nanoparticle. Default: 'Au'
        Energy       : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np           : No. of points with which the size distribution will be computed. Default: 20
        Nalf         : No. of Gauss-Legendre points used for the orientation averaging. Default: 100
        NrDep        : Energy dependence of the non-resonant element. Default= 'True' (Energy dependent), 'False' (Energy independent)
        dist         : The probability distribution function for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm         : The density of the nanoparticles in nanoMolar (nanoMoles/Liter)
        norm_err     : Percentage of error on normalization to simulated energy dependent SAXS data
        sbkg         : Constant incoherent background for SAXS-term
        cbkg         : Constant incoherent background for cross-term
        abkg         : Constant incoherent background for Resonant-term
        error_factor : Error-factor to simulate the error-bars
        term         : 'SAXS-term' or 'Cross-term' or 'Resonant-term' or 'Total'
        D            : Hard Sphere Diameter
        phi          : Volume fraction of particles
        U            : The sticky-sphere interaction energy
        SF           : Type of structure factor. Default: 'None'
        Rsig         : Width of the distribution of the equatorial radius of the nanoparticles (all the layers and the
                       height are scaled with it). Default: 0.0
        mpar         : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs, the equatorial thickness of each layer), and
                        Height to Radii ratio ('RzRatio', polar thickness/equatorial thickness of each layer)
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.Nalf=Nalf
        self.norm=norm
        self.norm_err=norm_err
        self.sbkg=sbkg
        self.cbkg=cbkg
        self.abkg=abkg
        self.dist=dist
        self.Np=Np
        self.Energy=Energy
        self.relement=relement
        self.NrDep=NrDep
        self.error_factor=error_factor
        self.D=D
        self.phi=phi
        self.U=U
        self.__mpar__=mpar #If there is any multivalued parameter
        self.SF=SF
        self.term=term
        self.Rsig=Rsig
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                      'term':['SAXS-term','Cross-term','Resonant-term','Total']} #If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
        self.__fit__=False
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params=Parameters()
        self.params.add('norm',value=self.norm,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('D', value=self.D, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg',value=self.sbkg,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rsig',value=self.Rsig,vary=0,min=0,max=np.inf,expr=None,brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey, key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)

    @lru_cache(maxsize=10)
    def calc_Rdist(self, R, Rsig, dist, N):
        """Calculate the radius distribution."""
        R = np.array(R)
        totalR = np.sum(R[:-1])
        if Rsig > 0.001:
            if dist == 'Gaussian':
                rmin, rmax = max(0.001, totalR - 5 * Rsig), totalR + 5 * Rsig
                dr = np.linspace(rmin, rmax, N)
            else:
                rmin, rmax = max(-3, np.log(totalR) - 5 * Rsig), np.log(totalR) + 5 * Rsig
                dr = np.logspace(rmin, rmax, N, base=np.exp(1.0))
            fdist = eval(f'{dist}.{dist}(x=0.001, pos=totalR, wid=Rsig)')
            fdist.x = dr
            rdist = fdist.y() / np.sum(fdist.y())
            return dr, rdist, totalR
        else:
            return [totalR], [1.0], totalR

    @lru_cache(maxsize=10)
    def new_ellipsoid(self, q, R, RzRatio, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=100):
        q = np.array(q)
        dr, rdist, totalR = self.calc_Rdist(R, Rsig, dist, Np)
        mu, wt = np.polynomial.legendre.leggauss(Nalf)
        mu, wt = (mu + 1) / 2.0, wt / 2.0  # Gauss-Legendre points for cos(alpha) in [0,1]
        pfac = (2.818e-5 * 1.0e-8) ** 2
        form, eiform, aform, cform = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)

        for i in range(len(dr)):
            r = np.array(R) * (1 + (dr[i] - totalR) / totalR)
            ff, eiff, aff, cff = ff_ellipsoid_ml(q, r, np.array(RzRatio), np.array(rho), np.array(eirho),
                                                 np.array(adensity), mu, wt)
            form += rdist[i] * ff
            eiform += rdist[i] * eiff
            aform += rdist[i] * aff
            cform += rdist[i] * cff

        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)  # in cm^2

    @lru_cache(maxsize=2)
    def new_ellipsoid_dict(self, q, R, RzRatio, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=100):
        form, eiform, aform, cform = self.new_ellipsoid(q, R, RzRatio, Rsig, rho, eirho, adensity, dist=dist, Np=Np,
                                                        Nalf=Nalf)
        result={
            'SAXS-term': eiform,
            'Resonant-term': aform,
            'Cross-term': cform,
            'Total': form
        }
        return result

    def update_params(self):
        """Update parameters based on fitting values."""
        mkey = self.__mkeys__[0]
        param_types = ['Density', 'SolDensity', 'Rmoles', 'R', 'RzRatio']
        for key in param_types:
            param_len = len(self.__mpar__[mkey][key])
            setattr(self, f'__{key}__',
                    [self.params[f'__{mkey}_{key}_{i:03d}'].value for i in range(param_len)])
        self.__Material__ = [self.__mpar__[mkey]['Material'][i] for i in range(param_len)]

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        scale = 1e27 / 6.022e23
        self.output_params = {'scaler_parameters': {}}
        self.update_params()
        NrDep = self.NrDep in (True, 'True')  # the choice can come as the string 'False' which is True for python
        rho, eirho, adensity, rhor, eirhor, adensityr, cdensityr = calc_rho(R=tuple(self.__R__),
                                                                      material=tuple(self.__Material__),
                                                                      relement=self.relement,
                                                                      density=tuple(self.__Density__),
                                                                      sol_density=tuple(self.__SolDensity__),
                                                                      Energy=self.Energy, Rmoles=tuple(self.__Rmoles__),
                                                                      NrDep=NrDep)
        major = np.sum(self.__R__)
        minor = np.sum(np.array(self.__R__) * np.array(self.__RzRatio__))
        self.output_params['scaler_parameters']['Diameter (Angstroms)'] = 2 * major
        self.output_params['scaler_parameters']['Height (Angstroms)'] = 2 * minor
        if self.dist == 'LogNormal':
            self.output_params['scaler_parameters']['Diameter_Std (Angstroms)'] = 2 * np.sqrt((np.exp(self.Rsig ** 2) - 1) * np.exp(2 * np.log(major) + self.Rsig ** 2))
            self.output_params['scaler_parameters']['Height_Std (Angstroms)'] = 2 * np.sqrt((np.exp(self.Rsig ** 2) - 1) * np.exp(2 * np.log(minor) + self.Rsig ** 2))
        else:
            self.output_params['scaler_parameters']['Diameter_Std (Angstroms)'] = 2 * self.Rsig
            self.output_params['scaler_parameters']['Height_Std (Angstroms)'] = 2 * self.Rsig * minor / major
        if type(self.x) == dict:
            sqf = {}
            key='SAXS-term'
            xref = self.x[key]
        else:
            xref = self.x
        if self.SF == 'Hard-Sphere':
            struct = hard_sphere_sf(xref, D=self.D, phi=self.phi)
        elif self.SF == 'Sticky-Sphere':
            struct = sticky_sphere_sf(xref, D=self.D, phi=self.phi, U=self.U, delta=0.01)
        else:
            struct = np.ones_like(xref)
        absnorm = self.norm * 1e-9 * 6.022e20 * struct
        tsqf, eisqf, asqf, csqf = self.new_ellipsoid(tuple(xref), tuple(self.__R__), tuple(self.__RzRatio__), self.Rsig,
                                                     tuple(rho), tuple(eirho), tuple(adensity), dist=self.dist,
                                                     Np=self.Np, Nalf=self.Nalf)
        sqfs = {'Total': absnorm * np.array(tsqf) + self.sbkg,  # in cm^-1
                'SAXS-term': absnorm * np.array(eisqf) + self.sbkg,
                'Cross-term': absnorm * np.array(csqf) + self.cbkg,
                'Resonant-term': absnorm * np.array(asqf) + self.abkg}
        if type(self.x) == dict:
            sqf = {key: sqfs[key] for key in self.x.keys()}
        else:
            sqf = sqfs[self.term]
        if not self.__fit__:  # Generate all the quantities below while not fitting
            signal = sqfs['Total']
            minsignal = np.min(signal)
            normsignal = signal / minsignal
            norm = np.random.normal(self.norm, scale = self.norm_err / 100.0)
            sqerr = np.random.normal(normsignal * norm, scale = self.error_factor)
            meta = {'Energy': self.Energy}
            if self.Energy is not None:
                self.output_params['simulated_w_err_%.4fkeV' % self.Energy] = {'x': xref, 'y': sqerr * minsignal,
                                                                               'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                                               'meta': meta,
                                                                               'names': ['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)']}
            else:
                self.output_params['simulated_w_err'] = {'x': xref, 'y': sqerr * minsignal,
                                                         'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                         'meta': meta,
                                                         'names': ['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)']}
            dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
            self.output_params['Distribution'] = {'x': dr, 'y': rdist, 'names': ['R (Angs)', 'P(R)']}
            for key in sqfs.keys():
                self.output_params[key] = {'x': xref, 'y': sqfs[key], 'names': ['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)']}
            self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                           'names': ['r (Angs)', 'Electron Density (el/Angs<sup>3</sup>)']}
            self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                             'names': ['r (Angs)', 'Electron Density (el/Angs<sup>3</sup>)']}
            self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                'names': ['r (Angs)', 'Density (Molar)']}  # in Molar
            self.output_params['cdensity_r'] = {'x': cdensityr[:, 0], 'y': cdensityr[:, 1],
                                                'names': ['r (Angs)', 'Density (g/cm<sup>3</sup>)']}
            self.output_params['Structure_Factor'] = {'x': xref, 'y': struct, 'names': ['q (Angs<sup>-1</sup>)', 'S(q)']}
            xtmp, ytmp = create_steps(x=list(self.__R__[:-1]), y=list(self.__Rmoles__[:-1]))
            self.output_params['Rmoles_radial'] = {'x': xtmp, 'y': ytmp, 'names': ['r (Angs)', 'Rmoles']}
            xtmp, ytmp = create_steps(x=list(self.__R__[:-1]), y=list(self.__RzRatio__[:-1]))
            self.output_params['RzRatio_radial'] = {'x': xtmp, 'y': ytmp, 'names': ['r (Angs)', 'RzRatio']}
            xtmp, ytmp = create_steps(x=list(self.__R__[:-1]), y=list(self.__Density__[:-1]))
            self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp, 'names': ['r (Angs)', 'Density (g/cm<sup>3</sup>)']}
        return sqf


if __name__=='__main__':
    x=np.logspace(-3,0.0,200)
    fun=Ellipsoid_Uniform(x=x)
    print(fun.y())
