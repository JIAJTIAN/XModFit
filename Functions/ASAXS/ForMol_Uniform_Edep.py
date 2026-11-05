####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_rountines'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
import copy
from io import StringIO
from xraydb import XrayDB
from itertools import combinations
import os
import mendeleev
# Older Biopython versions (Bio.PDB.SASA) still use the np.int alias which was removed from NumPy 1.24 onwards
if not hasattr(np, 'int'):
    np.int = int
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley



class ForMol_Uniform_Edep: #Please put the class name same as the function name
    re=2.818e-5 #Classical electron radius in Angstroms
    No=6.023e23 #Avagadro's number
    def __init__(self,x=0, relement='Pr', Rmoles=1.0, Energy=12.0,fname1='./Data/Pr4.pdb',eta1=1.0,fname2='None',eta2=0.0,sol=0.0,sig=0.0,
                 scale=1.0, norm=1, norm_err=0.01, bkg=0.0,mpar={},error_factor=1.0):
        """
        Calculates the form factor for two different kinds of  molecules in cm^-1 for which the XYZ coordinates of the all the atoms composing the molecules are known
        Reference: https://www.sciencedirect.com/science/article/pii/S2001037014600180?via%3Dihub
        x    	    : Scalar or array of reciprocal wave vectors
        relement    : Resonant element in the system
        Rmoles      : No. of moles of resonant element in the position of resonant element
        Energy    	: Energy of the X-rays at which the scattering pattern is measured
        fname1	    : Name with path of the .xyz file containing X, Y, Z coordinates of all the atoms of the molecule of type 1
        eta1 	    : Fraction of molecule type 1
        fname2	    : Name with path of the .xyz file containing X, Y, Z coordinates of all the atoms of the moleucule of type 2
        eta2 	    : Fraction of molecule type 2
        sol         : Electron density of solvent in el/Angs^3
        sig  	    : Debye-waller factor
        scale       : scale factor for uniform scaling of the co-ordinates
        norm 	    : Normalization constant which can be the molar concentration of the particles
        norm_err    : Percentage of error on normalization to simulated energy dependent SAXS data
        bkg 	    : Background
        error_factor: Error-factor to simulate the error-bars
        """
        if type(x)==list:
            self.x = np.array(x)
        else:
            self.x = np.array([x])
        if os.path.exists(fname1):
            self.fname1 = fname1
        else:
            self.fname1 = None
        self.eta1 = eta1
        if os.path.exists(fname2):
            self.fname2 = fname2
        else:
            self.fname2 = None
        self.relement = relement
        self.Rmoles = Rmoles
        self.eta2 = eta2
        self.norm = norm
        self.norm_err = norm_err
        self.error_factor = error_factor
        self.bkg = bkg
        self.Energy = Energy
        self.sol = sol
        self.sig = sig
        self.scale = scale
        self.__mpar__ = mpar #If there is any multivalued parameter
        self.choices = {} #If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
        self.__fnames__ = [self.fname1,self.fname2]
        self.__xdb__ = XrayDB()
        #if self.fname1 is not None:
        #    self.__Natoms1__,self.__pos1__,self.__f11__=self.readXYZ(self.fname1)
        #if self.fname2 is not None:
        #    self.__Natoms2__,self.__pos2__,self.__f12__=self.readXYZ(self.fname2)
        self.output_params={'scaler_parameters':{}}


    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params=Parameters()
        self.params.add('Rmoles', value=self.Rmoles, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('eta1',value=self.eta1,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('eta2',value=self.eta2,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('norm',value=self.norm,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('sig',value=self.sig,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('scale', value=self.scale, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)

    @lru_cache(maxsize=10)
    def readXYZ(self, fname):
        """
        Reads the xyz file to read the atomic positions
        """
        fh=open(fname,'r')
        lines = fh.readlines()
        fh.close()
        Natoms = eval(lines[0])
        elements = []
        pos = []
        # cen=np.zeros(3)
        for i in range(Natoms):
            line = lines[i+2].split()
            elements.append(line[0])
            pos.append(tuple((map(eval, line[1:]))))
        return Natoms, elements, pos

    @lru_cache(maxsize=10)
    def readPDB(self, fname):
        """
        Reads the PDB file to read the atomic positions, solvent accessible surface areas of each of the atoms
        """
        # Files written by Open Babel etc. repeat atom names and residue numbers (all carbons are named 'C', residues 1,2,1,2..).
        # Biopython silently drops such 'duplicate' atoms, so every atom gets a unique name and residue number first.
        lines = []
        with open(fname, 'r') as fh:
            for i, line in enumerate(fh):
                if line.startswith(('ATOM', 'HETATM')):
                    line = line.rstrip('\r\n').ljust(80)
                    line = line[:6] + '%5d' % (i + 1) + ' ' + '%4d' % (i + 1) + line[16:22] + '%4d' % (i + 1) + line[26:]
                lines.append(line)
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure('pdb', StringIO('\n'.join(lines)))
        elements = []
        pos = []
        sasa = []
        sr=ShrakeRupley()
        sr.compute(struct, level='A')
        # cen=np.zeros(3)
        for atom in struct.get_atoms():
            elements.append(atom.element.strip().capitalize())  # Biopython returns upper case symbols (PR)
            pos.append(atom.get_coord())
            sasa.append(atom.sasa)
        return len(elements), elements, pos, sasa

    @lru_cache(maxsize=10)
    def get_ff(self, q, elements, sasa, energy=None):
        """
        Get the atomic form factors at all the q-values for all the unique atoms
        """
        q = np.array(q)
        atom_ff = []
        atom_vol = {}
        unique_elements=[]
        contrast = []
        hydration = []
        sol_fac={}
        fw = 2 * self.__xdb__.f0('H', q) + self.__xdb__.f0('O', q)
        for i, element in enumerate(elements):
            if element not in unique_elements:
                unique_elements.append(element)
                vol = 4 * 3.14159 * eval('mendeleev.%s.vdw_radius' % element) ** 3 / 3.0 / 1.0e6
                sol_fac[element] = vol * np.exp((-q ** 2 * vol ** (2.0 / 3.0)) / 4.0 / np.pi)
                atom_vol[element] = vol
            f0 = self.__xdb__.f0(element, q)
            if energy is not None:
                f1 = self.__xdb__.f1_chantler(element = element, energy = energy * 1e3, smoothing = 0)
                f2 = self.__xdb__.f2_chantler(element = element, energy = energy * 1e3, smoothing = 0)
                ff = f0 + f1 + 1.0j * f2
            else:
                ff = f0 * (1.0 + 0j)
            atom_ff.append(ff)
            hydration.append(sasa[i] * fw)
            contrast.append(sol_fac[element])
        return unique_elements, atom_ff, hydration, contrast, atom_vol

    #@lru_cache(maxsize=10)
    def get_pairs(self, Natoms, pos):
        pairs = []
        distances = []
        for i in range(Natoms):
            for j in range(Natoms):
                pairs.append([i, j])
                distances.append(np.sqrt(np.sum((np.array(pos[i]) - np.array(pos[j]))**2)))
        return pairs, np.array(distances)

    def calc_formol(self, q, atom_ff, atom_pairs, atom_dist, atom_vol, sol, elements, relement, Rmoles):
        """
        Calculates the form factor of a molecule with known atomic positions
        :param q: reciprocal lattice vector in inv-Angs
        :param atom_ff: Array of atomic form factors of all the elements where rows represent each of the elements and cols
                        represents different form factor at different q values
        :param atom_pairs: array of pairs
        :param atom_dist: list of distance between the pairs
        :param atom_vol: Van Der Waals Atomic volume of each of the unique atoms in the system
        :param sol: solvent density in el/Angs^3
        :param elements: list of elements in each of the positions
        :param relement: resonant element in the system
        :param Rmoles: No. of moles of the resonant element in the system
        :return:
        """
        form = np.zeros_like(q)
        mole_frac = np.where(np.array(elements) == relement, Rmoles, 1.0)
        for k, atom in enumerate(atom_pairs):
            i, j = atom
            sinc = np.sinc(atom_dist[k]*q/np.pi)
            # form = form + np.real((atom_ff[i]-sol*atom_vol[i])*(atom_ff[j]-sol*atom_vol[j]).conjugate())# * sinc
            sol_faci = sol * atom_vol[elements[i]] * np.exp((-q ** 2 * atom_vol[elements[i]] ** (2.0 / 3.0)) / 4.0 / np.pi)
            sol_facj = sol * atom_vol[elements[j]] * np.exp((-q ** 2 * atom_vol[elements[j]] ** (2.0 / 3.0)) / 4.0 / np.pi)
            form = form + np.real((atom_ff[i]*mole_frac[i]-sol_faci)*(atom_ff[j]*mole_frac[j]-sol_facj).conjugate()) * sinc
        return form

    @lru_cache(maxsize=10)
    def calc_form(self, q, fname, sol, scale, relement, Rmoles, energy=None):
        Natoms, elements, pos, sasa = self.readPDB(fname)
        unique_elements, atom_ff, hydration, contrast, atom_vol = self.get_ff(q, tuple(elements), tuple(sasa), energy = energy)
        atom_pairs, atom_dist = self.get_pairs(Natoms, pos)
        return self.calc_formol(np.array(q), atom_ff, atom_pairs, scale*atom_dist, atom_vol, sol, elements, relement, Rmoles)


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        if type(self.x) == dict:
            sqf={}
            for key in self.x.keys():
                sq=[]
                Energy=float(key.split('_')[1].split('@')[1])
                if self.fname1 is not None:
                    form1 = self.re ** 2 * 1e-16 * self.No * self.calc_form(tuple(self.x[key]), self.fname1, self.sol,
                                                                            self.scale, self.relement, self.Rmoles, energy = Energy)
                # Contribution from second molecule
                if self.fname2 is not None:
                    form2 = self.re ** 2 * 1e-16 * self.No * self.calc_form(tuple(self.x[key]), self.fname2, self.sol,
                                                                            self.scale, self.relement, self.Rmoles, energy = Energy)

                self.__fnames__ = [self.fname1, self.fname2]

                if self.__fnames__[0] is not None and self.__fnames__[1] is not None:
                    self.output_params[os.path.basename(self.fname1) + '_%0.4fkeV'%Energy] = {'x': self.x[key],
                                                                                'y': self.norm * form1 * np.exp(
                                                                                    -self.x[key] ** 2 * self.sig ** 2),
                                                                                'names': ['q', 'Intensity']}
                    self.output_params[os.path.basename(self.fname2) + '_%0.4fkeV'%Energy] = {'x': self.x[key],
                                                                                'y': self.norm * form2 * np.exp(
                                                                                    -self.x[key] ** 2 * self.sig ** 2),
                                                                                'names': ['q', 'Intensity']}
                    self.output_params['bkg'] = {'x': self.x[key], 'y': self.bkg * np.ones_like(self.x[key])}
                    total = (self.eta1 * form1 + self.eta2 * form2) * self.norm * np.exp(
                        -self.x[key] ** 2 * self.sig ** 2) * 1e-3 + self.bkg
                elif self.__fnames__[0] is not None and self.__fnames__[1] is None:
                    self.output_params[os.path.basename(self.fname1) + '_%0.4fkeV'%Energy] = {'x': self.x[key],
                                                                                'y': self.norm * self.eta1 * form1 * np.exp(
                                                                                    -self.x[key] ** 2 * self.sig ** 2),
                                                                                'names': ['q', 'Intensity']}
                    self.output_params['bkg'] = {'x': self.x[key], 'y': self.bkg * np.ones_like(self.x)}
                    total = self.eta1 * form1 * self.norm * np.exp(-self.x[key] ** 2 * self.sig ** 2) * 1e-3 + self.bkg
                elif self.__fnames__[0] is None and self.__fnames__[1] is not None:
                    self.output_params[os.path.basename(self.fname2) + '_%0.4fkeV'%Energy] = {'x': self.x[key],
                                                                                'y': self.norm * self.eta2 * form2 * np.exp(
                                                                                    -self.x[key] ** 2 * self.sig ** 2),
                                                                                'names': ['q', 'Intensity']}
                    self.output_params['bkg'] = {'x': self.x[key], 'y': self.bkg * np.ones_like(self.x[key])}
                    total = self.eta2 * form2 * self.norm * np.exp(-self.x[key] ** 2 * self.sig ** 2) * 1e-3 + self.bkg
                else:
                    total = np.ones_like(self.x[key])
                sqf[key]=total[0:]
                signal = total[0:]
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
                sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
                meta = {'Energy': self.Energy}
                tkeys = list(self.output_params.keys())
                for tkey in tkeys:
                    if 'simulated_w_err' in tkey:
                        del self.output_params[tkey]
                if Energy is not None:
                    self.output_params['simulated_w_err_%.4fkeV' % Energy] = {'x': self.x[key], 'y': sqerr * minsignal,
                                                                              'yerr': np.sqrt(
                                                                                       normsignal) * minsignal * self.error_factor,
                                                                              'meta': meta,
                                                                              'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x[key], 'y': sqerr * minsignal,
                                                             'yerr': np.sqrt(
                                                                 normsignal) * minsignal * self.error_factor,
                                                             'meta': meta,
                                                             'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
        else:
            #Contribution from first molecule
            if self.fname1 is not None:
                form1 = self.re**2*1e-16*self.No*self.calc_form(tuple(self.x),self.fname1, self.sol, self.scale,
                                                                self.relement, self.Rmoles, energy=self.Energy)
            #Contribution from second molecule
            if self.fname2 is not None:
                form2 = self.re**2*1e-16*self.No*self.calc_form(tuple(self.x),self.fname2, self.sol, self.scale,
                                                                self.relement, self.Rmoles, energy=self.Energy)

            self.__fnames__=[self.fname1, self.fname2]

            if self.__fnames__[0] is not None and self.__fnames__[1] is not None:
                self.output_params[os.path.basename(self.fname1)+'_1']={'x':self.x,'y':self.norm*form1*np.exp(-self.x**2*self.sig**2),
                                                                        'names':['q','Intensity']}
                self.output_params[os.path.basename(self.fname2)+'_1']={'x':self.x,'y':self.norm*form2*np.exp(-self.x**2*self.sig**2),
                                                                        'names':['q','Intensity']}
                self.output_params['bkg']={'x':self.x,'y':self.bkg*np.ones_like(self.x)}
                total= (self.eta1*form1+self.eta2*form2)*self.norm*np.exp(-self.x**2*self.sig**2)*1e-3+self.bkg
            elif self.__fnames__[0] is not None and self.__fnames__[1] is None:
                self.output_params[os.path.basename(self.fname1)+'_1']={'x':self.x,'y':self.norm*self.eta1*form1*np.exp(-self.x**2*self.sig**2),
                                                                        'names':['q','Intensity']}
                self.output_params['bkg']={'x':self.x,'y':self.bkg*np.ones_like(self.x)}
                total= self.eta1*form1*self.norm*np.exp(-self.x**2*self.sig**2)*1e-3+self.bkg
            elif self.__fnames__[0] is None and self.__fnames__[1] is not None:
                self.output_params[os.path.basename(self.fname2)+'_1']={'x':self.x,'y':self.norm*self.eta2*form2*np.exp(-self.x**2*self.sig**2),
                                                                        'names':['q','Intensity']}
                self.output_params['bkg']={'x':self.x,'y':self.bkg*np.ones_like(self.x)}
                total= self.eta2*form2*self.norm*np.exp(-self.x**2*self.sig**2)*1e-3+self.bkg
            else:
                total= np.ones_like(self.x)
            sqf = total[0:]
            signal = total[0:]
            minsignal = np.min(signal)
            normsignal = signal / minsignal
            norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
            sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
            meta = {'Energy': self.Energy}
            tkeys = list(self.output_params.keys())
            for key in tkeys:
                if 'simulated_w_err' in key:
                    del self.output_params[key]
            if self.Energy is not None:
                self.output_params['simulated_w_err_%.3fkeV' % self.Energy] = {'x': self.x, 'y': sqerr * minsignal,
                                                                               'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                                               'meta': meta,
                                                                               'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
            else:
                self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal,
                                                         'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                         'meta': meta,
                                                         'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
        return sqf


if __name__=='__main__':
    x=np.logspace(-2,0.7,50)
    fun=ForMol_Uniform_Edep(x=x)
    fun.fname1='/media/sf_Mrinal_Bera/Documents/MA-Collab/XTal_data/P8W48.xyz'
    print(fun.y())
