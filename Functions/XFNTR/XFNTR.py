####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
import scipy.constants
import re
import cmath
from xraydb import XrayDB
xdb = XrayDB()
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_rountines'))
####Please do not remove lines above####

####Import your modules below if needed####



class XFNTR: #Please put the class name same as the function name
    def __init__(self,x=0.1,E=10.0, mpar={}, topchem='He', topden=1.78e-4, botchem='Sr50Cl100H110493.721O55246.86', botden=1.0032, element='Sr', line='Ka1', vslit= 0.04, detlen=10.5, qoff=0.0, yscale=1,int_bg=0, Rc=0, sur_den=0,ion_depth=0):
        """
        Calculates X-ray reflectivity from a system of multiple layers using Parratt formalism

        x     	: array of wave-vector transfer along z-direction
        E     	: Energy of x-rays in units of keV
        topchem : chemical formula for the top phase
        topden  : mass density for the top phase in the unit of g/ml
        botchem : chemical formula for the bottom phase
        botden  : mass density for the bottom phase in the unit of g/ml
        ele:    : target element, e.g., 'Sr'
        line:   : emission line, e.g., 'Ka1'
        vslit   : vertical slits size in unit of mm
        detlen  : detector size projected on the surface in the unit of mm
        qoff  	: q-offset to correct the zero q of the instrument
        yscale  : a scale factor for the fluorescence intensity in unit of 1E-3 /AA
        int_bg  : the background fluorescence intensity from the secondary scattering from the primary beam, should be zero for air/water interface
        Rc : the radius of the interfacial curvature in unit of meter; 0 means it's flat
        sur_den : the surface density of target element in unit of number per \AA^-2
        ion_depth : the monolayer thickness in unit of \AA where doesn't have any ions
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.E=E
        self.__mpar__ = mpar
        self.topchem = topchem
        self.topden = topden
        self.botchem = botchem
        self.botden = botden
        self.element = element
        self.line = line
        self.vslit = vslit
        self.detlen = detlen
        self.qoff = qoff
        self.yscale = yscale
        self.int_bg = int_bg
        self.Rc = Rc
        self.sur_den = sur_den
        self.ion_depth = ion_depth
        elelist = xdb.atomic_symbols
        linelist = list(xdb.xray_lines(98).keys())
        self.choices={'element':elelist,'line': linelist} #If there are choices available for any fixed parameters
        self.output_params = {}
        self.init_params()
        self.__fit__=False
        self.__avoganum__ = scipy.constants.Avogadro
        self.__eleradius__ = scipy.constants.physical_constants['classical electron radius'][0]*1e10 #classic electron radius in \AA

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params = Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('int_bg', self.int_bg, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rc', self.Rc, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sur_den', self.sur_den, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('ion_depth', self.ion_depth, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)

    def parseFormula(self, chemfor):
        a = re.findall(r'[A-Z][a-z]?|[0-9]+[.][0-9]+|[0-9]+', chemfor)
        if not a[-1].replace('.', '').isdigit():
            a.append('1')
        formula = {}
        i = 1
        while i <= len(a):
            if not a[i].replace('.', '').isdigit():
                a.insert(i, '1')
            if a[i - 1] in formula.keys():
                formula[a[i - 1]] = float(a[i]) + formula[a[i - 1]]
            else:
                formula[a[i - 1]] = float(a[i])
            i += 2
        return formula

    def getDelBet(self, energy, chemfor, massden):
        k0 = 2 * np.pi * energy / 12.3984
        formula=self.parseFormula(chemfor)
        #print(energy,chemfor,massden)
        molarmass = np.sum([xdb.molar_mass(key) * formula[key] for key in formula.keys()])
        massratio = {}
        for key in formula.keys():
            massratio[key] = xdb.molar_mass(key) * formula[key] / molarmass
        molarele = np.sum([xdb.atomic_number(key) * formula[key] for key in formula.keys()])
        eleden = massden / molarmass * molarele * self.__avoganum__ / 1e24
        tot_mu = np.sum([xdb.mu_elam(key, energy * 1000) * massratio[key] * massden for key in massratio.keys()])  #in unit of cm
        #print(eleden, 10000/tot_mu)
        return self.__eleradius__*2*np.pi/k0/k0*eleden, tot_mu/2/k0/1e8

    def getBulkCon(self, element, chemfor, massden):
        formula = self.parseFormula(chemfor)
        molarmass = np.sum([xdb.molar_mass(key) * formula[key] for key in formula.keys()])
        return massden/molarmass*formula[element]*1000  #return the bulk concentration in unit of M


    def fluCalFun(self, x):
        #surcur = flupara[5] * 1e10  # in unit of /AA
        #conbulk = float(self.ui.flubulLE.text())  # get the bulk concentration
        k0 = 2 * np.pi * self.E / 12.3984  # wave vector
        #slit = float(self.ui.flusliLE.text())  # get slits size

        detlen = self.detlen * 1e7  # get slits size in unit of /AA
        try:
            conbulk = self.getBulkCon(self.element,self.botchem,self.botden)
        except:
            conbulk = 0    # if the target element is not in the formula

        try:
            fluene = xdb.xray_lines(self.element)[self.line].energy/1000    #get the fluorescence energy in KeV
        except:
            fluene = 10.0    #if self.line is not in list of availble of lines for the target element.

        topdel, topbet=self.getDelBet(self.E, self.topchem, self.topden)  #get top del & bet for the incident beam
        botdel, botbet=self.getDelBet(self.E, self.botchem, self.botden)  #get bottom del & bet for the incident beam
        fludel, flubet=self.getDelBet(fluene, self.botchem, self.botden)  #get bottom del & bet for the fluorescent beam

        flumu = flubet*2*(2*np.pi*fluene/12.3984)
       # print(1/flumu/1e4)
        topd = 1 / (topbet * 2 * k0)  # get the absorption length in top phase in \AA
        alpha = x / 2 / k0  # get incident angle
        fprint = self.vslit / alpha * 1e7  # get the footprint in unit of /AA
        if self.Rc == 0:  # no surface curvature
            flu = []
            # p_d=[]
            for i in range(len(alpha)):
                z1 = (fprint[i] - detlen) / 2 * alpha[i]
                z2 = (fprint[i] + detlen) / 2 * alpha[i]
                effd, trans = self.frsnllCal(topdel, topbet, botdel, botbet, flumu, k0, alpha[i])
                if (fprint[i] > detlen):
                    effv = effd * topd * np.exp(-detlen / 2 / topd) * (detlen * effd * np.exp(z2 / alpha[i] / topd) * (
                            np.exp(-z1 / effd) - np.exp(-z2 / effd)) + topd * (np.exp(detlen / topd) - 1) * (
                                                                               z1 - z2)) / (
                                   detlen * effd + topd * (z1 - z2))
                else:
                    tempf1 = np.exp(- alpha[i] * (fprint[i] + detlen)/ 2 /effd + fprint[i]/ 2 / topd)
                    tempf2 = np.exp(- alpha[i] * (detlen - fprint[i]) / 2 / effd - fprint[i] / 2 / topd)
                    effv = 2 * topd * effd * np.sinh(fprint[i]/ 2 / topd) + topd * effd * effd * (tempf1 - tempf2) / (topd * alpha[i] - effd)
                int_sur = self.sur_den * topd * (np.exp(min(detlen, fprint[i]) / 2 / topd) - np.exp(-min(detlen, fprint[i]) / 2 / topd))  # surface intensity
                #print(x[i],topd*1e-10, detlen*1e-10, np.exp(detlen / 2 / topd) - np.exp(-detlen / 2 / topd))
                #print(topd*1e-10, fprint[i]*1e-10, np.exp(fprint[i] / 2 / topd) - np.exp(-fprint[i] / 2 / topd))
                int_bulk =  effv * self.__avoganum__ * conbulk / 1e27  # bluk intensity
                int_tot = np.exp(-self.ion_depth/effd) * self.yscale * 1e-3 * trans * (int_sur + int_bulk) + self.int_bg
                flu.append(int_tot)
            #   p_d.append(effd)
        else:  # with surface curvature
            flu = []
            for i in range(len(alpha)):
                bsum = 0
                ssum = 0
                steps = int((min(detlen, fprint[i]) + fprint[i]) / 2 / 1e6)  # use 0.1 mm as the step size
                stepsize = (min(detlen, fprint[i]) + fprint[i]) / 2 / steps
                x = np.linspace(-fprint[i] / 2 + stepsize / 2, min(detlen, fprint[i]) / 2 - stepsize / 2,steps)  # get the position fo single ray hitting the surface relative to the center of detector area with the step size "steps"
                for j in range(len(x)):
                    alphanew = alpha[i] - x[j] / self.Rc/1e10  # the incident angle at position x[j]
                    y1 = -detlen / 2 - x[j]
                    y2 = detlen / 2 - x[j]
                    effd, trans = self.frsnllCal(topdel, topbet, botdel, botbet, flumu, k0, alphanew)
                    if x[j] > -detlen / 2:
                        bsum = bsum + np.exp(-self.ion_depth/effd) * np.exp(-x[j] / topd) * trans * effd * (1.0 - np.exp(-y2 * alpha[i] / effd))
                        ssum = ssum + np.exp(-self.ion_depth/effd) * np.exp(-x[j] / topd) * trans
                    else:
                        bsum = bsum + np.exp(-self.ion_depth/effd) * np.exp(-x[j] / topd) * trans * effd * (np.exp(-y1 * alpha[i] / effd) - np.exp(
                            -y2 * alpha[i] / effd))  # surface has no contribution at this region
                int_bulk = bsum * stepsize * self.__avoganum__ * conbulk / 1e27
                int_sur = ssum * stepsize * self.sur_den
                int_tot =  self.yscale * 1e-3 * (int_bulk + int_sur) + self.int_bg
                flu.append(int_tot)
        return flu

    def frsnllCal(self, dett, bett, detb, betb, mub, k0, alpha):
        f1 = cmath.sqrt(complex(alpha * alpha, 2 * bett))
        fmax = cmath.sqrt(complex(alpha * alpha - 2 * (detb - dett), 2 * betb))
        length1 = 1 / mub
        length2 = 1 / (2 * k0 * fmax.imag)
        eff_d = length1 * length2 / (length1 + length2)
        trans = 4 * abs(f1 / (f1 + fmax)) * abs(f1 / (f1 + fmax))
        # frsnll=abs((f1-fmax)/(f1+fmax))*abs((f1-fmax)/(f1+fmax))
        return eff_d, trans


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        #print(xdb.xray_lines(self.element)[self.line].energy/1000)
        #self.output_params={}
        #print(self.getBulkCon(self.element,self.botchem,self.botden))
        x = self.x + self.qoff
        if not self.__fit__:
            self.output_params['scaler_parameters']={}
        return self.fluCalFun(x)
        #return self.x


if __name__=='__main__':
    x = np.linspace(0.015, 0.03, 400)
    fun=XFNTR(x=x)
    print(fun.y())
