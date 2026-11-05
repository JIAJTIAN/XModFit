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



class XFNTR_ADV: #Please put the class name same as the function name
    def __init__(self,x=0.1,E=20.0, mpar={'Top':{'Components':['Sr'],'Concentration':[0],'Radius':[1.25]}, 'Bottom':{'Components':['SrCl2'],'Concentration':[50.0],'Radius':[2.388]}},
                 topchem='C12H26', topden=0.75, botchem='H2O', botden=0.997, element='Sr', line='Ka1', beam_profile='Uniform',
                 x_axis = 'Qz', beamwid = 0.02, detlen=12.7, gl2=1238.0, qz=0.01, cen2win=50, samplesize=76, stepsize=100, qoff=0.0, shoff=0.0, gl2off=0.0, detoff=0.0, yscale=1, int_bg=0, Rc=300, sur_den=0, topextra=0.0, botextra=0.0):
        """
        Calculates X-ray reflectivity from a system of multiple layers using Parratt formalism

        x     	: array of wave-vector transfer along z-direction or "sh" (sample height)
        E     	: Energy of x-rays in units of keV
        topchem : chemical formula for the top phase
        topden  : mass density for the top phase in the unit of g/ml
        botchem : chemical formula for the bottom phase
        botden  : mass density for the bottom phase in the unit of g/ml
        ele:    : target element, e.g., 'Sr'
        line:   : emission line, e.g., 'Ka1'
        beam_profile : choice of the beam profile: Uniform | Gaussian
        x_axis  : choice of x axis: Qz | sh
        beamwid   : the width of the beam in unit of mm
        detlen  : detector size projected on the surface in the unit of mm
        gl2     : the distance btw the steering crystal and the sample in the unit of mm; needed if x axis is sh
        qz      : qz in unit of \AA^-1; needed if x axis is sh
        sampleszie : the interface size in unit of mm
        cen2win: the distance btw the sample center and side window in unit of mm
        stepsize : the step size along the footprint used for the numerical integration in the unit of um
        qoff  	: q-offset to correct the zero q of the instrument
        shoff   : sample height offset due to the alignment in the unit of mm
        gl2off  : gl2 offest due to the alignment in the unit of mm
        detoff  : The offset of the detector projection on the interface with respect to the sample center in the unit of mm
        yscale  : a scale factor for the fluorescence intensity
        int_bg  : the background fluorescence intensity from the secondary scattering from the primary beam, should be zero for air/water interface
        Rc : the radius of the interfacial curvature in unit of meter
        sur_den : the surface density of target element in unit of number per \AA^-2
        topextra : extra target ion concentration (mM) in the top phase on the top of the vaule listed in mpar 
        botextra : extra target ion concentration (mM) in the bottom phase on the top of the vaule listed in mpar 
        mpar:   : components with unknown concentrations(mM) and radius (\AA) in the top and bottom phase

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
        self.beam_profile = beam_profile
        self.x_axis = x_axis
        self.beamwid = beamwid
        self.detlen = detlen
        self.gl2 = gl2
        self.qz = qz
        self.samplesize = samplesize
        self.cen2win = cen2win
        self.stepsize = stepsize
        self.qoff = qoff
        self.shoff = shoff
        self.gl2off = gl2off
        self.detoff = detoff
        self.yscale = yscale
        self.int_bg = int_bg
        self.Rc = Rc
        self.sur_den = sur_den
        self.topextra = topextra
        self.botextra = botextra
        #self.ion_depth = ion_depth
        elelist = xdb.atomic_symbols
        linelist = list(xdb.xray_lines(98).keys())
        self.choices={'element':elelist,'line': linelist, 'beam_profile':['Uniform', 'Gaussian'], 'x_axis':['Qz', 'sh']} #If there are choices available for any fixed parameters
        self.output_params = {}
        self.init_params()
        self.__fit__=False
        self.__avoganum__ = scipy.constants.Avogadro
        self.__eleradius__ = scipy.constants.physical_constants['classical electron radius'][0]*1e10 #classic electron radius in \AA
        self.output_params = {'scaler_parameters': {}}

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params = Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('shoff', self.shoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('gl2off', self.gl2off, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('int_bg', self.int_bg, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rc', self.Rc, vary=0, min=10, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sur_den', self.sur_den, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('detoff', self.detoff, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('topextra', self.topextra, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('botextra', self.botextra, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)

        for mkey in self.__mpar__.keys():
            for key in self.__mpar__[mkey].keys():
                if key!='Components':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey,key, i), value=self.__mpar__[mkey][key][i], vary=0, min=0, max=np.inf, expr=None, brute_step=0.05)

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

    def formStr(self, chemfor):
        string=''
        for i in range(len(chemfor)):
            key=list(chemfor.keys())[i]
            if chemfor[key]>0:
                string=string+key+str('{0:.3f}'.format(chemfor[key]).rstrip('0').rstrip('.'))
        return string

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
        try:
            return massden/molarmass*formula[element]*1000  #return the bulk concentration in unit of M
        except:
            return 0


    def updatePhaseInfo(self, solchem, solden, phase):
        solformula = self.parseFormula(solchem)  #  solvent formula
        chemfor = []  # list of chemical in that phase
        concen = []   # list of concentrations for each row in that phase
        radius = []   # list of radius for each row in that phase
        volume = 0    # total volume of chemical in that phase
        for i in range(len(self.__mpar__[phase]['Components'])):
            chemfor.append(self.parseFormula(str(self.__mpar__[phase]['Components'][i])))
            concen.append(float(self.params['__%s_%s_%03d' % (phase, 'Concentration', i)].value))
            radius.append(float(self.params['__%s_%s_%03d' % (phase, 'Radius', i)].value))
        totalformula = {}
        for i in range(len(concen)):
            volume = volume + concen[i] * pow(radius[i], 3)
            for j in range(len(chemfor[i])):  # merge components at all rows.
                key = list(chemfor[i].keys())[j]
                if key in totalformula:
                    totalformula[key] = totalformula[key] + chemfor[i][key] * concen[i]
                else:
                    totalformula[key] = chemfor[i][key] * concen[i]

        totalvolume = volume / 1000 * self.__avoganum__ * 1e-27 * 4 / 3 * np.pi  # total volume of all components in unit of liter
        solvolume = 1 - totalvolume  # solvent volume in unit of liter

        solmolarmass = np.sum(
            [xdb.molar_mass(key) * solformula[key] for key in solformula.keys()])  # molar mass for the solvent
        solmolar = solvolume * solden / solmolarmass * 1e6  # solvent molarity in mM

        for key in solformula.keys():  # add solvent into the totalformula
            if key in totalformula:
                totalformula[key] = totalformula[key] + solformula[key] * solmolar
            else:
                totalformula[key] = solformula[key] * solmolar

        chemstr = self.formStr(totalformula)
        molarmass = np.sum([xdb.molar_mass(key) * totalformula[key] for key in totalformula.keys()]) / 1e6
        return chemstr, molarmass

    def fluCalFun(self, x, topdel, topbet, botdel, botbet, flutopdel, flutopbet, flubotdel, flubotbet, flutopmu, flubotmu, topmu, k0, conbot, contop):

        if self.x_axis == 'Qz':  # qz scan
            alpha = (x + self.qoff) / 2 / k0  # get incident angle
            delx = (-self.shoff / alpha + self.gl2off) * 1e7  # get the footprint shift due to "sh" off and "gl2" off in unit of /AA
        else:   # sh scan
            alpha = (self.qz + self.qoff)/2/k0 * np.ones_like(x)
            totshoff = self.shoff + x + self.gl2 * alpha  # total footprint shift from the center of the sample in unit of mm
            delx = (-totshoff / alpha + self.gl2off) * 1e7
        if self.beam_profile == 'Uniform':
            fprint = self.beamwid * 1e7 / alpha   # get the footprint in unit of /AA for the rectangular beam
        else:
            fprint = 6.0 * self.beamwid * 1e7 / alpha  # get the footprint in unit of /AA for the Gaussian beam


        flu = []
        top_con = []
        bot_con = []
        sur_con = []
        topabs = np.exp(-self.cen2win * 1e+7 * topmu) # account for the absorption of x-ray from the sample cell window to interface center for normalization
        #print('topabs', topabs, 'topmu', topmu)
        #top_ana = []
        for i in range(len(x)):
            steps = int(fprint[i]/(self.stepsize * 1e4))
            stepsize = fprint[i]/steps
            x0 = np.linspace(-fprint[i]/2 + delx[i] + stepsize/2, fprint[i]/2 + delx[i] - stepsize/2, steps)  # get the position of single ray hitting z=0 relative to the center of sample with the step size "steps"
            if self.beam_profile == 'Uniform':
                weight = 1 / (self.beamwid * 1e7) * np.ones_like(x0)  # get the beam intensity weight at given x0 for the rectangular beam
                #weight = 1 * np.ones_like(x0)
            else:
                weight = 1 / (self.beamwid * 1e7 * np.sqrt(2 * np.pi)) * np.exp(-(x0-delx[i])**2*alpha[i]**2/(2*self.beamwid**2*1e14)) # get the beam intensity weight for the Gaussian beam

            xprimetotal, zprimetotal = self.getxz(x0, self.Rc * 1e10, alpha[i])   # get x' and z', where the ray hits the interface from a given x0 and alpha, including the 'nan', where the ray misses the interface
            xprime = xprimetotal[np.where(np.isnan(xprimetotal)==False)]    # get x' and z' for the ray hitting the interface
            zprime = zprimetotal[np.where(np.isnan(xprimetotal)==False)]
            x0hit = x0[np.where(np.isnan(xprimetotal)==False)]    #get x0 for the ray hitting the interface
            x0miss = x0[np.where(np.isnan(xprimetotal)==True)]    #get x0 for the ray missing the interface, which has intensity contribution on the top phase
            weighthit = weight[np.where(np.isnan(xprimetotal)==False)] # get weight array for the ray hitting the interface
            weightmiss = weight[np.where(np.isnan(xprimetotal)==True)]  # get weight array for the ray missing the interface

            alphaprime = alpha[i] - xprime / (self.Rc * 1e10) # a' is the real incident angle at x' position
            x1 = xprime - zprime/(2 * alphaprime - alpha[i]) # the position of the reflected beam hitting z=0


            pend, trans, refl = self.frsnllCal(topdel, topbet, botdel, botbet, k0, alphaprime) # get penetration depth, the frsnll transmissivity, and the reflectivity from given a'

            mu_i = topmu / alpha[i] + flutopmu # the effective absorption coefficient of the incident beam
            mu_r = -topmu / (2*alphaprime - alpha[i]) + flutopmu # the effective absorption coefficient of the reflected beam
            mu_t = alphaprime / (alpha[i] * pend) + flubotmu # the effective absorption coefficient of the transimitted beam

            z_i1 = (x0hit - self.detlen * 1e7 / 2 - self.detoff * 1e7) * alpha[i]  # the position of the incident/transmitted beam hit the downstream edge of the detecting area
            z_i2 = (x0hit + self.detlen * 1e7 / 2 - self.detoff * 1e7) * alpha[i]  # the position of the incident/transmitted beam hit the upstream edge of the detecting area
            z_r1 = -(x1 - self.detlen * 1e7 / 2 - self.detoff * 1e7) * (2 * alphaprime - alpha[i]) # the position of the reflected beam hit the downstream edge of the detecting area
            z_r2 = -(x1 + self.detlen * 1e7 / 2 - self.detoff * 1e7) * (2 * alphaprime - alpha[i]) # the position of the reflected beam hit the upstream edge of the detecting area

            z_imed = np.median([z_i1, z_i2, zprime], axis=0)  # get median value of z_i1, z_i2, and zprime
            z_rmed = np.median([z_r1, z_r2, zprime], axis=0)  # get median value of z_r1, z_r2, and zprime

            int_top = 1.0 / mu_i * np.exp(-topmu * x0hit) * (np.exp(mu_i * z_i2) - np.exp(mu_i * z_imed)) + np.abs(1.0 / mu_r * refl * (np.exp(mu_r * z_r1 - topmu * x1) - np.exp(mu_r * z_rmed - topmu * x1)))
            int_bot = np.where(z_i1 > zprime, 0, 1.0 / mu_t * trans * np.exp(-topmu * xprime) * np.exp(-alphaprime * zprime / alpha[i] / pend) * (np.exp(mu_t * z_imed) - np.exp(mu_t * z_i1)))
            int_sur = np.where((xprime < self.detlen * 1e7 / 2 + self.detoff * 1e7) & (xprime > -self.detlen * 1e7 / 2 + self.detoff * 1e7), trans * np.exp(-topmu * xprime), 0)

            tot_top = topabs * self.yscale * stepsize * (contop+self.topextra*1e-3) * np.sum(np.where(xprime > -self.samplesize * 1e7 / 2, int_top * weighthit, 0)) * self.__avoganum__ / 1e27
            tot_bot = topabs * self.yscale * stepsize * (conbot+self.botextra*1e-3) * np.sum(np.where(xprime > -self.samplesize * 1e7 / 2, int_bot * weighthit, 0)) * self.__avoganum__ / 1e27
            tot_sur = topabs * self.yscale * stepsize * np.sum(int_sur * weighthit) * self.sur_den


            if len(x0miss)!=0:    # for the ray missing the interface, (all of them from the downsteam side)
                #print('missing the interface at x = ', x[i])
                z_i1miss = (x0miss - self.detlen * 1e7 / 2 - self.detoff * 1e7) * alpha[i]  # the position of the incident beam missing the interface hits the downstream edge of the detecting area
                z_i2miss = (x0miss + self.detlen * 1e7 / 2 - self.detoff * 1e7) * alpha[i]  # the position of the incident beam missing the interface hits the upstream edge of the detecting area
                int_topmiss = 1.0 / mu_i * (np.exp(mu_i * z_i2miss - topmu * x0miss) - np.exp(mu_i * z_i1miss - topmu * x0miss))
                tot_top = tot_top + topabs * self.yscale * stepsize * (contop+self.topextra*1e-3) * np.sum(int_topmiss * weightmiss) * self.__avoganum__ / 1e27

            int_tot = (tot_top + tot_bot + tot_sur) + self.int_bg
            top_con.append(tot_top)
            bot_con.append(tot_bot)
            sur_con.append(tot_sur)
            flu.append(int_tot)

            #########check the top phase with the analytic calculation ##########
            # l = self.detlen * 1e7
            # zz = -fprint[i] * alpha[i]    # get z1+z2
            # a = alpha[i]
            # D = 1 / topmu
            # g = 1 / flutopmu
            # incident = D*g*((-g*np.exp(-a*l/(2*g) - zz/(2*g) - l/(2*D))/(D*a + g) + g*np.exp(a*l/(2*g) - zz/(2*g) + l/(2*D))/(D*a + g) - np.exp(l/(2*D)))*np.exp(l/(2*D)) + 1)*np.exp(-l/(2*D))
            # pend, trans, refl = self.frsnllCal(topdel, topbet, botdel, botbet, k0, alpha[i])
            # reflected = refl*D*g*((-g*np.exp(-a*l/(2*g) - zz/(2*g) + l/(2*D))/(D*a - g) + g*np.exp(a*l/(2*g) - zz/(2*g) - l/(2*D))/(D*a - g) - np.exp(l/(2*D)))*np.exp(l/(2*D)) + 1)*np.exp(-l/(2*D))
            # top_ana.append((incident+reflected) * self.__avoganum__ * contop / 1e27)
        return flu, top_con, bot_con, sur_con

    def getxz(self, x0, curvature, a0):
        a0 = a0 * np.ones_like(x0)
        a = a0 * a0 + 1
        b = -(2 * curvature * a0 + 2 * x0 * a0 * a0)
        c = 2 * curvature * a0 * x0 + x0 * x0 * a0 * a0
        if curvature > 0:
            x = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
        else:
            x = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        z = a0 * (x0 - x)
        return x, z

    def frsnllCal(self, dett, bett, detb, betb, k0, alpha):
        f1 = np.sqrt((alpha * alpha+1j*2*bett))
        #fmax = np.sqrt(complex(alpha * alpha - 2 * (detb - dett), 2 * betb))
        fmax = np.sqrt((alpha * alpha - 2 * (detb - dett) + 1j * 2 * betb))
        eff_d = 1 / (2 * k0 * fmax.imag)
        trans = 4 * abs(f1 / (f1 + fmax)) * abs(f1 / (f1 + fmax))
        frsnll=abs((f1-fmax)/(f1+fmax))*abs((f1-fmax)/(f1+fmax))
        return eff_d, trans, frsnll


    def y(self):
        """
        Define the function in terms of x to return some value
        """

        newtopchem, newtopden = self.updatePhaseInfo(self.topchem, self.topden, 'Top')   # update the top phase chemical formula and density
        newbotchem, newbotden = self.updatePhaseInfo(self.botchem, self.botden, 'Bottom') # update the bottom phase chemical formula and density
        k0 = 2 * np.pi * self.E / 12.3984  # wave vector
        conbot = self.getBulkCon(self.element, newbotchem, newbotden)  # get the bottom bulk concentration of the target element
        contop = self.getBulkCon(self.element, newtopchem, newtopden)  # get the top bulk concentration of the target element

       # print(conbot, contop)
       # print(newtopchem, newtopden)

        fluene = xdb.xray_lines(self.element)[self.line].energy / 1000  # get the fluorescence energy in KeV

        topdel, topbet = self.getDelBet(self.E, newtopchem, newtopden)  # get top del & bet for the incident beam
        botdel, botbet = self.getDelBet(self.E, newbotchem, newbotden)  # get bottom del & bet for the incident beam
        flutopdel, flutopbet = self.getDelBet(fluene, newtopchem, newtopden)  # get top del & bet for the fluorescent beam
        flubotdel, flubotbet = self.getDelBet(fluene, newbotchem, newbotden)  # get bottom del & bet for the fluorescent beam

        flutopmu = flutopbet * 2 * (2 * np.pi * fluene / 12.3984)  # get the absorption coefficient for the fluorescent beam in the top phase  mu_eo  (\AA^-1)
        flubotmu = flubotbet * 2 * (2 * np.pi * fluene / 12.3984)  # get the absorption coefficient for the fluorescent beam in the bottom phase mu_ew (\AA^-1)
        topmu = topbet * 2 * k0  # get the absorption coefficient in top phase in \AA^-1 for the incident beam mu_io

        x = self.x
        flu, top, bot, sur = self.fluCalFun(x, topdel, topbet, botdel, botbet, flutopdel, flutopbet, flubotdel, flubotbet, flutopmu, flubotmu, topmu, k0, conbot, contop)
        #print(flu)
        #print(bot)
        if not self.__fit__:
            #flu, top, bot, sur = self.fluCalFun(x)
            self.output_params['Top Phase contribution'] = {'x': x, 'y': top}
            self.output_params['Bottom phase contribution'] = {'x': x, 'y': bot}
            self.output_params['Interface contribution'] = {'x': x, 'y': sur}
        return flu

        #return self.x


if __name__=='__main__':
    x=np.linspace(0.005,0.016,100)
    fun=XFNTR_ADV(x=x)
    print(fun.y())
