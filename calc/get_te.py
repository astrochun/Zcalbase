"""
get_te
====

Set of routines used by Zcalbase to compute electron temperatures

Requirements:
 PyNeb (http://www.iac.es/proyecto/PyNeb/)

"""

import os
import numpy as np
import pyneb as pn

from matplotlib import pyplot as plt
from pylab import subplots_adjust

# + on 12/12/2016
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

DataFileDict = {'O2': {'atom': 'o_ii_atom_Z82-WFD96.dat', 'coll': 'o_ii_coll_P06-T07.dat'},
                'O3': {'atom': 'o_iii_atom_FFT04-SZ00.dat', 'coll': 'o_iii_coll_SSB14.dat'}}

#'Ne3': {'atom': 'ne_iii_atom_GMZ97.dat', 'coll': 'ne_iii_coll_McLB00.dat'}}
#'N2': {'atom': 'n_ii_atom_GMZ97-WFD96.dat', 'coll': 'n_ii_coll_LB94.dat'},

pn.atomicData.setDataFileDict(DataFileDict)

def get_oiii(ratio0, n_e=100, default=False, silent=True):
    '''
    Function to compute electron temperature (T_e) of O++ from
    [OIII]4959,5007/OIII4363 based on PyNeb approach.

    Collisional and atomic data are set by pn.atomicData.setDataFileDict

    Parameters
    ----------
    ratio0 : array or array-like
      Flux ratio of [OIII] lines: I(4959+5007)/I(4363)

    n_e : array or array-like
      Electron density in cm^-3
    
    default: boolean
      Use default atomic and collision data adopted by PyNeb. Default: False
    
    silent : boolean
      Turns off stdout messages. Default: True

    Returns
    -------
    Te : array or array-like
      Electron temperature of O++ in Kelvins

    label0 : string
      Legend labeling string indicating adopted collisional and atomic data

    Notes
    -----
    Created by Chun Ly, 19 November 2016
    Modified by Chun Ly, 11 December 2016
     - Added labelling output
     - Documented code
     - Added n_e option
    Modified by Chun Ly, 12 December 2016
     - Change DataFileDict for default == True option
    '''

    # + on 12/12/2016
    if default == True:
        DataFileDict = {'O3': {'atom': 'o_iii_atom_SZ00-WFD96.dat',
                               'coll': 'o_iii_coll_AK99.dat'}}
        pn.atomicData.setDataFileDict(DataFileDict)
        label0 = 'PyNeb getTemDen - Collision: Aggarwal & Keenan 1999; \n'+\
                 'Atomic: Wiese, Fuhr & Deters 1996, Storey & Zeippen 2000'

    O3 = pn.Atom('O', 3)

    if silent == False:
        print '## O3.atomFile : ', O3.atomFile, ' O3.collFile : ', O3.collFile
        
    # + on 11/12/2016
    if O3.atomFile == 'o_iii_atom_FFT04-SZ00.dat' and \
       O3.collFile == 'o_iii_coll_SSB14.dat':
        label0 = 'PyNeb getTemDen - Collision: Storey+ 2014; \n'+\
                 'Atomic: Froese Fischer+ 2004, Storey & Zeippen 2000'
                 
        
    Te = O3.getTemDen(int_ratio=ratio0, den=n_e, to_eval='(L(5007) + L(4959)) / L(4363)')

    return Te, label0
#enddef

def nicholls14(palay=False):
    '''
    Function to compute electron temperature (T_e) of O++ from
    [OIII]4959,5007/OIII4363. Values are based on Eq. (2) in
    Nicholls et al. (2014), ApJ, 790, 75

    Parameters
    ----------
    palay : boolean
      Set to use the Palay et al. (2012) coefficients reported in Nicholls et al.
      (2013), ApJS, 207, 21. Default: False

    Returns
    -------
    10**iR : array or array-like
      Flux ratio of [OIII] lines: I(4959+5007)/I(4363)

    Te_OIII : array or array-like
      Electron temperature of O++ in Kelvins

    label0 : string
      Legend labeling string
    
    Notes
    -----
    Created by Chun Ly, 9 December 2016
    Modified by Chun Ly, 11 December 2016
     - Additional documentation
    Modified by Chun Ly, 12 December 2016
     - Added palay keyword option for Palay et al. (2012), MNRAS, 423, L35-L39
     - Added label string output
    '''

    # Mod on 12/12/2016
    if palay == False:
        # From Page 4
        print '## Using Nicholls et al. (2014)'
        a = 13205.
        b = 0.92506
        c = 0.98062
        label0 = 'Nicholls et al. (2014)'
    else:
        # From Table 4 on Page 16 of Nicholls et al. (2013)
        print '## Using Palay et al. (2012)'
        a = 13229.
        b = 0.92350
        c = 0.98196
        label0 = 'Palay et al. (2012), Nicholls et al. (2013)'
        
    iR = np.arange(1.0, 4.3, 0.01) # log([OIII]4959,5007/[OIII]4363)
    R = 1.0/10**(iR)               # [OIII]4363/[OIII]4959,5007
    
    Te_OIII = a*(-1*np.log10(R)-b)**(-c) # in Kelvin
    return 10**iR, Te_OIII, label0
#enddef

def eqn_for_logCt(x, t):
    '''
    Function to compute log(Ct) from Izotov et al. (2006), A&A, 448, 955

    Parameters
    ----------
    x : array or array-like
      electron density in terms of 1E-4 * n_e * t**-0.5

    t : array or array-like
      electron temperature in terms of 1E-4 T_e(OIII)
    
    Returns
    -------
    log(C_T): array or array-like
      Eq. 2 in Izotov et al. (2006)

    Notes
    -----
    Created by Chun Ly, 12 December 2016
    '''

    Ct = (8.44 - 1.09*t + 0.5*t**2 - 0.08*t**3)*((1+0.0004*x)/(1+0.044*x))
    return np.log10(Ct)
#enddef

def izotov06_te(T_e=None, n_e=100., silent=True):
    '''
    Function to compute electron temperature according to Izotov et al. (2006),
    A&A, 448, 955

    Parameters
    ----------
    T_e : array or array-like
      Electron temperature in terms of Kelvin. Default: None

    n_e : array or array-like
      Electron temperature in terms of 1E-4 T_e(OIII)
    
    Returns
    -------
    T_e : array or array-like
      Electron temperature in terms of Kelvin.

    flux2 : array or array-like
      Ratio of [OIII]4959,5007/[OIII]4363

    label0 : string
      Legend labeling string

    Notes
    -----
    Created by Chun Ly, 12 December 2016
    '''

    label0 = 'Izotov et al. (2006)'

    if T_e == None:
        T_e = np.arange(5000.0,1e5,10)

    t = 1e-4 * T_e
    x = 1e-4 * n_e * t**(-0.5)

    logCt = eqn_for_logCt(x, t)
    flux2 = 10**(1.432/t + logCt)

    return T_e, flux2, label0
#enddef
    
def plot_R_Te():
    '''
    Function to plot electron temperature (T_e) of O++ vs.
    [OIII]4959,5007/OIII4363 for different assumptions.

    Parameters
    ----------
    None

    Returns
    -------
    10**iR : array or array-like
      Flux ratio of [OIII] lines: I(4959+5007)/I(4363)

    Te_OIII : array or array-like
      Electron temperature of O++ in Kelvins

    Notes
    -----
    Created by Chun Ly, 9 December 2016
    Modified by Chun Ly, 11 December 2016
     - Additional documentation
     - Output plot to to PDF
    Modified by Chun Ly, 12 December 2016
     - Added default PyNeb for Te(OIII)
     - Added inset zoom in plot
     - Added Palay et al. (2012) calibration
    '''
    
    ratio0 = np.arange(1.0, 4.3, 0.01) # logarithm of values
    ratio0 = 10**ratio0
    
    Te, label1 = get_oiii(ratio0, silent=False)

    # + on 12/12/2016
    Te_def, label2 = get_oiii(ratio0, default=True, silent=False)

    # + on 12/12/2016
    Te_izo06, ratio0_izo06, label_izo06 = izotov06_te(silent=True)
    
    fig, ax = plt.subplots()

    ax.plot(ratio0, Te, 'b--', label=label1)
    ax.plot(ratio0, Te_def, 'r--',  label=label2) # + on 12/12/2016
    ax.plot(ratio0_izo06, Te_izo06, 'g:', label=label_izo06) # + on 12/12/2016
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\frac{I(\lambda4959) + I(\lambda5007)}{I(\lambda4363)}$', fontsize='16')
    ax.set_ylabel(r'$T_e({\rm O}^{++}$) (K)', fontsize='16')

    # Get Nicholls calibration
    Nic14_R, Nic14_Te, Nic14_label = nicholls14()
    ax.plot(Nic14_R, Nic14_Te, 'k-', label=Nic14_label)

    # Get Palay+(2012) calibration | + on 12/12/2016
    Pal12_R, Pal12_Te, Pal12_label = nicholls14(palay=True)
    ax.plot(Pal12_R, Pal12_Te, 'c-.', label=Pal12_label)
    
    ax.legend(loc='upper right', frameon=False, fontsize='x-small')

    # Mod on 11/12/2016
    ax.set_xlim([10,10**4.3])
    ax.set_ylim([3000,2E5])

    # zoom-in inset panel | + on 12/12/2016
    axins = zoomed_inset_axes(ax, 5, loc=7) #, bbox_to_anchor=[100,0.5]) # zoom = 6

    axins.plot(ratio0, Te, 'b--')
    axins.plot(ratio0, Te_def, 'r--')
    axins.plot(ratio0_izo06, Te_izo06, 'g:')

    axins.plot(Nic14_R, Nic14_Te, 'k-', label=Nic14_label)
    axins.plot(Pal12_R, Pal12_Te, 'c-.', label=Pal12_label)
    
    # sub region of the original image | + on 12/12/2016
    x1, x2, y1, y2 = 75, 140, 10000, 15000
    axins.set_xlim([x1, x2])
    axins.set_ylim([y1, y2])
    axins.set_xscale('log')
    axins.minorticks_on()

    # Draw inset box - dashed black line.
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k", ls='dashed', lw=0.5)
    
    # Write PDF file
    # + on 11/12/2016
    out_dir0 = os.path.dirname(__file__)+'/'
    outfile  = out_dir0 + 'Te_OIII.pdf'

    subplots_adjust(left=0.025, bottom=0.025, top=0.975, right=0.975)

    #fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')
#enddef

