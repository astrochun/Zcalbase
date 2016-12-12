"""
get_te
====

Set of routines used by Zcalbase to compute electron temperatures

Requirements:
 PyNeb (http://www.iac.es/proyecto/PyNeb/)

"""

import numpy as np
import pyneb as pn
from matplotlib import pyplot as plt

DataFileDict = {'O2': {'atom': 'o_ii_atom_Z82-WFD96.dat',   'coll': 'o_ii_coll_P06-T07.dat'},
                'O3': {'atom': 'o_iii_atom_FFT04-SZ00.dat', 'coll': 'o_iii_coll_SSB14.dat'}}

pn.atomicData.setDataFileDict(DataFileDict)

def get_oiii(ratio0, silent=True):
    O3 = pn.Atom('O', 3)
    if silent == False:
        print '## O3.atomFile : ', O3.atomFile, ' O3.collFile : ', O3.collFile

    Te = O3.getTemDen(int_ratio=ratio0, den=100., to_eval='(L(5007) + L(4959)) / L(4363)')
    return Te
#enddef

#'Ne3': {'atom': 'ne_iii_atom_GMZ97.dat', 'coll': 'ne_iii_coll_McLB00.dat'}}
#'N2': {'atom': 'n_ii_atom_GMZ97-WFD96.dat', 'coll': 'n_ii_coll_LB94.dat'},

def nicholls14():
    '''
    Function to compute electron temperature (T_e) of O++ from
    [OIII]4959,5007/OIII4363. Values are based on Eq. (2) in
    Nicholls et al. (2014), ApJ, 790, 75

    Parameters
    ----------
    None

    Returns
    -------
    10**iR : array or array-like
      Flux ratio of [OIII] lines: I(4959+5007)/I(4363)

    Te_OIII : array or array-like
      Electron temperature of O++ in Kelvins
    '''

    # From page 4
    a = 13205.
    b = 0.92506
    c = 0.98062
  
    iR = np.arange(1.0, 4.3, 0.01) # log([OIII]4959,5007/[OIII]4363)
    R = 1.0/10**(iR) # [OIII]4363/[OIII]4959,5007
    
    Te_OIII = a*(-1*np.log10(R)-b)**(-c) # in Kelvin
    return 10**iR, Te_OIII
#enddef
    
def plot_R_Te():
    ratio0 = np.arange(1.0, 4.3, 0.01) # logarithm of values
    ratio0 = 10**ratio0
    
    Te = get_oiii(ratio0, silent=False)

    fig, ax = plt.subplots()

    ax.plot(ratio0, Te, 'b--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\frac{I(\lambda4959) + I(\lambda5007)}{I(\lambda4363)}$')
    ax.set_ylabel(r'$T_e$ (K)')

    # Get Nicholls calibration
    Nic14_R, Nic14_Te = nicholls14()
    ax.plot(Nic14_R, Nic14_Te, 'k-')
    
    #ax.set_xlim([40,260])
    #plt.show()


