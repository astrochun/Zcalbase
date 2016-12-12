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

DataFileDict = {'O2': {'atom': 'o_ii_atom_Z82-WFD96.dat',   'coll': 'o_ii_coll_P06-T07.dat'},
                'O3': {'atom': 'o_iii_atom_FFT04-SZ00.dat', 'coll': 'o_iii_coll_SSB14.dat'}}

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

    Notes
    -----
    Created by Chun Ly, 9 December 2016
    Modified by Chun Ly, 11 December 2016
     - Additional documentation
    '''

    # From page 4
    a = 13205.
    b = 0.92506
    c = 0.98062
  
    iR = np.arange(1.0, 4.3, 0.01) # log([OIII]4959,5007/[OIII]4363)
    R = 1.0/10**(iR)               # [OIII]4363/[OIII]4959,5007
    
    Te_OIII = a*(-1*np.log10(R)-b)**(-c) # in Kelvin
    return 10**iR, Te_OIII
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
    Modified by Chun Ly, 11 December 2016
     - Added default PyNeb for Te(OIII)
    '''
    
    ratio0 = np.arange(1.0, 4.3, 0.01) # logarithm of values
    ratio0 = 10**ratio0
    
    Te, label1 = get_oiii(ratio0, silent=False)

    # + on 12/12/2016
    Te_def, label2 = get_oiii(ratio0, default=True, silent=False)

    fig, ax = plt.subplots()

    ax.plot(ratio0, Te, 'b--', label=label1)
    ax.plot(ratio0, Te_def, 'r:',  label=label2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\frac{I(\lambda4959) + I(\lambda5007)}{I(\lambda4363)}$', fontsize='16')
    ax.set_ylabel(r'$T_e({\rm O}^{++}$) (K)', fontsize='16')

    # Get Nicholls calibration
    Nic14_R, Nic14_Te = nicholls14()
    ax.plot(Nic14_R, Nic14_Te, 'k-', label='Nicholls et al. (2014)')

    ax.legend(loc='upper right', frameon=False, fontsize='x-small')

    # Mod on 11/12/2016
    ax.set_xlim([10,10**4.3])
    ax.set_ylim([1000,2E5])

    # Write PDF file
    # + on 11/12/2016
    out_dir0 = os.path.dirname(__file__)+'/'
    outfile  = out_dir0 + 'Te_OIII.pdf'

    fig.savefig(outfile, bbox_inches='tight')
#enddef

