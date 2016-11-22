"""
balmer_decrement
====

Determines Ha/Hb flux ratio based on T_e and n_e.

Requirements:
 PyNeb (http://www.iac.es/proyecto/PyNeb/)

"""

import sys, os

from chun_codes import systime

import pyneb as pn

#from os.path import exists
#import commands

import numpy as np

import matplotlib.pyplot as plt
import pylab

def HaHb_Te_ne_plot(silent=True, verbose=False):

    '''
    Function to plot Ha/Hb vs T_e and n_e. This is for illustrative purpose

    Parameters
    ----------

    silent : boolean
      Turns off stdout messages. Default: True

    verbose : boolean
      Turns on additional stdout messages. Default: False
	  
    Returns
    -------
    PDF called 'HaHb_Te_ne_grid.pdf' is saved in Zcalbase.calc directory path

    Notes
    -----
    Created by Chun Ly, 21 November 2016
    '''

    if silent == False: print '### Begin grid_plot() | '+systime()

    T_arr = np.arange(0.5,2.1,0.1) * 1E4

    N_arr  = np.logspace(1,5,3)
    style0 = ['blue', 'green', 'red']
    
    fig, ax = plt.subplots()
    for ii in range(len(N_arr)):
        ratio0, recfile = HaHb(T_arr, N_arr[ii], product=True)

        t_label = r'$n_e = 10^'+str(int(np.log10(N_arr[ii])))+'$'+r' cm$^{-3}$'
        ax.plot(T_arr / 1E4, ratio0, style0[ii], label=t_label)
    #endfor

    if recfile == 'h_i_rec_SH95.fits':
        txt0 = 'Storey and Hummer (1995)'
        ax.annotate(txt0, (0.01,0.01), xycoords='axes fraction',
                    ha='left', va='bottom')
        
    ax.set_xlabel(r'Electron Temperature, $T_e$ [$10^4$ K]', fontsize='16')
    ax.set_ylabel(r'I(H$\alpha$)/I(H$\beta$)', fontsize='16')
    ax.minorticks_on()
    
    ax.set_xlim([0.4,2.1])
    
    ax.legend(loc='upper right', frameon=False)

    out_dir0 = os.path.dirname(__file__)+'/'
    outfile  = out_dir0 + 'HaHb_Te_ne_grid.pdf'
    if silent == False: print '### Writing : ', outfile
    fig.savefig(outfile, bbox_inches='tight')

    if silent == False: print '### End grid_plot() | '+systime()

#enddef

def HaHb(t0, n0, product=False, silent=True, verbose=False):

    '''
    Function to obtain Ha/Hb flux ratio

    Parameters
    ----------
    t0 : array like
      Array of electron temperatures in units of K

    n0 : array like
      Array of electron densities in units of cm^-3

    product : boolean
      If True, all the combination of (t0, n0) are used. 
      If False (default), t0 and n0 must have the same size and are joined.      

    silent : boolean
      Turns off stdout messages. Default: True

    verbose : boolean
      Turns on additional stdout messages. Default: False
	  
    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 21 November 2016
    '''

    if silent == False: print '### Begin main | '+systime()

    H1 = pn.RecAtom('H', 1)

    if silent == False:
        print '### Using the following atomic data'
        print H1.recFitsFullPath
        
    Halpha = H1.getEmissivity(tem=t0, den=n0, lev_i=3, lev_j=2,
                              product=product)
    Hbeta  = H1.getEmissivity(tem=t0, den=n0, lev_i=4, lev_j=2,
                              product=product)

    if silent == False: print '### End main | '+systime()

    return Halpha/Hbeta, H1.recFitsFile
#enddef

