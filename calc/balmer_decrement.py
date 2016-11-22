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
from matplotlib.backends.backend_pdf import PdfPages

def ratios_Te_ne_plot(silent=False, verbose=True):
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
    Modified by Chun Ly, 22 November 2016
     - Labeling for type
     - Handles multiple Balmer decrements and produces a multi-page PDF
     - Renamed from HaHb_Te_ne_plot to ratios_Te_ne_plot
     - Output PDF has been changed.
    '''

    if silent == False:
        print '### Begin balmer_decrement.ratios_Te_ne_plot() | '+systime()

    T_arr = np.arange(0.5,2.1,0.1) * 1E4

    N_arr  = np.logspace(1,5,3)
    style0 = ['blue', 'green', 'red']

    # + on 22/11/2016
    r_type0  = ['HaHb', 'HgHb', 'HdHb']
    ylabel0 = [r'I(H$\alpha$)/I(H$\beta$)', r'I(H$\gamma$)/I(H$\beta$)',
               r'I(H$\delta$)/I(H$\beta$)']

    # Moved up on 22/11/2016 | outfile Mod on 22/11/2016
    out_dir0 = os.path.dirname(__file__)+'/'
    outfile  = out_dir0 + 'Balmer_decrement_ratios_Te_ne_grid.pdf'

    pp = PdfPages(outfile) # + on 22/11/2016

    # Mod on 22/11/2016
    for tt in range(len(r_type0)):
        fig, ax = plt.subplots()
        for ii in range(len(N_arr)):
            # Mod on 22/11/2016 to use intrinsic_ratios()
            ratio0, recfile = intrinsic_ratios(T_arr, N_arr[ii], r_type=r_type0[tt], 
                                               product=True)
            
            t_label = r'$n_e = 10^'+str(int(np.log10(N_arr[ii])))+'$'+r' cm$^{-3}$'
            ax.plot(T_arr / 1E4, ratio0, style0[ii], label=t_label)
        #endfor

        # + on 22/11/2016
        if recfile == 'h_i_rec_SH95.fits':
            txt0 = 'Storey and Hummer (1995)'
            ax.annotate(txt0, (0.01,0.01), xycoords='axes fraction',
                        ha='left', va='bottom')
        
        ax.set_xlabel(r'Electron Temperature, $T_e$ [$10^4$ K]', fontsize='16')
        ax.set_ylabel(ylabel0[tt], fontsize='16')
        ax.minorticks_on()
    
        ax.set_xlim([0.4,2.1])

        # Mod on 22/11/2016
        if r_type0[tt] == 'HaHb':
            loc0 = 'upper right'
        else: loc0 = 'lower right'
        ax.legend(loc=loc0, frameon=False)

        fig.savefig(pp, format='pdf', bbox_inches='tight')
        fig.clear()
    #endfor

    if silent == False: print '### Writing : ', outfile
    pp.close()

    if silent == False:
        print '### End balmer_decrement.ratios_Te_ne_plot() | '+systime()
#enddef

def get_k_values(wave, law='CCM89', silent=True, verbose=False):
    '''
    Function to get k(lambda):
     A(lambda) = k(lambda) * E(B-V)

    Parameters
    ----------
    wave : float or array like
      Wavelength in units of Angstroms

    law : string
      String for dust attenuation "law". Default: "CCM89".
      Full list available from RC.getLaws()
      Options are:
       'G03 LMC',  'K76', 'F99-like', 'F88 F99 LMC', 'No correction',
       'SM79 Gal', 'MCC99 FM90 LMC', 'CCM89 Bal07', 'CCM89 oD94',
       'S79 H83 CCM89', 'F99', 'CCM89'
 
    silent : boolean
      Turns off stdout messages. Default: True

    verbose : boolean
      Turns on additional stdout messages. Default: False
	  
    Returns
    -------
    PDF called 'HaHb_Te_ne_grid.pdf' is saved in Zcalbase.calc directory path

    Notes
    -----
    Created by Chun Ly, 22 November 2016
    '''

    if silent == False:
        print '### Begin balmer_decrement.get_k_values() | '+systime()

    RC = pn.RedCorr(E_BV = 1.0)
    RC.law = law

    if silent == False:
        print '### End balmer_decrement.get_k_values() | '+systime()

    return np.log10(RC.getCorr(wave)) / 0.4
#enddef

def intrinsic_ratios(t0, n0, r_type='HaHb', product=False, silent=True, verbose=False):
    '''
    Function to obtain Ha/Hb, Hg/Hb, or Hd/Hb flux ratios

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
     ratio0 : float or array like
       Intrinsic Balmer decrement ratios, Ha/Hb, Hg/Hb, Hd/Hb

     recFitsFile : string
       Name of FITS file containing recombination coefficients

    Notes
    -----
    Created by Chun Ly, 21 November 2016
    Modified by Chun Ly, 22 November 2016
     - Previously called HaHb() function
     - Fix to allow for other balmer decrement options
    '''

    if silent == False:
        print '### Begin balmer_decrement.intrinsic_ratios | '+systime()

    if r_type != 'HaHb' and r_type != 'HgHb' and r_type != 'HdHb':
        print "Invalid [r_type]"
        print "Exiting!!!"
        return
    #endif

    H1 = pn.RecAtom('H', 1)

    if silent == False:
        print '### Using the following atomic data'
        print H1.recFitsFullPath
        
    Halpha = H1.getEmissivity(tem=t0, den=n0, lev_i=3, lev_j=2, product=product)
    Hbeta  = H1.getEmissivity(tem=t0, den=n0, lev_i=4, lev_j=2, product=product)

    # + on 22/11/2016
    Hgamma = H1.getEmissivity(tem=t0, den=n0, lev_i=5, lev_j=2, product=product)
    Hdelta = H1.getEmissivity(tem=t0, den=n0, lev_i=6, lev_j=2, product=product)

    # + on 22/11/2016    
    if r_type == 'HaHb': ratio0 = Halpha/Hbeta
    if r_type == 'HgHb': ratio0 = Hgamma/Hbeta
    if r_type == 'HdHb': ratio0 = Hdelta/Hbeta
    
    if silent == False:
        print '### End balmer_decrement.intrinsic_ratios | '+systime()

    return ratio0, H1.recFitsFile
#enddef

def EBV_determine(HaHb_ratio, Te, ne, law='CCM89', silent=True, verbose=False):
    '''
    Function to get nebular E(B-V) from Ha/Hb ratio

    Parameters
    ----------
    HaHb_ratio : float or array like
      The observed Ha/Hb flux ratio from spectra

    Te : float or array like
      Electron temperature in Kelvin

    ne : float or array like
      Electron density in cm^-3

    law : string
      String for dust attenuation "law". Default: "CCM89".
      Full list available from RC.getLaws()
      Options are:
       'G03 LMC',  'K76', 'F99-like', 'F88 F99 LMC', 'No correction',
       'SM79 Gal', 'MCC99 FM90 LMC', 'CCM89 Bal07', 'CCM89 oD94',
       'S79 H83 CCM89', 'F99', 'CCM89'
 
    silent : boolean
      Turns off stdout messages. Default: True

    verbose : boolean
      Turns on additional stdout messages. Default: False
	  
    Returns
    -------
    EBV0 : float or array like
      Nebular E(B-V)

    Notes
    -----
    Created by Chun Ly, 22 November 2016
    '''

    if silent == False:
        print '### Begin balmer_decrement.EBV_determine() | '+systime()

    k_Ha = get_k_values(6562.80, law, silent=silent, verbose=verbose)
    k_Hb = get_k_values(4861.32, law, silent=silent, verbose=verbose)
    
    HaHb0, _ = HaHb(Te, ne, product=False, silent=True, verbose=False)

    EBV0 = np.log10(HaHb_ratio / HaHb0) / -0.4 / (k_Ha - k_Hb)

    if silent == False:
        print '### End balmer_decrement.EBV_determine() | '+systime()

    return EBV0, HaHb0
#enddef
