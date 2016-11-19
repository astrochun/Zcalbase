import pyneb as pn

DataFileDict = {'O2': {'atom': 'o_ii_atom_Z82-WFD96.dat',   'coll': 'o_ii_coll_P06-T07.dat'},
                'O3': {'atom': 'o_iii_atom_FFT04-SZ00.dat', 'coll': 'o_iii_coll_SSB14.dat'}}

pn.atomicData.setDataFileDict(DataFileDict)

def get_oiii(ratio0):
    O3 = pn.Atom('O', 3)
    O3.atomFile

    print O3.getTemDen(int_ratio=ratio0, den=100., to_eval='(L(5007)) / L(4363)')

#'Ne3': {'atom': 'ne_iii_atom_GMZ97.dat', 'coll': 'ne_iii_coll_McLB00.dat'}}
#'N2': {'atom': 'n_ii_atom_GMZ97-WFD96.dat', 'coll': 'n_ii_coll_LB94.dat'},
