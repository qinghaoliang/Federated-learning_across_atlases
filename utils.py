import numpy as np
import scipy.io as sio
from spectral_embedding import spectral_embedding

def load_data_sc(atlas, task):
    """load the structure connectomes"""
    path = './data/sc/'
    fname = path + task + "_" + atlas + ".mat"
    # the data is in shape (node, node, nsub)
    data = sio.loadmat(fname) 
    x = data['mats']
    print(atlas,task,x.shape)
    return x

def load_data_sc_transport(atlas_s, atlas_t, task):
    """load the structure connectomes
    and transform into target space
    """
    fname = "./map/map_fc_" + atlas_s + "_" + atlas_t + ".npz"
    mapping = np.load(fname)
    Trans = mapping['Trans']
    comT = load_data_sc(atlas_t, task) 
    nnode = np.shape(comT)[0]
    dim_t = np.shape(comT)[1]
    nsub = np.shape(comT)[2]
    comS = load_data_sc(atlas_s, task)
    comR = np.zeros((nnode, nnode, nsub))
    
    for i in range(nsub):
        com_s = comS[:,:,i]
        AS = com_s
        Xp, Xn = spectral_embedding(AS, p=50)
        Xtp = Trans.T@Xp
        Xtn = Trans.T@Xn
        ATp = Xtp@Xtp.T
        ATn = Xtn@Xtn.T
        Ar = ATp-ATn
        comR[:,:,i] = Ar

    return comR

def load_data_fc(atlas, task):
    """load the functional connectomes"""
    path = './data/fc/'
    fname = path + task + "_" + atlas + ".mat"
    # the data is in shape (node, node, nsub)
    data = sio.loadmat(fname)
    x = data['mats']
    print(atlas,task,x.shape)
    return x

