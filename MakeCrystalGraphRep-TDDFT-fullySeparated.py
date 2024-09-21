'''
@author: shapera 
contains code Copyright (c) 2018 Tian Xie from https://github.com/txie-93/cgcnn
'''
#Makes descriptors from crystal graph representation
#takes structures collected from qball
#need to make three sets of descriptors, velocity, position, and initial
import csv
import functools
import json
import os
import random
import warnings
import shutil
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torch.linalg
from pymatgen.io.cif import CifParser
import itertools
import sys

RunName = sys.argv[1]#"C1H3Cl1_T1000"#"C1H4_T1000"
RedirectFolder = ""

DataFolderPos = RedirectFolder + "DataPosition/" + RunName + "/"
DataFolderVel = RedirectFolder + "DataVelocity/" + RunName + "/"
DataFolderInit = RedirectFolder + "DataInit/" + RunName + "/"


Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]

for OutputFolder in [DataFolderPos,DataFolderVel,DataFolderInit]:
    shutil.copyfile("atom_init.json", OutputFolder+"/"+"atom_init.json")
    if not os.path.exists(OutputFolder): #check if DataFolder exists                                                                                                
        os.makedirs(OutputFolder)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.
    atom_init.json: a JSON file that stores the initialization vector for each
    element.
    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    Parameters
    ----------
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        #crystal = Structure.from_file(os.path.join(self.root_dir,cif_id+'.cif')) 
        crystal = CifParser(os.path.join(self.root_dir,cif_id+'.cif'), occupancy_tolerance=1000.0).get_structures()[0]
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.Tensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
    
    
#CalcTypes = ["Position", "Velocity"]
CalcTypes = ["Position","Initial"]


for pos in xzPos:
    if pos =="x_5_z_5":
        continue
    for speed in Speeds:
        print(pos,speed)
        for Calc in CalcTypes:           
            if Calc == "Velocity":
                DataFolder = DataFolderVel
            if Calc == "Position":
                DataFolder = DataFolderPos + pos + "/" + speed + "/"
                shutil.copyfile("atom_init.json", DataFolder+"/"+"atom_init.json")
            if Calc == "Initial":
                DataFolder = DataFolderInit + pos + "/" + speed + "/"
                shutil.copyfile("atom_init.json", DataFolder+"/"+"atom_init.json")
                shutil.copyfile(DataFolder + "id_propR.csv", DataFolder + "id_prop.csv")
            
            dataset = CIFData(root_dir = DataFolder)    
            #print(list(dataset)[0][0]) #list(dataset)[number][0] gives crystal graph rep for datapoint number, list(dataset)[number][2] is identifier of number
            #print(list(dataset)[0][0][0])#list(dataset)[number][0][0]this is representation of atom 0 in material number
            #print(list(dataset)[0][0][2])    
            #print(list(dataset)[0])#list(dataset)[n] gives the rep of material 'n' the index goes from 0 to the number of materials-1
            #                       #this has three components
            #                       #list(dataset)[n][0] is a representation of the material
            #                       #list(dataset)[0][1] is the value of the target quantity
            #                       #list(dataset)[0][2] is the id of the material
            #print(len(list(dataset)[0][0][0]))    
            
            #list(dataset)[n][0] is a 3-component rep of material n
            #list(dataset)[n][0][0] stores the vacuum representations of each atom in the material, has number of components equal to the number of atoms in the material
            #list(dataset)[n][0][0][L] is the representation of atom L in material n. this corresponds to the representation of the lone atom
            #list(dataset)[n][0][1] is tensor rep for the atoms in the material, number of components is number of atoms
            #list(dataset)[n][0][1][L] is the tensor rep for atom L in the material. this corresponds to the representation of the atom in the material. has size number of nearest neighbors x larger number agreed on between materials, can have many 0 values
            #list(dataset)[n][0][2] contains a matrix of nearest neighbor relations, has size natoms x number of nearest neighbors
            #list(dataset)[0][1] is the value of the target quantity
            #list(dataset)[0][2] is the id of the material
            
            
            #get number of materials and number of descriptors
            NumMaterials = len(list(dataset))
            NumDescriptor = []
            for i in range(0,NumMaterials):
                NumDescriptor.append(len(list(dataset)[i][0][0])*len(list(dataset)[i][0][1][0]))
            MaxDescriptors = max(NumDescriptor)
            
            counter = 0
            f=open(DataFolder + 'CrystalGraphDescriptors.csv','w')
            f.write("MaterialID")
            for i in range(0,MaxDescriptors):
                f.write(","+"Descriptor"+str(i))
            f.write("\n")
            for n in range(0,NumMaterials):
                Name = list(dataset)[n][2]
                print(Calc,n,Name)
                FullMat = torch.tensor([0])
                numAtoms = len(list(dataset)[n][0][0])
                for L in range(0,numAtoms):
                    #print(L)
                    #print(list(dataset)[0][0][1][L].shape)
                    FullMat = torch.block_diag(FullMat,list(dataset)[n][0][1][L])
                    #print(FullMat.shape)
                SVDValues = torch.linalg.svdvals(FullMat).tolist()
                f.write(Name+',')
                for m in range(0,MaxDescriptors):
                    try:
                        f.write(str(SVDValues[m])+',')
                    except:
                        f.write('0'+',')
                f.write('\n')
            
            f.close()
