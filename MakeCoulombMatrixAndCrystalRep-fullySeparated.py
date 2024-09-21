'''
@author: shapera
'''
#constructs coulomb matrix descriptors and descriptors of crystal structure
#run this after making the crystal graph representation
import pandas as pd
from pymatgen.io.cif import CifParser
import numpy as np
from numpy.linalg import eigvalsh
from numpy.linalg import det
import itertools
import sys

RunName = sys.argv[1]#"C1H3Cl1_T1000"#"C1H4_T1000"

RedirectFolder = ""
DataFolderPos = RedirectFolder + "DataPosition/" + RunName + "/"
DataFolderInit = RedirectFolder + "DataInit/" + RunName + "/"

Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]

DataFolders = []
for pos in xzPos:
    if pos =="x_5_z_5":
        continue
    for speed in Speeds:
        DataFolders.append(DataFolderInit + pos + "/" + speed + "/")
for pos in xzPos:
    if pos =="x_5_z_5":
        continue
    for speed in Speeds:
        DataFolders.append(DataFolderPos + pos + "/" + speed + "/")


for DataFolder in DataFolders:
    print(DataFolder)
    CrystalGraphFile = DataFolder + "CrystalGraphDescriptors.csv"

    NumCoulombMatrixelements = 40 #Number of Coulomb matrix elements, include padding, should be longer than number of atoms

    CrystalgraphDescriptors = pd.read_csv(CrystalGraphFile, index_col=False)

    #np.set_printoptions(precision=2)

    f = open(DataFolder + "CoulombDescriptors.csv",'w')
    f.write("MaterialID,")
    for i in range(0,NumCoulombMatrixelements):
        f.write("CMDescriptor" + str(i) +",")
    f.write("CMDescriptorTrace,CMDescriptorDeterminant,CMDescriptorNumPositive,CMDescriptorNumNegative" + "\n")

    g = open(DataFolder + "CrystalStructureDescriptors.csv",'w')
    g.write("MaterialID,ADescriptor,BDescriptor,CDescriptor,alphaDescriptor,betaDescriptor,gammaDescriptor" + "\n")

    for i in range(0,len(CrystalgraphDescriptors)):
        CurrentMat = CrystalgraphDescriptors.iloc[i]["MaterialID"]
        print(i, CurrentMat)
        #read .cif
        structure = CifParser(DataFolder + CurrentMat + ".cif").get_structures()[0]
        NumAtoms = len(structure.species)
        #MAke sure there are enough coulomb matrix elements allowed
        if NumAtoms > NumCoulombMatrixelements:
            print("Too few Coulomb matrix elments requested.")
            print(CurrentMat + " has " + str(NumAtoms) + " atoms")
            break
        #print(structure)
        #print(structure.species)
        #print(structure.atomic_numbers)
        #print(structure.distance_matrix) #verified this works with periodic boundaries
        #Calculate Coulomb matrix
        #only need diagonal and upper/lower triangle
        CMatTriangle=np.zeros((NumCoulombMatrixelements,NumCoulombMatrixelements)) #this format puts 0's in the middle
        CMatDiag=np.zeros((NumCoulombMatrixelements,NumCoulombMatrixelements))
        #first do diagonal
        for j in range(0,NumAtoms):
            CMatDiag[j][j] = 0.5 * structure.atomic_numbers[j] **2.4
    
        #do a triangle
        for j in range(0,NumAtoms-1):
            for k in range(j+1,NumAtoms):
                CMatTriangle[j][k] = structure.atomic_numbers[j] * structure.atomic_numbers[k] / structure.distance_matrix[j][k]
        #print(CMatTriangle) #verified correct
        #print(CMatTriangle.T)
        #print(CMatTriangle + CMatTriangle.T)
        CMat = CMatTriangle + CMatDiag + CMatTriangle.T
        #print(CMat)
        #print(CMatDiag)
        CMatEvals = eigvalsh(CMat)
        #print(CMatEvals)
    
        #write data to Coulomb matrix file
        f.write(CurrentMat + ",")
        for value in CMatEvals:
            f.write(str(value) + ",")
        f.write(str(np.trace(CMat)) + ",")
        NonZeroDet = np.abs(np.prod([a for a in CMatEvals if a !=0])) #tested correct
        ScaledNonZeroDet = np.sign(NonZeroDet) * NonZeroDet ** (1/NumAtoms)
        NumPositive = len([a for a in CMatEvals if a > 0])
        NumNegative = len([a for a in CMatEvals if a < 0])
        f.write(str(ScaledNonZeroDet) + "," + str(NumPositive) + "," + str(NumNegative) +"\n")

        #write data to crystal structure file
        #print(structure.lattice.a)
        #print(structure.lattice.alpha)
        g.write(CurrentMat + ",")
        g.write(str(structure.lattice.a) + ",")
        g.write(str(structure.lattice.b) + ",")
        g.write(str(structure.lattice.c) + ",")
        g.write(str(structure.lattice.alpha) + ",")
        g.write(str(structure.lattice.beta) + ",")
        g.write(str(structure.lattice.gamma) + "\n")


    f.close()
    g.close()








