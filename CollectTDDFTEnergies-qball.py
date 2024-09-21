'''
@author: shapera
'''
#collects the tddft energies from qball output
#takes from copies of files made by CollectDFTData-qball.py
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
import re
import sys

RunName = sys.argv[1]
RedirectFolder = ""

DataFolderDFT = RedirectFolder + "DataDFT/" + RunName + "/"
DataFolderEnergies = RedirectFolder + "DataEnergies/" + RunName + "/"

for OutputFolder in [DataFolderEnergies]:
    if not os.path.exists(OutputFolder): #check if DataFolder exists             
        os.makedirs(OutputFolder)

Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]


for pos in xzPos:
    if pos =="x_5_z_5":
        continue
    for speed in Speeds:
        print(pos+"/"+"v_"+speed+"/")
        origDataFolder = DataFolderDFT + pos + "/" + speed + "/"
        collectedDataFolder = DataFolderEnergies + pos + "/" + speed + "/"
        for OutputFolder in [collectedDataFolder]:
            if not os.path.exists(OutputFolder): #check if DataFolder exists             
                os.makedirs(OutputFolder)
        h = open(collectedDataFolder + "TDDFTEnergies.csv","w")
        h.write("MaterialID,Time,Energy")
        
        #MatID = TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0]
        #read DFT file line by line
        dftFile = open(origDataFolder + RunName + ".v_" + speed + "-" + pos + ".out",'r')
        count = 0
        Energies = []
        while True:
            count += 1
            line = dftFile.readline()
            if not line:
                break
            if "<etotal>" in line:
                Energies.append(line.split()[1])
            if "set dt" in line: #get time step
                timestep = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]
        #print(len(Energies))
        dftFile.close()
        
        for i in range(0,len(Energies)):
            h.write("\n")
            h.write(RunName + "." + "v_" + speed + "-" + pos + "-iter-" + str(int(i+1)) + "," + str(float(timestep) * i ) + "," + Energies[i])
            
            
        
        
        
        
        h.close()





























