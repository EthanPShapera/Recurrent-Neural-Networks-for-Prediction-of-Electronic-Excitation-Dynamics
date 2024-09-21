'''
@author: shapera
'''
#Collects time-dependent data from qball and stores structures, velocities as .cif files
import os
from pymatgen.core import Structure
import shutil
import subprocess
import itertools
import re
import collections
import sys

atomDict = {'carbon':'C','hydrogen':'H', 'chloride':'Cl'}


RunName = sys.argv[1]
TopDataFolder = ""#Data folder with raw TDDFT inputs/outputs
RedirectFolder = ""

DataFolderPos = RedirectFolder + "DataPosition/" + RunName + "/"
DataFolderVel = RedirectFolder + "DataVelocity/" + RunName + "/"
DataFolderInit = RedirectFolder + "DataInit/" + RunName + "/"
DataFolderDFT = RedirectFolder + "DataDFT/" + RunName + "/"
DataFolderOccupation = RedirectFolder + "DataOccupation/" + RunName + "/"

for OutputFolder in [DataFolderPos,DataFolderVel,DataFolderInit,DataFolderDFT,DataFolderOccupation]:
    if not os.path.exists(OutputFolder): #check if DataFolder exists             
        os.makedirs(OutputFolder)


TargetSystem = RunName#"ch4"
Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]

'''h = open(DataFolderOccupation + "TDDFTOccupations-" + TargetSystem + ".csv","w")
h.write("MaterialID,Time")
assumedOrbitals = 100 #guess number of orbitals
for i in range(0,assumedOrbitals):
    h.write(",Occ"+str(i))
h.write("\n")'''
'''pr = open(DataFolderPos + "id_prop.csv","w")
pv = open(DataFolderVel + "id_prop.csv","w")
p0r = open(DataFolderInit + "id_propR.csv","w")
p0v = open(DataFolderInit + "id_propV.csv","w")'''

'''for speed in Speeds:#["1.0"]:#Speeds:
    for pos in xzPos:'''
for pos in xzPos:
    for speed in Speeds:
        #print("v_"+speed+"/"+pos+"/")
        print(pos+"/"+"v_"+speed+"/")
        #origDataFolder = TopDataFolder + "v_" + speed + "/" + pos + "/"
        origDataFolder = TopDataFolder + pos + "/" + "v_" + speed + "/"
        print(origDataFolder)
        #make separate folders to hold cif and poscars for each xzPos and speed
        DataFolderPos = DataFolderPos + pos + "/" + speed + "/"
        DataFolderVel = DataFolderVel + pos + "/" + speed + "/"
        DataFolderInit = DataFolderInit + pos + "/" + speed + "/"
        DataFolderDFT = DataFolderDFT + pos + "/" + speed + "/"
        DataFolderOccupation = DataFolderOccupation + pos + "/" + speed + "/"
        for OutputFolder in [DataFolderPos,DataFolderVel,DataFolderInit,DataFolderDFT,DataFolderOccupation]:
            if not os.path.exists(OutputFolder): #check if DataFolder exists             
                os.makedirs(OutputFolder)
        pr = open(DataFolderPos + "id_prop.csv","w")
        pv = open(DataFolderVel + "id_prop.csv","w")
        p0r = open(DataFolderInit + "id_propR.csv","w")
        p0v = open(DataFolderInit + "id_propV.csv","w")
        h = open(DataFolderOccupation + "TDDFTOccupations-" + TargetSystem + ".csv","w")
        h.write("MaterialID,Time")
        assumedOrbitals = 100 #guess number of orbitals
        for i in range(0,assumedOrbitals):
            h.write(",Occ"+str(i))
        h.write("\n")
              
        #Copy over DFT results
        newDFTOutFile = DataFolderDFT + TargetSystem + "." + "v_" + speed + "-" + pos + ".out"
        #shutil.copyfile(origDataFolder + TargetSystem + ".out", newDFTOutFile)
        shutil.copyfile(origDataFolder + "OUTPUT", newDFTOutFile)
        
        #get unit cell dimensions:
        #correct format of data in qball output, a0->a, a1->b, a2->c
        subprocess.call(["sed", "-i", "s/a0=/a=/g", newDFTOutFile])
        subprocess.call(["sed", "-i", "s/a1=/b=/g", newDFTOutFile])
        subprocess.call(["sed", "-i", "s/a2=/c=/g", newDFTOutFile])
        sedCellCommand = "sed -n '/<cell/,/>/p' '" + newDFTOutFile +"'"
        Cell = subprocess.check_output(sedCellCommand,shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        #print(Cell)
        Cell = re.findall('\d*\.?\d+',Cell)
        sortedCell = new_list = [Cell[i:i+3] for i in range(0, len(Cell), 3)]
        print(sortedCell)
        
        #read through file line by line
        AtomPoses = []
        AtomVelocities = []
        AtList = []
        with open(newDFTOutFile,'r+') as DFTData:
            lines = DFTData.readlines()
            ReadingOccupations = False
            for lineNum in range(0,len(lines)):
                line = lines[lineNum]
                if "set dt" in line: #get time step
                    timestep = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]
                    print(timestep)
                if "iteration count" in line:
                    IterationNum = re.findall('\d*\.?\d+',line)
                    print(IterationNum)
                    AtomPoses = []
                    AtomVelocities = []
                    AtList = []
                if "atom name" in line:
                    atomSpecName = line.split('"')[3] #gets atomic species, written out
                    AtomSpec = atomDict[atomSpecName] #symbol for atom
                    AtList.append(AtomSpec)
                    AtPos = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",lines[lineNum+1])#,next(DFTData))
                    AtVel = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",lines[lineNum+2])#,next(next(DFTData)))
                    AtomPoses.append([AtomSpec,AtPos])
                    AtomVelocities.append([AtomSpec,AtVel])
                    #print(AtomSpec)
                    #print(AtPos)
                    #print(AtVel)          
                #Code to read occupation numbers
                if "Number of electron-hole pairs" in line:
                    ReadingOccupations = False
                    h.write("\n")
                if ReadingOccupations:
                    h.write(line.strip()+",")
                if "occupation numbers" in line: #start to list occupation numbers of orbitals one by one
                    ReadingOccupations = True
                    h.write(TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + "," + str(float(timestep) * (float(IterationNum[0])-1.0)) + ",") #occupation number file
                    pr.write(TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position" + "," + "1.0" + "\n") #write identifying information to id_prop.csv file
                    pv.write(TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity" + "," + "1.0" + "\n")
                    if IterationNum[0] == '1': #write to id_prop.csv for initial configuration only
                        p0r.write(TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position" + "," + "1.0" + "\n")
                        p0v.write(TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity" + "," + "1.0" + "\n")
                
                
                
                if "</iteration>" in line: #end of current iteration
                    #write position POSCAR file
                    f = open(DataFolderPos + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.POSCAR",'w')
                    f.write(TargetSystem+"\n")
                    f.write("1.0"+"\n")
                    for i in range(0,3):
                        for j in range(0,3):
                            f.write(sortedCell[i][j]+"\t")
                        f.write("\n")
                    #print(AtomPoses)
                    AtomPoses.sort()
                    AtomCountDict = dict(collections.Counter(AtList))
                    AtomCountDict = dict(sorted(AtomCountDict.items()))
                    for key in AtomCountDict:
                        f.write(key + "\t")
                    f.write("\n")
                    for key in AtomCountDict:
                        f.write(str(AtomCountDict[key]) + "\t")
                    f.write("\n")
                    f.write("Cart"+"\n")
                    for AtomItem in AtomPoses:
                        f.write(AtomItem[1][0] + "\t" + AtomItem[1][1] + "\t" + AtomItem[1][2] + "\n")
                    f.close()
                    
                    #write velocity POSCAR file
                    g = open(DataFolderVel + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.POSCAR",'w')
                    g.write(TargetSystem+"\n")
                    g.write("1.0"+"\n")
                    for i in range(0,3):
                        for j in range(0,3):
                            g.write(sortedCell[i][j]+"\t")
                        g.write("\n")
                    #print(AtomPoses)
                    AtomVelocities.sort()
                    AtomCountDict = dict(collections.Counter(AtList))
                    AtomCountDict = dict(sorted(AtomCountDict.items()))
                    for key in AtomCountDict:
                        g.write(key + "\t")
                    g.write("\n")
                    for key in AtomCountDict:
                        g.write(str(AtomCountDict[key]) + "\t")
                    g.write("\n")
                    g.write("Cart"+"\n")
                    for AtomItem in AtomVelocities:
                        g.write(AtomItem[1][0] + "\t" + AtomItem[1][1] + "\t" + AtomItem[1][2] + "\n")
                    g.close()  
                    
                    #convert position and velocity poscars to cif
                    structure1 = Structure.from_file(DataFolderPos + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.POSCAR")
                    #if int(IterationNum[0]) > 2370 and int(IterationNum[0]) < 2380:
                    #    print(structure1)
                    structure1.to(filename=DataFolderPos + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.cif") #writes POSCAR for input as .cif
                    structure2 = Structure.from_file(DataFolderVel + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.POSCAR")
                    structure2.to(filename=DataFolderVel + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.cif") #writes POSCAR for input as .cif
                    
                    #if on first iteration, copy files over to initial folder
                    if IterationNum[0] == "1":
                        shutil.copyfile(DataFolderPos + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.POSCAR", DataFolderInit + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.POSCAR")
                        shutil.copyfile(DataFolderVel + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.POSCAR", DataFolderInit + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.POSCAR")
                        shutil.copyfile(DataFolderPos + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.cif", DataFolderInit + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Position.cif")
                        shutil.copyfile(DataFolderVel + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.cif", DataFolderInit + TargetSystem + "." + "v_" + speed + "-" + pos + "-iter-" + IterationNum[0] + ".Velocity.cif")
        pr.close()
        pv.close()
        p0r.close()
        p0v.close()
        #clean up output folder names
        DataFolderPos = re.sub(pos + "/" + speed + "/", '', DataFolderPos)
        DataFolderVel = re.sub(pos + "/" + speed + "/", '', DataFolderVel)
        DataFolderInit = re.sub(pos + "/" + speed + "/", '', DataFolderInit)
        DataFolderDFT = re.sub(pos + "/" + speed + "/", '', DataFolderDFT)
        DataFolderOccupation = re.sub(pos + "/" + speed + "/", '', DataFolderOccupation)
        h.close()
'''
h.close()
pr.close()
pv.close()
p0r.close()
p0v.close()'''
