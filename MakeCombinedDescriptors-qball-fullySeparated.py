'''
@author: shapera
'''
#Combines Coulomb matrix, Crystal graph and crystal structure descriptors into a single file
#for use with qball
#Use with finding position versus time data, knowing position at all T
#Can also get energies in addition to occupations
import pandas as pd
import os
import numpy as np
import itertools
import sys


RunName = "C1H1Cl3_T1000"
RedirectFolder = ""

DataFolderPos = RedirectFolder + "DataPosition/" + RunName + "/"
DataFolderInit = RedirectFolder + "DataInit/" + RunName + "/"
DataFolderOcc = RedirectFolder + "DataOccupation/" + RunName + "/"
DataFolderEnergies = RedirectFolder + "DataEnergies/" + RunName + "/"
DataFolderComplete = RedirectFolder + "DataComplete/" + RunName + "/"
if not os.path.exists(DataFolderComplete): #check if DataFolder exists                                                                                                
    os.makedirs(DataFolderComplete)
numPrevious = 1 #number of prevous time steps to use for time t
OrbitalToUse = 'Energy' #Choose either an integer, HOMO, LUMO, Ground, Energy

AllDescriptorsFile = "AllDescriptors_" + RunName + "_" + str(numPrevious) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv"

#Make descriptors for position data
Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]




for pos in xzPos:
    if pos =="x_5_z_5":
        continue
    for speed in Speeds:
        DataFolder = DataFolderPos + pos + "/" + speed + "/"
        print(DataFolder)
        
        if OrbitalToUse ==  'Energy':
            DataFolderOccupations = DataFolderEnergies + pos + "/" + speed + "/"
        else:
            DataFolderOccupations = DataFolderOcc + pos + "/" + speed + "/"
        DataFolderCompleteRun = DataFolderComplete + pos + "/" + speed + "/"
        if not os.path.exists(DataFolderCompleteRun): #check if DataFolder exists                                                                                                
            os.makedirs(DataFolderCompleteRun)
        

        CoulombFile = DataFolder + "CoulombDescriptors.csv"
        CrystalGraphFile = DataFolder + "CrystalGraphDescriptors.csv"
        CrystalStructFile = DataFolder + "CrystalStructureDescriptors.csv"
    
        CoulombDescr = pd.read_csv(CoulombFile, index_col=False)
        CrystalGDescr = pd.read_csv(CrystalGraphFile, index_col=False)
        CrystalSDescr = pd.read_csv(CrystalStructFile, index_col=False)
        if OrbitalToUse ==  'Energy':
            OccDescr = pd.read_csv(DataFolderOccupations+"TDDFTEnergies.csv", index_col=False)
        else:
            OccDescr = pd.read_csv(DataFolderOccupations+"TDDFTOccupations-"+RunName+".csv", index_col=False).replace(np.nan, 0) #empty columns replaced with 0's
    
        CoulombDescrColumns = [item for item in list(CoulombDescr.columns) if "Descriptor" in item]
        CrystalGDescrColumns = [item for item in list(CrystalGDescr.columns) if "Descriptor" in item]
        CrystalSDescrColumns = [item for item in list(CrystalSDescr.columns) if "Descriptor" in item]
        #print(CoulombDescrColumns)
        #print(CrystalGDescrColumns)
        #print(CrystalSDescrColumns)
        #print(list(OccDescr.columns))
    
        #make columns for full data file
        FullDataColumns = ['MaterialID','Time','TimeStep']
        for i in range(0,numPrevious+1):
            if OrbitalToUse ==  'Energy':
                FullDataColumns.append("Energy"+"T-"+str(i))
            else:
                FullDataColumns.append("Occ"+str(OrbitalToUse)+"T-"+str(i))
        for i in range(0,numPrevious+1):
            for a in CoulombDescrColumns:
                FullDataColumns.append(a+"T-"+str(i))
            for a in CrystalGDescrColumns:
                FullDataColumns.append(a+"T-"+str(i))
        for a in CrystalSDescrColumns:
            FullDataColumns.append(a)
        for item in FullDataColumns:
            print(item)
        #FullDataFrame = pd.DataFrame(columns = FullDataColumns) #Will be the dataframe for the combined descriptors
        #print(FullDataFrame)
        DataPoints = OccDescr["MaterialID"].to_list() #Name of every data point in data set
        #print(DataPoints)
        DataPointsNoIterCount = [a.split("-iter-")[0] for a in DataPoints]
        DataRuns = set(DataPointsNoIterCount)
    
        Iterations = [DataPointsNoIterCount.count(a) for a in DataRuns] #number of iterations in each DataRun
        IterationsForRun = dict(zip(DataRuns,Iterations))
        print(IterationsForRun)
        #print(IterationsForRun)
        
        #Get HOMO orbital for each run
        if OrbitalToUse != 'Energy':
            HOMOs = {}
            for DataRun in DataRuns:
                InitialOccs = OccDescr.loc[OccDescr['MaterialID'] == DataRun + "-iter-1"]
                #print(InitialOccs)
                for j in range(0,len(OccDescr.columns)-2): #first two columns are for id and time
                    if OccDescr.loc[OccDescr['MaterialID'] == DataRun + "-iter-1", 'Occ'+str(j)].values[0] == 0.0: #in ground state, orbital above homo is empty, homo may be partially filled
                        #print(j)
                        HOMOs[DataRun] = j-1
                        break
        #now add data to frame
        #first create a list for a single time
        #then append list to FullDataFrame
        AllData = []
        loopcounter = 0
        for DataRun in DataRuns: #outer loop over runs
            for i in range(1+numPrevious,IterationsForRun[DataRun]+1): #loop over iteration number, indexing starts from 1 for iterations, need previous
                print(DataRun,i,str(loopcounter)+'/'+str(len(DataPointsNoIterCount)-numPrevious*len(DataRuns)))
                DataRow = [] #this is the list which will be appended to the FullDataFrame
                ThisMaterialID = DataRun + "-iter-" + str(i)
                PreviousMatIDs = [DataRun + "-iter-" + str(int(i-a)) for a in range(0,numPrevious+1)] #works backwards from current time step
                #print(ThisMaterialID)
                DataRow.append(ThisMaterialID) #MAterial ID
                DataRow.append(OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Time'].values[0]) #time at step
    
                DataRow.append(OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Time'].values[0]-OccDescr.loc[OccDescr['MaterialID'] == PreviousMatIDs[1], 'Time'].values[0]) #now working
                #loop over times to include descriptors and targets
                for TimeNum in range(0,numPrevious+1): #occupation numbers
                    if type(OrbitalToUse) == int: 
                        DataRow.append(OccDescr.loc[OccDescr['MaterialID'] == PreviousMatIDs[TimeNum], 'Occ'+str(OrbitalToUse)].values[0])
                    elif OrbitalToUse == 'HOMO':
                        DataRow.append(OccDescr.loc[OccDescr['MaterialID'] == PreviousMatIDs[TimeNum], 'Occ'+str(HOMOs[DataRun])].values[0])
                    elif OrbitalToUse == 'LUMO':
                        DataRow.append(OccDescr.loc[OccDescr['MaterialID'] == PreviousMatIDs[TimeNum], 'Occ'+str(HOMOs[DataRun]+1)].values[0])
                    elif OrbitalToUse == 'Ground': #number of electrons in the initial ground state orbitals
                        GroundElec = 0.0
                        for k in range(0,HOMOs[DataRun]+1):
                            GroundElec = GroundElec + OccDescr.loc[OccDescr['MaterialID'] == PreviousMatIDs[TimeNum], 'Occ'+str(k)].values[0]
                        DataRow.append(GroundElec)
                    elif OrbitalToUse == 'Energy':
                        DataRow.append(OccDescr.loc[OccDescr['MaterialID'] == PreviousMatIDs[TimeNum], 'Energy'].values[0])
                    else:
                        print("Not a valid orbital")
                        exit()
                for TimeNum  in range(0,numPrevious+1):
                    for ThisCoulombD in CoulombDescrColumns: #Coulomb matrix descriptor
                        DataRow.append(CoulombDescr.loc[CoulombDescr['MaterialID'] == PreviousMatIDs[TimeNum]+".Position", ThisCoulombD].values[0])
                    for ThisCrysD in CrystalGDescrColumns: #Crystal graph descriptors
                        DataRow.append(CrystalGDescr.loc[CrystalGDescr['MaterialID'] == PreviousMatIDs[TimeNum] + ".Position", ThisCrysD].values[0])
                for ThisStructD in CrystalSDescrColumns:
                    DataRow.append(CrystalSDescr.loc[CrystalSDescr['MaterialID'] == PreviousMatIDs[0] + ".Position", ThisStructD].values[0])
                #Add to DataFrame
                #FullDataFrame.loc[len(FullDataFrame.index)] = DataRow
                #Add to end of data
                AllData.append(DataRow)
                loopcounter = loopcounter + 1
                #print(DataRow)
        #write AllData to a dataframe
        FullDataFrame = pd.DataFrame(AllData, columns=FullDataColumns)
        #print(FullDataFrame)
        #write FullDataFrame to file
        FullDataFrame.to_csv(DataFolderCompleteRun+AllDescriptorsFile, index=False)
        

















