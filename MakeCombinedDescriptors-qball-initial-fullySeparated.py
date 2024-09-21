'''
@author: shapera
'''
#Combines Coulomb matrix, Crystal graph and crystal structure descriptors into a single file
#for use with qball
#Use with finding time data, knowing ONLY INITIAL conditions
import pandas as pd
import copy 
import itertools
import os

RunName = "C1H3Cl1_T1000"#"C1H4_T1000" #"prot1_Cl_run3"
RedirectFolder = ""
DataFolderPos = RedirectFolder + "DataPosition/" + RunName + "/"
DataFolderIn = RedirectFolder + "DataInit/" + RunName + "/"
DataFolderOcc = RedirectFolder + "DataOccupation/" + RunName + "/"
DataFolderComplete = RedirectFolder + "DataComplete/" + RunName + "/"

OrbitalToUse = 'Ground' #which orbital, index from 0, Choose either an integer, HOMO, LUMO, Ground

AllDescriptorsFile = "AllDescriptors_" + RunName + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv"

Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]



for pos in xzPos:
    for speed in Speeds:
        DataFolderInit = DataFolderIn + pos + "/" + speed + "/"
        DataFolderOccupations = DataFolderOcc + pos + "/" + speed + "/"
        DataFolderCompleteRun = DataFolderComplete + pos + "/" + speed + "/"
        if not os.path.exists(DataFolderCompleteRun): #check if DataFolder exists                                                                                                
            os.makedirs(DataFolderCompleteRun)

        CoulombFile = DataFolderInit + "CoulombDescriptors.csv"
        CrystalGraphFile = DataFolderInit + "CrystalGraphDescriptors.csv"
        CrystalStructFile = DataFolderInit + "CrystalStructureDescriptors.csv"
        OrientationFile = DataFolderInit + "OrientationDescriptors.csv"
        
        CoulombDescr = pd.read_csv(CoulombFile, index_col=False)
        CrystalGDescr = pd.read_csv(CrystalGraphFile, index_col=False)
        CrystalSDescr = pd.read_csv(CrystalStructFile, index_col=False)
        OrientationDescr = pd.read_csv(OrientationFile, index_col=False)
        OccDescr = pd.read_csv(DataFolderOccupations + "TDDFTOccupations-" + RunName + ".csv", index_col=False)
        
        #make headings for file descriptors
        FullDataColumns = ['MaterialID','Time']
        FullDataColumns.append("Occ" + str(OrbitalToUse))
        CoulombDescrColumns = [item for item in list(CoulombDescr.columns) if "Descriptor" in item]
        CrystalGDescrColumns = [item for item in list(CrystalGDescr.columns) if "Descriptor" in item]
        CrystalSDescrColumns = [item for item in list(CrystalSDescr.columns) if "Descriptor" in item]
        OrientationDescrColumns = [item for item in list(OrientationDescr.columns) if "Descriptor" in item]
        FullDataColumns = FullDataColumns + CoulombDescrColumns + CrystalGDescrColumns + CrystalSDescrColumns + OrientationDescrColumns
        for item in FullDataColumns:
            print(item)
        
        FullDataFrame = pd.DataFrame(columns = FullDataColumns) #Will be the dataframe for the combined descriptors
        
        DataPoints = OccDescr["MaterialID"].to_list() #Name of every data point in data set
        #print(DataPoints)
        DataPointsIterCount = [a.split("-iter-")[0] for a in DataPoints] #name of every point
        DataPointsInitialIterCount = [a.split("-iter-")[0] for a in DataPoints if a.endswith('-iter-1')] #name of every run
        #print(DataPointsInitialIterCount)
        #count number of iterations
        Iterations = [DataPointsIterCount.count(a) for a in DataPointsInitialIterCount] #contains number of iterations for each run
        #print(Iterations)
        IterationsForRun = dict(zip(DataPointsInitialIterCount,Iterations))
        print(IterationsForRun)
        
        #Get HOMO orbital for each run
        HOMOs = {}
        for DataRun in DataPointsInitialIterCount:
            InitialOccs = OccDescr.loc[OccDescr['MaterialID'] == DataRun + "-iter-1"]
            #print(InitialOccs)
            for j in range(0,len(OccDescr.columns)-2): #first two columns are for id and time
                if OccDescr.loc[OccDescr['MaterialID'] == DataRun + "-iter-1", 'Occ'+str(j)].values[0] == 0.0: #in ground state, orbital above homo is empty, homo may be partially filled
                    #print(j)
                    HOMOs[DataRun] = j-1
                    break
        
        AllData = []
        for DataRun in DataPointsInitialIterCount: #outer loop over different systems
            DataRow = [0] * len(FullDataColumns) #this will hold the data
            DataRow[0] = DataRun #name
            #Add in time-independent initial data
            #first the Coulomb descriptors
            for ThisCoulombD in range(0,len(CoulombDescrColumns)):  # Coulomb matrix descriptor
                DataRow[3+ThisCoulombD] = CoulombDescr.loc[CoulombDescr['MaterialID'] == DataRun + "-iter-1.Position", CoulombDescrColumns[ThisCoulombD]].values[0]
            #next Crystal graph descriptors
            for ThisCrystalGD in range(0,len(CrystalGDescrColumns)):  # Crystal graph descriptor
                DataRow[3+len(CoulombDescrColumns)+ThisCrystalGD] = CrystalGDescr.loc[CrystalGDescr['MaterialID'] == DataRun + "-iter-1.Position", CrystalGDescrColumns[ThisCrystalGD]].values[0]
            #crystal structure descriptors
            for ThisCrystalSD in range(0,len(CrystalSDescrColumns)):  # Crystal graph descriptor
                DataRow[3+len(CoulombDescrColumns)+len(CrystalGDescrColumns)+ThisCrystalSD] = CrystalSDescr.loc[CrystalSDescr['MaterialID'] == DataRun + "-iter-1.Position", CrystalSDescrColumns[ThisCrystalSD]].values[0]
            #Orientation descriptors
            for ThisOrientD in range(0,len(OrientationDescrColumns)):  # Crystal graph descriptor
                DataRow[3+len(CoulombDescrColumns)+len(CrystalGDescrColumns)+len(CrystalSDescrColumns)+ThisOrientD] = OrientationDescr.loc[OrientationDescr['MaterialID'] == DataRun + "-iter-1.Position", OrientationDescrColumns[ThisOrientD]].values[0]
        
            #Add in time-dependent data
            for i in range(1,IterationsForRun[DataRun]+1): #loop over number of timesteps
                ThisMaterialID = DataRun + "-iter-" + str(i)
                print(ThisMaterialID)
                #print(OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Time'].values[0])
                DataRow[1] = OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Time'].values[0] #time
                #now do orbital occupation
                if type(OrbitalToUse) == int:
                    DataRow[2] = OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Occ'+str(OrbitalToUse)].values[0]
                elif OrbitalToUse == "HOMO":
                    DataRow[2] = OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Occ'+str(HOMOs[DataRun])].values[0]
                elif OrbitalToUse == "LUMO":
                    DataRow[2] = OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Occ'+str(HOMOs[DataRun]+1)].values[0]
                elif OrbitalToUse == "Ground":
                    GroundElec = 0.0
                    for k in range(0,HOMOs[DataRun]+1):
                        GroundElec = GroundElec + OccDescr.loc[OccDescr['MaterialID'] == ThisMaterialID, 'Occ'+str(k)].values[0]
                    DataRow[2] = GroundElec
                else:
                    print("Not a valid orbital")
                    exit()
                
                #print(DataRow)
                #Add to Dataframe
                AllData.append(copy.copy(DataRow))
                #FullDataFrame.loc[len(FullDataFrame.index)] = DataRow
        
        #Write dataframe to file
        FullDataFrame = pd.DataFrame(AllData, columns=FullDataColumns)
        FullDataFrame.to_csv(DataFolderCompleteRun+AllDescriptorsFile, index=False)







