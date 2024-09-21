'''
@author: shapera
'''
#Run this after MakeCombinedDescriptors-qball-fullySeparated.py
#import pandas as pd
import os
import numpy as np
import itertools
import subprocess
import sys

RunName = sys.argv[1]
RedirectFolder = ""
numPrevious = 1 #number of prevous time steps to use for time t
OrbitalToUse = 'Energy' #Choose either an integer, HOMO, LUMO, Ground, Energy


CalculationType = "Previous" #Previous or Initial
PreviousSteps = numPrevious

DataFolderComplete = RedirectFolder + "DataComplete/" + RunName + "/"
FinalDataFolder = DataFolderComplete + "FullyCombined/"
if not os.path.exists(FinalDataFolder): #check if DataFolder exists                                                                                                
    os.makedirs(FinalDataFolder)
FinalDataFile = FinalDataFolder + "CombinedDescriptors_" + RunName + "_" + str(numPrevious) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv"

if CalculationType == "Previous":
    DataFile = "CombinedDescriptors_" + RunName + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv" #overall combined data file
    OneRunDescriptorsFile = "AllDescriptors_" + RunName + "_" + str(numPrevious) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv"  
elif CalculationType == "Initial":
    DataFile = "CombinedDescriptors_" + RunName + "_" + "InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv" #overall combined data file 
    OneRunDescriptorsFile = "AllDescriptors_" + RunName + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv"
#print(FinalDataFile)


Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(0,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
#Speeds = ["0.25","0.5"]
#XPosVals = [str(a) for a in range(0,2)]#11)]
#ZPosVals = [str(a) for a in range(0,2)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]



inputlines = 0
OutputFileCounter = 0
TotalIndividualLines = 0 #this will count the number of data rows in all folders, excludes heading
for i,pos in enumerate(xzPos):
    if pos =="x_5_z_5":
        continue
    for j,speed in enumerate(Speeds):
        if i==0 and j ==0:
            if CalculationType == "Previous":
                FinalDataFile = FinalDataFolder + "CombinedDescriptors_" + RunName + "_" + str(numPrevious) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital-Pt0.csv"
            elif CalculationType == "Initial":
                FinalDataFile = FinalDataFolder + "CombinedDescriptors_" + RunName + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital-Pt0.csv"
        DataFolder = DataFolderComplete + pos + "/" + speed + "/"
        IndividualFile = DataFolder + OneRunDescriptorsFile
        #print(IndividualFile)
        
        #count lines in new file
        LinesInNewCommand = "wc -l < " + "'" + IndividualFile + "'" 
        LinesInNew = float(subprocess.run(LinesInNewCommand, capture_output = True, shell=True).stdout)
        TotalIndividualLines = TotalIndividualLines + LinesInNew -1 
        if i==0 and j ==0: #no file, do not querry
            LinesInCombined = 0
        else:
            LinesInCombinedCommand = "wc -l < " + "'" + FinalDataFile + "'"
            LinesInCombined = float(subprocess.run(LinesInCombinedCommand, capture_output = True, shell=True).stdout)
        if LinesInNew + LinesInCombined >= 9000000: #when too many, open new data file
            OutputFileCounter = OutputFileCounter + 1
            if CalculationType == "Previous":
                FinalDataFile = FinalDataFolder + "CombinedDescriptors_" + RunName + "_" + str(numPrevious) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital-Pt" + str(OutputFileCounter) + ".csv"
            elif CalculationType == "Initial":
                FinalDataFile = FinalDataFolder + "CombinedDescriptors_" + RunName + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital-Pt" + str(OutputFileCounter) + ".csv"
            LinesInCombined = 0 #reset number of lines in combined
            #if making a new file, copy over, toinclude header
            Command = "cp " + "'" + IndividualFile + "'" + " " + "'" + FinalDataFile + "'"
        else:    
            if i ==0 and j == 0: #copy over first file
                Command = "cp " + "'" + IndividualFile + "'" + " " + "'" + FinalDataFile + "'"
            else:
                Command = "tail -n +2 " + "'" + IndividualFile + "'" + " >> " + "'" + FinalDataFile + "'"
        print(Command)
        os.system(Command)
                
        '''linecommand = "wc -l < " + IndividualFile
        lines = lines + float(subprocess.run(linecommand, capture_output = True, shell=True).stdout) -1 #-1 to remove header
        if i ==0 and j ==0: #add back in header for initial
            lines = lines + 1 #remove hea'''
        
print("Total number of data lines = ", TotalIndividualLines)

















