'''
@author: shapera
'''
#Active learning loop on neural network
#uses MAE per electron to choose which run to add in
#Use when there are multiple target sets, for training, interpolation, extrapolation
#Search for "#Increase this" to find values which must be adjusted for full learning run
#Tries to learn change in number of electrons
import pandas as pd
from sklearn.linear_model import ElasticNet
import re
import sys
from math import floor
from sklearn.neural_network import MLPRegressor
import os
import numpy as np
import shutil
import random
from heapq import nlargest
import time
from scipy.stats import spearmanr
import math

###FIXTHIS

#######System Information
MLRuns = ["C1H4_T1000","C1H1Cl3_T1000"] #Data runs to use for machine learning model training
ApplicationRuns = ["C1Cl4_T1000","C1H3Cl1_T1000"] #Data to use for applying model 
#MLRuns = 
#ApplicationRuns = 
RunName = "FullRunEnergyDiff-SpearmanCut-Refit-Seed" #Designation for this ML 
PreviousSteps = "1"
OrbitalToUse = "Energy" #Ground, HOMO or Energy
CalculationType = "Previous" #Previous or Initial
ElectronsInGround = {"C1H4":8,"C1H3Cl1":14,"C1H1Cl3":26,"C1Cl4":32}
if OrbitalToUse == "Energy":
    ElectronNormalizeCond = False
else:
    ElectronNormalizeCond = False

##########Active Learning Information
ActiveLearningRun = "14RELU" #which run number of active learning
ActiveLearningIterations = 15 #Increase this #number of active learning iterations to do 
ActiveLearningInitial = "x_5_z_4" 
RunsToAdd = 10 # number of new runs to add to ML data each active learning loop

###########Machine Learning Information
MLIterations = 10 #Increase this to 10 for full 
Valhidden_layer_sizes= (64,64,64,64) #Increase this based on ActiveLearningRun 
Valactivation='tanh'
Valsolver='adam'
Valalpha=0.0001 
Valbatch_size='auto' 
Vallearning_rate='constant' 
Vallearning_rate_init=0.001 
Valmax_iter=200
Valshuffle=True
Valrandom_state="Loop" 
Valtol=0.0001
Valverbose=False  
Valbeta_1=0.9
Valbeta_2=0.999 
Valepsilon=1e-08 
Valn_iter_no_change=10
refit = True
################Validation Information
FracInTesting = 0.1
FracInValid = 0.1
#################SpearmanCutoff
SpearmanCut = 0.05

#####################Active learning function
def MeanAbsoulteError(DataTable,TargetName,timeDep,ElectronNormalize): #Calculates error with options for time weighting and normalize by base number of electrons in orbital
    if timeDep == "None": 
        DataTable["TimeWeight"] = 1
    elif timeDep == "Linear":
        totTime = DataTable['Time'].max()
        DataTable["TimeWeight"] = DataTable['Time'].apply(lambda x: x / totTime)
    
    DataTable["Err"] = (DataTable[TargetName] - DataTable[TargetName+"-ML"]) * DataTable["TimeWeight"]
    DataTable["AbsErr"] = DataTable['Err'].apply(lambda x: np.abs(x))
    '''if ElectronNormalize:
        DataTable['Run'] = DataTable['MaterialID'].str.split('-iter').str[0]
        DataTable["TargetMat"] = DataTable['Run'].str.split('_').str[0]
        if OrbitalToUse =="Ground":
            DataTable["BaseElectrons"] = DataTable["TargetMat"].apply(lambda x: ElectronsInGround[x])
        elif OrbitalToUse == "Energy":
            DataTable["BaseElectrons"] = 1.0
        else:
            DataTable["BaseElectrons"] = 2.0     
        DataTable["AbsErr1"] = DataTable["AbsErr"] / DataTable["BaseElectrons"]
        MAE = DataTable["AbsErr1"].mean()
    else:
        MAE = DataTable["AbsErr"].mean()'''
    MAE = DataTable["AbsErr"].mean()
    return(MAE)


#####################Directory information
RedirectFolderData = ""

RedirectFolderResults = ""
if not os.path.exists(RedirectFolderResults): #check if DataFolder exists                                                                                                
    os.makedirs(RedirectFolderResults)



######################Start the new stuff

if CalculationType == "Previous":
    LearningFolder = RedirectFolderResults + RunName + "_" + PreviousSteps + "PreviousSteps_" + OrbitalToUse + "Orbital/"
elif CalculationType == "Initial":
    LearningFolder = RedirectFolderResults + RunName + "_" + "InitialConditionsOnly_" + OrbitalToUse + "Orbital/"
if not os.path.exists(LearningFolder): #check if DataFolder exists                                                                                                
    os.makedirs(LearningFolder)
LearningFolder = LearningFolder + "ActiveLearningRun" + ActiveLearningRun + "/"
if not os.path.exists(LearningFolder): #check if DataFolder exists                                                                                                
    os.makedirs(LearningFolder)
print(LearningFolder)
#make record of how each dataset is used
frecord = open(LearningFolder + "RunUsage.txt",'w')
frecord.write("Runs used for fitting machine learning model:" +"\n")
for item in MLRuns:
    frecord.write("\t" + item + "\n")
frecord.write("Runs used machine learning model is applied to:" +"\n")
for item in ApplicationRuns:
    frecord.write("\t" + item + "\n")
frecord.close()
#copy ML and Application datafiles to learning folder
MLRunDataFiles = []
for DataRun in MLRuns:
    DataFolderComplete = RedirectFolderData + "DataComplete/" + DataRun + "/FullyCombined/"
    if CalculationType == "Previous":
        #copy over datafile
        if not os.path.exists(LearningFolder + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv"):
            shutil.copyfile(DataFolderComplete + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv", LearningFolder + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv")
        MLRunDataFiles.append(LearningFolder + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv") #overall combined data file  
    elif CalculationType == "Initial":
        if not os.path.exists(LearningFolder + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv"):
            shutil.copyfile(DataFolderComplete + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv", LearningFolder + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv")
        MLRunDataFiles.append(LearningFolder + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv") #overall combined data file 
ApplicationRunDataFiles = []
for DataRun in ApplicationRuns:
    DataFolderComplete = RedirectFolderData + "DataComplete/" + DataRun + "/FullyCombined/"
    if CalculationType == "Previous":
        #copy over datafile
        if not os.path.exists(LearningFolder + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv"):
            shutil.copyfile(DataFolderComplete + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv", LearningFolder + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv")
        ApplicationRunDataFiles.append(LearningFolder + "CombinedDescriptors_" + DataRun + "_" + str(PreviousSteps) + "PreviousSteps_" + str(OrbitalToUse) + "Orbital.csv") #overall combined data file  
    elif CalculationType == "Initial":
        if not os.path.exists(LearningFolder + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv"):
            shutil.copyfile(DataFolderComplete + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv", LearningFolder + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv")
        ApplicationRunDataFiles.append(LearningFolder + "CombinedDescriptors_" + DataRun + "_InitialConditionsOnly_" + str(OrbitalToUse) + "Orbital.csv") #overall combined data file 


#set up pandas dataframe with all data
FullData = [pd.read_csv(f, index_col=False) for f in MLRunDataFiles]
FullData = pd.concat(FullData, ignore_index=True) #single data frame for ML Data
SeparatedApplicationData = [pd.read_csv(f, index_col=False) for f in ApplicationRunDataFiles]
FullApplicationData = pd.concat(SeparatedApplicationData, ignore_index=True) #single dataframe for Application Data
#add in some column data that was not previously included
FullData['Run'] = FullData['MaterialID'].str.split('-iter').str[0]
FullData["TargetMat"] = FullData['Run'].str.split('_').str[0]
if OrbitalToUse =="Ground":
    FullData["BaseElectrons"] = FullData["TargetMat"].apply(lambda x: ElectronsInGround[x])
elif OrbitalToUse == "Energy":
    FullData["BaseElectrons"] = 0.0 #1.0 always use differences with this code
else:
    FullData["BaseElectrons"] = 2.0 #2.0 we will always use diference from 
ColumnNames = FullData.columns
RunNames = list(set(FullData["MaterialID"].str.split('-iter').str[0]))
NumRunNames = len(RunNames)
RunCounts = FullData.Run.value_counts().to_dict()
#Add column for weight of each run
FullData['Weight'] = FullData['Run'].map(RunCounts)
FullData['Weight'] = FullData['Weight'].apply(lambda x: 1.0/x)
print("Full dataset size = "+ str(FullData.shape))
print(FullData.Run.value_counts())
print(NumRunNames)
###
FullApplicationData['Run'] = FullApplicationData['MaterialID'].str.split('-iter').str[0]
FullApplicationData["TargetMat"] = FullApplicationData['Run'].str.split('_').str[0]
if OrbitalToUse =="Ground":
    FullApplicationData["BaseElectrons"] = FullApplicationData["TargetMat"].apply(lambda x: ElectronsInGround[x])
elif OrbitalToUse == "Energy":
    FullApplicationData["BaseElectrons"] = 0.0 #1.0
else:
    FullApplicationData["BaseElectrons"] = 2.0 #2.0
ApplicationColumnNames = FullApplicationData.columns
ApplicationRunNames = list(set(FullApplicationData["MaterialID"].str.split('-iter').str[0]))
NumApplicationRunNames = len(ApplicationRunNames)
ApplicationRunCounts = FullApplicationData.Run.value_counts().to_dict()
#Add column for weight of each run
FullApplicationData['Weight'] = FullApplicationData['Run'].map(ApplicationRunCounts)
FullApplicationData['Weight'] = FullApplicationData['Weight'].apply(lambda x: 1.0/x)


#find target value differences from initial condition
if OrbitalToUse == "Energy":
    #get starting energy of each run
    if CalculationType == "Previous":
        FullDataStartingEnergies = {} 
        for RunE in RunNames:
            StartEnergy = FullData.loc[FullData["MaterialID"] == RunE+'-iter-'+ str(int(PreviousSteps)+1), 'EnergyT-'+PreviousSteps].iloc[0]            
            FullDataStartingEnergies[RunE] = StartEnergy
        #Shift energies at every timestep
        for i in range (0,int(PreviousSteps)+1):
            FullData["EnergyT-"+str(i)+"Diff"] = FullData["EnergyT-"+str(i)].sub(FullData['Run'].map(FullDataStartingEnergies))
        #Also shift energies for application
        FullApplicationDataStartingEnergies = {} 
        for RunE in ApplicationRunNames:
            StartApplicationEnergy = FullApplicationData.loc[FullApplicationData["MaterialID"] == RunE+'-iter-'+ str(int(PreviousSteps)+1), 'EnergyT-'+PreviousSteps].iloc[0]
            FullApplicationDataStartingEnergies[RunE] = StartApplicationEnergy
        #Shift energies at every timestep
        for i in range (0,int(PreviousSteps)+1):
            FullApplicationData["EnergyT-"+str(i)+"Diff"] = FullApplicationData["EnergyT-"+str(i)].sub(FullApplicationData['Run'].map(FullApplicationDataStartingEnergies))
else:
    if CalculationType == "Previous":
        FullData["Occ" + OrbitalToUse + "T-0Diff"] = FullData["Occ" + OrbitalToUse + "T-0"] - FullData["BaseElectrons"]
        FullApplicationData["Occ" + OrbitalToUse + "T-0Diff"] = FullApplicationData["Occ" + OrbitalToUse + "T-0"] - FullApplicationData["BaseElectrons"]
        for i in range(1,int(PreviousSteps)+1): #Also need to get difference for column at previous timesteps
            FullData["Occ" + OrbitalToUse + "T-" + str(i) + "Diff"] = FullData["Occ" + OrbitalToUse + "T-" + str(i)] - FullData["BaseElectrons"]
            FullApplicationData["Occ" + OrbitalToUse + "T-" + str(i) + "Diff"] = FullApplicationData["Occ" + OrbitalToUse + "T-" + str(i)] - FullApplicationData["BaseElectrons"]
    elif CalculationType == "Initial":
        FullData["Occ" + OrbitalToUse + "Diff"] = FullData["Occ" + OrbitalToUse] - FullData["BaseElectrons"]
        FullApplicationData["Occ" + OrbitalToUse + "Diff"] = FullApplicationData["Occ" + OrbitalToUse] - FullApplicationData["BaseElectrons"]
        













#Separate into data for machine learning and unused data
#data for machine learning will go into machine learning model, including 80-10-10 validation
#unused data will only have model applied to it.
InitialRunsForML = [a for a in RunNames if ActiveLearningInitial in a]
UnusedRuns = [a for a in RunNames if ActiveLearningInitial not in a]
print("Initial Runs for ML", InitialRunsForML)
MLData = FullData.iloc[np.where(FullData.Run.isin(InitialRunsForML))]
UnusedData = FullData.iloc[np.where(FullData.Run.isin(UnusedRuns))]
if UnusedData.empty:
    print('UnusedData DataFrame is empty!')
MLRunNames = list(set(MLData["MaterialID"].str.split('-iter').str[0]))
UnusedRunNames = list(set(UnusedData["MaterialID"].str.split('-iter').str[0]))


#write file for NN hyperparameters and active learning parameters
#write NN hyperparameters to file
fHyp = open(LearningFolder + "Hyperparameters.txt", 'w')
fHyp.write("hidden_layer_sizes = " + str(Valhidden_layer_sizes) + "\n")
fHyp.write("activation = " + str(Valactivation) + "\n")
fHyp.write("solver = " + str(Valsolver) + "\n")
fHyp.write("alpha = " + str(Valalpha) + "\n")
fHyp.write("batch_size = " + str(Valbatch_size) + "\n")
fHyp.write("learning_rate = " + str(Vallearning_rate) + "\n")
fHyp.write("learning_rate_init = " + str(Vallearning_rate_init) + "\n")
fHyp.write("max_iter = " + str(Valmax_iter) + "\n")
fHyp.write("shuffle = " + str(Valshuffle) + "\n")
fHyp.write("random_state = " + str(Valrandom_state) + "\n")
fHyp.write("tol = " + str(Valtol) + "\n")
fHyp.write("verbose = " + str(Valverbose) + "\n")
fHyp.write("beta_1 = " + str(Valbeta_1) + "\n")    
fHyp.write("beta_2 = " + str(Valbeta_2) + "\n")    
fHyp.write("epsilon = " + str(Valepsilon) + "\n")    
fHyp.write("n_iter_no_change = " + str(Valn_iter_no_change) + "\n")
fHyp.write("Refit if not at min loss = " + str(refit) + "\n")
fHyp.write("Spearman Cutoff = " + str(SpearmanCut) + "\n")    
fHyp.close()

#timing information
fTime = open(LearningFolder + "IterationTime.csv", 'w')
fTime.write("ActiveIteration,ValidationIteration,Time" + "\n")


for ActiveIter in range(0,ActiveLearningIterations): #loop over active learning iterations
    print("Active Learning Iteration "+ str(ActiveIter) + ", ML dataset size = "+ str(MLData.shape))
    print("Active Learning Iteration "+ str(ActiveIter) + ", Unused dataset size = "+ str(UnusedData.shape))
    ActiveIterationFolder = LearningFolder + "ActiveIteration" + str(ActiveIter) + "/"
    if not os.path.exists(ActiveIterationFolder): #check if DataFolder exists                                                                                                
        os.makedirs(ActiveIterationFolder)
    ActiveIterationResultsFolder = ActiveIterationFolder + "MLIterationAveragedResults/" # set up folder for writing iteration results
    if os.path.exists(ActiveIterationResultsFolder):
        print("Active Iteration already done")
        continue
    #partition off ML data into a testing set
    NumTesting = max(1,floor(len(MLRunNames) * FracInTesting))
    NumValid = max(1,floor(len(MLRunNames) * FracInValid))
    TestingRunNames = random.sample(MLRunNames, NumTesting)
    RunsNotInTesting = [a for a in MLRunNames if a not in TestingRunNames]
    TestingData = MLData.iloc[np.where(MLData.Run.isin(TestingRunNames))] #dataframe with testing points
    #print(TestingRunNames)
    #print(TestingData.shape)
    #Initialize dictionaries to hold MAEs for each run
    def InitializeErrorDict(runnames,iterations):
        tempdict = {}
        for rn in runnames:
            tempdict[rn] = [0 for i in range(iterations)] 
        return tempdict
    UnusedMAEs = InitializeErrorDict(UnusedRunNames, MLIterations) #holds mae data for every run and iteration of ML
    UnusedMAEsAvgd = InitializeErrorDict(UnusedRunNames, 1) #averages over ML iterations
    #Begin loop over validation process
    for MLIter in range(0,MLIterations):
        start = time.time()
        print("Active Learning Iteration "+ str(ActiveIter) + ", Validation Iteration " + str(MLIter))
        IterationFolder = ActiveIterationFolder + "ValidIteration" + str(MLIter) + "/" # set up folder for writing iteration results
        if not os.path.exists(IterationFolder):
            os.makedirs(IterationFolder)
        #select data for validation
        ValidRunNames = random.sample(RunsNotInTesting, NumValid)
        FittingRunNames = [a for a in RunsNotInTesting if a not in ValidRunNames]
        #print fitting, valid and testing run names to file
        fRun = open(IterationFolder + "DivisionofRuns.txt",'w')
        fRun.write("Fitting Runs: ")
        for item in FittingRunNames:
            fRun.write(item + ", ")
        fRun.write("\n")
        fRun.write("Validation Runs: ")
        for item in ValidRunNames:
            fRun.write(item + ", ")
        fRun.write("\n")
        fRun.write("Testing Runs: ")
        for item in TestingRunNames:
            fRun.write(item + ", ")
        fRun.close()
        #construct dataframes for fitting and validation
        ValidData = MLData.iloc[np.where(MLData.Run.isin(ValidRunNames))]
        FittingData = MLData.iloc[np.where(MLData.Run.isin(FittingRunNames))]
        #get column names of descriptors and target
        ColumnNames = FittingData.columns
        if OrbitalToUse == "Energy":
            if CalculationType == "Previous":
                DescriptorColumnsAll = ['Time', 'TimeStep']
                for i in range(1,int(PreviousSteps)+1):
                    DescriptorColumnsAll.append("EnergyT-" + str(i) + "Diff")
                TargetColumn = "EnergyT-0Diff"
            elif CalculationType == "Initial":
                DescriptorColumnsAll = ['Time']
                TargetColumn = "EnergyDiff"
        else:
            if CalculationType == "Previous":
                DescriptorColumnsAll = ['Time', 'TimeStep']
                for i in range(1,int(PreviousSteps)+1):
                    DescriptorColumnsAll.append("Occ" + OrbitalToUse + "T-" + str(i) + "Diff")
                TargetColumn = "Occ" + OrbitalToUse + "T-0Diff"
            elif CalculationType == "Initial":
                DescriptorColumnsAll = ['Time']
                TargetColumn = "Occ" + OrbitalToUse + "Diff"
        DescriptorColumnsAll.append("BaseElectrons") 
        DescriptorColumnsAll.extend([item for item in ColumnNames if "Descriptor" in item])
        NumStartingDescriptors = len(DescriptorColumnsAll)
        
        #Introduce SpearmanCutoff for descriptors
        #Only run check on first validation iteration
        if MLIter == 0:
            fSpear = open(ActiveIterationFolder + "SpearmanData.csv",'w')
            fSpearChosen = open(ActiveIterationFolder + "SpearmanChosenDescriptors",'w')
            for Descript in DescriptorColumnsAll:
                fSpear.write(Descript + ",")
            fSpear.write("\n")
            DescriptorColumsforML = []
            BannedDescriptors = ["ADescriptor", "BDescriptor","CDescriptor"]
            for Descript in DescriptorColumnsAll:
                SpearVal,p = spearmanr(FittingData[TargetColumn],FittingData[Descript])
                if math.isnan(SpearVal):
                    SpearVal = 0.0
                fSpear.write(str(SpearVal) + ",")
                if abs(SpearVal) >= SpearmanCut and Descript not in BannedDescriptors:
                    DescriptorColumsforML.append(Descript)
                    fSpearChosen.write(Descript + "\n")
            if "BaseElectrons" not in DescriptorColumsforML:
                DescriptorColumsforML.append("BaseElectrons")
                fSpearChosen.write("BaseElectrons" + "\n")
            fSpear.close()
            fSpearChosen.close()
            print(DescriptorColumsforML)
        
        
        #Set up dataframes to hold results
        FitResults = FittingData[["MaterialID","Time","Run",TargetColumn]].copy()
        ValidResults = ValidData[["MaterialID","Time","Run",TargetColumn]].copy()
        TestingResults = TestingData[["MaterialID","Time","Run",TargetColumn]].copy()
        UnusedResults = UnusedData[["MaterialID","Time","Run",TargetColumn]].copy()
        ApplicationResults = FullApplicationData[["MaterialID","Time","Run",TargetColumn]].copy()
        
        #set Seed
        if Valrandom_state == None:
            NNSeed = None
        elif Valrandom_state == 'Loop':
            NNSeed = 100* ActiveIter + MLIter
        #fit mlp regressor
        regr_1 = MLPRegressor(hidden_layer_sizes=Valhidden_layer_sizes, 
                              activation=Valactivation,
                              solver=Valsolver,
                              alpha=Valalpha, 
                              batch_size=Valbatch_size, 
                              learning_rate=Vallearning_rate, 
                              learning_rate_init=Vallearning_rate_init, 
                              max_iter=Valmax_iter, 
                              shuffle=Valshuffle, 
                              random_state=NNSeed, 
                              tol=Valtol, 
                              verbose=Valverbose,  
                              beta_1=Valbeta_1, 
                              beta_2=Valbeta_2, 
                              epsilon=Valepsilon, 
                              n_iter_no_change=Valn_iter_no_change)
        regr_1.fit(FittingData[DescriptorColumsforML].values,FittingData[TargetColumn].values)
        RegLoss = regr_1.loss_curve_
        #check if NN should be refit to get lower loss
        if refit == True:
            if RegLoss[-1] != min(RegLoss):
                MinLossIndex = RegLoss.index(min(RegLoss))
                regr_1 = MLPRegressor(hidden_layer_sizes = Valhidden_layer_sizes, 
                              activation = Valactivation,
                              solver = Valsolver,
                              alpha = Valalpha, 
                              batch_size = Valbatch_size, 
                              learning_rate = Vallearning_rate, 
                              learning_rate_init = Vallearning_rate_init, 
                              max_iter = MinLossIndex + 1, 
                              shuffle = Valshuffle, 
                              random_state = NNSeed, 
                              tol = Valtol, 
                              verbose = Valverbose,  
                              beta_1 = Valbeta_1, 
                              beta_2 = Valbeta_2, 
                              epsilon = Valepsilon, 
                              n_iter_no_change = Valn_iter_no_change)
                regr_1.fit(FittingData[DescriptorColumsforML].values,FittingData[TargetColumn].values)
                RegLoss2 = regr_1.loss_curve_
                floss2 = open(IterationFolder + "LossCurve2.csv",'w')
                floss2.write("Iteration,Loss" + "\n")
                for iloss in range(0,len(RegLoss2)):
                    floss2.write(str(iloss) + ',' + str(RegLoss2[iloss]) + "\n")
                floss2.close()
        FitResults[TargetColumn+"-ML"] = regr_1.predict(FittingData[DescriptorColumsforML].values)
        ValidResults[TargetColumn + "-ML"] = regr_1.predict(ValidData[DescriptorColumsforML].values)
        TestingResults[TargetColumn + "-ML"] = regr_1.predict(TestingData[DescriptorColumsforML].values)
        UnusedResults[TargetColumn + "-ML"] = regr_1.predict(UnusedData[DescriptorColumsforML].values)
        ApplicationResults[TargetColumn + "-ML"] = regr_1.predict(FullApplicationData[DescriptorColumsforML].values)
        
        #write ml iteration results to files
        FitResults.to_csv(IterationFolder + "FittingResults.csv", index=False)
        ValidResults.to_csv(IterationFolder + "ValidResults.csv", index=False)
        TestingResults.to_csv(IterationFolder + "TestingResults.csv", index=False)
        UnusedResults.to_csv(IterationFolder + "UnusedInMLResults.csv", index=False)
        ApplicationResults.to_csv(IterationFolder + "MLApplicationResults.csv", index=False)
        #record loss when training NN
        floss = open(IterationFolder + "LossCurve.csv",'w')
        floss.write("Iteration,Loss" + "\n")
        for iloss in range(0,len(RegLoss)):
            floss.write(str(iloss) + ',' + str(RegLoss[iloss]) + "\n")
        floss.close()
        
        #calculate MAE per electron for each run in Unused results
        for ThisRunName in UnusedRunNames:
            TempTable = UnusedResults.loc[UnusedResults['Run'] == ThisRunName].copy()
            UnusedMAEs[ThisRunName][MLIter] = MeanAbsoulteError(TempTable,TargetColumn,"None",ElectronNormalizeCond)
        
        #get timing information
        stop = time.time()
        duration = stop - start
        print("Active Learning Iteration "+ str(ActiveIter) + ", Validation Iteration " + str(MLIter) + " took " + str(duration) + " seconds.")
        fTime.write(str(ActiveIter) + "," + str(MLIter) + "," + str(duration) + "\n")
    #write MAEs per electron for unused to file
    
    if not os.path.exists(ActiveIterationResultsFolder):
        os.makedirs(ActiveIterationResultsFolder)
    fActive = open(ActiveIterationResultsFolder + "UnusedInMLResults-ActiveLearningObjectiveFunction.csv", "w")
    fActive.write("RunName,")
    for i in range(0,MLIterations):
        fActive.write("ValidationIteration" + str(i) +",")
    fActive.write("AveragedMAE" + "\n")
    for ThisRunName in UnusedRunNames:
        fActive.write(ThisRunName + ",")
        for i in range(0,MLIterations):
            fActive.write(str(UnusedMAEs[ThisRunName][i]) + ",")
        UnusedMAEsAvgd[ThisRunName][0] = np.mean(UnusedMAEs[ThisRunName]) #average the MAEs
        fActive.write(str(UnusedMAEsAvgd[ThisRunName][0]) + "\n")
    fActive.close()
    
    #Choose unused runs to add in to machine learning
    #select largest average MAE
    LargestMAE = nlargest(RunsToAdd, UnusedMAEsAvgd, key = UnusedMAEsAvgd.get)
    #write to file the chosen runs
    fLarge = open(ActiveIterationResultsFolder + "LargestErrorToAddToNextSet.txt",'w')
    for item in LargestMAE:
        fLarge.write(item + "\n")
    fLarge.close()
    
    #Add to ML data and remove from Unused
    for item in LargestMAE:
        MLRunNames.append(item)
    UnusedRuns = [a for a in RunNames if a not in MLRunNames]
    MLData = FullData.iloc[np.where(FullData.Run.isin(MLRunNames))]
    UnusedData = FullData.iloc[np.where(FullData.Run.isin(UnusedRuns))]
    UnusedRunNames = list(set(UnusedData["MaterialID"].str.split('-iter').str[0]))


fTime.close()















