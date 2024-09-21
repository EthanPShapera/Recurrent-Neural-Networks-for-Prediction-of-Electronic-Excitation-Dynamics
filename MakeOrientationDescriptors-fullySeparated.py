'''
@author: shapera 
'''
#Makes descriptors describing relative orientation of projectile and target at initial time
#descriptors must be invariant under translation, rotation
#Lattice vectors must be orthogonal
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
import numpy as np
import math
import itertools

RunName = "C1H3Cl1_T1000"#"C1H4_T1000"
RedirectFolder = ""

Speeds = ["0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5"]
XPosVals = [str(a) for a in range(5,11)]#11)]
ZPosVals = [str(a) for a in range(0,11)]#11)]
rxz = list(itertools.product(XPosVals, ZPosVals))
xzPos = ["x_"+a[0]+"_z_"+a[1] for a in rxz]

for pos in xzPos:
    for speed in Speeds:
        print(pos,speed)
        DataFolderPos = RedirectFolder + "DataPosition/" + RunName + "/" + pos + "/" + speed + "/"
        DataFolderVel = RedirectFolder + "DataVelocity/" + RunName + "/" + pos + "/" + speed + "/"
        DataFolderInit = RedirectFolder + "DataInit/" + RunName + "/" + pos + "/" + speed + "/"
        
        OrientationDescriptorsFile = "OrientationDescriptors.csv"
        OrientationColumns = ["MaterialID","ProjectileSpeedDescriptor","ProjectileCOMDistanceDescriptor","VelocityRelAngleDescriptor","AtomicMomentMagDescriptor","AtomicMomentRelAngleDescriptor"]
        InitFile = DataFolderInit + "CrystalGraphDescriptors.csv"
        InitDescr = pd.read_csv(InitFile, index_col=False)
        
        DataPoints = InitDescr["MaterialID"].to_list()
        DataPointsNoIterCount = [a.split("-iter-")[0] for a in DataPoints]
        #print(DataPointsNoIterCount)
        
        OrientDataFrame = pd.DataFrame(columns = OrientationColumns) #Will be the dataframe for the combined descriptors
        
        for DataRun in DataPointsNoIterCount:
            DataRow = [] #will hold data found from iteration, to add to OrientDataFrame
            PosData = Structure.from_file(DataFolderInit + DataRun + "-iter-1.Position.POSCAR")
            VelData = Structure.from_file(DataFolderInit + DataRun + "-iter-1.Velocity.POSCAR")
            #print(PosData)
            #print(PosData[1])
            #print(PosData.species[1])
            #print(Element(PosData.species[1]).Z) #gets atomic number of species i
            #print(VelData)
            #print(VelData[0].a) #gets x component
            #print(VelData.lattice.a)
            #find velocity vector for each atom, assume projectile is the fastest
            Velocities = []
            for i in range(0,len(VelData)):
                velo = [VelData[i].a * VelData.lattice.a, VelData[i].b * VelData.lattice.b, VelData[i].c * VelData.lattice.c ]
                Velocities.append(velo)
            #print(Velocities)
            VelMags = [np.linalg.norm(x) for x in Velocities]
            #print(VelMags)
            ProjectileSpeed = max(VelMags) #Speed of projectile
            ProjectileVelocity = Velocities[VelMags.index(max(VelMags))] #velocity of projectile
            #print(ProjectileSpeed,ProjectileVelocity)
        
            #Find which atom is the projectile, look at t=0, assume projectile is the farthest away
            distMat = [[PosData.get_distance(i, j) for i in range(0,len(PosData))] for j in range(0,len(PosData))]
            avgDist = [np.mean(a) for a in distMat]
            ProjectileNumber = avgDist.index(max(avgDist))
            ProjectilePos = [PosData[ProjectileNumber].a * PosData.lattice.a,PosData[ProjectileNumber].b * PosData.lattice.b,PosData[ProjectileNumber].c * PosData.lattice.c] #position of the projectile
            #ProjectilePos = PosData[ProjectileNumber]
            #print(ProjectilePos)
            #clean Projectile position
            '''if ProjectilePos[0] <= 0:
                ProjectilePos[0] = ProjectilePos[0] + PosData.lattice.a
            if ProjectilePos[1] <= 0:
                ProjectilePos[1] = ProjectilePos[1] + PosData.lattice.b
            if ProjectilePos[2] <= 0:
                ProjectilePos[2] = ProjectilePos[2] + PosData.lattice.c'''
        
            #find center of mass of target molecule
            #first clean positions so all have all positive
            #cleaning may not be necessary, leave commented out for now
            CleanedTargetPos = []
            TargetMasses = []
            TargetZ = []
            for i in range(0,len(PosData)):
                if i == ProjectileNumber: # do not include projectile
                    continue
                posHold = [PosData[i].a*PosData.lattice.a, PosData[i].b*PosData.lattice.b, PosData[i].c*PosData.lattice.c]
                TargetMasses.append(Element(PosData.species[i]).atomic_mass)
                TargetZ.append(Element(PosData.species[i]).Z)
                '''if posHold[0] < 0:
                    posHold[0] = posHold[0] + PosData.lattice.a
                if posHold[1] < 0:
                    posHold[1] = posHold[1] + PosData.lattice.b
                if posHold[2] < 0:
                    posHold[2] = posHold[2] + PosData.lattice.c'''
                CleanedTargetPos.append(posHold)
            #print(CleanedTargetPos)
            #print(TargetMasses)
            #get center of mass of target
            TargetCOM = np.average(CleanedTargetPos,axis=0,weights=TargetMasses)
            #print(TargetCOM)
        
            #Get distance between target CoM and Projectile
            #TargetCOM = [1,1,1] #confirmed this works if CoM is not at origin
            ProjTargetDist = math.dist(ProjectilePos,TargetCOM)
            #print(ProjectilePos)
            #print(TargetCOM)
            #print(ProjTargetDist)
            ProjTargetDisplacement = [b-a for a,b in zip(ProjectilePos,TargetCOM)] #points from projectile to COM
            #print("Projectile-Target displacement",ProjTargetDisplacement)
            #print("Projectile-Target distance",ProjTargetDist)
            def unit_vector(vector):
                """ Returns the unit vector of the vector.  """
                return vector / np.linalg.norm(vector)
            #print("ProjectileVelocity",ProjectileVelocity)
            VelocityRelAngle = np.arccos(np.clip(np.dot(unit_vector(ProjTargetDisplacement), unit_vector(ProjectileVelocity)), -1.0, 1.0))
            VelocityRelAngle = np.degrees(VelocityRelAngle)
            #print("Angle between velocity and displacement",VelocityRelAngle)
        
            #get atomic number moment
            Moment = [0,0,0]
            #TargetCOM = [1,1,1] #confirmed this works if CoM is not at origin
            for j in range(0,len(CleanedTargetPos)):
                RelativeTargetPos = [a-b for a,b in zip(CleanedTargetPos[j],TargetCOM)]
                Moment = [a + TargetZ[j]*b for a,b in zip(Moment,RelativeTargetPos)]
                #print("Added Target",CleanedTargetPos[j], RelativeTargetPos)
                #print("Moment",Moment)
            MomentMag = np.linalg.norm(Moment)
            #print("Moment",Moment)
            #print("Moment Magnitude",MomentMag)
            if MomentMag == 0:
                MomentRelAngle= 0.0
            else:
                MomentRelAngle = np.arccos(np.clip(np.dot(unit_vector(ProjTargetDisplacement), unit_vector(Moment)), -1.0, 1.0))
                MomentRelAngle = np.degrees(MomentRelAngle)
            #print("MomentRelAngle",MomentRelAngle)
        
            #Add data to DataRow
            #print(len(OrientationColumns))
            DataRow = [DataRun+"-iter-1.Position",ProjectileSpeed,ProjTargetDist,VelocityRelAngle,MomentMag,MomentRelAngle]
            #print(len(DataRow))
            OrientDataFrame.loc[len(OrientDataFrame.index)] = DataRow
        
        OrientDataFrame.to_csv(DataFolderInit+OrientationDescriptorsFile, index=False)

















