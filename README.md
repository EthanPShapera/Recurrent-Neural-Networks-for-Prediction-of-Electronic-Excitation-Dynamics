Summary of .py files:

1. ActiveLearning-SKLearnNN-MultipleTargets-ElecDiff-withSpearmanCutoff-refit.py - Main code, performs active learning and RNN fitting.
2. CollectDFTData-qball.py - Collects structural data and orbital occupations from QB@ll output files
3. CollectTDDFTEnergies-qball -  Collects TDDFT energies from QB@ll
4. MakeCombinedDescriptors-CombinetheSeparated.py - Provides postprocessing of the data file from MakeCombinedDescriptors-qball-fullySeparated.py.
5. MakeCombinedDescriptors-qball-fullySeparated.py - Combines data files containing crystal graph singular value and Coulomb matrix representations into a single data file.
6. MakeCombinedDescriptors-qball-initial-fullySeparated.py - Combines together descriptor files if only initial occupations are known.
7. MakeCoulombMatrixAndCrystalRep-fullySeparated.py - Constructs the Coulomb matrix representation for each structure.
8. MakeCrystalGraphRep-TDDFT-fullySeparated.py - Constructs the crystal graph singular value representations for each structure.
9. MakeOrientationDescriptors-fullySeparated.py - Constructs descriptors describing orientation of the target material.
