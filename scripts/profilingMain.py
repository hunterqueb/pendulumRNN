from qutils import qProfile

# folderLocation = 'test/'
folderLocation = './scripts/mamba/current/'
scriptName = "mambaCR3BP6d.py"
scriptName = "mambaGMATTest.py"

qProfile.runMemoryProfiling(folderLocation+scriptName)
qProfile.runPerformanceProfiling(folderLocation+scriptName,'profiled'+scriptName)