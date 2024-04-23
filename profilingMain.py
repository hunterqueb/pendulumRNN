from qutils import qProfile

# folderLocation = 'test/'
folderLocation = './'
scriptName = "mambaCR3BP6d"

qProfile.runMemoryProfiling(scriptName)
qProfile.runPerformanceProfiling(folderLocation+scriptName,'profiled'+scriptName)