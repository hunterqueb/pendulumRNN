import numpy as np

from qutils.orbital import  readGMATReport

problemDim = 6

gmatImportLEO = readGMATReport("gmat/data/reportLEO5050Prop.txt")
gmatImportLEOBackMamba = readGMATReport("gmat/data/reportLEO5050PropBackMamba.txt")
gmatImportLEOBackLSTM = readGMATReport("gmat/data/reportLEO5050PropBackLSTM.txt")

gmatImportHEO = readGMATReport("gmat/data/report5050Prop.txt")
gmatImportHEOBackMamba = readGMATReport("gmat/data/report5050PropBackMamba.txt")
gmatImportHEOBackLSTM = readGMATReport("gmat/data/report5050PropBackLSTM.txt")


initialConditionTruthLEO = gmatImportLEO[0,0:problemDim]
initialConditionTruthHEO = gmatImportHEO[0,0:problemDim]

backPropConditionMambaLEO = gmatImportLEOBackMamba[-1,0:problemDim]
backPropConditionMambaHEO = gmatImportHEOBackMamba[-1,0:problemDim]

backPropConditionLSTMLEO = gmatImportLEOBackLSTM[-1,0:problemDim]
backPropConditionLSTMHEO = gmatImportHEOBackLSTM[-1,0:problemDim]

def RTCerror(truthInitialConditions,backPropFinalCondition):
    r0Truth = truthInitialConditions[0:3]
    v0Truth = truthInitialConditions[3:6]
    rfNet = backPropFinalCondition[0:3]
    vfNet = backPropFinalCondition[3:6]
    
    norm = np.linalg.norm
    return 0.5*((norm(r0Truth-rfNet)/norm(r0Truth))+(norm(v0Truth-vfNet)/norm(v0Truth)))

JLEOMamba = RTCerror(initialConditionTruthLEO,backPropConditionMambaLEO)
JLEOLSTM = RTCerror(initialConditionTruthLEO,backPropConditionLSTMLEO)

JHEOMamba = RTCerror(initialConditionTruthHEO,backPropConditionMambaHEO)
JHEOLSTM = RTCerror(initialConditionTruthHEO,backPropConditionLSTMHEO)



print("RTC for LSTM in LEO",JLEOLSTM)
print("RTC for Mamba in LEO",JLEOMamba)

print("RTC for LSTM in HEO",JHEOLSTM)
print("RTC for Mamba in HEO",JHEOMamba)
