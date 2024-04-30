
from qutils.orbital import returnCR3BPIC

initialConditions = returnCR3BPIC('longPeriod',L=4,regime='cislunar')

print(initialConditions.x0)