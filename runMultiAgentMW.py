import os
import datetime
import sys
import numpy as np

#learnRates = [[0.0001,'0_0001'],[0.01,'0_01'],[0.1,'0_1'],[np.sqrt(np.log(8)/783),'opt']]
learnRates = [[np.sqrt(np.log(1)/1155),'opt']] # OPT for EXP3 is sqrt(log(nActions)/T)
#nAgents = [10,25,50,100,200]
nAgents = [2, 4, 8]
#nAgents = [4]
#for a in range(20):
#    nAgents.append(a+1)
#for a in [50,100,150,200,500,1000,2000]:
#    nAgents.append(a)
assets = [["BC",1]]
#assets = [["BC",1],["MB",1]]
#quoteSizes = [100,1000]
quoteSizes = [1]
maxInventorys=[5]
noLearnPercent = [0.0]
#noLearnPercent = [0.0,1.0]
#buySellSpread=[5,10,15,20,25,30,"var"] # for now just do symmetric
buySellSpread=["var"] # for now just do symmetric

scriptDir="C:\\classes\\UBC\\CPSC532L\\project\\code\\"
chartsDir=scriptDir+"charts\\"
scriptToRun="multiAgentMW.py"

for asset in assets:
    for qs in quoteSizes:
        for maxInv in maxInventorys:
            sym=asset[0]
            inverseTS=asset[1]
            chartFileNameRoot = sym+".20220307.20220311." # temp hardcoding
            for lrla in learnRates:
                lr=lrla[0]
                la=lrla[1]
                for nlp in noLearnPercent:
                    for spd in buySellSpread:
                        for na in nAgents:
                            # remove --unit if want to use a different random seed every time (--unit sets seed always to 0)
                            cmdStr = "python.exe "+scriptDir+scriptToRun+" --unit --learnRate "+str(lr)+ " --nAgents "+str(na)+\
                                " --asset "+sym+" --inverseTS "+str(inverseTS)+" --quoteSize "+str(qs)+" --maxInventory "+str(maxInv)+\
                                " --noLearnPercent "+str(nlp)
                            if(spd != "var"):    
                                cmdStr = cmdStr+" --buySpread "+str(spd)+" --sellSpread "+str(spd)
                            if(na<100):
                                cmdStr = cmdStr+" --plot"
                            print(datetime.datetime.now().strftime("%H:%M:%S")+": "+cmdStr)
                            os.system(cmdStr) 
                            chartFileName=chartFileNameRoot+str(na)
                            print(chartsDir+chartFileName)
                            if(os.path.isfile(chartsDir+chartFileName+".png")):
                                print("Found "+chartFileName)
                                cmdStr = "MOVE "+chartsDir+chartFileName+".png "+chartsDir+chartFileName+"."+la+\
                                    "."+str(qs)+"."+str(maxInv)+"."+str(spd)+"."+str(spd)+".png"
                                print("  "+cmdStr)
                                os.system(cmdStr)