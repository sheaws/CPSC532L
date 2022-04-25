import argparse
import glob
import sys
import os
import collections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

# settings and constants
validAssets = ["MB", "BC"]
asset="BC"
inverseTS=1
verbose = False 
doPlots = False
myEpsilon = 0.000001
myLargeValue = 999999999
isUnit = False # fixes the random seed if this is True
noLearnPercent = 0.0 # this is the percentage of players that do not run a no-regret strategy
csvFileName = "multiAgentMW.csv"
sellSpread = -1.0
buySpread = -1.0

###### define agents #####
# uncomment this section to hand specify different agents
#edgeSets = [[0.25,0.5,0.75,1.0],[0.25,0.5,0.75,1.0],[0.25,0.5,0.75,1.0]]
#maxInventorys = [10,10,10] # must be at least quoteSize
#quoteSizes = [1,1,1]
#learnRates = [0.001,0.001,0.001]
#nAgents = len(edgeSets)   
###

# uncomment this section to use n identical agents
edgeSets = []
maxInventorys = []
quoteSizes = []
learnRates = []
noLearn = []
inGame = []
nAgents = 5
defaultLearnRate = -1.0
defaultMaxInv = 20
defaultQS = 1
#defaultEdgeSet = [5,10,15,20,25,30]
defaultEdgeSet = [5]
defaultExitGameThreshold = -5000.0 # the amount to lose before deciding to quit game #TODO: command line override
###

#########################

# returns only the updates that are trade updates for the symbol given within cash equity hours
def getLastTradesBySym(sym):
    isFirstFile = True
    dateStr = ""
    numRec=0
    nBuckets = 0
    for filename in sorted(glob.iglob("./data/live.*.csv")):
        currDate = filename[12:20] 
        print("Looking at file for "+currDate)
        
        recData = pd.read_csv(filename)
        # valid means: price feed initalized so that last and close prices are populated
        # time between 6:30am and 1pm PST (i.e. cash markets are open)
        #  update type is 4 (last) or 5 (lastS)
        if(isFirstFile): 
            dateStr = currDate
            currSamples = recData[(recData["Sym"]==sym) & (recData["Last"]!=-1.0) & (recData["Close"]!=0.0)\
                & (recData["Time"].str.match(pat="([0][6]:[3-5][0-9]|[0][7-9]:[0-5][0-9]|[1][0-2]:[0-5][0-9]|[1][3]:[0][0-1])"))\
                & ((recData["Type"]==5) | (recData["Type"]==4))]    
        else:
            dateStr=dateStr+","+currDate
            newDf=recData[(recData["Sym"]==sym) & (recData["Last"]!=-1.0) & (recData["Close"]!=0.0)\
                & (recData["Time"].str.match(pat="([0][6]:[3-5][0-9]|[0][7-9]:[0-5][0-9]|[1][0-2]:[0-5][0-9]|[1][3]:[0][0-1])"))\
                & ((recData["Type"]==5) | (recData["Type"]==4))]
            numRec+=newDf.shape[0]
            currSamples = currSamples.append(newDf,ignore_index=True)
        isFirstFile = False
        nBuckets +=  1
    
    # hardcoded to match the time filtering above and the 1 minute bucketing in the main algo below
    nBuckets = nBuckets*390 # 390 minutes per North American cash equity trading day
    
    return dateStr,currSamples,nBuckets
        
def parseCmdLine():
    global nAgents, defaultLearnRate, doPlots, isUnit, asset, inverseTS, defaultMaxInv, defaultQS, noLearnPercent, sellSpread, buySpread
    parser = argparse.ArgumentParser(description='Runs a simulated multiagent, full bandit game where agents learn using MW')
    parser.add_argument("-n", "--nAgents", nargs=1, help="number of identical players/ agents")
    parser.add_argument("-l", "--learnRate", nargs=1, help="learning rate, defaults to "+str(defaultLearnRate))
    parser.add_argument("-p", "--plot", action="store_true", help="plot game details")
    parser.add_argument("-u", "--unit", action="store_true", help="is a unit test and sets random seed to 0")
    parser.add_argument("-a", "--asset", nargs=1, help="asset, defaults to "+asset)
    parser.add_argument("-t", "--inverseTS", nargs=1, help="inverse tick size. Needs to match asset. Defaults to "+str(inverseTS))
    parser.add_argument("-i", "--maxInventory", nargs=1, help="maxInventory, defaults to "+str(defaultMaxInv))
    parser.add_argument("-q", "--quoteSize", nargs=1, help="quote size, defaults to "+str(defaultQS))
    parser.add_argument("-r", "--noLearnPercent", nargs=1, help="percentage of players that do not learn between 0.0 and 1.0. Defaults to 0.0")
    parser.add_argument("-s", "--sellSpread", nargs=1, help="fuxes PS to sellSpread, defaults to variable.")
    parser.add_argument("-b", "--buySpread", nargs=1, help="fuxes PB to buySpread, defaults to variable.")
    
    args = parser.parse_args()

    if(args.nAgents):
        nAgents = int(args.nAgents[0])
        print("Setting the number of agents to "+str(nAgents))
    if(args.learnRate):
        defaultLearnRate = float(args.learnRate[0])
        print("Setting the learning rate to "+str(defaultLearnRate))
    if(args.plot):
        doPlots = True
        print("Turning on charting.")
    if(args.unit):
        isUnit = True
        np.random.seed(0)
        print("Running with fixed random seed of 0 for unit test.")
    if(args.asset):
        if(args.asset[0] in validAssets):
            asset = args.asset[0]
            print("Setting asset to "+asset)
    if(args.inverseTS):
        inverseTS = int(args.inverseTS[0])
        print("Setting inverse tick size to "+args.inverseTS[0])
    if(args.maxInventory):
        defaultMaxInv = int(args.maxInventory[0])
        print("Setting max inventory to "+args.maxInventory[0])
    if(args.quoteSize):
        defaultQS = int(args.quoteSize[0])
        print("Setting quote size to "+args.quoteSize[0])
    if(args.noLearnPercent):
        newVal = float(args.noLearnPercent[0])
        if(newVal>=0.0 and newVal<=1.0):
            noLearnPercent = newVal
            print("Setting percentage of players that do not learn to "+str(noLearnPercent))
    if(args.sellSpread):
        newVal = float(args.sellSpread[0])
        if(newVal>0.0):
            sellSpread = newVal
            print("Setting sellSpread to "+str(sellSpread))
    if(args.buySpread):
        newVal = float(args.buySpread[0])
        if(newVal>0.0):
            buySpread = newVal
            print("Setting buySpread to "+str(buySpread))

# prints out a summary of the game after the simulation finishes
def printAgentSummary(header,agentRewards,agentEdge,inventoryOverGame,inventoryT):
    nNegPL = sum(1 for r in agentRewards if r<0.0)
    print(header)
    print("  Total reward at time T per agent: average={:.0f}, sd={:.0f}".format(\
          np.average(agentRewards),np.std(agentRewards)))
    print("  Decision weight average at time T per agent: average={:.2f}, sd={:.2f}".format(\
          np.average(agentEdge),np.std(agentEdge)))
    print("  Average inventory size, i.e. abs(inventory), during game per agent: average={:.2f}, sd={:.2f}".format(\
          np.average(inventoryOverGame),np.std(inventoryOverGame)))
    print("  Average inventory size at time T per agent: average={:.2f}, sd={:.2f}".format(\
          np.average(inventoryT),np.std(inventoryT)))
    print("  {:d} agents with negative reward at time T ({:.1%})".format(nNegPL,(nNegPL/len(agentRewards))))

# main
# create agents
parseCmdLine()
noLearnSplit=int(nAgents*(1.0-noLearnPercent))
for a in range(nAgents):
    edgeSets.append(defaultEdgeSet) #TODO: add command line option
    maxInventorys.append(defaultMaxInv)
    quoteSizes.append(defaultQS)
    learnRates.append(defaultLearnRate)
    if(a>=noLearnSplit):
        noLearn.append(True)
    else:
        noLearn.append(False)
    inGame.append(True)

dateStr,symData,nBuckets = getLastTradesBySym(asset)
startDateStr = dateStr.split(',')[0]
endDateStr = dateStr.split(',')[-1]
numTicks = symData.shape[0]

# if learn rate was not pinned, set learn rates to the optimal for Exp3 algo
if(defaultLearnRate<0.0):
    for a in range(nAgents):
        learnRates[a] = np.sqrt(np.log(len(edgeSets[a]))/nBuckets)  

# sanity check on settings
for a in range(nAgents):
    if(quoteSizes[a]>maxInventorys[a]):
        print("ERROR: Player {:d} has quote size {:d} larger than max inventory {:d}. Exiting.".format(\
              a,quoteSizes[a],maxInventorys[a]))
        sys.exit(0)

print("Running game on {:s} data. Number of agents = {:d}. Ticks={:d}".format(\
      asset,nAgents,numTicks))

prevTime = symData.iloc[0]["Time"][0:5] # to keep track of time buckets
QBlimit = 0.0
QSlimit = 99999999999.0
QBsize = QSsize = nTrades = 0 # market demand sizes and number of last updates in data

# stuff to keep a history of for charts. len of these lists gives number of iterations
plotP = []
plotQBsize = []
plotQSsize = []
plotBSpread = []
plotSSpread = []
plotNInGame = []
timeOfDay = [] # x axis

# demand trading history
QBsizeFilled = []
QSsizeFilled = []
BpricePaid = []
SpricePaid = []
noBuyFills = noSellFills = 0

# agent trading history
decisions = {} # [agent] = []. arm number
buyFills = {} # [agent] = []
sellFills = {} # [agent] = []
inventorys = {} # [agent] = []
tradePl = {} # [agent] = []. not intuitively tradePl because matched inventory pl goes here.
posPl = {} # [agent] = []
rewards = {} # [agent] = []
histQuotes = {} # [agent] = [[pb,ps,qb,qs]] 

# agent bookkeeping variables
weightsList = {} # [agent] = [float]
probsList = {} # [agent] = [float]
cashFlow = {} # [agent] = float
exitGameThresh = {} # [agent] = float

# initialize agents
for a in range(nAgents):
    d = len(edgeSets[a])
    decisions[a] = []
    buyFills[a] = []
    sellFills[a] = []
    inventorys[a] = []
    inventorys[a].append(0)
    tradePl[a] = [] # per period netted qty. need history for total pl
    posPl[a] = [] # curr pos marked to mkt.
    rewards[a] = []
    histQuotes[a] = []
    weightsList[a] = [1.0]*d
    probsList[a] = [1.0/d]*d
    cashFlow[a] = 0.0    
    exitGameThresh[a] = defaultExitGameThreshold

# run simulation
for a in range(numTicks):
    # aggregate data a la signed volume
    currData = symData.iloc[a]
    currTime = currData["Time"][0:5]
    if(currTime==prevTime):
        if(currData["Type"]==5):
            #print(str(currData)+"\n*************")
            nTrades+=1
            lastS=currData["LastS"]
            last=currData["Last"]
            if(last>=currData["Ask"]):
                QBsize+=lastS
                QBlimit=max(QBlimit,last)
            elif(last<=currData["Bid"]):
                QSsize+=lastS
                QSlimit=min(QSlimit,last)
    else:
        if(nTrades>0 and QBlimit>1.0 and QSlimit<99999.0): # there were both buys and sells in one time bucket
            P = float(Fraction(int(round((QBlimit +QSlimit)/2.0*inverseTS)),inverseTS)) # round to the nearest tick
            timeOfDay.append(prevTime)
            plotP.append(P)
            plotNInGame.append(sum(int(i) for i in inGame))
            plotQBsize.append(QBsize)
            plotQSsize.append(QSsize)
            currSpread = 0.0
            if(buySpread<0.0):
                currSpread += QBlimit-P
                plotBSpread.append((QBlimit-P))
            else:
                currSpread += buySpread
                plotBSpread.append(buySpread)
            if(sellSpread<0.0):
                currSpread += P-QSlimit
                plotSSpread.append((P-QSlimit))
            else:
                currSpread += sellSpread
                plotSSpread.append(sellSpread)
            if(verbose):
                print("{:d}: time={:s} QB=({:.2f},{:d}), QS=({:.2f},{:d}), P={:.2f}, spread={:.2f}, nTrades={:d}".format(\
                  len(timeOfDay),prevTime,QBlimit,QBsize,QSlimit,QSsize,P,currSpread,nTrades))
            
            # run MW for each agent
            strategyProfile=[] # [pb, ps, qb, qs]
            buyPrices = []
            buyersAtPricePoint = []
            sellPrices = []
            for b in range(nAgents):
                if(inGame[b]):
                    # update probabilities
                    denom=sum(weightsList[b])
                    numArms = len(edgeSets[b])
                    for c in range(numArms):
                        probsList[b][c] = (1.0-learnRates[b])*weightsList[b][c]/denom+learnRates[b]/numArms
                    
                    # sample from distribution and make arm choice
                    arms=list(range(0,numArms))
                    currEdgeIdx=np.random.choice(arms,1,p=probsList[b])[0] 
                    currEdge=edgeSets[b][currEdgeIdx]    
                    decisions[b].append(currEdgeIdx)
                    
                    # calculate x=(pb,ps,qb,qs)
                    pb = P-currEdge
                    ps = P+currEdge
                    qb=min(quoteSizes[b],maxInventorys[b]-inventorys[b][-1])
                    qs=min(quoteSizes[b],maxInventorys[b]+inventorys[b][-1])    
                    strategyProfile.append([pb,ps,qb,qs])
                    histQuotes[b].append([pb,ps,qb,qs])
                    if(pb not in buyPrices):
                        buyPrices.append(pb)
                    if(ps not in sellPrices):
                        sellPrices.append(ps)
                else:
                    # pick the largest value in action set
                    currEdgeIdx=len(edgeSets[b])-1
                    currEdge=edgeSets[b][currEdgeIdx]
                    decisions[b].append(currEdgeIdx)
                    strategyProfile.append([P-currEdge,P+currEdge,0,0])
                    histQuotes[b].append([P-currEdge,P+currEdge,0,0])
                
            # matching/ auction
            for b in range(nAgents):
                buyFills[b].append(0)
                sellFills[b].append(0)            

            # agents buy
            sizeRemaining = QSsize
            averageSellPrice = 0.0
            willingnessToPay = QSlimit
            if(sellSpread>0.0):
                willingnessToPay = P-sellSpread
            for bid in reversed(sorted(buyPrices)):
                if(bid >= willingnessToPay and sizeRemaining>0):
                    # build book at price point. note: agents only submit 1 bid and 1 ask
                    totalQtyAtPricePoint = 0
                    agentsAtPricePoint = []
                    for b in range(nAgents):
                        if(inGame[b] and strategyProfile[b][0]==bid):
                            totalQtyAtPricePoint += strategyProfile[b][2]
                            agentsAtPricePoint.append(b)
                    
                    # allocations
                    if(totalQtyAtPricePoint<=sizeRemaining): # sum(agentSize) <= demand -> fill fully
                        for b in agentsAtPricePoint:
                            buyFills[b][-1]=strategyProfile[b][2]
                            sizeRemaining-=buyFills[b][-1]
                            averageSellPrice += buyFills[b][-1]*(P-bid)
                    else: # not enough demand to fill all agents
                        sumQty=0
                        # pro-rata
                        for b in agentsAtPricePoint:
                            qty = int(strategyProfile[b][2]/totalQtyAtPricePoint*sizeRemaining) # round down
                            buyFills[b][-1]=qty
                            averageSellPrice += qty*(P-bid)
                            sumQty+=qty
                        sizeRemaining -= sumQty
                        # clean up after rounding: allocate sizeRemaining to random agents at price point
                        while(sizeRemaining>0):
                            luckyDude=np.random.choice(agentsAtPricePoint,1,\
                                p=[1/len(agentsAtPricePoint)]*len(agentsAtPricePoint))[0]
                            sizeAdj = min(sizeRemaining,strategyProfile[luckyDude][2]-buyFills[luckyDude][-1])
                            buyFills[luckyDude][-1]+=sizeAdj
                            averageSellPrice += sizeAdj*(P-bid)
                            sizeRemaining-=sizeAdj
                            agentsAtPricePoint.remove(luckyDude)
                else: # rest of bids too low or demand filled
                    break
            qtySold=QSsize-sizeRemaining
            QSsizeFilled.append(qtySold/QSsize)
            if(qtySold >0):
                SpricePaid.append(averageSellPrice/qtySold)
            else:
                noSellFills += 1
                # don't append anything - basically no data
            
            # agents sell - mirrors agents buy
            sizeRemaining = QBsize
            averageBuyPrice = 0.0
            willingnessToPay = QBlimit
            if(buySpread>0.0):
                willingnessToPay = P+buySpread
            for ask in sorted(sellPrices):
                if(ask <= willingnessToPay and sizeRemaining>0):
                    totalQtyAtPricePoint = 0
                    agentsAtPricePoint = []
                    for b in range(nAgents):
                        if(inGame[b] and strategyProfile[b][1]==ask):
                            totalQtyAtPricePoint += strategyProfile[b][3]
                            agentsAtPricePoint.append(b)
                    
                    if(totalQtyAtPricePoint<=sizeRemaining):
                        for b in agentsAtPricePoint:
                            sellFills[b][-1]=strategyProfile[b][3]
                            sizeRemaining-=sellFills[b][-1]
                            averageBuyPrice += sellFills[b][-1]*(ask-P)
                    else: 
                        sumQty=0
                        for b in agentsAtPricePoint:
                            qty = int(strategyProfile[b][3]/totalQtyAtPricePoint*sizeRemaining) # round down
                            sellFills[b][-1]=qty
                            averageBuyPrice += qty*(ask-P)
                            sumQty+=qty
                        sizeRemaining -= sumQty
                        while(sizeRemaining>0):
                            luckyDude=np.random.choice(agentsAtPricePoint,1,\
                                p=[1/len(agentsAtPricePoint)]*len(agentsAtPricePoint))[0]
                            sizeAdj = min(sizeRemaining,strategyProfile[luckyDude][3]-sellFills[luckyDude][-1])
                            sellFills[luckyDude][-1]+=sizeAdj
                            averageBuyPrice += sizeAdj*(ask-P)
                            sizeRemaining-=sizeAdj
                            agentsAtPricePoint.remove(luckyDude)  
                else:
                    break
            qtyBought=QBsize-sizeRemaining
            QBsizeFilled.append(qtyBought/QBsize)  
            if(qtyBought>0):
                BpricePaid.append(averageBuyPrice/qtyBought)
            else:
                noBuyFills += 1
                
            # sanity check
            failedCheck = False
            for b in range(nAgents):
                if(any(qty<0 for qty in buyFills[b])):
                    print("Negative quantity in buyFills for "+str(b))
                    print(buyFills[b])
                    failedCheck=True
            
            for b in range(nAgents):
                if(any(qty<0 for qty in sellFills[b])):
                    print("Negative quantity in sellFills for "+str(b))
                    print(sellFills[b])
                    failedCheck=True            
            
            # should check that fill prices are not outside MM bid/ask spreads or market limits
            
            if(failedCheck):
                sys.exit(0)
                
            # strategyProfile has not been changed so it could be stored for charting
            # agents receives fills, calculates loss, i.e. -reward, updates weights
            for b in range(nAgents):
                if(inGame[b]):
                    nettedQty=min(buyFills[b][-1],sellFills[b][-1])
                    excessQty=buyFills[b][-1]-sellFills[b][-1]
                    spread = strategyProfile[b][1]-strategyProfile[b][0]
                    
                    # update tradePl, posPl, cash balance and inventory
                    tradePl[b].append(spread*nettedQty)
                    posPl[b].append(0.0)
                    try:
                        newInv = inventorys[b][-1]
                    except IndexError:
                        newInv = 0
                    if((newInv<0 and excessQty>0) or (newInv>0 and excessQty<0)): # risk reducing net fills
                        absInv = abs(newInv)
                        absEx = abs(excessQty)
                        if(absInv>absEx): # partially offset inventory
                            adj = cashFlow[b]*absEx/absInv
                            cashFlow[b] -= adj
                            if(newInv>0): # reduce long position
                                tradePl[b][-1] += adj + absEx*strategyProfile[b][1]
                            else: # reduce short position
                                tradePl[b][-1] += adj - absEx*strategyProfile[b][0]
                            newInv += excessQty
                            excessQty = 0
                        elif(absInv<absEx): # offsets inventory and more
                            adj = absEx-absInv
                            if(newInv > 0): # sold and went short
                                tradePl[b][-1] += cashFlow[b] + absInv*strategyProfile[b][1]
                                cashFlow[b] = adj*strategyProfile[b][1]
                                newInv = excessQty = -adj
                            else: # covered and went long
                                tradePl[b][-1] += cashFlow[b] - absInv*strategyProfile[b][0]
                                cashFlow[b] = -adj*strategyProfile[b][0]
                                newInv = excessQty = adj
                        else: # went flat
                            if(newInv > 0): # closed long
                                tradePl[b][-1] += cashFlow[b] + absEx*strategyProfile[b][1]
                            else: # closed short
                                tradePl[b][-1] += cashFlow[b] - absEx*strategyProfile[b][0]
                            cashFlow[b] = 0.0
                            newInv = excessQty = 0.0
                    elif(excessQty<0 or excessQty>0): # risk increasing net fills
                        newInv += excessQty
                        if(excessQty<0):
                            cashFlow[b] -= excessQty*strategyProfile[b][1]
                        else:
                            cashFlow[b] -= excessQty*strategyProfile[b][0]
                                
                    inventorys[b].append(int(newInv))
                    if(newInv>0):
                        posPl[b][-1]=newInv*(max(buyPrices)-abs(cashFlow[b]/newInv))
                    elif(newInv<0):
                        posPl[b][-1]=-newInv*(abs(cashFlow[b]/newInv)-min(sellPrices))
                else:
                    tradePl[b].append(0.0)
                    posPl[b].append(0.0)
                    inventorys[b].append(0)
                # calculate reward
                rewards[b].append(sum(tradePl[b])+posPl[b][-1])
                
                currRet = 0.0
                currMult = 1.0
                # Exp3 algorithm requires reward in [0,1] but I could only think of a way for reward to be in [-1,1]
                if(inGame[b] and not noLearn[b]):
                    # only weight to change is for the currently used arm
                    if(rewards[b][-1]<exitGameThresh[b]): 
                        tradePl[b][-1]+=posPl[b][-1] #liquidate position
                        posPl[b][-1]=0.0
                        cashFlow[b] = 0.0
                        inGame[b] = False
                    else:
                        #if(len(rewards[b])>1 and rewards[b][-2]!=0.0):
                        #    currRet = (rewards[b][-1]-rewards[b][-2])/rewards[b][-2] 
                        avgTradePl = np.average(tradePl[b])
                        if(avgTradePl>0):
                            currRet=tradePl[b][-1]/np.average(tradePl[b])
                        # Exp3 requires this to be in [0,1]. 
                        # So essentially losses are treated as just breakeven and large returns are discounted
                        # Otherwise, exp blows up
                        currRet = max(min(1.0, currRet),0.0)
                        currArm = decisions[b][-1]
                        currMult = np.exp(learnRates[b]*(currRet/probsList[b][currArm])/len(edgeSets[b]))                        
                        weightsList[b][currArm]=weightsList[b][currArm]*currMult
                    #remCap[b] = rewards[b][-1]
                       
                # show per iteration per agent data
                if(verbose):
                    print("  [{:d}]: arm={:d}, currRet={:.2f}, currMult={:.2f}, weights: [".format(\
                          b,decisions[b][-1],currRet,currMult),end="")
                    for w in weightsList[b]:
                        print("{:<8.0f}".format(w),end="")
                    print("], probs: [", end="")
                    for p in probsList[b]:
                        print("{:<8.3f}".format(p),end="")
                    print("]", end="")
                    print(" Inventory: {:d}. Reward: {:<10.1f}.".format(inventorys[b][-1],rewards[b][-1]))    
                    
        #reset
        QBlimit = 0.0
        QSlimit = 99999999999.0
        QBsize = QSsize = nTrades = 0
        if(currData["Type"]==5):
            nTrades+=1
            lastS=currData["LastS"]
            last=currData["Last"]
            if(last>=currData["Ask"]):
                QBsize=lastS
                QBlimit=last
            elif(last<=currData["Bid"]):
                QSsize=lastS
                QSlimit=last
        prevTime=currTime

# end of data reached    
print("Reached end of data, T={:d}.".format(len(timeOfDay)))
if(verbose):
    print("Agent summaries:")
nSurvivors = sum(int(i) for i in inGame)
agentRewards = []
agentEdge = []
inventoryOverGame = []
inventoryT = []
agentTotalBidQty = []
agentTotalAskQty = []
for a in range(nAgents):
    agentRewards.append(rewards[a][-1])
    if(inGame[a]):
        agentEdge.append(sum([e*w for e,w in zip(edgeSets[a],probsList[a])]))
        inventoryT.append(np.abs(inventorys[a][-1]))
    agentTotalBidQty.append(sum([row[2] for row in histQuotes[a]])) # this should be adjusted for non-survivers too but
    agentTotalAskQty.append(sum([row[3] for row in histQuotes[a]])) #  actually, this metric is not what i really want measured
    inventoryOverGame.append(np.average([abs(inv) for inv in inventorys[a]]))
    if(verbose):
        print("  [{:d}]: final inventory={:d} and reward={:0.1f}. Weights={:s}. probs={:s}.".format(\
              a,inventorys[a][-1],rewards[a][-1],str(['%.0f' % w for w in weightsList[a]]),\
              str(['%.2f' % p for p in probsList[a]])))
if(nSurvivors==0):
    agentEdge.append(0)
    inventoryT.append(0)

if(verbose):
    print("*****")
if(noLearnPercent>0.0 and noLearnPercent<1.0):
    printAgentSummary("Summary for the {:d} agents that ran no-regret strategies:".format(noLearnSplit),\
                     agentRewards[:noLearnSplit],agentEdge[:noLearnSplit],inventoryOverGame[:noLearnSplit],\
                     inventoryT[:noLearnSplit])

    printAgentSummary("Summary for the {:d} agents that ran random strategies:".format(nAgents-noLearnSplit),\
                     agentRewards[noLearnSplit:],agentEdge[noLearnSplit:],inventoryOverGame[noLearnSplit:],\
                     inventoryT[noLearnSplit:])
else:
    if(noLearnPercent<=0.0):
        printAgentSummary("Summary for the {:d} agents that ran no-regret strategies:".format(nAgents),\
                          agentRewards,agentEdge,inventoryOverGame,inventoryT)
    else: #noLearnPercent==1.0
        printAgentSummary("Summary for the {:d} agents that ran random strategies:".format(nAgents),
                          agentRewards,agentEdge,inventoryOverGame,inventoryT)

print("Average demand percentage filled: buy={:.2f}, sell={:.2f}".format(\
          np.average(QBsizeFilled),np.average(QSsizeFilled)))
print("Average demand price paid: buy={:.4f}, sell={:.4f}".format(\
      np.average(BpricePaid),np.average(SpricePaid)))
print("Number of periods with no fills: buy={:d}, sell={:d} out of {:d} time periods".format(\
      noBuyFills, noSellFills,len(timeOfDay)))

# write CSV
if(os.path.isfile(csvFileName)):
    outfile = open(csvFileName,'a')
else:
    outfile = open(csvFileName,'w')
    outfile.write("nAgents,isUnit,asset,noLearnPct") # game setting
    outfile.write(",data,nTime,open,high,low,close") # misc
    outfile.write(",avgQtyBuy,stdQtyBuy,avgQtySell,stdQtySell") # market conditions
    outfile.write(",avgBuySpread,stdBuySpread,avgSellSpread,stdSellSpread") # market conditions
    outfile.write(",avgLearnRate,stdLearnRate,avgMaxInv,stdMaxInv,avgQuoteSize,stdQuoteSize") # agent distribution
    outfile.write(",actionSets") # agent distribution
    outfile.write(",avgBidQty,stdBidQty,avgAskQty,stdAskQty,avgAction,stdAction") # competitiveness metrics
    outfile.write(",avgBuyPrcPaid,stdBuyPrcPaid,avgSellPrcPaid,stdSellPrcPaid") # competitiveness metrics
    outfile.write(",avgBidPctFilled,stdBidPctFilled,avgAskPctFilled,stdAskPctFilled,nNoBuys,nNoSells") # liquidity metrics
    outfile.write(",avgReward,stdReward,nNegReward,nInGame,avgInvOverGame,stdInvOverGame,avgInvT,stdInvT") # sustainability metrics
    outfile.write("\n")

outfile.write("{:d},{:d},{:s},{:f}".format(nAgents,int(isUnit),asset,noLearnPercent))
outfile.write(",{:s} to {:s},{:d},{:f},{:f},{:f},{:f}".format(\
    startDateStr,endDateStr,len(timeOfDay),plotP[0],max(plotP),min(plotP),plotP[-1]))
outfile.write(",{:.0f},{:.0f},{:.0f},{:.0f}".format(\
    np.average(plotQBsize),np.std(plotQBsize),np.average(plotQSsize),np.std(plotQSsize)))
outfile.write(",{:.2f},{:.2f},{:.2f},{:.2f}".format(\
    np.average(plotBSpread),np.std(plotBSpread),np.average(plotSSpread),np.std(plotSSpread)))
outfile.write(",{:f},{:f},{:f},{:f},{:f},{:f}".format(\
    np.average(learnRates),np.std(learnRates),np.average(maxInventorys),np.std(maxInventorys),\
    np.average(quoteSizes),np.std(quoteSizes)))
outfile.write(",{:s} to {:s}".format(str(defaultEdgeSet[0]),str(defaultEdgeSet[-1])))
#outfile.write(",{:s}".format("1.0 to 8.0"))
outfile.write(",{:.0f},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f}".format(\
    np.average(agentTotalBidQty),np.std(agentTotalBidQty),
    np.average(agentTotalAskQty),np.std(agentTotalAskQty),
    np.average(agentEdge),np.std(agentEdge)))
outfile.write(",{:.2f},{:.2f},{:.2f},{:.2f}".format(\
    np.average(BpricePaid),np.std(BpricePaid),
    np.average(SpricePaid),np.std(SpricePaid)))
outfile.write(",{:.2f},{:.2f},{:.2f},{:.2f},{:d},{:d}".format(\
    np.average(QBsizeFilled),np.std(QBsizeFilled),
    np.average(QSsizeFilled),np.std(QSsizeFilled),noBuyFills,noSellFills))
outfile.write(",{:.0f},{:.0f},{:d},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}".format(\
    np.average(agentRewards),np.std(agentRewards),sum(1 for r in agentRewards if r<0.0),\
    nSurvivors,np.average(inventoryOverGame),np.std(inventoryOverGame),\
    np.average(inventoryT),np.std(inventoryT)))
outfile.write("\n")
    
outfile.close()

if(not doPlots):
    sys.exit(0)

# plot results
fig, axs = plt.subplots(6,1,constrained_layout=False,figsize=(12,12))
fig.subplots_adjust(wspace=0.05,hspace=0.05,top=0.95,bottom=0.15)
axs[0].plot(plotP, label="P",color='tab:orange')
axs[0].set_ylabel("Asset price")
axs[0].set_ylim(bottom=(min(plotP)-10*(1.0/inverseTS)))
axs0_1 = axs[0].twinx()
axs0_1.set_ylabel("Demand size")
axs0_1.plot(plotQBsize,label="QB",color='tab:blue')
axs0_1.plot(plotQSsize,label="QS",color='tab:red')
axs0_1.set_ylabel("Demand size")
axs[0].set_xticks([])
axs[0].legend()
axs0_1.legend(loc=0)
    
axs[1].plot(plotNInGame,label="nSurvivors")
axs[1].set_ylabel("N Active Agents")
axs[1].set_ylim(bottom=0)
axs[1].set_xticks([])
axs[1].legend()

axs[2].plot(plotBSpread,label="QB^limit-P")
axs[2].plot(plotSSpread,label="P-QS^limit")
axs[2].set_ylabel("Demand spread")
axs[2].set_xticks([])
axs[2].legend()
    
for a in range(nAgents):    
    axs[3].plot(decisions[a],label=str(a+1))
axs[3].set_ylabel("Decision")
axs[3].set_xticks([])
if(nAgents<6):
    axs[3].legend()

for a in range(nAgents):
    axs[4].plot(inventorys[a],label=str(a+1))
axs[4].set_ylabel("Inventory")
axs[4].set_xticks([])
if(nAgents<6):
    axs[4].legend()
        
for a in range(nAgents):
    axs[5].plot(rewards[a],label=str(a+1))
axs[5].set_ylabel("Total Reward")
xloc = np.arange(1,len(timeOfDay),int(len(timeOfDay)/10))
axs[5].set_xticks(xloc)
axs[5].set_xticklabels([timeOfDay[i] for i in xloc])
axs[5].set_xlabel("Time")
if(nAgents<6):
    axs[5].legend()
    
# labelling
suptitleText = "Simulated multi player using EXP3 in bandit environment with "+str(nAgents)+\
    " agents ({:.0%} do not learn)\n".format(noLearnPercent)
captionText = "Asset = "+asset+". Historical data aggregated into 1 minute intervals between 06:30 and 13:00 PST starting on "+\
    startDateStr+" and ending on "+endDateStr+".\n"
captionText = captionText + "P, QB and QS are from data. Original data size="+str(numTicks)+". Aggregated data size="+\
    str(len(plotP))+". Agent settings:\n"
if(nAgents>5):
    for a in range(2):
        captionText = captionText + "  [{:d}]: kmax={:d}, qmax={:d}, learning rate={:f}, decision set={:s}\n".format(\
            a+1,maxInventorys[a],quoteSizes[a],learnRates[a],str(edgeSets[a]))        
    captionText = captionText + ".....\n"
    for a in range(nAgents-2,nAgents):
        captionText = captionText + "  [{:d}]: kmax={:d}, qmax={:d}, learning rate={:f}, decision set={:s}\n".format(\
            a+1,maxInventorys[a],quoteSizes[a],learnRates[a],str(edgeSets[a]))        
else:
    for a in range(nAgents):
        captionText = captionText + "  [{:d}]: kmax={:d}, qmax={:d}, learning rate={:f}, decision set={:s}\n".format(\
            a+1,maxInventorys[a],quoteSizes[a],learnRates[a],str(edgeSets[a]))
    
filenameText = "./charts/"+asset+"."+startDateStr+"."+endDateStr+"."+str(nAgents)+".png"
fig.suptitle(suptitleText)
fig.text(0.5,0.0,captionText, ha='center')
plt.savefig(filenameText)    
