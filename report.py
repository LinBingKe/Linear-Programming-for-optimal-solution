import numpy as np
import numpy.random as Rnd
import copy
fitnessMax = 1.e3
maxLoop = 1000
noInPool = 1000      # number of chromosome in the beginning
lenGene = 100        # the segment length for each design variable
penalty = 100.
def initPopulation(noInPool,lenGene,noDV):
    '''To create individual string list in (noInPool, noGenes)-shape'''
    #return [''.join(generateChromosome(lenGene,noDV)) for x in range(noInPool)]
    return [generateChromosome(lenGene,noDV) for x in range(noInPool)]
def generateChromosome(lenGene, noDV):
    '''To generate chromosome for all design variable. 
       The length of chromosome is 
       (number of variables)*(length of genes for each variable)'''
    lenChromosome = lenGene * noDV
    chroTemp = Rnd.random(lenChromosome)
    chroTemp[chroTemp > 0.5] = '1'
    chroTemp[chroTemp <= 0.5] = '0'
    chromosome = ''
    chromosome = ''.join(str(chroTemp))
    #print("尚未去除")
    #print(chromosome)
    chromosome = deleteNoUseSymbol(chromosome)
    #print("已去除")
    #print(chromosome)
    return chromosome
def deleteNoUseSymbol(chromosome):
    temp = chromosome.replace('[','')
    temp = temp.replace(']','')
    temp = temp.replace('.','')
    temp = temp.replace(',','')
    temp = temp.replace(' ','')
    temp = temp.replace('\n','')
    return temp

def deCode(chromosome, lenGene, xbound,noDV):
    '''To convert chromosome into design variable value'''
    noInPool = len(chromosome)
    x = np.empty((noInPool, noDV))
    fullSpan = 2**lenGene - 1
    if noDV == 1:
        for i in range(noInPool):
            x[i] = int(chromosome[i],2)/fullSpan * \
                   (xbound[1] - xbound[0]) + xbound[0]
    else:
        for i in range(noInPool):
            x[i][:] = [int(chromosome[i][k*lenGene:(k+1)*lenGene],2)/fullSpan \
                       * (xbound[k][1]-xbound[k][0]) + xbound[k][0] for k in range(noDV)]
    return x       

def mutant(offspring,lenChromosome):
    '''Perform mutation operation'''
    for k in range(len(offspring)):
        index = Rnd.randint(lenChromosome)
        s1 = list(offspring[k])
        s1[index] = '1' if s1[index] == '0' else '0'
        offspring[k] = ''.join(s1)
    return offspring
def crossOver(parents):
    '''Perform crossover opeartion'''
    offspring = [0,0]
    p1 = parents[0]
    p2 = parents[1]
    index1 = Rnd.randint(len(p1))
    index2 = Rnd.randint(len(p1))
    in1 = min(index1,index2)
    in2 = max(index1,index2)
    offspring[0] = p1[0:in1]
    offspring[0] += p2[in1:in2]
    offspring[0] += p1[in2:]
    offspring[1] = p2[0:in1]
    offspring[1] += p1[in1:in2]
    offspring[1] += p2[in2:]
    return offspring
def simpleMating(chromosome,ind):
    '''Perform mating operation according to fitness ranking'''
    parents = [0,0]
    parents[0] = chromosome[ind[-1]]
    parents[1] = chromosome[ind[-2]] 
    return parents
def randomMating(chromosome):
    '''Perform mating operation randomlly'''
    parents = [0,0]
    in1 = Rnd.randint(len(chromosome))
    in2 = Rnd.randint(len(chromosome))
    parents[0] = chromosome[in1]
    parents[1] = chromosome[in2] 
    return parents
def roulettePool(chromosome,f):
    '''The more fitness,the more chance to in pool'''
    parents = [0,0]
    copy_f = f / np.sum(f)
    copy_f = np.add.accumulate(copy_f)
    in1 = np.searchsorted(copy_f, Rnd.rand())
    in2 = np.searchsorted(copy_f, Rnd.rand())
    parents[0] = chromosome[in1]
    parents[1] = chromosome[in2] 
    return parents
def fitness(fct,x):
    return fitnessMax - fct(x)
def fitnessPshenichny(x, noEq, noConstr, penalty, fct, constr):
    """ Pshenichny's function"""
    f1 = fct(x)
    g1 = constr(x)
    maxg1 = np.zeros(len(x))
    ag = np.zeros((len(x),1))
    if noConstr == 1:
        if noEq != 0:
            print("原本的g1")
            print()
            g1[:] = np.abs(g1[:])
            #print("g1絕對值以後")
            #print(g1)

        g1 = g1.reshape((len(x),1))
        print("g1 reshape以後")
        print(g1)
        ag = np.hstack([ag,g1])
        print("ag堆疊以後")
        print(ag)
    else:
        ag = np.hstack([ag,g1])
    maxg1[:] = np.max(ag, axis=-1)

    print ("fitnessMas")
    print (fitnessMax)
    print("f1")
    print(f1)
    print("penalty")
    print(penalty)
    print("maxg1")
    print(maxg1)
    return fitnessMax - (f1 + penalty * maxg1)

def gaMainLoop(fct,constr,noEq,noInEq,noDV,xbound,maxEpoch):
    '''Main loop of genetic algorithm for unconstrained problem'''
    global penalty
    noConstr = noEq + noInEq
    chromosome = initPopulation(noInPool,lenGene,noDV)
    #print("chromosome")
    #print(chromosome)
    lenChromosome = lenGene*noDV
    x = deCode(chromosome,lenGene,xbound,noDV)
    print("解碼數字")
    print(x)
    print("進入適合度")
    f = fitnessPshenichny(x,noEq,noConstr,penalty,fct,constr)
    print("計算完適合度")
    print(f)
    f = np.asarray(f).reshape(-1)
    #ind = np.argsort(f,axis=0)
    ind = np.argsort(f)
    print("合適度排列",ind)
    #ind = ind.reshape((1,noInPool))
    best = {'cost': - np.inf, 'design': ''}
    in1 = ind[-1]
    print("ind[-1]說明",ind[-1])
    best['cost'] = f[in1]
    best['design'] = chromosome[in1]
    ik = 0
    while ik < maxLoop:
        parents1 = []
        parents2 = []
        parents1 = simpleMating(chromosome, ind) 
        #Parents are picked according fitness
        parents2 = roulettePool(chromosome, f)
        #Parents are picked in a way of roulette
        kk = 0
        while kk < maxEpoch:
            '''Pair one'''
            offspring1 = crossOver(parents1)
            if Rnd.random() > 0.9:
                offspring1 = mutant(offspring1,lenChromosome)
            x1 = deCode(offspring1, lenGene, xbound, noDV)
            f1 = fitnessPshenichny(x1,noEq,noConstr,penalty,fct,constr)
            f1 = np.asarray(f1).reshape(-1)
            ind1 = np.argsort(f1)
            if f1[ind1[-1]] > best['cost']:
               best['cost'] = f1[ind1[-1]]
               best['design'] = offspring1[ind1[-1]]
               print("From offspring 1, cost = {}".format(best['cost']))
            parents1 = offspring1
            '''Pair two'''
            offspring2 = crossOver(parents2)
            if Rnd.random() > 0.9:
                offspring2 = mutant(offspring2,lenChromosome)
            x1 = deCode(offspring2, lenGene, xbound, noDV)
            f1 = fitnessPshenichny(x1,noEq,noConstr,penalty,fct,constr)
            f1 = np.asarray(f1).reshape(-1)
            ind1 = np.argsort(f1)
            if f1[ind1[-1]] > best['cost']:
               best['cost'] = f1[ind1[-1]]
               best['design'] = offspring2[ind1[-1]]
               print("From offspring 2, cost = {}".format(best['cost']))
            parents2 = offspring2
            kk += 1
        ik += 1
    x1 = [best['design']]
    xk = deCode(x1, lenGene, xbound, noDV)
    fk = fct(xk)
    result={'function':fk,'design':xk}
    return result
def fct(x):  # Objective function of test example
    a=x.shape
    fungoal=np.zeros((a[0],1))
    fungoal2=np.zeros((a[0],1))
    fungoal3=np.zeros((a[0],1))
    fungoal[:,0]=3*x[:,0]+(0.0094*x[:,5]+2.4)*x[:,0]/0.6\
           +2*x[:,1]+(0.0214*x[:,6]+3.0)*x[:,1]/1.0\
           +1.2*x[:,2]+(0.01874*x[:,7]+2.8)*x[:,2]/1.0\
           +17*x[:,3]+(0.04*x[:,8]+3.5)*x[:,8]/0.7\
           +34*x[:,4]+(0.04*x[:,9]+3.5)*x[:,9]/0.75
    fungoal2[:,0]=50.07*0.6*x[:,0]+28.08*0.6*x[:,1]+30.08*0.65*x[:,2]+153.12*0.54*x[:,3]+204.75*0.55*x[:,4]
    fungoal3[:,0]=fungoal[:,0]/fungoal2[:,0]

    return fungoal3[:,0]

def constr(x): # Constraint of test example
    #return x[:,0] - x[:,1] + 1.0
    #return 0*x[:,0]
    b=x.shape
    fungoalb=np.zeros((b[0],7))
    fungoalb[:,0]=0.1-x[:,0]
    fungoalb[:,1]=0.1-x[:,1]
    fungoalb[:,2]=0.1-x[:,2]
    fungoalb[:,3]=0.1-x[:,3]
    fungoalb[:,4]=0.1-x[:,4]
    fungoalb[:,5]=x[:,0]-0.5
    fungoalb[:,6]=x[:,1]-0.5
    fungoalb[:,7]=x[:,2]-0.5
    fungoalb[:,8]=x[:,3]-0.5
    fungoalb[:,9]=x[:,4]-0.5
    fungoalb[:,10]=((0.25*x[:,0]+0.08*x[:,1]+0.07*x[:,2]+0.294*x[:,3]+0.35*x[:,4])\
                  /(x[:,0]+x[:,1]+x[:,2]+x[:,3]+x[:,4]))-0.2
    return fungoalb[:,:]


if __name__ == '__main__':
    xbound = np.array([[0,20000],[0,20000],[0,20000],[0,20000],[0,20000],\
                       [0,100],[0,100],[0,100],[0,100],[0,100]])
    res = gaMainLoop(fct,constr,0,11,10,xbound,1)
    print(res)
    print(fct(res['design']))
