import numpy as np
from scipy.sparse import diags
import pandas as pd
from  math import ceil



def initializePsi(valueX, sigmaX, centerX, k_0):
    
    normal = 1/(2*np.pi * sigmaX**2)**(1/4)
    gauss = np.exp(-(valueX-centerX)**2/(2*sigmaX**2))


    psi = normal*gauss*np.exp(1j*(k_0*valueX))

    realPsi = np.real(psi)
    imaginaryPsi = np.imag(psi)

    realPsi[0], realPsi[-1] = 0.0, 0.0
    imaginaryPsi[0], imaginaryPsi[-1] = 0.0, 0.0


    return realPsi, imaginaryPsi

def freeParticlePotential(xArray, sizeParameter = []):
    potentialArray = np.zeros((len(xArray)), dtype=np.float64)


    return potentialArray

def initialConditions(valueArray):

    e = valueArray[0]
    hbar = valueArray[1]
    m = valueArray[2]
    initialEnergy = valueArray[3]
    sigmaX = valueArray[4]
    centerX = valueArray[5]
    L = valueArray[6]
    potentialMax = valueArray[7]
    scaleTime = valueArray[8]
    bigT = valueArray[9]
    Nsteps = valueArray[10]
    spatialScale = valueArray[11]

    k_0 = np.sqrt(2*m*initialEnergy)/hbar

    stepL = min(0.8e-9, (np.pi)/k_0)
    Dx = (spatialScale)*stepL
    Nx = ceil(L/Dx) + 1

    # Nx = 2*max(int(2*k_0*L) + 1, Nsteps)
    # Dx = L/(Nx-1)

    x = np.linspace(0.0, L, Nx, dtype=np.float64)
    omega = initialEnergy/hbar
    v_g = hbar * k_0/m
    Dt = scaleTime * hbar /((hbar**2/(2*m*Dx**2))+potentialMax)
    T = (bigT)*L/v_g
    Nt = int(T/Dt)

    valueDict = {
        "charge" : e,
        "hbar" : hbar,
        "mass" : m,
        "initialEnergy" : initialEnergy,
        "sigmaX" : sigmaX,
        "centerX" : centerX,
        "L" : L,
        "potentialMax" : potentialMax,
    }


    valueDict["k_0"] = k_0
    valueDict["Nx"] = Nx
    valueDict["Dx"] = Dx
    valueDict["x"] = x
    valueDict["omega"] = omega
    valueDict["v_g"] = v_g
    valueDict["Dt"] = Dt
    valueDict["T"] = T
    valueDict["Nt"] = Nt

    return valueDict


def findCharFreq(potentialParameters, initialEnergy):
    m = 9.1093837015e-31
    freq = (np.sqrt(2*(potentialParameters["vMax"]-initialEnergy)))/np.sqrt(m)*potentialParameters["barrierWidth"]
    return freq

def staticWavePropagation(initPsi, potentialFunction, initialArray, potentialParameters, frameRate):

    initialDictionary= initialConditions(initialArray)
    x = initialDictionary["x"]
    centerX = initialDictionary["centerX"]
    sigmaX = initialDictionary["sigmaX"]
    k_0 = initialDictionary["k_0"]
    Dx = initialDictionary["Dx"]

    potentialParameters["minWidth"] = Dx

    potential = potentialFunction(x, potentialParameters)
    psiReal, psiImaginary = initPsi(x, sigmaX, centerX, k_0)


    length = len(psiReal)

    Dt = initialDictionary["Dt"]
    Nt = initialDictionary["Nt"]


    tT = 0
    for i in range(Nt-1):
        if((i%frameRate == 0) or (i==Nt-1)):
            tT += 1


    saveReal = np.zeros((tT, length), dtype = np.float64, order = 'C')
    saveImaginary = np.zeros((tT, length), dtype = np.float64, order = 'C')

    hbar = initialDictionary["hbar"]
    m = initialDictionary["mass"]


    saveReal[0,:] = (psiReal)
    saveImaginary[0, :] = psiImaginary

    diagonal = np.full(length, -2)*(hbar * Dt / (2*m *Dx**2))
    semidiagonal = np.full(length - 1, 1)*(hbar * Dt / (2*m *Dx**2))
    derivative = [diagonal, semidiagonal, semidiagonal]

    sparseDerivative = diags(derivative, [0, -1, 1]).tocsr()

    updateReal = psiReal
    updateImaginary = psiImaginary

    pot = diags((Dt/hbar) * potential).tocsr()
    mat = -sparseDerivative + pot

    tCounter = 0
    stepCount = np.zeros((tT, length))
    tList = np.zeros((tT, length))
    xVal = np.zeros((tT, length))
    potentialList = np.zeros((tT, length))

    potentialList[0,:] = (potential)

    tList[0,:] = np.zeros(length)
    stepCount[0,:] = np.zeros(length)
    xVal[0,:] = x



    for t in range(Nt-1):

        updateReal += mat.dot(updateImaginary)
        updateImaginary += -mat.dot(updateReal)

        if ((t%frameRate == 0) or (t==Nt-1)):
            tStep = list(np.full((length), (t+1)*Dt))
            step = list(np.full((length), (tCounter + 1)))
            saveReal[tCounter,:] = updateReal
            saveImaginary[tCounter,:] = updateImaginary
            tList[tCounter,:] = tStep
            xVal[tCounter,:] = x
            stepCount[tCounter,:] = step
            potentialList[tCounter,:] = potential
            tCounter += 1


    potentialList = np.array(potentialList)*3e23

    xVal = xVal.flatten()
    saveReal = saveReal.flatten()
    saveImaginary = saveImaginary.flatten()
    potentialList = potentialList.flatten()
    tList = tList.flatten()
    stepCount = stepCount.flatten()


    data = np.stack((xVal, saveReal, saveImaginary, potentialList, tList, stepCount), axis = 1)
    col = ['x', 'real', 'imag', 'pot', 'dt', 'step']

    df = pd.DataFrame(data, columns=col)

 

    return df

def cantor(x, sizeParameter):

    begining = sizeParameter["begining"]
    end = sizeParameter["end"]
    potentialHeight = sizeParameter["vMax"]
    iterations = sizeParameter["iterations"]
    minWidth = sizeParameter["minWidth"]
    partition = sizeParameter["partition"]
    
    potential = np.full(len(x), potentialHeight)
    beginingIndex = np.searchsorted(x, begining, side = 'right')
    endIndex = np.searchsorted(x, end, side = 'right')
    
    potential[:beginingIndex] = 0
    potential[endIndex:] = 0

    middle = potential[beginingIndex:endIndex]
    beginingList = list(potential[:beginingIndex])
    endingList = list(potential[endIndex:])

    def cantorStep(middle, steps, partition):

        thirds = np.array_split(middle, partition)
        first = thirds[0]
        mid = thirds[1:-1]
        last = thirds[-1]


        if ((steps == 0)):
            return middle
        else:
            first = cantorStep(first, steps-1, partition)
            mid = np.array(mid).flatten()
            mid[:] = 0
            last = cantorStep(last, steps-1, partition)

        first = list(first)
        mid = list(mid)
        last = list(last)

        middle = []

        middle.extend(first)
        middle.extend(mid)
        middle.extend(last)


        return middle
    

    middle = cantorStep(middle, iterations, partition)
    middleList = list(middle)
    potentialList = []

    potentialList.extend(beginingList)
    potentialList.extend(middleList)
    potentialList.extend(endingList)

    return np.array(potentialList)

def probabilityDensity(waveDF, initialArray, potentialParameters, f = 3/4):

    frac = np.max(waveDF["step"])
    step=int(f*frac)

    initialDictionary = initialConditions(initialArray)

    realWave = np.array(waveDF.loc[(waveDF["step"] == step)]["real"])
    imaginaryWave = np.array(waveDF.loc[(waveDF["step"] == step)]["imag"])
    wave = realWave + 1j*imaginaryWave

    begin = potentialParameters["begining"]
    end = potentialParameters["end"]
    Dx = initialDictionary["Dx"]
    x = initialDictionary["x"]

    beginingIndex = np.searchsorted(x, begin, side = 'right')
    endIndex = np.searchsorted(x, end, side = 'right')

    probabilityTot = np.abs(wave)**2
    probabilityTrans = np.abs(wave[endIndex:])**2
    probabilityRef = np.abs(wave[:beginingIndex])**2

    normalTrans = np.sum(probabilityTrans)*Dx
    normalRef = np.sum(probabilityRef)*Dx

    return probabilityTot, normalRef, normalTrans

def transmitionAmplitude(energy, b, omega, vMaxx = 120e-3):


    e = 1.602176634e-19
    hbar = 1.054571817e-34
    m = 9.1093837015e-31
    initialEnergy = energy * e
    vMax = vMaxx * e

    k = np.sqrt((2*m*initialEnergy)/(hbar**2))
    kNull = np.sqrt((2*m*vMax)/(hbar**2))

    kappa = np.emath.sqrt(kNull**2 - k**2)

    sigma = (kappa/k) + (k/kappa)
    delta = (kappa/k) - (k/kappa)

    def A(k):
        d = D(k)
        return np.exp(-2j*k*b)/d

    def D(k):
        ledd1 = np.cosh(kappa*b)**2
        ledd2 = (1/4)*(np.sinh(kappa*b)**2)*((sigma**2)*np.cos(2*k*omega) - (delta**2))
        ledd3 = 1j*np.sinh(kappa*b)*(delta*np.cosh(kappa*b) + (1/4)*(sigma**2)*np.sinh(kappa*b)*np.sin(2*k*omega))

        return ledd1 + ledd2 + ledd3
    
    result = A(k)
    result = np.abs(result)**2
    return result

def dbTD(x, sizeParameter, time):

    center = sizeParameter["center"]
    potentialHeight = sizeParameter["vMax"]
    wellWidth = sizeParameter["wellWidth"]
    barrierWidth = sizeParameter["barrierWidth"]

    potentialVar = sizeParameter["potentialVar"]
    freqRight = sizeParameter["freqRight"]
    freqLeft = sizeParameter["freqLeft"]
    phaseRight = sizeParameter["phaseRight"]
    phaseLeft = sizeParameter["phaseLeft"]

    potential = np.zeros(len(x))

    p1 = center - (0.5*wellWidth + barrierWidth)
    p2 = center - (0.5*wellWidth)
    p3 = center + (0.5*wellWidth)
    p4 = center + (0.5*wellWidth + barrierWidth)

    p1i = np.searchsorted(x, p1, side = 'right')
    p2i = np.searchsorted(x, p2, side = 'right')
    p3i = np.searchsorted(x, p3, side = 'right')
    p4i = np.searchsorted(x, p4, side = 'right')


    potential[p1i:p2i], potential[p3i:p4i] = potentialHeight + potentialVar*np.sin(freqLeft*time + phaseLeft), potentialHeight + potentialVar*np.sin(freqRight*time + phaseRight)


    return potential


def aatTI(x, sizeParameter):

    center = sizeParameter["center"]
    potentialHeight = sizeParameter["vMax"]
    wellWidth = sizeParameter["wellWidth"]
    barrierWidth = sizeParameter["barrierWidth"]
    stepWidth = sizeParameter["stepWidth"]
    stepHeight = sizeParameter["stepHeight"]


    potential = np.zeros(len(x))

    p1 = center - (0.5*wellWidth + barrierWidth)
    p2 = center - (0.5*wellWidth)
    p3 = center + (0.5*wellWidth)
    p4 = center + (0.5*wellWidth + barrierWidth)

    s1 = center - (0.5*wellWidth + barrierWidth + stepWidth)
    s2 = center + (0.5*wellWidth + barrierWidth + stepWidth)

    p1i = np.searchsorted(x, p1, side = 'right')
    p2i = np.searchsorted(x, p2, side = 'right')
    p3i = np.searchsorted(x, p3, side = 'right')
    p4i = np.searchsorted(x, p4, side = 'right')

    s1i = np.searchsorted(x, s1, side = 'right')
    s2i = np.searchsorted(x, s2, side = 'right')

    potential[s1i: p1i], potential[p4i: s2i] = stepHeight, stepHeight
    potential[p1i:p2i], potential[p3i:p4i] = potentialHeight, potentialHeight

    return potential

def aatTD(x, sizeParameter, time):

    center = sizeParameter["center"]
    potentialHeight = sizeParameter["vMax"]
    wellWidth = sizeParameter["wellWidth"]
    barrierWidth = sizeParameter["barrierWidth"]
    stepWidth = sizeParameter["stepWidth"]
    stepHeight = sizeParameter["stepHeight"]
    potentialVar = sizeParameter["potentialVar"]
    frequency = sizeParameter["freq"]


    potential = np.zeros(len(x))

    p1 = center - (0.5*wellWidth + barrierWidth)
    p2 = center - (0.5*wellWidth)
    p3 = center + (0.5*wellWidth)
    p4 = center + (0.5*wellWidth + barrierWidth)

    s1 = center - (0.5*wellWidth + barrierWidth + stepWidth)
    s2 = center + (0.5*wellWidth + barrierWidth + stepWidth)

    p1i = np.searchsorted(x, p1, side = 'right')
    p2i = np.searchsorted(x, p2, side = 'right')
    p3i = np.searchsorted(x, p3, side = 'right')
    p4i = np.searchsorted(x, p4, side = 'right')

    s1i = np.searchsorted(x, s1, side = 'right')
    s2i = np.searchsorted(x, s2, side = 'right')

    potential[s1i: p1i], potential[p4i: s2i] = stepHeight, stepHeight
    potential[p1i:p2i], potential[p3i:p4i] = potentialHeight + potentialVar*np.sin(frequency*time), potentialHeight + potentialVar*np.sin(frequency*time)

    return potential


def dbTI(x, sizeParameter):

    center = sizeParameter["center"]
    potentialHeight = sizeParameter["vMax"]
    wellWidth = sizeParameter["wellWidth"]
    barrierWidth = sizeParameter["barrierWidth"]

    potential = np.zeros(len(x))

    p1 = center - (0.5*wellWidth + barrierWidth)
    p2 = center - (0.5*wellWidth)
    p3 = center + (0.5*wellWidth)
    p4 = center + (0.5*wellWidth + barrierWidth)

    p1i = np.searchsorted(x, p1, side = 'right')
    p2i = np.searchsorted(x, p2, side = 'right')
    p3i = np.searchsorted(x, p3, side = 'right')
    p4i = np.searchsorted(x, p4, side = 'right')

    potential[p1i:p2i], potential[p3i:p4i] = potentialHeight, potentialHeight
    
    return potential


def tdWavePropagation(initPsi, potentialFunction, initialArray, potentialParameters, frameRate):

    initialDictionary= initialConditions(initialArray)
    x = initialDictionary["x"]
    centerX = initialDictionary["centerX"]
    sigmaX = initialDictionary["sigmaX"]
    k_0 = initialDictionary["k_0"]
    Dx = initialDictionary["Dx"]

    potentialParameters["minWidth"] = Dx

    potential = potentialFunction(x, potentialParameters, 0)
    psiReal, psiImaginary = initPsi(x, sigmaX, centerX, k_0)


    length = len(psiReal)

    Dt = initialDictionary["Dt"]
    Nt = initialDictionary["Nt"]


    tT = 0
    for i in range(Nt-1):
        if((i%frameRate == 0) or (i==Nt-1)):
            tT += 1


    saveReal = np.zeros((tT, length), dtype = np.float64, order = 'C')
    saveImaginary = np.zeros((tT, length), dtype = np.float64, order = 'C')

    hbar = initialDictionary["hbar"]
    m = initialDictionary["mass"]


    saveReal[0,:] = (psiReal)
    saveImaginary[0, :] = psiImaginary

    diagonal = np.full(length, -2)*(hbar * Dt / (2*m *Dx**2))
    semidiagonal = np.full(length - 1, 1)*(hbar * Dt / (2*m *Dx**2))
    derivative = [diagonal, semidiagonal, semidiagonal]

    sparseDerivative = diags(derivative, [0, -1, 1]).tocsr()

    updateReal = psiReal
    updateImaginary = psiImaginary

    pot = diags((Dt/hbar) * potential).tocsr()
    mat = -sparseDerivative + pot

    tCounter = 0
    stepCount = np.zeros((tT, length))
    tList = np.zeros((tT, length))
    xVal = np.zeros((tT, length))
    potentialList = np.zeros((tT, length))

    potentialList[0,:] = (potential)

    tList[0,:] = np.zeros(length)
    stepCount[0,:] = np.zeros(length)
    xVal[0,:] = x



    for t in range(Nt-1):


        potential = potentialFunction(x, potentialParameters, t*Dt)
        pot = diags((Dt/hbar) * potential).tocsr()
        mat = -sparseDerivative + pot

        updateReal += mat.dot(updateImaginary)
        updateImaginary += -mat.dot(updateReal)

        if ((t%frameRate == 0) or (t==Nt-1)):
            tStep = list(np.full((length), (t+1)*Dt))
            step = list(np.full((length), (tCounter + 1)))
            saveReal[tCounter,:] = updateReal
            saveImaginary[tCounter,:] = updateImaginary
            tList[tCounter,:] = tStep
            xVal[tCounter,:] = x
            stepCount[tCounter,:] = step
            potentialList[tCounter,:] = potential
            tCounter += 1


    potentialList = np.array(potentialList)*3e23

    xVal = xVal.flatten()
    saveReal = saveReal.flatten()
    saveImaginary = saveImaginary.flatten()
    potentialList = potentialList.flatten()
    tList = tList.flatten()
    stepCount = stepCount.flatten()


    data = np.stack((xVal, saveReal, saveImaginary, potentialList, tList, stepCount), axis = 1)
    col = ['x', 'real', 'imag', 'pot', 'dt', 'step']

    df = pd.DataFrame(data, columns=col)

 

    return df


def fourierTransform(waveDF, initialArray, step, potentialParameters):

    if(step == -1):
        step = np.max(waveDF["step"])

    initialDictionary = initialConditions(initialArray)
    realWave = np.array(waveDF.loc[waveDF["step"] == step]["real"])
    imaginaryWave = np.array(waveDF.loc[waveDF["step"] == step]["imag"])
    wave = realWave + 1j*imaginaryWave
    
    begin = potentialParameters["begining"]
    end = potentialParameters["end"]
    Dx = initialDictionary["Dx"]
    x = initialDictionary["x"]

    beginingIndex = np.searchsorted(x, begin, side = 'right')
    endIndex = np.searchsorted(x, end, side = 'right')

    transformed = np.fft.fft(wave, n = 1000000)
    transformedReflected = np.fft.fft(wave[:beginingIndex], n = 1000000)
    transformedTransmited = np.fft.fft(wave[endIndex:], n = 1000000)

    N = len(transformed)
    freq = np.fft.fftfreq(N, Dx)

        

    freq = 2*np.pi*freq

    transformedReflected = np.abs(transformedReflected)**2
    transformedTransmited = np.abs(transformedTransmited)**2
            
    return freq, transformedReflected, transformedTransmited


def makeArrayFromDF(df, wave):
    columnDim = len(np.array(df[df["step"] == 1]["step"]))
    rowDim = len(set((df["step"])))

    waveArray = np.zeros((rowDim, columnDim))

    if(wave == "total"):
        for i in range(rowDim):
            wewo = np.abs(np.array(df[df["step"] == i+1]["real"]) + 1j * np.array(df[df["step"] == i+1]["imag"]))
            waveArray[i] = wewo

    elif(wave == "pot"):
        for i in range(rowDim):
            waveArray[i] = np.array(df[df["step"] == i+1]["pot"])

    elif(wave == "real"):
        for i in range(rowDim):
            waveArray[i] = np.array(df[df["step"] == i+1]["real"])

    elif(wave == "imag"):
        for i in range(rowDim):
            waveArray[i] = np.array(df[df["step"] == i+1]["imag"])

    return waveArray



def analyticalTransProb(energyArray, b, omega, vMaxx = 120e-3):
    transmissionArray = np.zeros(len(energyArray))

    for i in range(len(energyArray)):
        transmissionArray[i] = transmitionAmplitude(energyArray[i], b, omega, vMaxx)

    transmissionArray = transmissionArray
    return transmissionArray


def resonanceFreq(n):
    # for particle in box
    hbar = 1.054571817e-34
    m = 9.1093837015e-31
    eRes = (n*np.pi*hbar)**2/(2*m*(6e-9)**2)

    return eRes


