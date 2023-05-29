import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


from workhorse import *


def plotWave(df):
    df['Num']=df['Num'].astype(int)

    fig=go.Figure()
    for num, group in df.groupby('Num'):
        fig.add_trace(go.Scattergl(x=group['x'], y=group['y'],
                        name = num, mode='lines'))

    fig.update_layout(
        title_text = 'Real and Imaginary part of wavefunction',
        xaxis_title_text = 'x',
        yaxis_title_text = 'y'
    )

    fig.show()

def potentialPlotOnly(potentialFunction, initialArray, potentialParameters):

    initialDictionary= initialConditions(initialArray)
    x = initialDictionary["x"]
    Dx = initialDictionary["Dx"]
    potentialParameters["minWidth"] = Dx
    potential = potentialFunction(x, potentialParameters)


    fig=go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=potential, name="Potential", mode="lines"))
    fig.show()


def waveAnimation(waveDF, wave, timeDepPot = False, title = None, figsize_input = [5, 5], limits = [-1e-30, 2000e-9,-5000,5000]):
    
    fig, ax = plt.subplots(figsize=(figsize_input[0], figsize_input[1]))
    
    l_1 = ax.plot([], [])[0]
    l_2 = ax.plot([], [], linewidth=3, color="black", label="V")[0]

    waveArray = makeArrayFromDF(waveDF, wave)
    potentialArray = makeArrayFromDF(waveDF, "pot")

    if(not (timeDepPot)):
        potentialArray = potentialArray[0]
    xArray = np.array(waveDF[waveDF["step"]==1]["x"])

    ax.set_xlabel('Length')
    ax.set_ylabel('Height')
    
    if title == None:
        ax.set_title(f'Animated plot')
    else:
        ax.set_title(title)
        
    
    ax.axis([limits[0], limits[1], limits[2], limits[3]])

    ax2 = ax.twinx()
    ax2.set_ylabel("$V(x)$")
    # v_max = np.min(potentialArray) + 1.1 * (np.max(potentialArray) - np.min(potentialArray)) + 1 # + 1 if v = const
    x_ext = np.concatenate(([xArray[1]], xArray, [xArray[-1]]))

    if (timeDepPot == False):

        v_ext = np.concatenate(([potentialArray[1]], potentialArray, [potentialArray[-1]]))
        ax2.plot(x_ext, v_ext, linewidth=3, color="black", label="V")
        

        #A function that plots all particles for frame i.
        #This function is run for all frames (see FuncAnimation function)
        def animate_system_frame(i):
            l_1.set_data(xArray, waveArray[i])

    elif (timeDepPot == True):

        def animate_system_frame(i):
            l_1.set_data(xArray, waveArray[i])
            v_ext = np.concatenate(([potentialArray[1][1]], potentialArray[i], [potentialArray[-1][-1]]))
            l_2.set_data(x_ext, v_ext)


    FPS = np.arange(0, len(waveArray[:, 0]), 1)
    ani = matplotlib.animation.FuncAnimation(fig, animate_system_frame, frames = FPS)

    plt.close()

    HTML(ani.to_jshtml())
    return ani


def printTransmitionAmplitude(energyArray, transmissionArray, limits = [0, 3e-1, 0, 1.1], figsizeInput = [12, 6]):

    plt.figure(figsize = (figsizeInput[0], figsizeInput[1]))
    plt.plot(energyArray, transmissionArray, label = 'Analytical')

    plt.xlim(limits[0], limits[1])
    plt.ylim(limits[2], limits[3])
    plt.xlabel('Energi/eV')
    plt.title("Analytical Transmission Probability")
    plt.legend()
    plt.show()
    
def energySpectrum(freq, refl, trans, title = None, show = "both", limits = [-2e-9, 4e-3, -2e11, 2e11], figsizeInput = [12, 6]):
    e = 1.602176634e-19
    hbar = 1.054571817e-34
    m = 9.1093837015e-31


    energySpec = ( (freq)* hbar)**2/(2 * m)

    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=energySpec/e, y=np.abs(refl)**2, name="Reflected", mode="lines"))
    # fig.add_trace(go.Scattergl(x=energySpec/e, y=np.abs(trans)**2, name="Transmited", mode="lines"))
    # fig.show()

    plt.figure(figsize = (figsizeInput[0], figsizeInput[1]))

    if title == None:
        plt.title("Energy Spectrum")
    else:
        plt.title(title)

    if(show == "refl"):
        plt.plot(energySpec/e, refl, label = 'Reflected')

    elif(show == "trans"):
        plt.plot(energySpec/e, trans, label = 'Transmited')

    elif(show == "both"):
        plt.plot(energySpec/e, refl, label = 'Reflected')
        plt.plot(energySpec/e, trans, label = 'Transmited')

    plt.xlim(limits[0], limits[1])
    plt.ylim(limits[2], limits[3])
    plt.xlabel('Energi/eV')
    plt.ylabel('Intensity')
    plt.grid()
    plt.legend()
    plt.show()

def fourierPlot(freq, refl, trans, show = "both", limits = [-3e9, 3e9, -5e5, 5e5], figsizeInput = [12, 6]):

    plt.figure(figsize = (figsizeInput[0], figsizeInput[1]))

    if(show == "refl"):
        plt.plot(freq, refl, label = 'Reflected')

    elif(show == "trans"):
        plt.plot(freq, trans, label = 'Transmited')

    elif(show == "both"):
        plt.plot(freq, refl, label = 'Reflected')
        plt.plot(freq, trans, label = 'Transmited')

    plt.xlim(limits[0], limits[1])
    plt.ylim(limits[2], limits[3])
    plt.title("Fourier Transform")
    plt.legend()
    plt.show()

