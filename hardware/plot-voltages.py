import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("alexa-voltages.csv", 'r') as f:
        r = csv.reader(f, delimiter = ",")
        voltages = np.array(list(r)).flatten().astype(float)


    x = np.array(range(len(voltages))) / 10

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, voltages, label="data")
    ax.plot(x, [870]*len(voltages), 'r--', label="threshold")
    ax.set_title("Voltages Measured from Alexa Sensor Array at 10Hz")
    ax.set_ylabel("Raw Values")
    ax.set_xlabel("Seconds")
    ax.legend(loc='lower left')

    ax.set_ylim((840, 980))

    ax.annotate("", xy=(0,960), xytext=(2.5,960), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("Off", xy=(1.25, 970), ha='center', va='center')

    ax.annotate("", xy=(2.5,960), xytext=(5.6,960), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("User Query", xy=(4, 970), ha='center', va='center')

    ax.annotate("", xy=(5.6,960), xytext=(17.2,960), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("Alexa Response", xy=(11.6, 970), ha='center', va='center')

    ax.annotate("", xy=(17.2,960), xytext=(21.3,960), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("Off", xy=(19.2, 970), ha='center', va='center')

    plt.savefig('voltages.png')
    plt.show()