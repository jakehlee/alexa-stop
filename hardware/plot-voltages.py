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
    ax.plot(x, [900]*len(voltages), 'r--', label="threshold")
    ax.set_title("Voltages Measured from Alexa Sensor Array at 10Hz")
    ax.set_ylabel("Raw Values")
    ax.set_xlabel("Seconds")
    ax.legend(loc='lower left')

    ax.set_ylim((870, 1000))

    ax.annotate("", xy=(0,980), xytext=(4.9,980), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("Off", xy=(2.5, 990), ha='center', va='center')

    ax.annotate("", xy=(4.9,980), xytext=(8.6,980), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("User Query", xy=(7.0, 990), ha='center', va='center')

    ax.annotate("", xy=(8.6,980), xytext=(19.5,980), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("Alexa Response", xy=(14.0, 990), ha='center', va='center')

    ax.annotate("", xy=(19.5,980), xytext=(24.8,980), xycoords='data', textcoords='data', 
        arrowprops={'arrowstyle':'|-|'})
    ax.annotate("Off", xy=(22.0, 990), ha='center', va='center')

    plt.savefig('voltages.png')
    plt.show()