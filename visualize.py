##########################################################################################
#
# Desc: Visualization tools
#
###########################################################################################
import matplotlib.pyplot as plt
import numpy as np


def writeCountToTxt(count):
    the_file = open('data.txt', 'a')
    output = 'count' + ',' + str(count) + '\n'
    the_file.write(output)
    the_file.close()


def writeSpeedToTxt(speed):
    the_file = open('data.txt', 'a')
    speed_record = str(np.around(speed))
    the_file.write("speed," + speed_record + '\n')
    the_file.close()


def visualization():
    number = 0
    speed_list = []
    avg_list = []
    number_list = []
    with open('data.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            if 'speed' in line:
                number += 1
                _, speed = line.split(',')
                speed = float(speed)
                speed_list.append(speed)
            if 'count' in line:
                try:
                    avg = sum(speed_list) / float(len(speed_list))
                except:
                    continue
                number_list.append(number)
                avg_list.append(avg)
                speed_list = []
                number = 0

    # for average speed
    x = []
    for i in range(len(avg_list)):
        k = i + 1
        x.append(k)

    plt.plot(x, avg_list)
    plt.xlabel('time')
    plt.ylabel('average speed')
    plt.title('Average Speed')
    plt.show()

