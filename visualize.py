import matplotlib.pyplot as plt
import numpy as np


def writeCountToTxt(count):
    print("count:", count)
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
            # print(line)
            if 'speed' in line:
                number += 1
                _, speed = line.split(',')
                speed = float(speed)
                speed_list.append(speed)
                # print(speed)
            if 'count' in line:
                #print('number: ', number)
                # print(speed_list)
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

    '''
    x = []
    for i in range(len(number_list)):
        k = i + 1
        x.append(k)
    plt.plot(x, number_list, linestyle='dashed', linewidth=3,
            marker='o', markerfacecolor='blue', markersize=12)

    plt.ylim(0, 5)
    plt.xlim(1, len(number_list))

    plt.xlabel('time')
    plt.ylabel('numbers')
    plt.title('numbers of cars')
    plt.show()
    '''
