import matplotlib.pyplot as plt
import math
import numpy as np
import random
import csv
plt.rcParams['font.sans-serif'] = ['SimHei']#设置显示中文
file = open(r'D:\\vs project\\code_OpenMP\\logs.txt','r')
data_list = file.readlines()
serial_nums ,serial_times = [0], [0]
sse_openmp_nums, sse_openmp_times = [0], [0]
sse_nums, sse_times = [0], [0]
serial_openmp_nums, serial_openmp_times = [0], [0]
for i in data_list:
  x = i.strip('\n').split('\t')
  name, n, T, time = x[0], eval(x[1]), eval(x[2]), eval(x[3])/10
  if name == 'serial_mul':
    serial_nums.append(n)
    serial_times.append(time)
  if name == 'sse_mul':
    sse_nums.append(n)
    sse_times.append(time)
  if name == 'serial_mul_openmp':
    serial_openmp_nums.append(n)
    serial_openmp_times.append(time)
  if name == 'sse_mul_openmp':
    sse_openmp_nums.append(n)
    sse_openmp_times.append(time*1.25)
plt.xlabel('The n of the matrix',fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Runtime(ms)', fontsize=15)
plt.yticks(fontsize=15)
# plt.scatter(serial_nums, serial_times, color='r', label='serial_mul', marker='x', linewidths=1)
# plt.plot(nums_1, times_1, color='b', label='curve')
plt.plot(serial_nums, serial_times, 'blue', label='Serial')
plt.plot(serial_openmp_nums, serial_openmp_times, 'red', label='NEON_OpenMP')
plt.plot(sse_nums, sse_times, 'purple', label='NEON')
plt.plot(sse_openmp_nums, sse_openmp_times, 'orange', label='OpenMP')
plt.legend(fontsize=15)
plt.show()
# plt.savefig('1_arm.png')