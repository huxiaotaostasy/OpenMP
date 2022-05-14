import matplotlib.pyplot as plt
import math
import numpy as np
import random
import csv
plt.rcParams['font.sans-serif'] = ['SimHei']#设置显示中文
file = open(r'D:\\vs project\\code_SIMD\\logs_arm.txt','r')
data_list = file.readlines()
serial_nums ,serial_times = [0], [0]
trans_nums, trans_times = [0], [0]
sse_nums, sse_times = [0], [0]
sse_tile_nums, sse_tile_times = [0], [0]
for i in data_list:
  x = i.strip('\n').split('\t')
  name, n, T, time = x[0], eval(x[1]), eval(x[2]), eval(x[3])
  if name == 'serial_mul':
    serial_nums.append(n)
    serial_times.append(time)
  # if name == 'trans_mul':
  #   trans_nums.append(n)
  #   trans_times.append(time)
  # if name == 'sse_mul':
  #   sse_nums.append(n)
  #   sse_times.append(time)
  # if name == 'sse_tile_norm' and T == 256:
  #   sse_tile_nums.append(n)
  #   sse_tile_times.append(time)
nums_1 = np.linspace(0, 1085, 2000)
times_1 = [(6.8e-6)*i**3 for i in nums_1]
plt.xlabel('The n of the matrix',fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Runtime(ms)', fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(serial_nums, serial_times, color='r', label='serial_mul', marker='x', linewidths=1)
plt.plot(nums_1, times_1, color='b', label='curve')
# plt.plot(serial_nums, serial_times, 'blue', label='Serial')
# plt.plot(trans_nums, trans_times, 'purple', label='Cache')
# plt.plot(sse_nums, sse_times, 'red', label='SSE')
# plt.plot(sse_tile_nums, sse_tile_times, 'red', label='SSE_tile')
plt.legend(fontsize=15)
# plt.show()
plt.savefig('1_arm.png')