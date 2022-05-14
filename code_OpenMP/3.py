import matplotlib.pyplot as plt
import math
import numpy as np
import random
import csv
# plt.rcParams['font.sans-serif'] = ['SimHei']#设置显示中文

serial_nums ,serial_times = [1024, 2048, 4096, 8192, 16384], [64.014, 255.257, 1031.23, 4150.63, 17073.7]
sse_nums, sse_times = [1024, 2048, 4096, 8192, 16384], [27.1056, 109.224, 431.497, 1722.49, 7025.84]
# nums_1 = np.linspace(0, 1085, 2000)
# times_1 = [(4.1381044e-6)*i**3 for i in nums_1]
plt.xlabel('The n of the input matrix',fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Runtime(ms)', fontsize=15)
plt.yticks(fontsize=15)
# plt.scatter(nums, times, color='r', label='serial_mul', marker='x', linewidths=0.65)
plt.plot(serial_nums, serial_times, 'blue', label='Serial')
# plt.plot(trans_nums, trans_times, 'purple', label='Cache')
# plt.plot(sse_nums, sse_times, 'red', label='SSE')
plt.plot(sse_nums, sse_times, 'red', label='AVX')
plt.legend(fontsize=15)
plt.show()
# plt.savefig('4.png')