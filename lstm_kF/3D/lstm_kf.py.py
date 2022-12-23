# !/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
file_name = r'E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\xyz\xyzpose.xls'
flight_data  = pd.read_excel(file_name)
flight_data.shape #(144, 3)

# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 15
# fig_size[1] = 5
# plt.rcParams["figure.figsize"] = fig_size
# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.xlabel('Months')
# plt.grid(True)
# plt.autoscale(axis='x',tight=True)
# plt.plot(flight_data['y'])
# plt.show()

x_data = flight_data['x'].values.astype(float)
y_data = flight_data['y'].values.astype(float)
z_data = flight_data['z'].values.astype(float)
all_data = np.zeros((len(x_data),3))
all_data[:,0] = x_data
all_data[:,1] = y_data
all_data[:,2] = z_data
# print(all_data)
#将数据区分为训练数据和测试数据
test_data_size = 30
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]


# 由于训练数据存在相差较大的，因此使用min/max尺度变换对训练数据进行归一化
# 注意只对训练数据进行归一化，为了防止有些信息从训练数据泄露到的测试数据

 
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 3))

#现在可以看到所有训练值都在[-1,1]的范围内
# print(train_data_normalized[:5])
# print(train_data_normalized[-5:])

# 将数据转换为torch tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)

# 由于我们使用的是以月为单位的乘客数据，所以很自然想到的是
# 使用12 作为一个时间序列，我们使用12个月的数据来预测第13个月的乘客量
# 定义如下函数来构建输入输出
# @tw 为sequence length 
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_window = 50
# print(train_data_normalized)
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
# print("train_inout_seq",train_inout_seq)
class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        # 定义lstm 层
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 2)

        # 定义线性层，即在LSTM的的输出后面再加一个线性层
        self.linear = nn.Linear(hidden_size, output_size)
    
    # input_seq参数表示输入sequence
    def forward(self, input_seq):
    
        # lstm默认的输入X是（sequence_legth,bacth_size,input_size）
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        
        # lstm_out的默认大小是（sequence_legth,bacth_size，hidden_size）
        # 转化之后lstm_out的大小是(sequence_legth, bacth_size*hidden_size)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
        # 由于bacth_size = 1, 可知predictions 的维度为(sequence_legth, output_size)
        # [-1] 表示的是取最后一个时间步长对应的输出
        return predictions[-1]

model_path = r'E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\xyz\lstm_model_xyz.pt'
model = torch.load(model_path)

fut_pred = 1
test_inputs = train_data_normalized[-train_window:].tolist()


model.eval()



font = {'family': 'SimSun',  # 宋体
        # 'weight': 'bold',  # 加粗
        'size': '10.5'  # 五号
        }
plt.rc('font', **font)
plt.rc('axes', unicode_minus=False)

# k = np.diag(np.ones((1, 3))[0, :], 3)
# print(k)
R = np.diag(np.ones(2)) * 0.1
print(R)
# plt.rcParams['figure.facecolor'] = "#FFFFF0"  # 设置窗体颜色
# plt.rcParams['axes.facecolor'] = "#FFFFF0"  # 设置绘图区颜色
 
class Kf_Params:
    B = 0  # 外部输入为0
    u = 0  # 外部输入为0
    K = float('nan')  # 卡尔曼增益无需初始化
    z = float('nan')  # 这里无需初始化，每次使用kf_update之前需要输入观察值z
    P = np.diag(np.ones(4))  # 初始P设为0 ??? zeros(4, 4)
 
    # 初始状态：函数外部提供初始化的状态，本例使用观察值进行初始化，vx，vy初始为0
    x = []
    G = []

    # 状态转移矩阵A
    # 和线性系统的预测机制有关，这里的线性系统是上一刻的位置加上速度等于当前时刻的位置，而速度本身保持不变
    A = np.eye(6) + np.diag(np.ones((1, 3))[0, :], 3)

    # 预测噪声协方差矩阵Q：假设预测过程上叠加一个高斯噪声，协方差矩阵为Q
    # 大小取决于对预测过程的信任程度。比如，假设认为运动目标在y轴上的速度可能不匀速，那么可以把这个对角矩阵
    # 的最后一个值调大。有时希望出来的轨迹更平滑，可以把这个调更小
    Q = np.diag(np.ones(6)) * 0.1
 
    # 观测矩阵H：z = H * x
    # 这里的状态是（坐标x， 坐标y，坐标z, 速度x， 速度y,速度z），观察值是（坐标x， 坐标y,坐标z），所以H = eye(3, 6)
    H = np.eye(3, 6)

    # 观测噪声协方差矩阵R：假设观测过程上存在一个高斯噪声，协方差矩阵为R
    # 大小取决于对观察过程的信任程度。比如，假设观测结果中的坐标x值常常很准确，那么矩阵R的第一个值应该比较小
    R = np.diag(np.ones(3)) * 0.1
 
 
def kf_init(px, py,pz, vx, vy,vz):
    # 本例中，状态x为（坐标x， 坐标y， 速度x， 速度y），观测值z为（坐标x， 坐标y）
    kf_params = Kf_Params()
    kf_params.B = 0
    kf_params.u = 0
    kf_params.K = float('nan')
    kf_params.z = float('nan')
    kf_params.P = np.diag(np.ones(6))
    kf_params.x = [px, py,pz , vx, vy,vz]
    kf_params.G = [px, py,pz, vx, vy,vz]
    kf_params.A = np.eye(6) + np.diag(np.ones((1, 3))[0, :], 3)
    kf_params.Q = np.diag(np.ones(6)) * 0.1
    kf_params.H = np.eye(3, 6)
    kf_params.R = np.diag(np.ones(3)) * 0.1
    return kf_params
 
 
def kf_update(kf_params,t):
    # 以下为卡尔曼滤波的五个方程（步骤）
    # print(kf_params.x)

    a1 = np.dot(kf_params.A, kf_params.x)
    a2 = kf_params.B * kf_params.u
    x_ = np.array(a1) + np.array(a2)
    # x_[0:2] = A
    ####lstm预测####
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        # print("seq",seq)
        # 模型评价时候关闭梯度下降
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            list_model_seq = [model(seq)[0].item(),model(seq)[1].item(),model(seq)[2].item()]
            # print("list_model_seq",list_model_seq)
            test_inputs.append(list_model_seq)
    # print(len(test_inputs))
    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 3))
    # print(actual_predictions)
    # print(len(actual_predictions))
    # print(t)
    # print(x_)
    x_[0] = actual_predictions[t-1][0]
    x_[1] = actual_predictions[t-1][1]
    x_[2] = actual_predictions[t-1][2]

    # print(x_)
    # print([x_[t-1],x_[t-1]])
    

    # print(kf_params.A,kf_params.B)
    # print(kf_params.P)
    b1 = np.dot(kf_params.A, kf_params.P)
    b2 = np.dot(b1, np.transpose(kf_params.A))
    p_ = np.array(b2) + np.array(kf_params.Q)
 
    c1 = np.dot(p_, np.transpose(kf_params.H))
    c2 = np.dot(kf_params.H, p_)
    c3 = np.dot(c2, np.transpose(kf_params.H))
    c4 = np.array(c3) + np.array(kf_params.R)
    c5 = np.linalg.matrix_power(c4, -1)
    kf_params.K = np.dot(c1, c5)
    # print("kf_params.H",kf_params.H)
    # print("x_",x_)
    d1 = np.dot(kf_params.H, x_)
    # print(d1)
    # print("d1",d1)
    # print("kf_params.z",kf_params.z)
    d2 = np.array(kf_params.z) - np.array(d1)
    # print("d2",d2)
    d3 = np.dot(kf_params.K, d2)
    # print("d3",d3 )
    kf_params.x = np.array(x_) + np.array(d3)

    e1 = np.dot(kf_params.K, kf_params.H)
    e2 = np.dot(e1, p_)
    kf_params.P = np.array(p_) - np.array(e2)
    # print(kf_params.x)
    kf_params.G = x_
    return kf_params
 
 
def accuracy(predictions, labels):
    res = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            res = res + abs(predictions[i][j] - labels[i][j]) 
    return res/(len(predictions)*len(predictions[0]))
 
 
if __name__ == '__main__':
    # 先验路径
    # path = r'E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\predict.csv'
    # data_A = pd.read_csv(path)
    # data_A_x = list(data_A.iloc[::, 0])
    # data_A_y = list(data_A.iloc[::, 1])
    # A = np.array(list(zip(data_A_x, data_A_y)))

    # plt.subplot(131)
    plt.figure()
    # plt.plot(data_A_x, data_A_y, 'b-+')
    # plt.title('实际的真实路径')

    # 检测到的路径
    path = r'E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\xyz\true.csv'
    # data_B = pd.read_excel(path, header=None)
    data_B = pd.read_csv(path)
    data_B_x = list(data_B.iloc[::, 0])
    data_B_y = list(data_B.iloc[::, 1])
    data_B_z = list(data_B.iloc[::, 2])
    B = np.array(list(zip(data_B_x, data_B_y,data_B_z)))

    # plt.subplot(132)
    x_tick = np.arange(0, 30, 1)
    # plt.plot(data_B_x, data_B_y, color = "blue")
    # print(x_tick)
    # print(data_B_x)


    plt.plot(x_tick, data_B_x, color = "red") #红色真实值
    plt.plot(x_tick, data_B_y, color = "red") 
    plt.plot(x_tick, data_B_z, color = "red") 
    # plt.plot(x_tick, data_A_x, color = "yellow")#黄色lstm
    # plt.plot(x_tick, data_A_y, color = "yellow",linestyle='--')

    # plt.title('检测到的路径')

    # 卡尔曼滤波
    kf_params_record = np.zeros((len(data_B), 6))
    kf_params_p = np.zeros((len(data_B), 6))
    t = len(data_B)

    kalman_filter_params = kf_init(data_B_x[0], data_B_y[0],data_B_z[0], 0, 0 ,0)
    for i in range(t):

        if i == 0:
            kalman_filter_params = kf_init(data_B_x[i], data_B_y[i],data_B_z[i], 0, 0,0 )  # 初始化
        else:
            # print([data_B_x[i], data_B_y[i]])
            # print(1111111)
            kalman_filter_params.z = np.transpose([data_B_x[i], data_B_y[i],data_B_z[i]])  # 设置当前时刻的观测位置
            kalman_filter_params = kf_update(kalman_filter_params,i)  # 卡尔曼滤波
            
        kf_params_record[i, ::] = np.transpose(kalman_filter_params.x)
        kf_params_p[i, ::] = np.transpose(kalman_filter_params.G)

    kf_trace = kf_params_record[::, :3]
    kf_trace_1 = kf_params_p[::, :3]

    # plt.subplot(133)
    # print(kf_trace)
    # plt.plot(kf_trace[::, 0], kf_trace[::, 1], 'g-+')
    # plt.plot(kf_trace_1[1:26, 0], kf_trace_1[1:26, 1], 'm-+')

    
    plt.plot(x_tick, kf_trace[::, 0], color = "blue",linestyle='--')
    plt.plot(x_tick, kf_trace[::, 1], color = "blue",linestyle='--') #蓝色卡尔曼
    plt.plot(x_tick, kf_trace[::, 2], color = "blue",linestyle='--') #蓝色卡尔曼

    
    # path = r'E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\last12fps.csv'
    # # data_B = pd.read_excel(path, header=None)
    # data_C = pd.read_csv(path)
    
    # data_C_x = list(data_C.iloc[::, 0])
    # data_C_y = list(data_C.iloc[::, 1])

    # # print(data_C_x)
    # data_C_x = [i/640 for i in data_C_x]
    # data_C_y = [i/480 for i in data_C_y]
    # print(data_C_x)
    # print(data_C_y)
    # plt.plot(x_tick, data_C_x, color = "black",linestyle='--')
    # plt.plot(x_tick, data_C_y, color = "black",linestyle='--')

    legend = ['观测的x中心值', '观测的y中心值', '卡尔曼滤波加lstm的x值', '卡尔曼滤波加lstm的y值']

    plt.legend(legend, loc="best", frameon=False)
    plt.title('卡尔曼滤波后的效果')
    plt.savefig('result.svg', dpi=600)

    plt.show()
    # plt.close()

    p = accuracy(kf_trace, B)
    print(p)