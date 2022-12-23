import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
#  r"D:\PixivWallpaper\catavento.png"
file_name = r'E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\xyz\xyzpose.xls'
print(file_name)
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
test_data_size = 12
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

# 由于训练数据存在相差较大的，因此使用min/max尺度变换对训练数据进行归一化
# 注意只对训练数据进行归一化，为了防止有些信息从训练数据泄露到的测试数据

 
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 3))

#现在可以看到所有训练值都在[-1,1]的范围内
print(train_data_normalized[:5])
print(train_data_normalized[-5:])

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
# 模型实例化并定义损失函数和优化函数



loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
print(model)



# epochs = 150
# print(train_inout_seq[0])
# print(type(train_inout_seq))
# for i in range(epochs):
#     for seq, labels in train_inout_seq:
#         optimizer.zero_grad()
#         # 下面这一步是对隐含状态以及细胞转态进行初始化
#         # 这一步是必须的，否则会报错 "RuntimeError: Trying to backward through the graph a     
#         # second time, but the buffers have already been freed"
#         model.hidden_cell = (torch.zeros(2, 1, model.hidden_size),
#                            torch.zeros(2, 1, model.hidden_size))

#         y_pred = model(seq)
        

#         single_loss = loss_function(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()
 
#     if i%25 == 1:
#         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
 
# print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
 
# 以train data的最后12个数据进行预测
fut_pred = 30
test_inputs = train_data_normalized[-train_window:].tolist()


model.eval()
#基于最后12个数据来预测第133个数据，并基于新的预测数据进一步预测
# 134-144 的数据
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    # print("seq",seq)
    # 模型评价时候关闭梯度下降
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        print(model(seq))
        list_model_seq = [model(seq)[0].item(),model(seq)[1].item(),model(seq)[2].item()]
        # print("list_model_seq",list_model_seq)
        test_inputs.append(list_model_seq)        
# print(test_inputs)
test_inputs[fut_pred:]
 
##save

# torch.save(model, model_name)  # save entire net


# with open(model_name, 'wb') as f:
#     torch.save(model.state_dict(), f)  # 保留模型参数即可，这样保存的模型比较小






# 通过反向尺度变换，将预测数据抓换成非归一化的数据
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 3))
# print(actual_predictions)
print("len test_inputs",len(test_inputs))
print("len actual_predictions",len(actual_predictions))
x_predict = []
y_predict = []
z_predict = []
for i in range(len(actual_predictions)):
    x_predict.append(actual_predictions[i][0])
    y_predict.append(actual_predictions[i][1])
    z_predict.append(actual_predictions[i][2])
# 绘制图像查看预测的[133-144]的数据和实际的133-144 之间的数据差别
# print(x_predict,y_predict,z_predict)
x = np.arange(165, 195, 1)
# print(x)


# fig=plt.figure()
# ax1 = Axes3D(fig)





dataframe = pd.DataFrame({'x_predict':x_predict,'y_predict':y_predict,'z_predict':z_predict})
dataframe.to_csv(r"E:\View of Delft dataset\Pytorch-lstm-forecast-main\Pytorch-lstm-forecast-main\predict2.csv",index=False,sep=',')

# ax1.scatter3D(x_data[165:195],y_data[165:195],z_data[165:195], cmap='Blues')  #绘制散点图

# ax1.plot3D(x_data[165:195],y_data[165:195],z_data[165:195],'gray',linewidth = 1)    #绘制空间曲线

# ax1.scatter3D(x_predict,y_predict,z_predict, cmap='Blues')  #绘制散点图

# ax1.plot3D(x_predict,y_predict,z_predict,'red',linestyle='--',linewidth = 5)    #绘制空间曲线



my_x_ticks = np.arange(0, 194, 1)
plt.title('tt flight')
plt.ylabel('Total Passengers')
plt.grid(True)

plt.xticks(my_x_ticks)
plt.plot(my_x_ticks,x_data,color = "red")
plt.plot(my_x_ticks,y_data,color = "blue")
plt.plot(my_x_ticks,z_data,color = "yellow")

plt.plot(x,x_predict,color = "red",linestyle='--')
plt.plot(x,y_predict,color = "blue",linestyle='--')
plt.plot(x,z_predict,color = "yellow",linestyle='--')

plt.show()






