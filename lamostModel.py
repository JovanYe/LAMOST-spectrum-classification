"""

Editor : jovan_ye

This is a temporary script file,aim is to process data and visualization.
take note of that the pytoch vision is gpu_11.3,cpu vision is also okay at this program.

"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding_size = 1  # 输入维数
        self.feature_size1 = 32  # 所含卷积核个数
        self.feature_size2 = 64  # 所含卷积核个数
        self.feature_size3 = 32  # 所含卷积核个数
        self.max_len = 3000  # 数据点个数
        self.k_size = 49    #提取中等大小特征
        self.maxPooling_size = 3  # 下采样尺寸
        # self.acceptance_threshold = 0.8
        self.window_sizes1 = [5, 9, 17, 33,
                              69, 73, 81, 97]
            # [3, 7, 21, 49, 84, 140, 252]  # 卷积核尺寸列表
        #        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_size) #构造一个词嵌入模型
        # 卷积层池化层集合

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embedding_size,
                                    out_channels=self.feature_size1,
                                    kernel_size=h),  #output: 1*(3000-h+1)  3001-h
                          nn.BatchNorm1d(self.feature_size1),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=2, stride=2),  #output: 1*[(hi-size)/stride+1] (3001-h)/2

                          nn.Conv1d(in_channels=self.feature_size1,
                                    out_channels=self.feature_size2,
                                    kernel_size=self.k_size),  # 1×49卷积大卷积核 output: 1*(last_input-k_size+1) (3001-h)/2-k_size+1
                          nn.BatchNorm1d(self.feature_size2),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=4, stride=4), #大池化核  #[(3001-h)/2-k_size+1]/4

                          nn.Conv1d(in_channels=self.feature_size2,
                                    out_channels=self.feature_size2,
                                    kernel_size=3,
                                    stride=2),  # 1×3卷积小卷积核,步长为2    ([(3001-h)/2-k_size+1]/4-2)/2
                          nn.BatchNorm1d(self.feature_size2),
                          nn.ReLU(),
                          nn.Conv1d(in_channels=self.feature_size2,
                                    out_channels=self.feature_size3,
                                    kernel_size=1),  #1×1卷积，通道交流，增强非线性，降维
                          nn.BatchNorm1d(self.feature_size3),
                          nn.ReLU(),

                          nn.MaxPool1d(kernel_size= int((int((int((3001-h)/2)-self.k_size+1)/4)-2)/2)) )
            for h in self.window_sizes1
        ])
        # self.uniform_fmap = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        self.fcdropout = nn.Dropout(p=0.25)  # 随机丢失部分神经元，使之失活，防止过拟合
        self.ln = nn.Linear(in_features=len(self.window_sizes1), out_features=3)
        self.softMax = nn.Softmax()

    # 激活函数既可以使用nn，又可以调用nn.functional
    def forward(self, x):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,dtype=torch.float)
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        # print(x.shape)
        out = [F.adaptive_avg_pool1d(conv(x).permute(0, 2, 1), 1)  for conv in self.convs]  # 卷积层池化层输出   con(x) shape torch.Size([64, 8, 1])

        out = torch.cat(out, dim=1)

        out = out.squeeze(2)  # 拼接

        out = self.fcdropout(out)  # 失活
        out = self.ln(out)  # 输出层输出
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    pass
