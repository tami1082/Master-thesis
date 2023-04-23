import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    DLinear
    """
    def __init__(self, seq_len,pred_len,moving_avg):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = moving_avg
        self.decompsition = series_decomp(kernel_size)
        # self.channels = enc_in

        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch, nodes, time, features = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x_dl = x.permute(0,2,1,3).reshape(batch,time,nodes*features)

        seasonal_init, trend_init = self.decompsition(x_dl)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        seasonal_output = seasonal_output.reshape(batch,nodes,-1,features)
        trend_output = trend_output.reshape(batch,nodes,-1,features)

        return trend_output, seasonal_output


class WaveNet(nn.Module):
    def __init__(self, num_nodes, dropout, in_dim, out_dim,out_len, residual_channels, dilation_channels, skip_channels,
                 end_channels, blocks, output_window_dl, layers, kernel_size, batch_size):
        super(WaveNet, self).__init__()
        self.batch_size = batch_size
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_len = out_len
        self.num_nodes = num_nodes
        self.output_window_dl = output_window_dl

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        receptive_filed = 0

        """ WaveNet. """
        for i in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for j in range(self.layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_filed += additional_scope
                additional_scope *= 2

        self.receptive_filed = receptive_filed
        self.end_conv = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,
                                  # kernel_size=(1, 288 - self.receptive_filed),
                                  kernel_size=(1, 1),
                                  bias=True)

        self.end_features = Linear(end_channels, self.out_dim)

        self.predict = nn.Linear(self.output_window_dl-self.receptive_filed, self.out_len)

    def forward(self, input_o):
        input = input_o.permute(0,3,1,2)#BFNT
        in_len = input.size(3)
        output = []
        if in_len < self.receptive_filed:
            x = nn.functional.pad(input, (self.receptive_filed - in_len, 0, 0, 0))
        else:
            x = input
        output.append(x)
        x = self.start_conv(x)
        skip = 0

        """ WaveNet Layers"""
        for i in range(self.blocks):
            for j in range(self.layers):

                count = i * 2 + j
                residual = x
                # dilated convolution
                filter = self.filter_convs[count](residual)
                filter = torch.tanh(filter)
                gate = self.gate_convs[count](residual)
                gate = torch.sigmoid(gate)
                x = filter * gate

                # parametrized skip connection
                s = x
                s = self.skip_convs[count](s)
                try:
                    skip = skip[:, :, :, -s.size(3):]
                except:
                    skip = 0
                skip = s + skip
                # if count % 2 == 1:
                #     output.append(s)

                x = self.residual_convs[count](x)
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[count](x)
            # every block
            output.append(s)


        x = skip
        x = self.end_conv(x)
        x = self.end_features(x)

        x = self.predict(x.squeeze())
        # x = x.permute() #BNT
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class Linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(Linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order):
        super(GCN,self).__init__()
        self.nconv = nconv()
        self.c_in = c_in
        self.c_out = c_out
        self.l_in = (order * support_len + 1) * c_in
        self.mlp = Linear(self.l_in,c_out)
        self.dropout = dropout
        self.order = order
        self.support_len = support_len

    def forward(self,x_o,support):
        x = x_o.permute(0,3,1,2).to(torch.float) #BFNT
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = h.permute(0,2,3,1) #BNTF
        return h

class Pinjie(nn.Module):
    def __init__(self, config, adj_mx):
        super(Pinjie, self).__init__()
        self.feature_dim = config["len_select"]
        self.dropout = config['dropout']
        self.num_nodes = config['num_nodes']
        self.output_window = config["output_len"]

        # dl
        self.input_window = config["input_len"]  # 144
        self.output_window_dl = config["output_len_dl"]
        self.moving_avg = config["moving_avg"]

        self.dl = DLinear(self.input_window, self.output_window_dl, self.moving_avg)

        # gcn
        self.adj_mx = torch.tensor(adj_mx).to(config.device).to(torch.float)
        self.embed_dim = config['embed_dim']
        self.gcn_out = config['gcn_out']
        self.order = config['order']

        self.e_1 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim)).to(config.device)
        self.e_2 = nn.Parameter(torch.randn(self.embed_dim, self.num_nodes)).to(config.device)
        dj_adj = torch.mm(self.e_1, self.e_2).to(config.device).to(torch.float)

        self.support = []
        self.support.append(self.adj_mx)
        self.support.append(dj_adj)
        len_support = len(self.support)

        self.t_gcn = GCN(self.feature_dim, self.gcn_out, self.dropout, len_support, self.order)
        self.s_gcn = GCN(self.feature_dim, self.gcn_out, self.dropout, len_support, self.order)
        self.raw_gcn = GCN(self.feature_dim, self.gcn_out, self.dropout, len_support, self.order)


        self.f_out = config["f_out"]
        self.x_Linear = Linear(self.feature_dim, self.gcn_out)
        self.x_time = Linear(self.input_window, self.output_window_dl)
        self.f_Linear = nn.Linear(self.gcn_out, self.f_out)

        # wave
        self.residual_channels = config['residual_channels']
        self.dilation_channels = config['dilation_channels']
        self.skip_channels = config['skip_channels']
        self.end_channels = config['end_channels']
        self.blocks = config['blocks']
        self.layers = config['layers']
        self.kernel_size = config['kernel_size']
        self.batch_size = config['batch_size']

        self._logger = getLogger()
        out_dim = 1
        self.tcn = WaveNet(self.num_nodes, self.dropout,self.f_out,out_dim,self.output_window,self.residual_channels,
                           self.dilation_channels, self.skip_channels, self.end_channels, self.blocks,
                           self.output_window_dl,self.layers, self.kernel_size, self.batch_size)

    def forward(self, x, y, m, s):
        inputs = x  # (batch_size, num_nodes, input_window, feature_dim)
        inputs = inputs[:, :, :, 2:]
        inputs = (inputs - m) / s

        tx, sx = self.dl(x) #BNTF

        tx_g = self.t_gcn(tx, self.support)
        sx_g = self.s_gcn(sx, self.support)

        # x_l = x.permute(0,3,2,1) #BFTN
        # f_x = self.x_Linear(x_l)
        # f_x = self.x_time(f_x.permute(0,2,1,3)) #BTFN
        # f_x = f_x.permute(0,3,1,2) #BNTF
        # m = tx_g + sx_g + f_x

        x_l = x.permute(0,2,1,3) #BTNF
        x_l = self.x_time(x_l)
        x_lg = self.raw_gcn(x_l.permute(0,2,1,3),self.support)
        m = tx_g + sx_g + x_lg

        fusion_x = self.f_Linear(m)

        wx = self.tcn(fusion_x) #BNT
        output = torch.sigmoid(wx)+torch.tanh(wx)
        return output


if __name__ == "__main__":
    x = torch.rand([32,134,144,7])
    # p = Pinjie()


    dl = DLinear(144,72,5)
    # x = x.reshape(32,144,-1)
    r,y = dl(x) # B N T F
    # print('r')
    # # r = r.reshape(32,72,134,-1)
    w = WaveNet(134,0.2,7,1,144,32,32,128,64,4,72,2,2,32)
    # r = r.permute(0,3,2,1)
    rw = w(r)
    #
    # # support = torch.randn([134,134])
    # # supports = []
    # # supports.append(support)
    # # g = GCN(7,32,0.2,len(supports),2)
    # # gr = g(r,supports)
    # print('u')