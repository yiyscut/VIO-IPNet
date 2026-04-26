import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.depth_wise_1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm1d(input_dim)

        self.depth_wise_2 = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_2 = nn.BatchNorm1d(output_dim)

        self.point_wise = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=stride)
        
        self.active_1 = nn.PReLU()
        self.active_2 = nn.PReLU()
       
    def forward(self, x):
        residual = self.point_wise(x)
        out = self.depth_wise_1(x)
        out = self.bn_1 (out)
        out = self.active_1(out)
        out = self.depth_wise_2(out)
        out = self.bn_2 (out)

        out += residual.clone()
        out = self.active_2(out)
        return out

class IPNet(nn.Module):
    def __init__(self, conf):
        super(IPNet, self).__init__()
        self.conf = conf

        # Encoder
        self.encoder = nn.Sequential(
            Transpose(-1, -2),
            ResBlock(6, 32, 7, 1, 3),
            nn.MaxPool1d(2),
            ResBlock(32, 64, 7, 1, 3),
            nn.MaxPool1d(2),
            ResBlock(64, 128, 7, 1, 3),
            nn.MaxPool1d(2),
            ResBlock(128, 256, 7, 1, 3),
            nn.MaxPool1d(2),
            Transpose(-1, -2),
        )

        self.gru1 = nn.GRU(input_size = 256, hidden_size = 128, num_layers = 2, batch_first = True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size = 32, num_layers = 2, batch_first = True)

        self.attention2 = nn.MultiheadAttention(embed_dim=128, num_heads=4, add_bias_kv=True, add_zero_attn=True, batch_first=True)

        # Decoder
        self.gyro_decoder = nn.Sequential(nn.Linear(32, 3), Transpose(-1, -2), nn.Linear(62, 50), Transpose(-1, -2))
        self.acc_decoder = nn.Sequential(nn.Linear(32, 3), Transpose(-1, -2), nn.Linear(62, 50), Transpose(-1, -2))

        # Not in use yet
        # self.gyro_noise_decoder = nn.Sequential(nn.Linear(32, 3), Transpose(-1, -2), nn.Linear(62, 1), Transpose(-1, -2))
        # self.acc_noise_decoder = nn.Sequential(nn.Linear(32, 3), Transpose(-1, -2), nn.Linear(62, 1), Transpose(-1, -2))
    
    def forward(self, input_data):
        feature = torch.cat([input_data["acc"], input_data["gyro"]], dim=-1)                                # [32, 1000, 6]
        output = self.encoder(feature)                                                                      # [32, 125, 128]

        output, _ = self.gru1(output)
        output, _ = self.attention2(output, output, output)
        output, _ = self.gru2(output)                                                      # [32, 125, 3]

        gyro_bias = self.gyro_decoder(output)
        acc_bias = self.acc_decoder(output)

        gyro_noise = 0#input_data["acc"] - input_data["acc"]
        acc_noise = 0#input_data["acc"] - input_data["acc"]
        return gyro_bias, gyro_noise, acc_bias, acc_noise