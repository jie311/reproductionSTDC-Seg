import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math


# add indexnet 添加
class HolisticIndexBlock(nn.Module):
    def __init__(self,
                 inp,
                 use_nonlinear=False,
                 use_context=False,
                 batch_norm=None):
        super(HolisticIndexBlock, self).__init__()

        BatchNorm2d = batch_norm

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            self.indexnet = nn.Sequential(
                nn.Conv2d(inp,
                          2 * inp,
                          kernel_size=kernel_size,
                          stride=2,
                          padding=padding,
                          bias=False), BatchNorm2d(2 * inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(2 * inp,
                          4,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False))
        else:
            self.indexnet = nn.Conv2d(inp,
                                      4,
                                      kernel_size=kernel_size,
                                      stride=2,
                                      padding=padding,
                                      bias=False)

    # 输入inf 通道输出 idx_en 和idx_de
    def forward(self, x):
        x = self.indexnet(x)

        y = torch.sigmoid(x)
        z = F.softmax(y, dim=1)

        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


# 老熟人了ConvX
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel,
                              stride=stride,
                              padding=kernel // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2,
                          out_planes // 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes,
                          in_planes,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          groups=in_planes,
                          bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2,
                          out_planes // 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


# STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 pretrain_model='',
                 use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        # index net 包后面引入
        index_block = HolisticIndexBlock
        BatchNorm2d = nn.BatchNorm2d
        self.use_conv_last = use_conv_last
        # 此为conx到最小的块
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16),
                            max(1024, base * 16),
                            bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        self.index0 = index_block(base // 2,
                                  use_nonlinear=True,
                                  use_context=True,
                                  batch_norm=BatchNorm2d)
        self.index1 = index_block(base,
                                  use_context=True,
                                  use_nonlinear=True,
                                  batch_norm=BatchNorm2d)
        self.index2 = index_block(base * 4,
                                  use_context=True,
                                  use_nonlinear=True,
                                  batch_norm=BatchNorm2d)

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    # 解析预训练包
    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]  # channal 64 -> 8
        features += [ConvX(base // 2, base, 3, 2)]  # channal 8-> 64

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)),
                              base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)),
                              base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    # 在此处添加

    def forward(self, x):
        # /2
        feat2 = self.x2(x)
        # 生成index
        idx0_en, idx0_de = self.index0(feat2)
        # 点乘
        idx_feat2 = idx0_en * feat2

        # /4
        feat4 = self.x4(idx_feat2)
        idx1_en, idx1_de = self.index1(feat4)
        idx_feat4 = idx1_en * feat4

        feat8 = self.x8(idx_feat4)
        idx2_en, idx2_de = self.index2(feat8)
        idx_feat8 = idx2_en * feat8
        feat16 = self.x16(idx_feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32, idx0_de, idx1_de, idx2_de

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


# STDC1Net
class STDCNet813(nn.Module):
    def __init__(self,
                 base=64,
                 layers=[2, 2, 2],
                 block_num=4,
                 type="cat",
                 num_classes=1000,
                 dropout=0.20,
                 pretrain_model='',
                 use_conv_last=False):
        super(STDCNet813, self).__init__()
        # block类型
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16),
                            max(1024, base * 16),
                            bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        # 生成Stage1&2
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]
        # 批量生成Stage3, 4, 5,i为层,j为层中的操作
        for i, layer in enumerate(layers):
            for j in range(layer):
                # Stage3的第一个是65->65 * 4
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)),
                              base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)),
                              base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    torch.save(model.state_dict(), 'cat.pth')
    print(y.size())
