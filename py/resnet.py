import numpy as np
import go
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def conv5x5(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, 5, stride=1, padding=2, bias=False)


class Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual_block, self).__init__()

        self.conv = conv5x5(in_channel, out_channel)
        # self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x) + x
        out = F.relu(out, True)
        # out = self.bn(out)

        return out


class Resnet(nn.Module):
    def __init__(self, num_planes):
        super(Resnet, self).__init__()

        self.entry_block = conv5x5(2, num_planes)

        self.residual_layers = nn.Sequential(
            Residual_block(num_planes, num_planes),
            Residual_block(num_planes, num_planes),
            Residual_block(num_planes, num_planes),
            Residual_block(num_planes, num_planes),
            Residual_block(num_planes, num_planes)
        )

        self.classifier = nn.Conv2d(num_planes, 4, 1, stride=1)
        self.fc = nn.Linear(go.LN * 4, go.LN + 1)

    def forward(self, x):
        out = self.entry_block(x)
        out = self.residual_layers(out)
        out = self.classifier(out)
        out = out.view(-1, go.LN * 4)
        out = self.fc(out)

        return out


halo_resnet = Resnet(30)
if torch.cuda.is_available():
    halo_resnet.cuda()


def print_input():
    i = go.N
    s = "\n"
    while i > 0:
        s += str(i).zfill(2) + " "
        i -= 1
        j = 0
        while j < go.N:
            s += str(int(go.INPUT_BOARD[0, 0, i, j])) + " "
            j += 1
        s += "\n"

    s += "   "
    while i < go.N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1

    i = go.N
    s += "\n"
    while i > 0:
        s += str(i).zfill(2) + " "
        i -= 1
        j = 0
        while j < go.N:
            s += str(int(go.INPUT_BOARD[0, 1, i, j])) + " "
            j += 1
        s += "\n"

    s += "   "
    while i < go.N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1

    s += "\n"
    print(s)


def get(position):
    position.input_board()

    # print("=====================")
    # print(position.text())
    # print_input()

    x = None
    y = None
    if torch.cuda.is_available():
        x = Variable(go.INPUT_BOARD).cuda()
        y = halo_resnet(x).data.cpu().numpy()
    else:
        x = Variable(go.INPUT_BOARD)
        y = halo_resnet(x).data.numpy()

    positions = []

    pos = go.POSITION_POOL.pop()
    position.move2(pos, 0)
    pos.prior = -1000000 #y[0, go.LN]
    positions.append(pos)

    i = 0
    while i < go.LN:
        v = go.COORDS[i]
        if go.FLAG_BOARD[v]:
            pos = go.POSITION_POOL.pop()
            position.move2(pos, v)
            pos.prior = y[0, i]
            positions.append(pos)
        i += 1

    positions = sorted(positions, key=lambda pos: pos.prior)

    return positions


def sim():
    go.SIM_POS.input_board()

    best_move = 0
    best_score = -1000000

    # if has_resonable:
    x = None
    y = None
    if torch.cuda.is_available():
        x = Variable(go.INPUT_BOARD).cuda()
        y = halo_resnet(x).data.cpu().numpy()
    else:
        x = Variable(go.INPUT_BOARD)
        y = halo_resnet(x).data.numpy()

    i = 0
    while i < go.LN:
        v = go.COORDS[i]
        if go.FLAG_BOARD[v]:
            score = y[0, i]
            if score > best_score:
                best_score = score
                best_move = v
        i += 1

    # score = y[0, go.LN]
    # if score > best_score:
    #     best_score = score
    #     best_move = 0

    go.SIM_POS.move2(go.SIM_POS, best_move)
    go.HASH_SIM[go.SIM_POS.hash_code] = 0

    return go.SIM_POS.hash_code


criterion = None
if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
optimizer = optim.SGD(halo_resnet.parameters(), lr=0.000001)


def train(position, best_move):
    position.input_board()

    target_data = torch.zeros(1, go.LN + 1)
    if best_move == 0:
        target_data[0, go.LN] = 1
    else:
        j, i = go.toXY(best_move)
        j -= 1
        i -= 1
        v = i * go.N + j
        target_data[0, v] = 1

    optimizer.zero_grad()

    x = None
    y = None
    target = None
    if torch.cuda.is_available():
        x = Variable(go.INPUT_BOARD).cuda()
        target = Variable(target_data).cuda()
    else:
        x = Variable(go.INPUT_BOARD)
        target = Variable(target_data)

    y = halo_resnet(x)

    loss = criterion(y, target)
    loss.backward()
    optimizer.step()
