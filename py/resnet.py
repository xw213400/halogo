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
        
        self.classifier = nn.Conv2d(num_planes, 1, 1, stride=1)

    def forward(self, x):
        out = self.entry_block(x)
        out = self.residual_layers(out)
        out = self.classifier(out)

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
            s += str(int(go.INPUT_BOARD[0,0,i,j])) + " "
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
            s += str(int(go.INPUT_BOARD[0,1,i,j])) + " "
            j += 1
        s += "\n"

    s += "   "
    while i < go.N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1
        
    s += "\n"
    print(s)

def get(position):
    for v in go.COORDS:
        j, i = go.toXY(v)
        j -= 1
        i -= 1
        p = 1
        go.FLAG_BOARD[v] = False
        if position.resonable(v) and position.move2(go.MOVE_POS, v):
            p = 2
            go.FLAG_BOARD[v] = True
        go.INPUT_BOARD[0,0,i,j] = position.board[v] + 2
        go.INPUT_BOARD[0,1,i,j] = p

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
    pos.prior = -1000000
    positions.append(pos)

    for v in go.COORDS:
        j, i = go.toXY(v)
        j -= 1
        i -= 1
        if go.FLAG_BOARD[v]:
            pos = go.POSITION_POOL.pop()
            position.move2(pos, v)
            pos.prior = y[0, 0, i, j]
            positions.append(pos)

    positions = sorted(positions, key=lambda pos:pos.prior)
    
    return positions

def sim():
    has_resonable = False
    for v in go.COORDS:
        j, i = go.toXY(v)
        j -= 1
        i -= 1
        p = 1
        go.FLAG_BOARD[v] = False
        if go.SIM_POS.resonable(v) and go.SIM_POS.move2(go.MOVE_POS, v):
            p = 2
            has_resonable = True
            if go.MOVE_POS.hash_code not in go.HASH_SIM:
                go.FLAG_BOARD[v] = True

        go.INPUT_BOARD[0,0,i,j] = go.SIM_POS.board[v] + 2
        go.INPUT_BOARD[0,1,i,j] = p

    best_move = 0
    best_score = -1000000
    if has_resonable:
        x = None
        y = None
        if torch.cuda.is_available():
            x = Variable(go.INPUT_BOARD).cuda()
            y = halo_resnet(x).data.cpu().numpy()
        else:
            x = Variable(go.INPUT_BOARD)
            y = halo_resnet(x).data.numpy()

        for v in go.COORDS:
            j, i = go.toXY(v)
            j -= 1
            i -= 1
            if go.FLAG_BOARD[v]:
                score = y[0, 0, i, j]
                if score > best_score:
                    best_score = score
                    best_move = v

    go.SIM_POS.move2(go.SIM_POS, best_move)
    go.HASH_SIM[go.SIM_POS.hash_code] = 0

    return go.SIM_POS.hash_code

criterion = None
if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
optimizer = optim.SGD(halo_resnet.parameters(), lr=0.001)

def train(position, best_move):
    for v in go.COORDS:
        j, i = go.toXY(v)
        j -= 1
        i -= 1
        p = 2 if position.resonable(v) and position.move2(go.MOVE_POS, v) else 1
        go.INPUT_BOARD[0,0,i,j] = position.board[v] + 2
        go.INPUT_BOARD[0,1,i,j] = p

    target_data = torch.zeros(1, 1, go.N, go.N)
    j, i = go.toXY(best_move)
    j -= 1
    i -= 1
    target_data[0, 0, i, j] = 1

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
