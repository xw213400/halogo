import numpy as np
import go
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os.path
import random
import math

MOVES = [0] * go.LN

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
            # Residual_block(num_planes, num_planes),
            # Residual_block(num_planes, num_planes),
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


class Policy():
    def __init__(self, PUCT=1, pars='../data/resnet_pars.pkl'):
        self.PUCT = PUCT
        self.resnet = Resnet(32)
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.resnet.cuda()
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.SGD(self.resnet.parameters(), lr=0.001, momentum=0.9)
        if os.path.isfile(pars):
            self.resnet.load_state_dict(torch.load(pars))

    def get(self, position):
        position.input_board()

        x = None
        out = None
        if torch.cuda.is_available():
            x = Variable(go.INPUT_BOARD).cuda()
            out = self.resnet(x).data.cpu().numpy()
        else:
            x = Variable(go.INPUT_BOARD)
            out = self.resnet(x).data.numpy()

        positions = []

        pos = position.move(0)
        pos.prior = out[0, go.LN]
        positions.append(pos)

        c = 0
        x = 0
        y = 0
        while c < go.LN:
            if go.INPUT_BOARD[0, 1, y, x] == 1:
                v = go.COORDS[c]
                pos = position.move(v)
                pos.prior = out[0, c]
                positions.append(pos)

            x += 1
            if x == go.N:
                y += 1
                x = 0
            c = y * go.N + x

        positions = sorted(positions, key=lambda pos: pos.prior)

        return positions

    def sim(self, position):
        global MOVES
        position.input_board()

        x = None
        y = None
        if torch.cuda.is_available():
            x = Variable(go.INPUT_BOARD).cuda()
            y = self.resnet(x).data.cpu().numpy()
        else:
            x = Variable(go.INPUT_BOARD)
            y = self.resnet(x).data.numpy()

        cs = np.argsort(y[0])
        move = 0
        n = 0
        i = go.LN

        while i >= 0:
            c = cs[i]
            if c < go.LN:
                x, y = go.toXY(c)
                if go.INPUT_BOARD[0, 1, y, x] == 1:
                    MOVES[n] = go.COORDS[c]
                    n += 1
            i -= 1

        if n > 0:
            r = random.random()
            r = int(r * r * n)
            move = MOVES[r]
        
        pos = position.move(move)

        return pos

    def train(self, positions, epoch=1):
        for e in range(epoch):
            batch = 10
            n = math.floor(len(positions)/batch)
            i = 0
            running_loss = 0.0
            while i < n:
                j = 0
                target_data = torch.LongTensor(batch)
                input_data = torch.zeros(batch, 2, go.N, go.N)
                while j < batch:
                    k = i * batch + j
                    pos = positions[k]
                    v = go.LN
                    if pos.vertex != 0:
                        p, q = go.toJI(pos.vertex)
                        v = q * go.N + p - go.N - 1

                    target_data[j] = v

                    pos.parent.input_board()
                    input_data[j].copy_(go.INPUT_BOARD[0])

                    j += 1

                self.optimizer.zero_grad()

                x = Variable(input_data)
                t = Variable(target_data)
                if torch.cuda.is_available():
                    x = x.cuda()
                    t = t.cuda()

                y = self.resnet(x)

                loss = self.criterion(y, t)
                loss.backward()
                self.optimizer.step()

                i += 1
                running_loss += loss.data[0]
                if i % 100 == 0 or i == n:
                    print('epoch: %d, i:%d, loss %.3f' % (e, i*batch, running_loss / 100))
                    running_loss = 0.0

    def test(self, positions):
        right = 0
        for pos in positions:
            v = go.LN
            if pos.vertex != 0:
                p, q = go.toJI(pos.vertex)
                v = q * go.N + p - go.N - 1

            pos.parent.input_board()

            y = None
            if torch.cuda.is_available():
                x = Variable(go.INPUT_BOARD).cuda()
                y = self.resnet(x).data.cpu()
            else:
                x = Variable(go.INPUT_BOARD)
                y = self.resnet(x).data

            _, predicted = torch.max(y, 1)

            if v == predicted[0]:
                right += 1
        
        n = len(positions)
        print('Right:%d, N:%d, Rate: %.1f' % (right, n, right/n*100))

