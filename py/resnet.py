import numpy as np
import go
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os.path
import random


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


class Policy():
    def __init__(self, PUCT=1, pars='../data/resnet_pars.pkl'):
        self.PUCT = PUCT
        self.resnet = Resnet(30)
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.resnet.cuda()
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.SGD(self.resnet.parameters(), lr=0.001, momentum=0.9)
        if os.path.isfile(pars):
            self.resnet.load_state_dict(torch.load(pars))

    def get(self, position):
        position.input_board()

        x = None
        y = None
        if torch.cuda.is_available():
            x = Variable(go.INPUT_BOARD).cuda()
            y = self.resnet(x).data.cpu().numpy()
        else:
            x = Variable(go.INPUT_BOARD)
            y = self.resnet(x).data.numpy()

        positions = []

        pos = position.move(0)
        pos.prior = y[0, go.LN]
        positions.append(pos)

        i = 0
        while i < go.LN:
            v = go.COORDS[i]
            if go.FLAG_BOARD[v]:
                pos = position.move(v)
                pos.prior = y[0, i]
                positions.append(pos)
            i += 1

        positions = sorted(positions, key=lambda pos: pos.prior)

        return positions

    def sim(self, position):
        position.input_board()

        x = None
        y = None
        if torch.cuda.is_available():
            x = Variable(go.INPUT_BOARD).cuda()
            y = self.resnet(x).data.cpu().numpy()
        else:
            x = Variable(go.INPUT_BOARD)
            y = self.resnet(x).data.numpy()

        i = 0
        best_score = go.WORST_SCORE
        sum_score = 0
        move = 0
        moves = []
        while i < go.LN:
            v = go.COORDS[i]
            if go.FLAG_BOARD[v]:
                score = y[0, i]
                if score > 0:
                    sum_score += score
                    moves.append((v, sum_score))
                elif best_score < score:
                    best_score = score
                    move = v
            i += 1

        if sum_score > 0:
            n = len(moves)
            if n > 1:
                rand = random.random() * sum_score
                for v, s in moves:
                    if s >= rand:
                        move = v
                        break
            else:
                move, s = moves[0]

        pos = position.move(move)

        return pos

    def train(self, positions):
        n = len(positions)
        k = 0
        c = 0
        while k + c < n:
            pos = positions[k+c]
            v = None

            if pos.vertex == 0:
                c += 1
                v = go.LN
            else:
                k += 1
                j, i = go.toXY(pos.vertex)
                j -= 1
                i -= 1
                v = i * go.N + j

                ###########################
                target_data = torch.zeros(1, go.LN + 1)
                target_data[0, v] = 1
                self.optimizer.zero_grad()

                x = None
                y = None
                target = None
                pos.input_board()
                if torch.cuda.is_available():
                    x = Variable(go.INPUT_BOARD).cuda()
                    target = Variable(target_data).cuda()
                else:
                    x = Variable(go.INPUT_BOARD)
                    target = Variable(target_data)

                y = self.resnet(x)

                loss = self.criterion(y, target)
                loss.backward()
                self.optimizer.step()
                ###########################

            if k % 100 == 0 or k + c == n:
                print('%d: loss %.3f' % (k, loss))
