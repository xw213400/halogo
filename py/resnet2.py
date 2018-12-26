#!/usr/bin/python

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
# from torch.jit import ScriptModule, script_method, trace

MOVES = [0] * go.LN
INPUT_BOARD = torch.zeros(1, 2, go.N, go.N)


def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1, bias=False)


class Resnet(nn.Module):
    def __init__(self, num_planes):
        super(Resnet, self).__init__()

        self.entryblock = conv3x3(2, num_planes)
        self.rbconv = conv3x3(num_planes, num_planes)
        self.classifier = nn.Conv2d(num_planes, 4, 1, stride=1)
        self.fc = nn.Linear(go.LN * 4, go.LN + 1)

    def forward(self, x):
        out = self.entryblock(x)
        out = F.relu(self.rbconv(out) + out)
        out = F.relu(self.rbconv(out) + out)
        out = F.relu(self.rbconv(out) + out)
        out = F.relu(self.rbconv(out) + out)
        # out = F.relu(self.rbconv(out) + out)
        # out = F.relu(self.rbconv(out) + out)
        # out = F.relu(self.rbconv(out) + out)
        # out = F.relu(self.rbconv(out) + out)
        out = self.classifier(out)
        out = out.view(-1, go.LN * 4)
        out = self.fc(out)

        return out

# # cpp version module
# class Resnet(ScriptModule):
#     def __init__(self, num_planes):
#         super(Resnet, self).__init__()

#         self.entryblock = trace(conv5x5(1, num_planes), torch.rand(1, 1, go.N, go.N))
#         self.rbconv = trace(conv5x5(num_planes, num_planes), torch.rand(1, num_planes, go.N, go.N))
#         self.classifier = trace(nn.Conv2d(num_planes, 4, 1, stride=1), torch.rand(1, 32, go.N, go.N))
#         self.fc = trace(nn.Linear(go.LN * 4, go.LN + 1), torch.rand(324))

#     @script_method
#     def forward(self, x):
#         out = self.entryblock(x)
#         out = F.relu(self.rbconv(out) + out)
#         out = F.relu(self.rbconv(out) + out)
#         out = F.relu(self.rbconv(out) + out)
#         out = F.relu(self.rbconv(out) + out)
#         out = F.relu(self.rbconv(out) + out)
#         out = self.classifier(out)
#         out = out.view(-1, go.LN * 4)
#         out = self.fc(out)

#         return out


class Policy():
    def __init__(self, PUCT=0.5, pars='../data/goai.pth'):
        self.PUCT = PUCT
        self.HASH = {}
        self.resnet = Resnet(96)
        if torch.cuda.is_available():
            self.resnet.cuda()
        if os.path.isfile(pars):
            self.resnet.load_state_dict(torch.load(pars))

    # prepare input plane for resnet
    # INPUT_BOARD[0]: board:1, ko: -1
    # INPUT_BOARD[1]: enemy:-1, empty:0, self:1
    def input_board(self, position):
        global INPUT_BOARD

        i = 0
        x = 0
        y = 0
        while i < go.LN:
            v = go.COORDS[i]
            c = position.board[v]

            if v == position.ko:
                INPUT_BOARD[0, 0, y, x] = -1
            else:
                INPUT_BOARD[0, 0, y, x] = 1
            
            INPUT_BOARD[0, 1, y, x] = c * position.next

            x += 1
            if x == go.N:
                y += 1
                x = 0
            i = y * go.N + x

    def get(self, position):
        self.input_board(position)

        x = None
        out = None
        if torch.cuda.is_available():
            x = Variable(INPUT_BOARD).cuda()
            out = self.resnet(x)[0].data.cpu().numpy()
        else:
            x = Variable(INPUT_BOARD)
            out = self.resnet(x)[0].data.numpy()

        position.update_group()
        positions = [position.move(0)]

        cs = np.argsort(out)

        for c in cs:
            if c < go.LN:
                v = go.COORDS[c]
                pos = position.move(v)
                if pos is not None:
                    positions.append(pos)

        return positions

    def sim(self, position):
        global MOVES

        score = self.HASH.get(position.hash_code)
        if score is not None:
            return score

        pos = position

        while pos.pass_count() < 2:
            self.input_board(pos)

            x = None
            out = None
            if torch.cuda.is_available():
                x = Variable(INPUT_BOARD).cuda()
                out = self.resnet(x)[0].data.cpu().numpy()
            else:
                x = Variable(INPUT_BOARD)
                out = self.resnet(x)[0].data.numpy()

            cs = np.argsort(out)
            ppp = [None]
            pos.update_group()

            i = go.LN
            while i >= 0:
                c = cs[i]
                if c < go.LN:
                    v = go.COORDS[c]
                    ppp = pos.move(v)
                    if ppp is not None:
                        break
                i -= 1

            if ppp is None:
                pos = pos.move(0)
            else:
                pos = ppp

        score = pos.score()
        while pos is not position:
            pos.release()
            pos = pos.parent

        self.HASH[position.hash_code] = score

        return score

    def clear(self):
        self.HASH = {}

    def train(self, trainset, estimset, epoch):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.resnet.parameters(), lr=0.001, momentum=0.9)
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

        for e in range(epoch):
            batch = 10
            n = math.floor(len(trainset) / batch)
            i = 0
            running_loss = 0.0
            random.shuffle(trainset)
            while i < n:
                j = 0
                target_data = torch.LongTensor(batch)
                input_data = torch.zeros(batch, 2, go.N, go.N)
                while j < batch:
                    k = i * batch + j
                    pos = trainset[k]
                    v = go.LN
                    if pos.vertex != 0:
                        p, q = go.toJI(pos.vertex)
                        v = q * go.N + p - go.N - 1

                    target_data[j] = v

                    self.input_board(pos.parent)
                    input_data[j].copy_(INPUT_BOARD[0])

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
                running_loss += loss.item()

            rights = self.test(estimset)
            result = ''
            right = 0
            for r in range(10):
                right += rights[r]
                result += '%d:%.1f, ' % (r, right / len(estimset) * 100)

            print('epoch: %d, loss %.3f, Est: %s' % (e, running_loss / n, result))
            torch.save(self.resnet.state_dict(), '../data/goai_%d.pth' % e)

    def test(self, estimset):
        rights = [0] * 10
        for pos in estimset:
            v = go.LN
            if pos.vertex != 0:
                p, q = go.toJI(pos.vertex)
                v = q * go.N + p - go.N - 1

            self.input_board(pos.parent)

            y = None
            if torch.cuda.is_available():
                x = Variable(INPUT_BOARD).cuda()
                y = self.resnet(x).data.cpu()
            else:
                x = Variable(INPUT_BOARD)
                y = self.resnet(x).data

            predicted = np.argsort(y[0].numpy())
            prediction = predicted[::-1]

            i = 0
            while i < 10:
                if v == prediction[i]:
                    rights[i] += 1
                    break
                i += 1

        return rights


def print_input(self):
    i = go.N
    s = "\n"
    while i > 0:
        s += str(i).zfill(2) + " "
        i -= 1
        j = 0
        while j < go.N:
            s += str(int(INPUT_BOARD[0, 0, i, j])) + " "
            j += 1
        s += "\n"

    s += "   "
    while i < go.N:
        s += "{} ".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[i])
        i += 1

    s += "\n"
    print(s)
