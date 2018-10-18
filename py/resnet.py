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

        self.entry_block = conv5x5(1, num_planes)

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
        self.HASH = {}
        self.resnet = Resnet(32)
        if torch.cuda.is_available():
            self.resnet.cuda()
        if os.path.isfile(pars):
            self.resnet.load_state_dict(torch.load(pars))

    def get(self, position):
        position.input_board()

        x = None
        out = None
        if torch.cuda.is_available():
            x = Variable(go.INPUT_BOARD).cuda()
            out = self.resnet(x)[0].data.cpu().numpy()
        else:
            x = Variable(go.INPUT_BOARD)
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
            pos.input_board()

            x = None
            out = None
            if torch.cuda.is_available():
                x = Variable(go.INPUT_BOARD).cuda()
                out = self.resnet(x)[0].data.cpu().numpy()
            else:
                x = Variable(go.INPUT_BOARD)
                out = self.resnet(x)[0].data.numpy()

            cs = np.argsort(out)
            ppp = None
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

    def train(self, positions, epoch=1):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.resnet.parameters(), lr=0.001, momentum=0.9)
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

        for e in range(epoch):
            batch = 10
            n = math.floor(len(positions)/batch)
            i = 0
            running_loss = 0.0
            random.shuffle(positions)
            while i < n:
                j = 0
                target_data = torch.LongTensor(batch)
                input_data = torch.zeros(batch, 1, go.N, go.N)
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

