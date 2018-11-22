#!/usr/bin/python

import sys
import torch
from torch.autograd import Variable
import go
import resnet

def main(path):
    policy = resnet.Policy(0.5, path)
    dummy_input = Variable(torch.randn(1, 1, go.N, go.N))
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    torch.onnx.export(policy.resnet, dummy_input, "../data/goai.onnx")

if __name__ == '__main__':
    path = 'goai.pth'
    if len(sys.argv) >= 2:
        path = sys.argv[1]

    main('../data/'+path)