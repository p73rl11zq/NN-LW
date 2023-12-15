import random

import torch
import torch.optim as optim
from torch.autograd import Variable


def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    endpoints = []
    start = 0
    for stop in range(0, len(args[0]), n):
        if stop - start > 0:
            endpoints.append((start, stop))
            start = stop
    random.shuffle(endpoints)
    for start, stop in endpoints:
        yield [a[start:stop] for a in args]


class Wrapper:
    def __init__(self, model, cuda=True, epochs=200, batchsize=4096):
        self.batchsize = batchsize
        self.epochs = epochs
        self.cuda = cuda
        self.model = model
        if cuda:
            self.device =  ('cuda' if torch.cuda.is_available() else 'cpu')
            print(self.device)
            self.model.to(self.device)
        # TODO: добавьте оптимизатор
        self.optimizer = optim.Adam(model.parameters(), lr=1e-1)

    def fit(self, *args):
        self.model.train()
        if self.cuda:
            self.model.to(self.device)
        for epoch in range(self.epochs):
            total = 0.0
            for itr, datas in enumerate(chunks(self.batchsize, *args)):
                datas = [Variable(torch.from_numpy(data)) for data in datas]
                if self.cuda:
                    datas = [data.to(self.device) for data in datas]
                # TODO: вычислите логиты модели и градиенты от datas
                # datas = [pij, i, j]
                # pij - значения сходства между точками данных
                # i, j - индексы точек
                self.optimizer.zero_grad()
                loss = self.model(*datas)
                loss.backward()
                self.optimizer.step()
                total += loss.data
            if epoch % (self.epochs //5) == 0:
                msg = 'Train Epoch: {} \tLoss: {:.3e}'
                msg = msg.format(epoch, total / (len(args[0]) * 1.0))
                print(msg)