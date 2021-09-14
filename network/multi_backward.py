class MultiBackwardBak:
    def __init__(self, optimizer, batch_size):
        self.all_loss = []
        self.optimizer = optimizer
        self.batch_size = batch_size
    def backward(self):
        if len(self.all_loss) > 0:
            mean_loss = sum(self.all_loss) / len(self.all_loss)
            mean_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.all_loss = []
    def add_loss(self, loss):
        if len(self.all_loss) == 0:
            self.optimizer.zero_grad()
        if len(self.all_loss) == self.batch_size:
            self.backward()
        self.all_loss.append(loss)
    def __del__(self):
        if len(self.all_loss) > 0:
            self.backward()

class MultiBackward:
    def __init__(self, optimizer, batch_size):
        self.all_loss = []
        self.optimizer = optimizer
        self.batch_size = batch_size
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.all_loss = []
    def add_loss(self, loss):
        if len(self.all_loss) == 0:
            self.optimizer.zero_grad()
        loss.backward()
        if len(self.all_loss) == self.batch_size:
            self.step()
        self.all_loss.append(loss)
    def __del__(self):
        if len(self.all_loss) > 0:
            self.step()


