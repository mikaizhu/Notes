import torch
from pathlib import Path

class Trainer:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")
        self.model = kwargs.get("model").to(self.device)
        self.optimizer = kwargs.get("optimizer")
        self.scheduler = kwargs.get("scheduler")
        self.criterion = kwargs.get("criterion")
        self.epochs = kwargs.get("epochs")
        self.train_loader = kwargs.get("train_loader")
        self.test_loader = kwargs.get("test_loader")
        self.logger = kwargs.get('logger')
        self.model_save_name = kwargs.get('model_save_name')

    def train_step(self):
        self.model.train()
        self.train_loss = 0
        self.train_acc_num = 0
        for feature, label in self.train_loader:
            feature = feature.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(feature)
            loss = self.criterion(preds, label)
            loss.backward()
            self.optimizer.step()
            self.train_acc_num += (preds.argmax(1) == label).sum()
            self.train_loss += loss.item() / len(self.train_loader)
        return self.train_loss, self.train_acc_num

    def test_step(self):
        self.model.eval()
        self.test_loss = 0
        self.test_acc_num = 0
        with torch.no_grad():
            for feature, label in self.test_loader:
                feature = feature.to(self.device)
                label = label.to(self.device)
                preds = self.model(feature)
                self.test_loss += self.criterion(preds, label) / len(self.test_loader)
                self.test_acc_num += ((preds.argmax(1) == label).sum())
        return self.test_loss, self.test_acc_num

    def train(self):
        self.best_score = -float('inf')
        for epoch in range(self.epochs):
            train_loss, train_acc_num = self.train_step()
            test_loss, test_acc_num = self.test_step()
            self.scheduler.step(test_loss)
            self.logger.info(f'Epoch:{epoch:2} | Train Loss:{train_loss:6.4f} | Train Acc:{train_acc_num/len(self.train_loader.dataset):6.4f} | Test Loss:{test_loss:6.4f} | Test Acc:{test_acc_num/len(self.test_loader.dataset):6.4f}')
            score = test_acc_num / len(self.test_loader.dataset)
            if score > self.best_score:
                self.best_score = score
                self.model_save(self.model_save_name)


    def model_save(self, file_name):
        path = Path('./model')
        if not path.exists():
            path.mkdir()
        path = str(path / Path(file_name))
        torch.save(self.model.state_dict(), path)
