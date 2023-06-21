import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

torch.manual_seed(0)


class MyDataLoader:
    def __init__(self, *file_paths):
        self.file_paths = file_paths
        self.data = []

    def load_data(self):
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                data = list(csv.reader(file))
                self.data.append(np.array(data, float))
        return [torch.from_numpy(data.T).to(torch.float32) for data in self.data]


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(420, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyTrainer:
    def __init__(self, model, input_data, input_data2, test_data, num_epochs=6000, learning_rate=0.001, patience=10):
        self.model = model
        self.input_data = input_data
        self.input_data2 = input_data2
        self.test_data = test_data
        self.num_epochs = num_epochs
        self.patience = patience
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.best_loss = np.inf
        self.early_stopping_counter = 0
        self.current_loss = None

    def train(self):
        for epoch in range(self.num_epochs):
            output_data = self.model(self.input_data)
            loss = self.criterion(output_data, self.input_data2)
            self.current_loss = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {self.current_loss}")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            output_data = self.model(self.test_data)
        return output_data


if __name__ == "__main__":
    file_paths = ["Prb_data/Prb02data01.csv", "Prb_data/Prb02data02.csv", "Prb_data/Prb02data03.csv"]
    data_loader = MyDataLoader(*file_paths)
    input_data, input_data2, test_data = data_loader.load_data()

    model = MyModel()
    trainer = MyTrainer(model, input_data, input_data2, test_data, learning_rate=0.001, patience=20)
    trainer.train()
    print("Final Loss:", trainer.current_loss)

    test_output = trainer.test()

    print("Test Output Size:", test_output.size())
    print("Test Output:", test_output)