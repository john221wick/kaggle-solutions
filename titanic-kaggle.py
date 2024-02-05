import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

X_train = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1).values.astype(np.float32)
y_train = train_df['Survived'].values.astype(np.float32)


test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

X_test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1).values.astype(np.float32)


X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).view(-1, 1)

X_test_tensor = torch.tensor(X_test)


mean = X_train_tensor.mean(0, keepdim=True)
std = X_train_tensor.std(0, unbiased=False, keepdim=True)
std[std == 0] = 1
X_train_tensor = (X_train_tensor - mean) / std
X_test_tensor = (X_test_tensor - mean) / std  # Use training mean and std for normalization


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


class titanic_predictor(nn.Module):
    def __init__(self, num_features):
        super(titanic_predictor, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


model = titanic_predictor(X_train_tensor.shape[1])


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, train_loader, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


train_model(model, criterion, optimizer, train_loader, epochs=100)

model.eval() 
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    probabilities = torch.sigmoid(test_outputs)
    predictions = (probabilities >= 0.5).int()  # Convert probabilities to 0 or 1

submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions.view(-1).numpy()  
})


submission_path = "/kaggle/working/submission.csv"  
submission_df.to_csv(submission_path, index=False)
print("Your submission was successfully saved!")