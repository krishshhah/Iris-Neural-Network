import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

column_names = [
    'area',
    'perimeter',
    'compactness',
    'kernel_length',
    'kernel_width',
    'asymmetry_coefficient',
    'groove_length',
    'class' # 3 classes: 1,2,3
]

# Load the dataset from the URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
df = pd.read_csv(url, sep='\s+', header=None, names=column_names)

# Display the first few rows of the DataFrame
print(df.head())

df['class'] = df['class'].replace(1, 0)
df['class'] = df['class'].replace(2, 1)
df['class'] = df['class'].replace(3, 2)
print(df.head())

# classes have been set to start from 0, for index purposes

df.describe()

# Create a model class to inherit the nn.Module

class Model(nn.Module):
  # input layer contains 4 inputs of the model
  # 1st hidden layer (l1) will have 10 neurons
  # 2nd hidden layer (l2) will have 10 neurons
  # output layer has 3 neurons to pick a flower

  def __init__(self, in_features=7, L1=20, L2=20, L3=20, out_features=3):
    super().__init__() # inherites from the superclass (nn.Module)
    self.fc1 = nn.Linear(in_features, L1) #fc1 = fully connected
    self.fc2 = nn.Linear(L1, L2) # stars with input features, moving FORWARD to the next stage
    self.fc3 = nn.Linear(L2, L3) # stars with input features, moving FORWARD to the next stage
    self.out = nn.Linear(L2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x)) # relu = rectified linear unit. If output<0, it is 0. if >0, then use the output
    x = F.relu(self.fc2(x)) # moves the object forward
    x = F.relu(self.fc3(x)) # moves the object forward
    x = F.relu(self.out(x))

    return x

# Create random seed for randomisation
torch.manual_seed(41)
model = Model() # creates an instance

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(10, 6))
plt.scatter(df['class'], df['area'], alpha=0.6, edgecolors='k')
plt.title('Class vs. Area')
plt.xlabel('Class')
plt.ylabel('Area')
plt.xticks([0, 1, 2], ['Class 1', 'Class 2', 'Class 3'])  # Adjust y-axis ticks for clarity
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['class'], df['perimeter'], alpha=0.6, edgecolors='k')
plt.title('Class vs. Perimeter')
plt.xlabel('Class')
plt.ylabel('Perimeter')
plt.xticks([0, 1, 2], ['Class 1', 'Class 2', 'Class 3'])  # Adjust y-axis ticks for clarity
plt.grid()
plt.show()

#train, test, split: set features (X) and targets (y)
X = df.drop('class', axis = 1) # selects only the targets, axis (columns)
y = df['class']

#convert to numpy arrays
X = X.values
y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41) #80% train, 20% test

X_train = torch.FloatTensor(X_train) # converts numpy arrays to floatTensors (all features are decimals)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train) # converts numpy arrays to longTensors (all targets are integers)
y_test = torch.LongTensor(y_test)

# set a criterion of the model to measure the error, to measure how far off the predictions are from the actual data
criterion = nn.CrossEntropyLoss()
# choose an optimiser - using Adam optimiser, lr (learning rate) = if learning rate does not go down through each iteration, it should be lowered to learn slower
optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # lower learning rate takes longer to train the model
#model.parameters() are the layers: fc1, fc2, fc3, out


# Train model, determining how many epochs needed. Epoch = 1 run through the entire network
epoch = 1000
losses = [] # to keep track of the losses to track progress
for i in range(epoch):
  # Go forward and get prediciton
  y_pred = model.forward(X_train) # using the features to move forward and get predicitive results

  # Measure the loss
  loss = criterion(y_pred, y_train) #predicted value vs y-train value
  losses.append(loss.detach().numpy())

  if i % 100 == 0:
    print(f'{epoch}: {i} and the loss is {loss}')

  # Back propogation = take the error rate of the forward propogation and feed it back through the neural network to tweak the weights
  optimiser.zero_grad() # Clears the old gradient values, ensuring gradients from the previous iterations don’t accumulate.
  loss.backward() # Performs backpropagation to calculate the new gradients of the loss with respect to each parameter (weights)
  optimiser.step() # Uses the calculated gradients to update the model's parameters, moving them toward values that minimize the loss

# Graphing out the losses
plt.plot(range(epoch), losses) # y-axis = losses
plt.ylabel("Loss/error")
plt.xlabel("Epoch")
plt.show()

# Evaluate deep learning results on our test data
with torch.no_grad():
   #turns off back propogation
   y_eval = model.forward(X_test) # testing the model on the new dataset, features from test set on predictions
   loss = criterion(y_eval, y_test) # find the loss error

loss

correct = 0
diff = []
with torch.no_grad():
  for i, data in enumerate(X_test): #features test set
    y_val = model.forward(data) #outcome of testing

    # tells us what type of flower out network thinks it is

    print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}') # compares feature test to target test

    # correct or not
    if y_val.argmax().item() == y_test[i]: # if the index of the max tensor in the feature test == target test value
      correct +=1

    x = abs((y_val.argmax().item() - y_test[i]))
    diff.append(x)

print(f'We got {correct} correct!. Accuracy: {correct}/{len(y_test)} ({((correct/y_test.size(0))*100):.2f}%)')
print(f'The mean difference between predicted and actual quality: {sum(diff) / len(diff):.4f}')

# Allows a new alcohol to be passed through the model to guess what the quality is
def testing(area, perimeter, compactness, kernel_length, kernel_width, asymmetry_coefficient, groove_length):
  new_seed = torch.tensor([area, perimeter, compactness, kernel_length, kernel_width, asymmetry_coefficient, groove_length])
  with torch.no_grad(): # without back propogation
    z = model(new_seed)
    z_quality = z.argmax().item()  # max index in tensor
    print(z)
    print(f'Class guess is: {z_quality} => ({z_quality+1})')


testing(14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956)

# saving the model as a dictionary
torch.save(model.state_dict(), 'my_seeds_nn.pt')

# loading the saved model
new_model = Model()
new_model.load_state_dict(torch.load('my_seeds_nn.pt'))

# making sure it loaded successfully (checking instance variables)
new_model.eval()
