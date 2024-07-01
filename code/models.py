
class CNNCifar(torch.nn.Module):
    
    def __init__(self, num_channels=3, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
    
        self.fc1 = nn.Linear(64*26*26, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'conv3': self.conv3, 'fc1': self.fc1, 'fc2': self.fc2}

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
                    