import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms


class ResNet18(nn.Module):

    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


batch_size = 32

# Define the image transformations
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the images from the directories
train_dataset = datasets.ImageFolder(root="D:/Mihail/Resized 224/Train", transform=transform)
val_dataset = datasets.ImageFolder(root="D:/Mihail/Resized 224/Validation", transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def train_model(model, criterion, optimizer, num_epochs):

    # GPU only
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)


    for epoch in range(num_epochs):

        start_time = time.time()

        train_acc, train_loss, val_acc, val_loss = [], [], [], []

        # Train the model
        iter = 0
        running_acc = 0
        running_loss = 0
        data_size = 0

        model.train()
        
        for inputs, labels in train_loader:

            iter = iter + 1
            # GPU only
            inputs, labels = inputs.to(device), labels.to(device)
          
            # Clear the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Evaluation
            _, predicted = torch.max(outputs.data, 1)
            running_acc += torch.sum(predicted == labels)
            running_loss += loss.item() * inputs.size(0) 

            data_size += inputs.size(0)

            print(int((iter*batch_size)/12138*100), "% complet")

        epoch_acc = running_acc / data_size
        epoch_loss = running_loss / data_size

        train_acc.append(int(epoch_acc))
        train_loss.append(epoch_loss)

        # Evaluate the model
        iter = 0
        running_acc = 0
        running_loss = 0
        data_size = 0

        model.eval()

        with torch.no_grad():
            for inputs, labels in val_loader:

                iter = iter + 1
                # GPU only
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)

                # Evaluation
                _, predicted = torch.max(outputs.data, 1)
                running_acc += torch.sum(predicted == labels)
                running_loss += loss.item() * inputs.size(0) 

                data_size += inputs.size(0)   

                print(int((iter*batch_size)/315*100), "% complet")
        
        epoch_acc = running_acc / data_size
        epoch_loss = running_loss / data_size
        val_acc.append(epoch_acc)
        val_loss.append(epoch_loss)


        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")


        torch.save(model.state_dict(), 'D:/Mihail/m_'+str(epoch+1)+'_'+str(epoch_loss)[:5]+'_'+str(epoch_acc)[7:12]+'.pth')

    # Save the accuracy and loss
    # np.save("D:/Mihail/train_acc.npy", train_acc)
    # np.save("D:/Mihail/train_loss.npy", train_loss)
    # np.save("D:/Mihail/val_acc.npy", val_acc)
    # np.save("D:/Mihail/val_loss.npy", val_loss)


model = ResNet18()

# GPU only
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

num_epochs = 50

train_model(model, criterion, optimizer, num_epochs)