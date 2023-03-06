import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from Parameters import *


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


def classify(params:Parameters, detections, scores, file_names):

    threshold = params.threshold_task2

    new_detections = [[] for k in range(4)]
    new_scores = [[] for k in range(4)]
    new_file_names = [[] for k in range(4)]
    model = ResNet18()
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Lambda(lambda x: x.unsqueeze(0))])


    model.load_state_dict(torch.load(params.dir_models+'/task2_resnet18.pth', map_location=torch.device('cpu')))

    softmax = torch.nn.Softmax(dim = 0)

    for k in range(len(detections)):
        
        print("Classifing face ", k+1, " of ", len(detections))

        x_min, y_min, x_max, y_max = detections[k]
        img = Image.open(os.path.join(params.dir_test_examples, file_names[k]))
        img = img.crop((x_min, y_min, x_max, y_max))
        img = img.resize((224, 224))
        img = transform(img)
        output = model(img).flatten()
        output = softmax(output)
        output = output.detach().numpy()
                
        i = np.argmax(output)

        if output[i] > threshold:

            new_detections[i].append(detections[k])
            new_scores[i].append(scores[k])
            new_file_names[i].append(file_names[k])


    prediction = ["andy", "louie", "ora", "tommy"]

    for k in range(4):
        np.save(params.dir_solutions+"/task2/"+"detections_"+prediction[k], new_detections[k])
        np.save(params.dir_solutions+"/task2/"+"scores_"+prediction[k], new_scores[k])
        np.save(params.dir_solutions+"/task2/"+"file_names_"+prediction[k], new_file_names[k])

    return new_detections, new_scores, new_file_names