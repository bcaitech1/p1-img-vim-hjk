import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from timm.models.layers.classifier import ClassifierHead
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=18)
        
    def forward(self, x):
        x = self.net(x)
        return x
    

class MaskClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=3, num_layers=3):
        super(MaskClassifier, self).__init__()
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)      
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
    def forward(self, x):
        x = self.layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GenderClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=2, num_layers=3):
        super(GenderClassifier, self).__init__()
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)       
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AgeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=3, num_layers=3):
        super(AgeClassifier, self).__init__()
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.layers(x)        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, coefficient=0):
        super(CustomModel, self).__init__()
        self.model_names = [
            'efficientnet-b0',
            'efficientnet-b1',
            'efficientnet-b2',
            'efficientnet-b3',
            'efficientnet-b4',
            'efficientnet-b5',
            'efficientnet-b6',
            'efficientnet-b7'
        ]
        self.in_channels = [1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.backbone = EfficientNet.from_pretrained(self.model_names[coefficient])
        self.mask_classifier = MaskClassifier(in_channels=self.in_channels[coefficient])
        self.gender_classifier = GenderClassifier(in_channels=self.in_channels[coefficient])
        self.age_classifier = AgeClassifier(in_channels=self.in_channels[coefficient])

               
        torch.nn.init.xavier_uniform_(self.mask_classifier.fc.weight)
        self.mask_classifier.fc.bias.data.fill_(0.01)
        
        torch.nn.init.xavier_uniform_(self.gender_classifier.fc.weight)
        self.gender_classifier.fc.bias.data.fill_(0.01)
        
        torch.nn.init.xavier_uniform_(self.age_classifier.fc.weight)
        self.age_classifier.fc.bias.data.fill_(0.01)

        
    def forward(self, x):
        x = self.backbone.extract_features(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)

        return mask, gender, age
    
    
class MultiSampleDropout(nn.Module):
    def __init__(self, num_classes, dropout_num, dropout_p):
        super(MultiSampleDropout, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1792, num_classes)
        
    def forward(self, x, y=None, loss_fn=None):
        feature = self.backbone.extract_features(x)
        feature = self.gap(feature)
        if len(self.dropouts) == 0:
            out = feature.view(feature.size(0), -1)
            out = self.fc(out)
            if loss_fn is not None:
                loss = loss_fn(out, y)
                return out, loss
            return out, None
        else:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(feature)
                    out = out.view(out.size(0), -1)
                    out = self.fc(out)
                    if loss_fn is not None:
                        loss = loss_fn(out, y)
                else:
                    temp_out = dropout(feature)
                    temp_out = temp_out.view(temp_out.size(0), -1)
                    out = out + self.fc(temp_out)
                    if loss_fn is not None:
                        loss = loss + loss_fn(temp_out, y)
            if loss_fn is not None:
                return out / len(self.dropouts), loss / len(self.dropouts)
            return out, None
        
    
class MultiFCModel(nn.Module):
    def __init__(self, model_arch, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.num_features
        self.mask_classifier = ClassifierHead(n_features, 3)
        self.gender_classifier = ClassifierHead(n_features, 2)
        self.age_classifier = ClassifierHead(n_features, 3)
        
    def forward(self, x):
        x = self.model.forward_features(x)
        pred_mask = self.mask_classifier(x)
        pred_gender = self.gender_classifier(x)
        pred_age = self.age_classifier(x)
        
        return pred_mask, pred_gender, pred_age