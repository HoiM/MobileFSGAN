import torch
import torchvision


class VGGLoss(torch.nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.vgg.eval()
        self.normalize = Normalize()
        self.criterion = torch.nn.L1Loss()
        self.weights = [0, 0.6, 0.8, 1, 1.2]#[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        """
        :param x: real images (bs, 3, h, w)
        :param y: fake images (bs, 3, h, w)
        :return:
        """
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        x = x[:, [2, 1, 0], :, :]
        y = y[:, [2, 1, 0], :, :]
        x = self.normalize(x)
        y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(y_vgg[i], x_vgg[i])
        return loss


class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))

    def forward(self, x):
        bs, c, h, w = x.shape
        m = self.mean.repeat([bs, 1, h, w])
        s = self.std.repeat([bs, 1, h, w])
        x = (x - m) / s
        return x


class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
