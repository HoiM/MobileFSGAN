import torch
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm


class NLayerDiscriminator(torch.nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=32,
                 n_layers=3,
                 norm_layer=torch.nn.BatchNorm2d,
                 use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[spectral_norm(torch.nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                     torch.nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                spectral_norm(torch.nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(nf), torch.nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            spectral_norm(torch.nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(nf),
            torch.nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[spectral_norm(torch.nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))]]

        if use_sigmoid:
            sequence += [[torch.nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), torch.nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = torch.nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(torch.nn.Module):
    def __init__(self,
                 input_nc=3,
                 ndf=16,
                 n_layers=4,
                 norm_layer=torch.nn.BatchNorm2d,
                 use_sigmoid=False,
                 num_D=3,
                 getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class MultiScaleGradientDiscriminator(torch.nn.Module):
    def __init__(self, num_channels, num_layers):
        super(MultiScaleGradientDiscriminator, self).__init__()
        self.discriminator_256 = NLayerDiscriminator(3, num_channels, num_layers, torch.nn.BatchNorm2d, False, False)
        self.discriminator_128 = NLayerDiscriminator(3, num_channels, num_layers, torch.nn.BatchNorm2d, False, False)
        self.discriminator_64  = NLayerDiscriminator(3, num_channels, num_layers, torch.nn.BatchNorm2d, False, False)

    def forward(self, images256, images128, images64):
        out256 = self.discriminator_256(images256)
        out128 = self.discriminator_128(images128)
        out64 = self.discriminator_64(images64)
        return out256, out128, out64
