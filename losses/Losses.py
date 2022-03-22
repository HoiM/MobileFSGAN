import torch


class IdentityLoss(torch.nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, predicted, target, weights=None):
        """
        :param predicted:
        :type predicted: (torch tensor) [bs, 512]
        :param target:
        :type target: (torch tensor) [bs, 512]
        :param weights: weights on different sample losses
        :type weights: (torch tensor) [bs, ]
        :return:
        :rtype:
        """
        loss = torch.nn.functional.relu((1 - self.cosine_similarity(predicted, target)) - 0.02)
        if not (weights is None):
            loss = loss * weights
            if torch.sum(weights) == 0:
                return torch.sum(loss)
            else:
                return torch.sum(loss) / torch.sum(weights)
        else:
            count = 0.0
            for i in range(predicted.shape[0]):
                if loss[i] > 0.0:
                    count += 1.0
            if count > 0:
                loss = torch.sum(loss) / count
            else:
                loss = torch.sum(loss)
            return loss


class AdversarialLoss(torch.nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    @staticmethod
    def _hinge_loss(X, is_real=True):
        if is_real:
            return torch.relu(1 - X).mean()
        else:
            return torch.relu(X + 1).mean()

    def forward(self, X, is_real):
        """
        :param X: output from the discriminator
        :type X: (list of list of torch tensor)
        :param positive: if treated as from real data
        :type positive: bool
        :return:
        :rtype:
        """
        loss = 0.0
        for x in X:
            loss += self._hinge_loss(x[0], is_real)
        return loss


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, l=256):
        super(ReconstructionLoss, self).__init__()
        outside_mask = torch.zeros([3, l, l])
        self.n1 = int(80.0 / 256.0 * l)
        self.n2 = int(40.0 / 256.0 * l)
        #outside_mask[:, :n1, :] = 1.0
        #outside_mask[:, :, :n2] = 1.0
        #outside_mask[:, :, -n2:] = 1.0
        #self.register_buffer("outside_mask", outside_mask)

    def forward(self, pred, targ, weights=None):
        bs = pred.shape[0]
        if weights is None:
            loss = torch.mean(0.5 * torch.mean(torch.pow(pred - targ, 2).reshape(bs, -1), dim=1))
        else:
            with torch.no_grad():
                masks = torch.where(torch.sum(targ, 1, keepdim=True) < -2.999,
                                    torch.zeros([], dtype=targ.dtype, device=targ.device),
                                    torch.ones([], dtype=targ.dtype, device=targ.device))
                masks[:, :, :self.n1, :] = 1.0
                masks[:, :, :, :self.n2] = 1.0
                masks[:, :, :, -self.n2:] = 1.0
            loss = 0.5 * torch.sum(torch.pow(pred - targ, 2) * masks) / torch.sum(masks)
        return loss

