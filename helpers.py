import torch
import torchvision


def get_embeddings(images, identity_encoder, with_grad):
    """
    :param images: images from which embeddings are extracted
    :type images: torch tensor [bs, 3, 256, 256]
    :param identity_encoder: the face recognition model
    :type identity_encoder:
    :param with_grad: if True, keep track of the gradients
    :type with_grad: bool
    :return: embeddings
    :rtype: torch tensors [bs, 512]
    """
    if images.shape[2] == 112 and images.shape[3] == 112:
        if with_grad:
            embeddings = identity_encoder(images)
            return embeddings
        else:
            with torch.no_grad():
                embeddings = identity_encoder(images)
                return embeddings
    else:
        if with_grad:
            cropped = images[:, :, 26:230, 26:230]
            resized = torch.nn.functional.interpolate(cropped, [112, 112], mode='bilinear', align_corners=True)
            embeddings = identity_encoder(resized)
            return embeddings
        else:
            with torch.no_grad():
                cropped = images[:, :, 26:230, 26:230]
                resized = torch.nn.functional.interpolate(cropped, [112, 112], mode='bilinear', align_corners=True)
                embeddings = identity_encoder(resized)
                return embeddings


def get_grid_image(X):
    if X.shape[0] > 8:
        X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_images(*args):
    images = list()
    for im in args:
        images.append(get_grid_image(im))
    return torch.cat(images, dim=1).numpy()


def cal_mask():
    """19"""
    mask19 = torch.ones([1, 1, 19, 19])
    # background
    mask19[:, :, :2, :] = 0.5
    mask19[:, :, :, :2] = 0.5
    mask19[:, :, :, 16:] = 0.5
    mask19[:, :, 2, 2] = 0.5
    mask19[:, :, 2, 16] = 0.5
    # mouth
    mask19[:, :, 13:16, 7:12] = 4
    # eyes
    mask19[:, :, 8:10, 5:8] = 4
    mask19[:, :, 8:10, 10:13] = 4
    # nose
    mask19[:, :, 11:13, 8:11] = 2
    """11"""
    mask11 = torch.ones([1, 1, 11, 11])
    # mouth
    mask11[:, :, 8, 4:7] = 2
    # eyes
    mask11[:, :, 5, 3:5] = 2
    mask11[:, :, 5, 6:8] = 2
    """7"""
    mask7 = torch.ones([1, 1, 7, 7])
    # mouth
    mask7[:, :, 5, 3] = 2
    # eyes
    mask7[:, :, 3, 2] = 2
    mask7[:, :, 3, 4] = 2
    # nose
    mask7[:, :, 4, 3] = 2
    return mask19, mask11, mask7

