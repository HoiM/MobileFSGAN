import os
import time
import argparse
import PIL.Image
import numpy as np
import torch

from modules import EncoderDecoder
from modules import IdentityEncoder
from modules import MultiScaleGradientDiscriminator
from dataset import FaceShifterDataset
from losses import AdversarialLoss
from losses import IdentityLoss
from losses import ReconstructionLoss
from losses import VGGLoss
from helpers import get_embeddings
from helpers import make_images


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size per GPU")
arg_parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for data loader")
arg_parser.add_argument("--lr_G", type=float, default=1e-4,
                        help="learning rate for generator")
arg_parser.add_argument("--lr_D", type=float, default=1e-4,
                        help="learning rate for discriminator")
arg_parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of epochs")
arg_parser.add_argument("--print_iter", type=int, default=200,
                        help="print info every n iterations")
arg_parser.add_argument("--save_dir", type=str, default="output/",
                        help="directory to save results")
arg_parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank for distributed data parallel")
arg_parser.add_argument("--ngf", type=int, default=64,
                        help="number of channels for generator")
arg_parser.add_argument("--ndf", type=int, default=16,
                        help="number of channels for discriminator")
arg_parser.add_argument("--d_layers", type=int, default=4,
                        help="number of layers for discriminator")
args = arg_parser.parse_args()


def main():
    """device-related"""
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:%d" % args.local_rank)
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(device)
    """directories"""
    model_save_path = os.path.join(args.save_dir, 'saved_models')
    gen_images_path = os.path.join(args.save_dir, 'gen_images')
    if args.local_rank == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(gen_images_path):
            os.makedirs(gen_images_path)
    """dataset"""
    dataset = FaceShifterDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             sampler=sampler,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             drop_last=True)
    total_iter = len(dataloader)
    """models"""
    G = EncoderDecoder(args.ngf, 512).to(device)
    D = MultiScaleGradientDiscriminator(args.ndf, args.d_layers).to(device)
    identity_encoder = IdentityEncoder()
    identity_encoder.load_state_dict(torch.load("params/RGB_model_mobilefacenet.pth", map_location="cpu"))
    identity_encoder = identity_encoder.to(device)
    identity_encoder.eval()
    """distributed data parallel"""
    G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
    D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
    G = torch.nn.parallel.DistributedDataParallel(G, [args.local_rank], args.local_rank, find_unused_parameters=True)
    D = torch.nn.parallel.DistributedDataParallel(D, [args.local_rank], args.local_rank, find_unused_parameters=True)
    """losses"""
    id_loss = IdentityLoss().to(device)
    adv_loss = AdversarialLoss().to(device)
    #attr_loss = AttributeLoss().to(device)
    rec_loss = ReconstructionLoss().to(device)
    vgg_loss = VGGLoss().to(device)
    """optimizer"""
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999))
    """training"""
    for epoch in range(args.max_epoch):
        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            sources, targets, gts, with_gt, src_as_true = data
            sources = sources.to(device)
            targets = targets.to(device)
            gts = gts.to(device)
            with_gt = with_gt.to(device)
            """train G"""
            opt_G.zero_grad()
            source_embeddings = get_embeddings(sources, identity_encoder, False)
            fake256, fake128, fake64, _ = G(targets, source_embeddings)
            fake256_embeddings = get_embeddings(fake256, identity_encoder, True)
            fake128_embeddings = get_embeddings(torch.nn.functional.interpolate(fake128[:, :, 13:115, 13:115], [112, 112], mode='bilinear', align_corners=True), identity_encoder, True)
            fake64_embeddings = get_embeddings(torch.nn.functional.interpolate(fake64[:, :, 7:57, 7:57], [112, 112], mode='bilinear', align_corners=True), identity_encoder, True)
            fake256_disc_out, fake128_disc_out, fake64_disc_out = D(fake256, fake128, fake64)
            # losses
            loss_adv256 = adv_loss(fake256_disc_out, True)
            loss_adv128 = adv_loss(fake128_disc_out, True)
            loss_adv64 = adv_loss(fake64_disc_out, True)
            loss_id256 = id_loss(fake256_embeddings, source_embeddings)
            loss_id128 = id_loss(fake128_embeddings, source_embeddings)
            loss_id64 = id_loss(fake64_embeddings, source_embeddings)
            loss_gt256 = rec_loss(fake256, gts, with_gt)
            loss_vgg256 = vgg_loss(fake256, targets)
            # total
            loss_G_256 = 1 * loss_adv256 + 20 * loss_id256 + 10 * loss_gt256 + 4 * loss_vgg256
            loss_G_128 = 0.02 * loss_adv128 + 20 * loss_id128
            loss_G_64 = 0.02 * loss_adv64 + 20 * loss_id64
            loss_G = 1 * loss_G_256 + 1 * loss_G_128 + 1 * loss_G_64
            loss_G.backward()
            opt_G.step()
            """train D"""
            opt_D.zero_grad()
            real_data_1 = sources[src_as_true]
            real_data_2 = targets[torch.bitwise_not(src_as_true)]
            real_data = torch.cat([real_data_1, real_data_2], 0)
            # data
            fake_data_256 = fake256.detach()
            real_data_256 = real_data
            fake_data_128 = fake128.detach()
            real_data_128 = torch.nn.functional.interpolate(real_data, [128, 128], mode='bilinear', align_corners=True)
            fake_data_64 = fake64.detach()
            real_data_64 = torch.nn.functional.interpolate(real_data, [64, 64], mode='bilinear', align_corners=True)
            # discriminator
            fake_256_disc_out, fake_128_disc_out, fake_64_disc_out = D(fake_data_256, fake_data_128, fake_data_64)
            real_256_disc_out, real_128_disc_out, real_64_disc_out = D(real_data_256, real_data_128, real_data_64)
            # loss 256
            loss_real_256 = adv_loss(real_256_disc_out, True)
            loss_fake_256 = adv_loss(fake_256_disc_out, False)
            loss_D_256 = 0.5 * (loss_real_256 + loss_fake_256)
            # loss 128
            loss_real_128 = adv_loss(real_128_disc_out, True)
            loss_fake_128 = adv_loss(fake_128_disc_out, False)
            loss_D_128 = 0.5 * (loss_real_128 + loss_fake_128)
            # loss 64
            loss_real_64 = adv_loss(real_64_disc_out, True)
            loss_fake_64 = adv_loss(fake_64_disc_out, False)
            loss_D_64 = 0.5 * (loss_real_64 + loss_fake_64)
            # total loss_D
            loss_D = 1 * loss_D_256 + 0.02 * loss_D_128 + 0.02 * loss_D_64
            loss_D.backward()
            opt_D.step()
            # info
            batch_time = time.time() - start_time
            if args.local_rank == 0 and (iteration + 1) % args.print_iter == 0:
                fake_others = torch.zeros_like(fake256)
                fake_others[:, :,    :128   ,    :128   ] = fake128
                fake_others[:, :, 128:128+64, 128:128+64] = fake64
                image = make_images(sources, targets, fake256, gts, fake_others)
                image = image.transpose([1, 2, 0]) * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
                gen_images_name = os.path.join(gen_images_path, '%03d_%05d.jpg' % (epoch, iteration + 1))
                PIL.Image.fromarray(image).save(gen_images_name)
                print('[GAN] Epoch: %d Iter: %d/%d lossD: %.6f lossG: %.6f time: %.2f' %
                      (epoch, iteration + 1, total_iter, loss_D.item(), loss_G.item(), batch_time))
                print('[G] L_adv_256: %.6f L_adv_128: %.6f L_adv_64: %.6f' %
                      (loss_adv256.item(), loss_adv128.item(), loss_adv64.item()))
                print('[G] L_id256: %.6f L_id128: %.6f L_id64: %.6f' %
                      (loss_id256.item(), loss_id128.item(), loss_id64.item()))
                print('[G] L_gt256: %.6f L_gt128: %.6f L_gt64: %.6f' %
                      (loss_gt256.item(), 0, 0))
                print('[G] L_vgg256: %.6f L_vgg128: %.6f L_vgg64: %.6f' %
                      (loss_vgg256.item(), 0, 0))
                print('[D] L_real_256: %.6f L_real_128: %.6f L_real_64: %.6f' %
                      (loss_real_256.item(), loss_real_128.item(), loss_real_64.item()))
                print('[D] L_fake_256: %.6f L_fake_128: %.6f L_fake_64: %.6f' %
                      (loss_fake_256.item(), loss_fake_128.item(), loss_fake_64.item()))
        if args.local_rank == 0:
            model_save_path_G = os.path.join(model_save_path, '%03d_G.pth' % (epoch + 1))
            model_save_path_D = os.path.join(model_save_path, '%03d_D.pth' % (epoch + 1))
            torch.save(G.state_dict(), model_save_path_G)
            torch.save(D.state_dict(), model_save_path_D)


if __name__ == '__main__':
    main()
