import argparse
import os
import sys
import torch.backends.cudnn as cudnn
from torch.autograd import grad
from load_dataset import ORARDataset
from model import MRDN, Discriminator
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utilis import CharbonnierLoss, SSIMLoss, VGGFeatureExtractor
import torch
from torch.utils.data import DataLoader
import warnings
import torch.nn as nn
import time
import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

def gradient_penalty(gp_discriminator, real_samples, fake_samples, gp_device='cuda'):
    batch_size = real_samples.size(0)
    gp_alpha = torch.rand(batch_size, 1, 1, 1, device=gp_device)

    interpolated = gp_alpha * real_samples + ((1 - gp_alpha) * fake_samples)
    interpolated = interpolated.to(gp_device)
    interpolated.requires_grad_(True)

    d_interpolated = gp_discriminator(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, requires_grad=False)
    gradients = grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs = grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_factor', type=int, default=6, help='Downsampling factor')
    parser.add_argument('--PreFlag', type=bool, default=True, help='whether to use pre-trained module')
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--patchSize', type=int, default=192, help='input batch size')
    parser.add_argument('--StartEpochs', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--nPreEpochs', type=int, default=10, help='number of epochs to Pretrain for')
    parser.add_argument('--nEpochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
    parser.add_argument('--generatorWeights', type=str, default='',
                        help='path to generator weights (to continue training)')
    parser.add_argument('--discriminatorWeights', type=str, default='',
                        help="path to discriminator weights (to continue training)")
    parser.add_argument('--datapath', type=str, default='dataset', help='folder of dataset')
    parser.add_argument('--out', type=str, default='checkpoint', help='folder to output model checkpoints')
    opt = parser.parse_args()
    print(opt)
    try:
        os.makedirs(opt.out)
    except OSError:
        pass
    cudnn.benchmark = True
    # initial TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp_{timestamp}"
    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    # load dataset
    enh_dir = f"AR_Enh_{opt.scale_factor}x/"
    train_dataset = ORARDataset(__data_dir__=opt.datapath, __enh_dir__=enh_dir, __scale_factor__=opt.scale_factor, __patch_size__=opt.patchSize, d_flag=True)
    train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle=True, num_workers=8)
    len_train = len(train_loader)

    # load model
    device = torch.device('cuda')
    generator = MRDN(scale_factor=opt.scale_factor).to(device)
    discriminator = Discriminator(__patch_size__=opt.patchSize).to(device)

    # define Loss
    pixel_criterion = CharbonnierLoss().to(device)
    ssim_criterion = SSIMLoss().to(device)
    # pretrain
    optim_generator = optim.Adam(generator.parameters(), lr=0.0001)
    pretrain_losses = []
    if opt.PreFlag:
        print('Generator training')
        for epoch in range(opt.nPreEpochs):
            generator.train()
            epoch_loss = 0.0
            for batch_idx, (lor_data, enh_data, or_data) in enumerate(train_loader):
                # Generate data
                optim_generator.zero_grad()
                # Generate real and fake inputs
                or_data = or_data.to(device)
                or_fake = generator(lor_data.to(device), enh_data.to(device))
                pixel_loss = pixel_criterion(or_fake, or_data)
                pixel_loss.backward()
                optim_generator.step()
                epoch_loss += pixel_loss.item()
            epoch_loss /= len_train
            pretrain_losses.append(epoch_loss)
            sys.stdout.write('\r[%d/%d] Pre_Generator_Loss: %.4f\n' % (epoch + 1, opt.nPreEpochs, epoch_loss))
            writer.add_scalar('Train/pre_Generator_Loss', epoch_loss, epoch + 1)
        # Save checkpoint
        torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)

    if opt.generatorWeights != '':
        print('Loading generator weights')
        generator.load_state_dict(torch.load(opt.generatorWeights))
    else:
        generator.load_state_dict(torch.load('%s/generator_pretrain.pth' % opt.out))
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, betas=(0.0, 0.9))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR, betas=(0.0, 0.9))
    feat_criterion = nn.MSELoss().to(device)
    l_fea_w = 0.05
    l_adv_w = 0.01
    ssim_weight = 0.5
    l_fidelity_w = 1.0
    vgg_extractor = VGGFeatureExtractor(layer_index=7).to(device)
    generator_losses = []
    print('Training Start')
    for epoch in range(opt.StartEpochs, opt.nEpochs):
        epoch_start_time = time.time()

        generator.train()
        discriminator.train()
        mean_fidelity_loss = 0.0
        mean_adversarial_loss = 0.0
        mean_feat_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0
        for batch_idx, (lor_data, enh_data, or_data) in enumerate(train_loader):
            or_data = or_data.to(device)

            optim_discriminator.zero_grad()
            or_fake = generator(lor_data.to(device), enh_data.to(device)).detach()
            pred_d_real = discriminator(or_data)
            pred_d_fake = discriminator(or_fake)
            discriminator_loss_gp = gradient_penalty(discriminator, or_data, or_fake)

            d_loss = -torch.mean(pred_d_real) + torch.mean(pred_d_fake) + 10 * discriminator_loss_gp
            d_loss.backward()
            optim_discriminator.step()
            mean_discriminator_loss += d_loss.item()

            # --- Train Generator ---
            optim_generator.zero_grad()
            or_fake = generator(lor_data.to(device), enh_data.to(device))

            fidelity_loss = pixel_criterion(or_fake, or_data) + ssim_weight * ssim_criterion(or_fake, or_data)
            g_adv_loss = -discriminator(or_fake).mean()
            feat_loss = feat_criterion(vgg_extractor(or_fake), vgg_extractor(or_data))

            generator_total_loss = l_fidelity_w * fidelity_loss + l_adv_w * g_adv_loss + l_fea_w * feat_loss
            generator_total_loss.backward()
            optim_generator.step()
            mean_generator_total_loss += generator_total_loss.item()
            mean_fidelity_loss += l_fidelity_w * fidelity_loss.item()
            mean_adversarial_loss += l_adv_w * g_adv_loss.item()
            mean_feat_loss += l_fea_w * feat_loss.item()

        mean_discriminator_loss = mean_discriminator_loss / len_train
        mean_generator_total_loss = mean_generator_total_loss / len_train
        mean_fidelity_loss = mean_fidelity_loss / len_train
        mean_adversarial_loss = mean_adversarial_loss / len_train
        mean_feat_loss = mean_feat_loss / len_train
        generator_losses.append(mean_generator_total_loss)
        epoch_time = time.time() - epoch_start_time
        sys.stdout.write('\r[%d/%d] Generator_Loss: %.4f, Time: %.2f seconds\n' % (
        epoch + 1, opt.nEpochs, mean_generator_total_loss, epoch_time))

        writer.add_scalar('Train/Generator_Loss', mean_generator_total_loss, epoch + 1)
        writer.add_scalar('Train/Mean_fidelity_loss', mean_fidelity_loss, epoch + 1)
        writer.add_scalar('Train/Mean_adversarial_loss', mean_adversarial_loss, epoch + 1)
        writer.add_scalar('Train/Mean_feat_loss', mean_feat_loss, epoch + 1)
        writer.add_scalar('Train/Mean_discriminator_loss', mean_discriminator_loss, epoch + 1)
        if ((epoch + 1) % 1000 == 0):
            torch.save(generator.state_dict(), f'{opt.out}/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'{opt.out}/discriminator_epoch_{epoch + 1}.pth')
    writer.close()


    plt.figure("Generator_Loss", (18, 6))
    plt.title("Generator_Loss")
    x = [i + 1 for i in range(len(generator_losses))]
    y = [generator_losses[i] for i in range(len(generator_losses))]
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")

    plt.show()