import torch
from flim.experiments import utils
from model.model import UNet, UnetLoss, get_device, train_batch
from data.dataset import SegmDataset, ToTensor
from torchvision import transforms
from torch import optim
from torch_snippets import *
import click


@click.command()
@click.option('--arch-path', '-a', required=True, type=str, help='Architecture description file')
@click.option('--images_datapath', '-id', required=True, type=str, help='Path with original .png images')
@click.option('--gt_datapath', '-gd',required=True, type=str, help='Path with _label.png images')
@click.option('--output_model', '-o', default='unet.pt', type=str, help='Output .pt encoder (default=unet.pt)')
@click.option('--n_epochs', '-ne', default=20, type=int, help='Number of epochs')
@click.option('--encoder_model', '-e', default='encoder.pt', type=str, help='Encoder .pt model')
def main(arch_path, encoder_model, images_datapath, gt_datapath, output_model, n_epochs):

    device = get_device()
    arch = utils.load_architecture(arch_path)

    encoder = utils.build_model(arch, input_shape=[3])

    checkpoint = torch.load(encoder_model)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])


    num_classes = 2
    u_net = UNet(encoder=encoder, out_channels=num_classes)

    model = u_net.to(device)
    criterion = UnetLoss

    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)


    transform = transforms.Compose([ToTensor()])
    trn_ds = SegmDataset(images_datapath, gt_datapath, transform=transform)

    trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True)

    # train decoder with backpropagation
    log = Report(n_epochs)
    log_trn_loss = []
    log_trn_iou  = []
    for ex in range(n_epochs):
    
        model.decoder.train()
        N = len(trn_dl)
        tmp_loss = 0
        tmp_iou  = 0
        for bx, data in enumerate(trn_dl):
            loss, acc, iou = train_batch(model, (data['img'], data['gt']), optimizer, criterion, device=device)
            log.record(ex+(bx+1)/N, trn_loss=loss, trn_acc=acc, iou=iou, end='\r')
            tmp_loss += loss
            tmp_iou  += iou
            
        log_trn_loss.append(tmp_loss/(bx+1))
        log_trn_iou.append(tmp_iou/(bx+1))

        log.report_avgs(ex+1)
    

    torch.save({
            'epoch': n_epochs,
            'decoder_state_dict': model.decoder.state_dict(),
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': model.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, output_model)




    

if __name__ == '__main__':
    main()
