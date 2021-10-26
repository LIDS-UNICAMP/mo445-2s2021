import torch
from flim.experiments import utils
from model.model import UNet, UnetLoss, get_device, validate_batch, IoU
from data.dataset import SegmDataset, ToTensor
from torchvision import transforms
from scipy.ndimage.morphology import binary_erosion
from torch import optim
from torch_snippets import *
import matplotlib.pyplot as plt
import click
import cv2
import os
from skimage.color import rgb2lab, lab2rgb

def save_output(image, pred_img, base_name):
    eroded = binary_erosion(pred_img, structure=np.ones([3,3])).astype(np.int64)
    lines = pred_img-eroded

    lined_image = image.squeeze(0).numpy()

    lined_image = lab2rgb(lined_image.transpose(1,2,0))
    lined_image = cv2.cvtColor(lined_image, cv2.COLOR_RGB2BGR)*255

    lined_image[:,:,1][lines==1] = 255

    cv2.imwrite(base_name +'.png', pred_img*255)
    cv2.imwrite(base_name + '_segm.png', lined_image)




@click.command()
@click.option('--arch-path', '-a', required=True, type=str, help='Architecture description file')
@click.option('--images_datapath', '-id', required=True, type=str, help='Path with original .png images')
@click.option('--gt_datapath', '-gd',required=True, type=str, help='Path with _label.png images')
@click.option('--output-dir', '-o', required=False, type=str, help='Output dir, if defined will save the segmentations')
@click.option('--unet_model', '-u', default='unet.pt', type=str, help='Input .pt unet model (default=unet.pt)')
def main(arch_path, images_datapath, gt_datapath, unet_model, output_dir):

    device = get_device()
    arch = utils.load_architecture(arch_path)

    encoder = utils.build_model(arch, input_shape=[3])

    num_classes = 2
    u_net = UNet(encoder=encoder, out_channels=num_classes)
    model = u_net.to(device)
   
   
    checkpoint = torch.load(unet_model)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])


   
    criterion = UnetLoss

    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)
    n_epochs = 3


    transform = transforms.Compose([ToTensor()])
    val_ds = SegmDataset(images_datapath, gt_datapath, transform=transform)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    # eval model
    log_val_loss = []
    log_val_iou  = []
    all_preds = None
    all_true_labels = None
        
    model.decoder.eval()
    N = len(val_dl)
    tmp_loss = 0
    tmp_iou  = 0
    for bx, data in enumerate(val_dl):
        loss, acc, preds, true_labels, iou = validate_batch(model, (data['img'], data['gt']), criterion, device=device)

        if all_preds is None:
            all_preds = preds.detach().cpu().numpy().flatten()
            all_true_labels = true_labels.detach().cpu().numpy().flatten()
        else:
            all_preds = np.concatenate((all_preds, preds.detach().cpu().numpy().flatten()))
            all_true_labels = np.concatenate((all_true_labels, true_labels.detach().cpu().numpy().flatten()))
        tmp_loss += loss
        tmp_iou  += iou

        if output_dir is not None:
            tmp_img = preds.detach().cpu().squeeze(0).numpy()
            save_output (data['img'], tmp_img, os.path.join(output_dir, data['name'][0]))
        
    log_val_loss.append(tmp_loss/(bx+1))
    log_val_iou.append(tmp_iou/(bx+1))

    
    print("IoU of dataset is ", IoU(all_true_labels, all_preds))

    



    

if __name__ == '__main__':
    main()
