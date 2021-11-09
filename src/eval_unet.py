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
import warnings, traceback
import inspect

try:
    import pyift.pyift as ift
    
    if ("CreateImageFromNumPy" not in  dir(ift)):
        raise NotImplementedError("CreateImageFromNumPy not in ift")
    if ("CloseBasins" not in  dir(ift)):
        raise NotImplementedError("CloseBasins not in ift")
    if ("SelectCompAboveArea" not in  dir(ift)):
        raise NotImplementedError("SelectCompAboveArea not in ift")
except NotImplementedError as e:
    print("Warning:", e)

except:
    ift = None
    warnings.warn("PyIFT is not installed.")


def complete_img_tensor(image, area=2000):
    out_img = image.clone()
    if ift is not None:
        out_img = out_img.detach().cpu().squeeze(0).numpy().astype(np.int32)

        try:
            iftimage = ift.CreateImageFromNumPy(np.ascontiguousarray(out_img), is3D=False)
            iftimage = ift.CloseBasins(iftimage, None, None)
            iftimage = ift.SelectCompAboveArea(iftimage, None, area)
            out_img  =  iftimage.AsNumPy()
        except Exception:
            traceback.print_exc()

        out_img = torch.tensor(out_img).unsqueeze(0)

    return out_img


def save_output(image, pred_img, output_dir, base_name, inputpath):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    eroded = binary_erosion(pred_img, structure=np.ones([5,5])).astype(np.int64)
    lines = pred_img-eroded

    lined_image  = cv2.imread(os.path.join(inputpath, base_name+".png"))
    
    lined_image[:,:,1][lines==1] = 255

    cv2.imwrite(os.path.join(output_dir, base_name +'.png'), pred_img*255)
    cv2.imwrite(os.path.join(output_dir, base_name + '_segm.png'), lined_image)




@click.command()
@click.option('--arch-path', '-a', required=True, type=str, help='Architecture description file')
@click.option('--images_datapath', '-id', required=True, type=str, help='Path with original .png images')
@click.option('--gt_datapath', '-gd',required=True, type=str, help='Path with _label.png images')
@click.option('--output-dir', '-o', required=False, type=str, help='Output dir, if defined will save the segmentations')
@click.option('--unet_model', '-u', default='unet.pt', type=str, help='Input .pt unet model (default=unet.pt)')
@click.option('--min-area', '-ma', default=2000, type=int, help='Minimal area (used only if pyift installed)')
def main(arch_path, images_datapath, gt_datapath, unet_model, output_dir, min_area):

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
        closed_img = complete_img_tensor(preds, min_area)

        if all_preds is None:
            all_preds = closed_img.detach().cpu().numpy().flatten()
            all_true_labels = true_labels.detach().cpu().numpy().flatten()
        else:
            all_preds = np.concatenate((all_preds, closed_img.detach().cpu().numpy().flatten()))
            all_true_labels = np.concatenate((all_true_labels, true_labels.detach().cpu().numpy().flatten()))
        tmp_loss += loss
        tmp_iou  += iou

        if output_dir is not None:
            tmp_img = closed_img.detach().cpu().squeeze(0).numpy()
            save_output (data['img'], tmp_img, output_dir, data['name'][0], images_datapath)
            
    log_val_loss.append(tmp_loss/(bx+1))
    log_val_iou.append(tmp_iou/(bx+1))

    
    print("IoU of dataset is ", IoU(all_true_labels, all_preds))

    



    

if __name__ == '__main__':
    main()
