import torch
from torch import nn
from collections import OrderedDict
import functools
from sklearn.metrics import jaccard_score
from torch_snippets import *

def get_device():
    gpu = torch.cuda.is_available()

    if not gpu:
        device = torch.device('cpu')
    else:
        device = torch.device(0)

    return device




# U-Net


def maybe_resize2d(x, shape):
        if x.shape[-2:] != shape[-2:]:
            x = F.interpolate(x, size=shape[-2:],
                            mode="bilinear", align_corners=True)
        return x
    
def maybe_resize3d(x, shape):
        if x.shape[-3:] != shape[-3:]:
            x = F.interpolate(x, size=shape[-3:],
                            mode="trilinear", align_corners=True)
        return x


def rgetattr(obj, attr, *args):
    
    def _getattr (obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

class IntermediateLayerGetter:
    def __init__(self, model, return_layers):
        self._model = model
        self._return_layers = return_layers

    def __call__(self, x):
        outputs = OrderedDict()
        handles = []

        for name, out_name in self._return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, out_name=out_name):
                outputs[out_name] = output

            handle = layer.register_forward_hook(hook)

            handles.append(handle)
        
        self._model(x)

        for handle in handles:
            handle.remove()

        return outputs

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )
    
def _layers_before_downscale(model):
    last_layer_name = None
    last_out_channel = None
    layer_names = []
    last_out_channels = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Sequential)):
            continue
        if isinstance(layer, nn.Conv2d):
            last_out_channel = layer.out_channels
        if isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
            layer_names.append(last_layer_name)
            last_out_channels.append(last_out_channel)
        last_layer_name = name

    layer_names.append(last_layer_name)
    last_out_channels.append(last_out_channel)

    return layer_names, last_out_channels

class UNet(nn.Module):
    def __init__(self, encoder, out_channels=2, train_encoder=False):
        super().__init__()

        self.train_encoder=train_encoder
        self.encoder = encoder

        encoder_block_names, block_out_channels = _layers_before_downscale(encoder)

        layer_names = {layer_name: layer_name for layer_name in encoder_block_names[:-1]}
        layer_names[encoder_block_names[-1]] = "bottleneck"

        self._encoder_blocks = IntermediateLayerGetter(self.encoder, layer_names)

        self.decoder = nn.Module()

        bottleneck_out_channels = block_out_channels[-1] * 2
        self.decoder.add_module("conv_bottleneck", conv(block_out_channels[-1], bottleneck_out_channels))

        last_conv_out_channels = bottleneck_out_channels
        for i, _out_channels in enumerate(reversed(block_out_channels[:-1])):

            self.decoder.add_module(f"up_conv{i}", up_conv(last_conv_out_channels, last_conv_out_channels//2))
            self.decoder.add_module(f"conv{i}", conv(last_conv_out_channels//2 + _out_channels, _out_channels))

            last_conv_out_channels = _out_channels
        
        self.decoder.add_module("output_layer", nn.Conv2d(last_conv_out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        if not self.train_encoder:
            self.encoder.eval()
            with torch.no_grad():
                encoder_outputs = self._encoder_blocks(x)

                block_names = reversed(encoder_outputs.keys())

                bottleneck = encoder_outputs[next(block_names)]
        else:
            encoder_outputs = self._encoder_blocks(x)

            block_names = reversed(encoder_outputs.keys())

            bottleneck = encoder_outputs[next(block_names)]

        x = bottleneck

        for name, layer in self.decoder.named_children():
            if "up_conv" in name:
                block_output = encoder_outputs[(next(block_names))]
                x = layer(x)
                block_output = maybe_resize2d(block_output, x.shape)
                
                x = torch.cat([x, block_output], dim=1)
            else:
                x = layer(x)

        return x


# metrics
def IoU(gt, pred, ignore_label=-1, average='binary'):
    mask = gt != ignore_label
    iou = jaccard_score(gt[mask].flatten(), pred[mask].flatten(), average=average)
    return iou

ce = nn.CrossEntropyLoss()


def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    pred_labels = torch.max(preds, 1)[1]    
    acc = (pred_labels == targets).float().mean()
    
    iou = IoU(targets.cpu(), pred_labels.cpu()) 
    return ce_loss, acc, iou


def train_batch(model, data, optimizer, criterion, device='cpu'):
    ims, ce_masks = data[0].to(device), data[1].to(device)
    _masks = model(ims)
    _masks = maybe_resize2d(_masks, ce_masks.shape)
    optimizer.zero_grad()
    loss, acc, iou = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item(), iou.item()

@torch.no_grad()
def validate_batch(model, data, criterion, device='cpu'):
    ims, masks = data[0].to(device), data[1].to(device)
    _masks = model(ims)
    _masks = maybe_resize2d(_masks, masks.shape)
    loss, acc, iou = criterion(_masks, masks)
    return loss.item(), acc.item(), torch.max(_masks, 1)[1], masks, iou.item()


def _layers_before_downscale(model):
    last_layer_name = None
    last_out_channel = None
    layer_names = []
    last_out_channels = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Sequential)):
            continue
        if isinstance(layer, nn.Conv2d):
            last_out_channel = layer.out_channels
        if isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
            layer_names.append(last_layer_name)
            last_out_channels.append(last_out_channel)
        last_layer_name = name

    layer_names.append(last_layer_name)
    last_out_channels.append(last_out_channel)

    return layer_names, last_out_channels