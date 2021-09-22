from flim.experiments import utils
from model.model import get_device, _layers_before_downscale
import torch
import click
import matplotlib.pyplot as plt
from collections import OrderedDict
import functools
import os
import shutil

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



def save_features(out_channels, output_dir):


    if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        #os.rmdir(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.mkdir(output_dir)

    for block in out_channels:

        blockname = block['block']
        feats     = block['out']


        innerdir = os.path.join(output_dir, blockname)
        os.mkdir(innerdir)

        #saving mimage
        try:
            utils.save_mimage(innerdir + ".mimg", feats.transpose(1,2,0) )
        except Exception as e:
            print('An exception occurred when saving mimage: {}'.format(e))

        nfeats = feats.shape[0]
        for i in range(nfeats):
            tmp = feats[i,:,:]

            output_img = os.path.join(innerdir, str(i) + ".png")
            plt.imsave(output_img, tmp, cmap='gray')




def forward_encoder(encoder, x):
    encoder_block_names, block_out_channels = _layers_before_downscale(encoder)

    layer_names = {layer_name: layer_name for layer_name in encoder_block_names[:-1]}
    layer_names[encoder_block_names[-1]] = "bottleneck"

    encoder_blocks = IntermediateLayerGetter(encoder, layer_names)



    encoder_outputs = encoder_blocks(x)

    block_names = reversed(encoder_outputs.keys())


    ret = []
    for name in block_names:

        block_output = encoder_outputs[(name)]
        listname = name.split('.')
        if len(listname) > 1:
            outname = listname[1]
        else:
            outname = listname[0]

        tmp = {"block": outname, "out": block_output.detach().cpu().squeeze(0).numpy()}
        ret.append(tmp)

    return ret





@click.command()
@click.option('--arch-path', '-a', required=True, type=str, help='Architecture json description file')
@click.option('--input-image', '-i', required=True, type=str, help='Input .png image')
@click.option('--output-dir', '-o', required=True, type=str, help='Output features dir')
@click.option('--model', '-m', default='encoder.pt', type=str, help='Input encoder model, default=encoder.pt')
def main(arch_path, model, input_image, output_dir):


    arch = utils.load_architecture(arch_path)

    encoder = utils.build_model(arch, input_shape=[3])

    checkpoint = torch.load(model)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    image = torch.tensor(utils.load_image(input_image))
    image = image.unsqueeze(0).permute(0,3,1,2).float()
    
    
    out_channels = forward_encoder(encoder, image)
    
    save_features(out_channels, output_dir)

    print(f"Done. All images saved to {output_dir}")

if __name__ == '__main__':
    main()