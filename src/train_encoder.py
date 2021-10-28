from flim.experiments import utils
from model.model import get_device
import torch
import click


@click.command()
@click.option('--arch-path', '-a', required=True, type=str, help='Architecture json description file')
@click.option('--input-dir', '-i', required=True, type=str, help='Input dir. with .png images and .txt markers')
@click.option('--output-model', '-o', default='encoder.pt', type=str, help='Output .pt model name (default is encoder.pt)')
def main(arch_path, input_dir, output_model):

    device = get_device()
    arch = utils.load_architecture(arch_path)

    images, markers = utils.load_images_and_markers(input_dir)

    # train FLIM-Encoders
    print("Building model..")
    encoder = utils.build_model(arch, images, markers, images[0].shape, device=device)

    torch.save({'encoder_state_dict': encoder.state_dict()}, output_model)
    print(f"Done. Model save to {output_model}")

    

if __name__ == '__main__':
    main()