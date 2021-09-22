from flim.experiments import utils
from model.model import get_device
import torch
import click


def save_features(feats, output_dir):
    nfeats = feats.shape[1]

    print(f"max min ", feats.max(), feats.min())

    for i in range(nfeats):
        tmp = feats[0,i,:,:]
        print(tmp.shape)


        output_img = output_dir + "/" + str(i) + ".png"
        print("saving images to ", output_img)
        #plt.imsave(output_img, tmp.numpy(), cmap='gray')

        plt.imshow(tmp, cmap='gray')
        plt.show()

        break


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