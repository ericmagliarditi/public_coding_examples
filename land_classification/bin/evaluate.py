import os
import torch
from torchvision.transforms import transforms
import landpy
import argparse
from tqdm import tqdm, trange
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt


def evaluate(args):
    '''
    This script is used to generate a evaluation figure.
    '''

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("GPU Enabled")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()}")

    #Create the data
    t = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    ])

    t2 = transforms.Compose([
    transforms.Resize(args.mask_size),
    transforms.ToTensor(),
    ])

    data_set = landpy.MyDataLoader(args.data_dir, args.mask_size,
        image_transforms=t, mask_transforms=t2)

    #Set Batch Size
    batch_size = 1

    dataset_size = len(data_set)
    print(f"Number of Images {dataset_size}")

    train_loader, test_loader = landpy.create_data_loaders(
        data_set, 0.8, batch_size)

    unet_model = landpy.UNet(3,7)
    if use_gpu:
        unet_model = unet_model.cuda()
        unet_model.load_state_dict(torch.load(args.model_path))
    else:
        unet_model.load_state_dict(torch.load(args.model_path,
            map_location=torch.device('cpu')))

    unet_model.eval()

    num_images = 3
    
    figure_row = 0
    sat_image, actual_mask, label = next(iter(test_loader))
    
    fig, ax = plt.subplots(nrows=num_images, ncols=3, figsize=(20,20))
    for i, (test_image, labels, class_labels) in enumerate(test_loader):
        if use_gpu:
            test_image = test_image.cuda()
            torch.cuda.empty_cache()

        predictions = unet_model(test_image)
        soft_max_output = torch.nn.LogSoftmax(dim=1)(predictions)

        if use_gpu:
            soft_max_output = soft_max_output.cpu()
        
        numpy_output = soft_max_output.data.numpy()
        final_prediction = np.argmax(numpy_output,axis=1)
        prediction_img = landpy.construct_image(final_prediction[0], args.mask_size)

        ax[figure_row][0].imshow(np.transpose(test_image[0].data.numpy(), (1,2,0)))
        ax[figure_row][0].set_title("Real Image")
        ax[figure_row][1].imshow(np.transpose(labels[0].data.numpy(), (1,2,0)))
        ax[figure_row][1].set_title("Actual Mask")
        ax[figure_row][2].imshow(np.transpose(prediction_img, (1,2,0)))
        ax[figure_row][2].set_title("Prediction Mask")
        
        figure_row += 1
        if figure_row == num_images:
            break

    fig_path = os.path.join(args.figure_path, 'evaluation_figure.png')
    plt.savefig(fig_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='Directory that contains images and masks')
    parser.add_argument('--model-path', type=str,
                        help='File path that contains model weights')
    parser.add_argument('--figure-path', type=str, help='Where to save the figure')
    parser.add_argument('--input-size', type=int, default=612,
        help="Input size needed for evaluation - to be modified within code base")
    parser.add_argument('--mask-size', type=int, default=420,
        help="Mask size needed for evaluation to be modified within code base")
    
    args = parser.parse_args()

    evaluate(args)
