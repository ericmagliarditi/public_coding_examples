import os
import torch
from torchvision.transforms import transforms
import landpy
import argparse
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def evaluate(args):
    '''
    This script is used to make a single prediction mask.
    '''

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("GPU Enabled")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()}")

    #Create Image for Prediction
    raw_image_transformation = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    ])

    image = landpy.get_image_for_prediction(image_path=args.image_path,
        transformation=raw_image_transformation)

    #Bring in Model
    unet_model = landpy.UNet(3,7)
    if use_gpu:
        unet_model = unet_model.cuda()
        unet_model.load_state_dict(torch.load(args.model_path))
    else:
        unet_model.load_state_dict(torch.load(args.model_path,
            map_location=torch.device('cpu')))

    unet_model.eval()

    if use_gpu:
        image = image.cuda()
    print("Generating Prediction\n")    
    predictions = unet_model(image)
    print("Made Prediction")
    soft_max_output = torch.nn.LogSoftmax(dim=1)(predictions)

    if use_gpu:
        soft_max_output = soft_max_output.cpu()
    
    numpy_output = soft_max_output.data.numpy()
    final_prediction = np.argmax(numpy_output,axis=1)
    landpy.calculate_percentages(final_prediction[0], args.output_path)
    prediction_img = landpy.construct_image(final_prediction[0], args.mask_size)

    #Create Figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))

    ax[0].imshow(np.transpose(image[0].data.numpy(), (1,2,0)))
    ax[0].set_title("Real Image")
    
    ax[1].imshow(np.transpose(prediction_img, (1,2,0)))
    ax[1].set_title("Prediction Mask")

    # plt.show()
    
    fig_path = os.path.join(args.output_path, 'prediction.png')
    plt.savefig(fig_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str,
                        help='File path that points to the image')
    parser.add_argument('--model-path', type=str,
                        help='File path that contains model weights')
    parser.add_argument('--output-path', type=str, help='Where to save the figure')
    parser.add_argument('--input-size', type=int, default=612,
        help="Input size needed for evaluation - to be modified within code base")
    parser.add_argument('--mask-size', type=int, default=420,
        help="Mask size needed for evaluation to be modified within code base")
    
    args = parser.parse_args()

    evaluate(args)
