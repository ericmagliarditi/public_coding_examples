import os
import torch
from torchvision.transforms import transforms
import landpy
import argparse
from tqdm import tqdm, trange
import pandas as pd
import time
import numpy as np


def execute(args):
    '''
    Train the Model

    ..notes:
    Standard input image size: 2448 x 2448

    Label construction size is dependent on the input image size
    and the kernels used
        label_size = 196 #3x3 kernel sizes for all three decode layers and input size 284
        label_size = 180 #7x7, 5x5, 3x3 kernel sizes for decode layers and input size 284
        label_size = 172 #9x9, 6x6, 3x3 kernel sizes for decode layers and input size 284
        label_size = TBD #9x9, 6x6, 3x3 kernel sizes for decode layers and input size 1224
        label_size = 500 #9x9, 6x6, 3x3 kernel sizes for decode layers and input size 612
        label_size = 420 #7x7, 3x3, 3x3, 3x3, kernel size for decode layers and input size 612
    '''

    # Satellite Image Transformations
    t = transforms.Compose([
        transforms.Resize(args.input_image_size),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
    ])

    # Mask Transformations
    t2 = transforms.Compose([
        transforms.Resize(args.label_size),
        transforms.ToTensor(),
    ])

    data_set = landpy.MyDataLoader(
        args.data_dir, args.label_size, image_transforms=t, mask_transforms=t2)

    train_loader, validation_loader = landpy.create_data_loaders(
        data_set, args.training_split, args.batch_size)

    # Establish the UNet Model & Training parameters
    unet_model = landpy.UNet(3, 7)
    if args.start_new_model == 0:
        unet_path = os.path.join(args.model_paths, f"{args.model_to_load}.pt")
        unet_model.load_state_dict(torch.load(unet_path))

    loss_weights = torch.tensor(
        [0.145719925, 0.022623007,  0.133379898, 0.098588677, 0.36688587, 0.222802623, 0.01])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.empty_cache()
        print("GPU Enabled")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()}")
        print("Making Model GPU Based")
        unet_model = unet_model.cuda()
        loss_weights = loss_weights.cuda()

    loss = torch.nn.NLLLoss(weight=loss_weights)

    optimizer = torch.optim.SGD(unet_model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum
                                )
    # optimizer = torch.optim.Adam(unet_model.parameters(), lr=args.learning_rate)

    final_path = os.path.join(args.model_paths, f"{args.final_model_name}.pt")

    print(
        f"Number of Images for Training: {int(len(data_set)*args.training_split)}")
    print(
        f"Number of Images for Validation: {int(len(data_set)*(1-args.training_split))}")
    print(f"Number of Epochs Used: {args.epochs}")
    print(f"Batch Size Used: {args.batch_size}")
    print(f"Learning Rate Used: {args.learning_rate}")
    print(f"Momentum for Optimizer: {args.momentum}")
    print(f"Final Model Name: {args.final_model_name}")
    print(f"Loss Weights by Class: {loss_weights}")
    print("\n")

    epoch_losses = {}
    checkpoint_idx = 1
    print("Begin Training")
    for epoch in trange(args.epochs):
        if use_gpu:
            torch.cuda.empty_cache()

        t0 = time.time()
        total_training_loss = 0
        with torch.set_grad_enabled(True):
            for i, (batch_x_images, batch_y_mask, match_y_class_mask) in enumerate(train_loader):
                unet_model.train()
                if use_gpu:
                    batch_x_images = batch_x_images.cuda()
                    match_y_class_mask = match_y_class_mask.cuda()

                batch_loss = landpy.train_step(
                    batch_x_images, match_y_class_mask, optimizer, loss, unet_model)
                total_training_loss += batch_loss

        t1 = time.time()
        print(
            f"Total Training Loss for Epoch {epoch} is: {total_training_loss}")

        total_validation_loss = 0
        total_mean_iou = []
        with torch.no_grad():
            if use_gpu:
                torch.cuda.empty_cache()
            for j, (batch_val_x_images, batch_val_y_mask, match_val_y_class_mask) in enumerate(validation_loader):
                unet_model.eval()
                if use_gpu:
                    batch_val_x_images = batch_val_x_images.cuda()
                    match_val_y_class_mask = match_val_y_class_mask.cuda()

                outputs = unet_model(batch_val_x_images)
                soft_max_output = torch.nn.LogSoftmax(dim=1)(outputs)
                val_batch_loss = loss(
                    soft_max_output, match_val_y_class_mask.long())
                total_validation_loss += val_batch_loss

                batch_mean_iou = landpy.mean_IOU(
                    soft_max_output, match_val_y_class_mask)
                total_mean_iou.append(batch_mean_iou)

        epoch_losses[epoch] = {"Training Loss": total_training_loss.item(), "Validation Loss": total_validation_loss.item(),
                               "Mean IOU": np.mean(np.array(total_mean_iou)), "Execution Time": (t1-t0)}
        print(
            f"Total Validation Loss for Epoch {epoch} is: {total_validation_loss.item()}")

        if epoch % args.checkpoint == 0:
            # Checkpoint Save
            checkpoint_path = os.path.join(
                args.model_paths, f"{args.final_model_name}_chp_{checkpoint_idx}.pt")
            torch.save(unet_model.state_dict(), checkpoint_path)
            checkpoint_idx += 1

    print("\n")
    print("Completed Training; Saving Model")
    torch.save(unet_model.state_dict(), final_path)

    print("Saving Epoch Losses to DF")
    epoch_losses_path = os.path.join(args.epoch_loss_dir, args.final_model_name+"_epoch_losses.csv")
    df = pd.DataFrame.from_dict(epoch_losses, orient='index')
    df.to_csv(epoch_losses_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../Data/training_data',
                        help='Directory that contains images and masks')
    parser.add_argument('--start-new-model', type=int, default=1,
                        help='1 if start new 0 if not start new. Note: If 0 then need to supplement model')
    parser.add_argument('--model-to-load', type=str, default='unet_model')
    parser.add_argument('--final-model-name', type=str,
                        default='unet_model', help='Name your trained model!')
    parser.add_argument('--model-paths', type=str, default='../Data/models',
                        help='Directory where models are to be saved')
    parser.add_argument('--epoch-loss-dir', type=str,
                        default='../Data', help='Directory to save epoch loss csv')
    parser.add_argument('--input-image-size', type=int,
                        default=284, help='Resized satellite image size')
    parser.add_argument('--label-size', type=int, default=196,
                        help='Generated label size based on kernel size')
    parser.add_argument('--batch-size', type=int,
                        default=20, help='Size of the batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Numebr of epochs for training')
    parser.add_argument('--learning-rate', type=float,
                        default=0.1, help='Learning rate of model')
    parser.add_argument('--momentum', type=float,
                        default=0.0, help='Momentum value for SGD')
    parser.add_argument('--training-split', type=float, default=0.9,
                        help='Percentage of data to be used for training')
    parser.add_argument('--checkpoint', type=int, default=20,
                        help="After this many epochs, the most current model will save")

    args = parser.parse_args()

    execute(args)
