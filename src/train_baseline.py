import argparse

# my modules
from models.LFI_conv3D import build_model
from train import train


if __name__ == "__main__":
    # define model
    model = build_model()
    print(model.summary())

    # args settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', '-m', default='')
    parser.add_argument('--frame_length', '-fl', type=int, default=5)
    parser.add_argument('--model_name', default='baseline')
    parser.add_argument('--train_list', default='../patch_data_fl5/train_data.txt')
    parser.add_argument('--valid_list', default='../patch_data_fl5/validation_data.txt')
    args=parser.parse_args()

    # start train
    train(model=model, args=args)