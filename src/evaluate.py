import numpy as np
import csv, datetime, argparse
from pathlib import Path

# my modules
from models import LFI_conv3D, LFV_conv3D_STCLSTM
from mygenerator import test_generator
from metrics import calc_metrics


# predicted patch video -> full video
frame_length = 5
patch_size = 32
stride_size = 16
x_patch_n = (1024 - (patch_size-stride_size)) // stride_size
y_patch_n = (436 - (patch_size-stride_size)) // stride_size

def create_fullmap(patches):
    fullmap = np.zeros((frame_length, 432, 1024))
    for i, patch in enumerate(patches):
        ix = i % x_patch_n
        iy = i // x_patch_n
        px = ix*stride_size
        py = iy*stride_size
        fullmap[:, py:py+patch_size, px:px+patch_size] += patch
    # divide the area where multiple patches overlap to obtain the average value
    fullmap[:,  stride_size:-stride_size,  stride_size:-stride_size] /= 4  # center
    fullmap[:,  stride_size:-stride_size,             : stride_size] /= 2  # top center
    fullmap[:,  stride_size:-stride_size, -stride_size:            ] /= 2  # bottom center
    fullmap[:,             : stride_size,  stride_size:-stride_size] /= 2  # center left
    fullmap[:, -stride_size:            ,  stride_size:-stride_size] /= 2  # center right
    return np.float32(fullmap)


def evaluate_model(model_weights_path, test_list):
    # load model weights
    if 'baseline' in model_weights_path:
        model = LFI_conv3D.build_model()
        model.load_weights(model_weights_path)
    elif 'CLSTM' in model_weights_path:
        model = LFV_conv3D_STCLSTM.build_model()
        model.load_weights(model_weights_path)
    else:
        raise Exception(f'model weights path error.')
    print(model.summary())

    # path settings
    save_dir = Path(model_weights_path).parent / 'evaluated'
    save_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_path = save_dir / f'_metrics_{now}.csv'
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        metrics_dict = calc_metrics(pred=None, true=None)
        writer.writerow(['scene_name', *[metric for metric in metrics_dict.keys()]])

    # evaluate
    test_gen = test_generator(test_list)
    preds = []
    gts = []
    scene = {'name':'', 'frame_n':0}
    for inputs, gt, scenes in test_gen.flow_from_directory():
        pred = model.predict(inputs)
        for i, scene_name in enumerate(scenes):
            preds.append(pred[i])
            gts.append(gt[i])
            if len(preds) == x_patch_n*y_patch_n:
                fullmap_pred = create_fullmap(preds)
                fullmap_gt = create_fullmap(gts)
                # npz_save
                if not scene['name'] == scene_name:
                    scene['frame_n'] = 0
                scene['name'] = scene_name
                save_name = save_dir / f"{scene['name']}_{scene['frame_n']:03}.npz"
                np.savez_compressed(save_name, pred=fullmap_pred, gt=fullmap_gt)
                print('saved:', save_name)
                scene['frame_n'] += 1
                preds = []
                gts = []
                # metrics output
                metrics_dict = calc_metrics(fullmap_pred, fullmap_gt)
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([scene_name, *metrics_dict.values()])


if __name__ == "__main__":
    # args settings
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weights_path')
    parser.add_argument('--test_list', default='../patch_data_fl5/test_data.txt')
    args=parser.parse_args()

    evaluate_model(model_weights_path=args.model_weights_path, test_list=args.test_list)