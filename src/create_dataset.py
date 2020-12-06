import numpy as np
import pathlib, argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--frame_length', '-fl', type=int, default=5)

frame_length = parser.parse_args().frame_length
w_size = 1024
h_size = 436
full_data_root = pathlib.Path('../Sintel_LF')
patch_data_root = pathlib.Path(f'../patch_data_fl{frame_length}')

def create_seq_EPI_patch(save_dir, seq_v, seq_h, seq_disp, patch_size=32, stride=16):
    save_dir.mkdir(parents=True, exist_ok=True)
    y_patch_n = (h_size - (patch_size-stride)) // stride
    x_patch_n = (w_size - (patch_size-stride)) // stride
    # save patch binary
    for key_frame_idx in range(len(seq_disp) - (frame_length-1)):
        numbering = y_patch_n * x_patch_n * key_frame_idx
        for iy in range(0, y_patch_n):
            for ix in range(0, x_patch_n):
                np.savez_compressed(save_dir / f'{numbering:05}.npz', 
                        v=seq_v[      key_frame_idx:key_frame_idx+frame_length, :, iy*stride:iy*stride+patch_size, ix*stride:ix*stride+patch_size],
                        h=seq_h[      key_frame_idx:key_frame_idx+frame_length, :, iy*stride:iy*stride+patch_size, ix*stride:ix*stride+patch_size],
                        disp=seq_disp[key_frame_idx:key_frame_idx+frame_length,    iy*stride:iy*stride+patch_size, ix*stride:ix*stride+patch_size])
                numbering += 1
    # save full binary
    np.savez_compressed(save_dir / 'full.npz', v=seq_v, h=seq_h, disp=seq_disp)

def main():
    for scene in full_data_root.glob('**/04_04'):
        center_disp_list = [p for p in scene.glob('*.npy')]
        scene_frame_n = len(center_disp_list)
        # prepare all data
        seq_v = np.zeros((scene_frame_n, 9, h_size, w_size, 3), dtype=np.uint8)
        seq_h = np.zeros((scene_frame_n, 9, h_size, w_size, 3), dtype=np.uint8)
        seq_disp = np.zeros((scene_frame_n, h_size, w_size), dtype=np.float32)
        for i, disp_path in enumerate(center_disp_list):
            seq_disp[i] = np.asarray(Image.fromarray(np.load(disp_path)).resize((w_size, h_size)))
            for iy in range(9):
                img_path = str(disp_path).replace('04_04', f'{iy:02}_04').replace('npy', 'png')
                seq_v[i, iy] = np.asarray(Image.open(img_path).resize((w_size, h_size)))
            for ix in range(9):
                img_path = str(disp_path).replace('04_04', f'04_{ix:02}').replace('npy', 'png')
                seq_h[i, ix] = np.asarray(Image.open(img_path).resize((w_size, h_size)))
        # create input patch
        create_seq_EPI_patch(save_dir=patch_data_root / center_disp_list[0].parent.parent.name, 
                            seq_v=seq_v, seq_h=seq_h, seq_disp=seq_disp, patch_size=32, stride=16)
        print(f'done: {center_disp_list[0].parent.parent.name}')


if __name__ == "__main__":
    main()