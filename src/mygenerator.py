import csv
import numpy as np
import pathlib
import re


class input_generator_fl5():
    def __init__(self, class_list_path, val_mode=False,
                gammma=False, horizontal_flip=False, vertical_flip=False, rotation=False):
        self.clear()
        self.val_mode = val_mode
        with open(class_list_path) as f:
            reader = csv.reader(f)
            scene_names = np.array([row for row in reader]).flatten()
        scene_dirs = [pathlib.Path(class_list_path).parent / scene_name
                        for scene_name in scene_names]
        self.data_paths = [npz_path for scene_dir in scene_dirs
                                    for npz_path in scene_dir.glob('*[!full].npz')]
        self.gammma = gammma
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
    
    def clear(self):
        self.seq_h = []
        self.seq_v = []
        self.seq_disp = []
    
    def convert(self, seq_batch_h, seq_batch_v, seq_batch_disp):
        # any process
        return seq_batch_h, seq_batch_v, seq_batch_disp

    def augmentation(self, seq_batch_h, seq_batch_v, seq_batch_disp):
        for i in range(len(seq_batch_disp)):
            if self.gammma:
                gamma_rand = 0.4 * np.random.rand() + 0.8
                seq_batch_h[i] = pow(seq_batch_h[i], gamma_rand)
                seq_batch_v[i] = pow(seq_batch_v[i], gamma_rand)
            if self.horizontal_flip:
                flip_rand = np.random.randint(0,2)
                if flip_rand == 1:
                    seq_batch_h[i] = seq_batch_h[i][:, ::-1, :, ::-1, :]  # (frame, angle, height, width, channel)
                    seq_batch_v[i] = seq_batch_v[i][:,    :, :, ::-1, :]
                    seq_batch_disp[i] = seq_batch_disp[i][:, :, ::-1]     # (frame, height, width)
            if self.vertical_flip:
                flip_rand = np.random.randint(0,2)
                if flip_rand == 1:
                    seq_batch_h[i] = seq_batch_h[i][:,    :, ::-1, :, :]  # (frame, angle, height, width, channel)
                    seq_batch_v[i] = seq_batch_v[i][:, ::-1, ::-1, :, :]
                    seq_batch_disp[i] = seq_batch_disp[i][:, ::-1, :]     # (frame, height, width)
            if self.rotation:
                rot90_rand = np.random.randint(0,4)
                for rot_step in range(rot90_rand):
                    seq_batch_h_rot90 = np.copy(np.rot90(seq_batch_h[i], 1, (2, 3)))
                    seq_batch_v_rot90 = np.copy(np.rot90(seq_batch_v[i], 1, (2, 3)))
                    seq_batch_h[i] = seq_batch_v_rot90[:, ::-1]
                    seq_batch_v[i] = seq_batch_h_rot90
                    seq_batch_disp[i] = np.copy(np.rot90(seq_batch_disp[i], 1, (1, 2))) 
        return seq_batch_h, seq_batch_v, seq_batch_disp
    
    def flow_from_directory(self, batch_size=64, seed=None):
        np.random.seed(seed)
        while True:
            if not self.val_mode:
                np.random.shuffle(self.data_paths)
            for target_path in self.data_paths:
                loaded_data = np.load(target_path)
                self.seq_h.append(loaded_data['h'])
                self.seq_v.append(loaded_data['v'])
                self.seq_disp.append(loaded_data['disp'])
                if len(self.seq_disp) == batch_size:
                    seq_batch_h = np.array(self.seq_h, dtype=np.float32) / 255.0
                    seq_batch_v = np.array(self.seq_v, dtype=np.float32) / 255.0
                    seq_batch_disp = np.array(self.seq_disp, dtype=np.float32)
                    seq_batch_h, seq_batch_v, seq_batch_disp = self.convert(seq_batch_h, seq_batch_v, seq_batch_disp)
                    if not self.val_mode:
                        seq_batch_h, seq_batch_v, seq_batch_disp = self.augmentation(seq_batch_h, seq_batch_v, seq_batch_disp)
                    self.clear()
                    yield [seq_batch_h, seq_batch_v], seq_batch_disp

class input_generator_fl3(input_generator_fl5):
    def convert(self, seq_batch_h, seq_batch_v, seq_batch_disp):
        if self.val_mode:
            return seq_batch_h[:,:3], seq_batch_v[:,:3], seq_batch_disp[:,:3]
        rand = np.random.randint(0,3)
        return seq_batch_h[:,rand:rand+3], seq_batch_v[:,rand:rand+3], seq_batch_disp[:,rand:rand+3]

class input_generator_fl4(input_generator_fl5):
    def convert(self, seq_batch_h, seq_batch_v, seq_batch_disp):
        if self.val_mode:
            return seq_batch_h[:,:4], seq_batch_v[:,:4], seq_batch_disp[:,:4]
        rand = np.random.randint(0,2)
        return seq_batch_h[:,rand:rand+4], seq_batch_v[:,rand:rand+4], seq_batch_disp[:,rand:rand+4]


class test_generator():
    def __init__(self, class_list_path):
        self.clear()
        with open(class_list_path) as f:
            reader = csv.reader(f)
            scene_names = np.array([row for row in reader]).flatten()
        scene_dirs = [pathlib.Path(class_list_path).parent / scene_name
                        for scene_name in scene_names]
        self.data_paths = [npz_path for scene_dir in scene_dirs
                                    for npz_path in scene_dir.glob('*[!full].npz')]
    
    def clear(self):
        self.seq_h = []
        self.seq_v = []
        self.seq_disp = []
        self.taerget_scenes = []
    
    def flow_from_directory(self, batch_size=64):
        for target_path in self.data_paths:
            loaded_data = np.load(target_path)
            self.seq_h.append(loaded_data['h'])
            self.seq_v.append(loaded_data['v'])
            self.seq_disp.append(loaded_data['disp'])
            self.taerget_scenes.append(re.search(r'[a-z]+_\d', str(target_path)).group())
            if len(self.seq_disp) == batch_size:
                seq_batch_h = np.array(self.seq_h, dtype=np.float32) / 255.0
                seq_batch_v = np.array(self.seq_v, dtype=np.float32) / 255.0
                seq_batch_disp = np.array(self.seq_disp, dtype=np.float32)
                scenes_batch = self.taerget_scenes
                self.clear()
                yield [seq_batch_h, seq_batch_v], seq_batch_disp, scenes_batch
        # return remaining data
        yield [seq_batch_h, seq_batch_v], seq_batch_disp, scenes_batch