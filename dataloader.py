import glob
import torch
import h5py
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.nn import DataParallel
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from math import sqrt
#####################################################################################################################
#Datalaoder
class GetFluidDataset(Dataset):
    def __init__(self, location, train, transform, upscale_factor, noise_ratio, std, patch_size, n_patches, method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std), 1, 1)
        self.transform = transform
        self._get_files_stats()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.method = method
        if (train == True) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.patch_size[0] / upscale_factor), int(self.patch_size[1] / upscale_factor)), Image.BICUBIC)
        elif (train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.img_shape_x / upscale_factor), int(self.img_shape_y / upscale_factor)), Image.BICUBIC)

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['fields'].shape[0]
            self.n_in_channels = _f['fields'].shape[1]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_files * self.n_samples_per_file
        self.files = [None for _ in range(self.n_files)]
        print("Number of samples per file: {}".format(self.n_samples_per_file))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['fields']  

    def __len__(self):
        if self.train:
            return self.n_samples_total * self.n_patches
        else:
            return self.n_samples_total

    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx) 
        if self.files[file_idx] is None:
            self._open_file(file_idx)
        y = self.transform(self.files[file_idx][local_idx])

        if self.train:
            patches = self.get_patches(y)
            return torch.stack(patches)  
        else:
            X = self.get_X(y)
            return X, y

    def get_indices(self, global_idx):
        if self.train:
            file_idx = int(global_idx / (self.n_samples_per_file * self.n_patches))
            local_idx = int((global_idx // self.n_patches) % self.n_samples_per_file)
        else:
            file_idx = int(global_idx / self.n_samples_per_file)
            local_idx = int(global_idx % self.n_samples_per_file)
        return file_idx, local_idx

    def get_X(self, y):
        if self.method == "uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.method == "noisy_uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
            X = X + self.noise_ratio * self.std * torch.randn(X.shape)
        elif self.method == "bicubic":
            X = self.bicubicDown_transform(y)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        return X

    def get_patches(self, y):
        patches = []
        patch_size_x, patch_size_y = self.patch_size
        for i in range(self.n_patches):
            rand_x = torch.randint(0, y.shape[1] - patch_size_x + 1, (1,))
            rand_y = torch.randint(0, y.shape[2] - patch_size_y + 1, (1,))
            patch = y[:, rand_x:rand_x + patch_size_x, rand_y:rand_y + patch_size_y]
            patches.append(patch)
        return patches

def get_data_loader(data_path, data_tag, train, upscale_factor, noise_ratio, crop_size, method, batch_size, n_patches, std):
    transform = torch.from_numpy
    dataset = GetFluidDataset(data_path + data_tag, train, transform, upscale_factor, noise_ratio, std, crop_size, n_patches, method) 
    dataloader = DataLoader(dataset,
                            batch_size=int(batch_size),
                            num_workers=2,
                            shuffle=train,
                            drop_last=False,
                            pin_memory=torch.cuda.is_available())
    return dataloader

def getData(upscale_factor, noise_ratio, crop_size, method, std, data_path="/ssd_scratch/flow_data/nskt16000_1024/", n_patches=8):
    train_loader = get_data_loader(data_path, '/train', True, upscale_factor, noise_ratio, crop_size, method, 32, n_patches, std)
    val1_loader = get_data_loader(data_path, '/valid_1', True, upscale_factor, noise_ratio, crop_size, method, 32, n_patches, std)
    val2_loader = get_data_loader(data_path, '/valid_2', True, upscale_factor, noise_ratio, crop_size, method, 32, n_patches, std) 
    test1_loader = get_data_loader(data_path, '/test_1', False, upscale_factor, noise_ratio, crop_size, method, 1, n_patches, std)
    test2_loader = get_data_loader(data_path, '/test_2', False, upscale_factor, noise_ratio, crop_size, method, 1, n_patches, std)
    return train_loader, val1_loader, val2_loader, test1_loader, test2_loader 

train_loader, val1_loader, val2_loader, test1_loader, test2_loader = getData(8, 0.05, (128, 128), 'bicubic', [0.6703, 0.6344, 8.3615], "/ssd_scratch/flow_data/nskt16000_1024/", 8)

# print(len(train_loader))
# print(len(val1_loader))
# print(len(val2_loader))
# print(len(test1_loader))
# print(len(test2_loader))