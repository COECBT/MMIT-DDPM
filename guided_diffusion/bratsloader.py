import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel as nib
import blobfile as bf
import matplotlib.pyplot as plt

def load_data(data_dir, batch_size, image_size, test_flag=False, class_cond=False):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    """
    
    if not data_dir:
        raise ValueError("unspecified data directory")
        
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[3] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = BRATSDataset(all_files,classes, test_flag)
    return dataset


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        results.append(full_path)
    return results



class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, all_files, classes, test_flag, shard=0, num_shards=1):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = all_files
        self.local_images = all_files[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.test_flag=test_flag
        
    

    def __getitem__(self, x):
        file=self.local_images[x]
        path=file
        nib_img=nib.load(file)
        image=nib_img.get_fdata()
        if (np.count_nonzero(image)==0):
            norm=image
        else:
            norm=2*((image - np.min(image)) / (np.max(image) - np.min(image)))-1
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[x], dtype=np.int64)
        
        if(self.test_flag==False):
           norm=norm[..., 8:-8, 8:-8,:]  #crop to 224*224
           norm_tensor=torch.tensor(norm)
           return norm_tensor.permute(2,0,1),out_dict
        else:
            norm=norm[..., 8:-8, 8:-8]
            norm_tensor=torch.tensor(norm)
            norm_tensor=norm_tensor.unsqueeze(0)
            
            return norm_tensor,out_dict, path
        

    def __len__(self):
        return len(self.local_images)

    