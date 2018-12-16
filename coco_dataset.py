# Pytorch imports.
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def get_coco_captions_data_set(image_dir, anno_dir):
    """
    Use TorchVision's CoCoCaptions Data Set
    Args:
        image_dir (str)
        anno_dir (str)
    Returns:
        cap (Dataset)
    """
    cap = dset.CocoCaptions(root=image_dir,
                            annFile=anno_dir,
                            transform=transforms.ToTensor())
    return cap

def get_coco_captions_data_loader(image_dir, anno_dir, batch_size=4):
    """
    Get the data loader object for MS-COCO
    Args:
        image_dir (str)
        anno_dir (str)`
        batch_size (int): Number of examples to load at a given iteration

    Returns:
        data_loader (DataLoader)
    """
    data_set = get_coco_captions_data_set(image_dir, anno_dir)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader