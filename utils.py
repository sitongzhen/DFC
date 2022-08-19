import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys, math, random
import os.path as osp
import torch

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        color_uni_label = np.unique(train_color_label)
        thermal_uni_label = np.unique(train_thermal_label)
        self.n_classes = len(color_uni_label)
        
        
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            color_batch_idx = np.random.choice(color_uni_label, batchSize, replace = False)
            thermal_batch_idx = np.random.choice(thermal_uni_label, batchSize, replace=False)

            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[color_batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[thermal_batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.35, r1=1.0, r2=3.0, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            # aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img