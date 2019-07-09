import torch
import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

file_extensions = ['jpg','jpeg','JPG','JPEG','png','PNG']

# make dataset
def datafiles(datadir):
    img = []
    for file in os.listdir(datadir):
        if os.path.basename(file).split('.')[-1] in file_extensions:
            img.append(os.path.join(datadir,file))
    return img

class ImageFolder(data.Dataset):
    # loading image and processing
    def __init__(self,datadir,transform=None,return_paths=False):
        if len(datafiles(datadir)==0):
            raise(RuntimeError('Found no data in %s' % datadir))
        else:
            self.data = datafiles(datadir) 
        self.transform = transform
        self.return_paths = return_paths
        
    def __getitem__(self,index):
        imagefile = self.img[index]
        image = Image.open(imagefile).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.return_paths:
            return image,imagefile
        else:
            return iamge
            
    def __len__(self):
        return len(data)
        
def transform(new_size=None,crop_size=256,crop=True,is_train=True):
    transform_list = [transforms.ToTensor(),transforms.Normlize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    if crop:
        transform_list += [transforms.RandomCrop((crop_size,crop_size))]
    if new_size:
        transform_list += [transforms.Resize(new_size)] 
    if is_train:
        transform_list += [transforms.RandomHorizontalFlip()]
    transform = transforms.Compose(transform_list)
    return transform
    
def get_train_data(datadir,batch_size,num_workers,new_size=None,crop_size=256,crop=True,is_train=True):
    trainA_path = os.path.join(datadir,'trainA')
    trainB_path = os.path.join(datadir,'trainB')
    image_process = transform(new_size=None,crop_size=256,\
                    crop=True,is_train=True)
    trainA_data = ImageFolder(trainA_path,transform=image_process)
    trainB_data = ImageFolder(trainB_path,transform=image_process)
    trainA = DataLoader(dataset=trainA_data,batch_size=batch_size,\
             shuffle=is_train,num_workers=num_workers)
    trainB = DataLoader(dataset=trainB_data,batch_size=batch_size,\
             shuffle=is_train,num_workers=num_workers)
    return trainA, trainB
    
def get_test_data(datadir,batch_size,num_workers,new_size=None,crop_size=256,crop=True,is_train=False):
    testA_path = os.path.join(datadir,'testA')
    testB_path = os.path.join(datadir,'testB')
    image_process = transform(new_size=None,crop_size=256,\
                    crop=True,is_train=True)
    testA_data = ImageFolder(testA_path,transform=image_process)
    testB_data = ImageFolder(testB_path,transform=image_process)
    testA = DataLoader(dataset=testA_data,batch_size=batch_size,\
            shuffle=is_train,num_workers=num_workers)
    testB = DataLoader(dataset=testB_data,batch_size=batch_size,\
            shuffle=is_train,num_workers=num_workers)
    return testA, testB
    
def save_image(images,batch_size,save_path):
    images_grid = vutils.make_gird(iamges.data,nrow=batch_size,padding=0,normalize=True)
    vutils.save_image(image_grid, save_path, nrow=1)
