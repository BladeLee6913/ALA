from torch.utils.data import Dataset
import PIL.Image as Image
import os

# 训练集数据载入
def make_dataset(root, images, labels):# 训练集目标载入
    imgs=[]
    
    file_path = os.path.join(root, images)    
    n=len(os.listdir(file_path))        
    for i in range(n):
        if i<1000:
            img=os.path.join(root,images,"%07d.png"%i)
            mask=os.path.join(root,labels,"%07d.png"%i)
        elif i>=1000:
            img=os.path.join(root,images,"%08d.png"%i)
            mask=os.path.join(root,labels,"%08d.png"%i)            
        imgs.append((img,mask))
    return imgs

# 测试集数据载入
# def make_dataset(root, images, labels):# 训练集目标载入
#     imgs=[]
    
#     file_path = os.path.join(root, images)    
#     n=len(os.listdir(file_path))        
#     for i in range(n):
#         img=os.path.join(root,images,"%08d.png"% (i+1883))
#         mask=os.path.join(root,labels,"%08d.png"% (i+1883))
        
#         imgs.append((img,mask))       
#     return imgs

class crackDataset(Dataset):
    def __init__(self, root, images, labels,transform=None, target_transform=None):
        imgs = make_dataset(root, images, labels)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

def make_target_dataset(root, images):# 训练集目标载入
    imgs=[]   
    file_path = os.path.join(root, images)    
    n=len(os.listdir(file_path))        
    for i in range(n):       
        img=os.path.join(root,images,"%d.png"%i)                       
        imgs.append(img,)
    return imgs


class fractureDataset(Dataset):
     def __init__(self, root, images, transform=None, target_transform=None):
         imgs = make_target_dataset(root, images)
         self.imgs = imgs
         self.transform = transform
         self.target_transform = target_transform

     def __getitem__(self,index):   # __getitem__()接收一个索引，返回索引对应的样本和标签
         x_path = self.imgs[index]
         img_x = Image.open(x_path)
         if self.transform is not None:
            img_x = self.transform(img_x)       
         return img_x

     def __len__(self):
         return len(self.imgs)


