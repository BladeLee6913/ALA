from __future__ import print_function
import os
import torch
from PIL import Image
from Generator import generator
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

class Transformer(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img_):
        img_ = img_.resize(self.size, self.interpolation)
        img_ = self.toTensor(img_)  
        img_.sub_(0.5).div_(0.5)   
        return img_
    
# 将输入的图片转换为tensor
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model_unet = generator(1,1)
model_path_unet = './models_weights/XXX.pth' 
model_unet.load_state_dict(torch.load(model_path_unet, map_location='cpu'))
imgs = os.listdir('./source_data/target_images/')


threshold = 175
table = []
for i in range(256):
    if i < threshold:
        table.append(0)  # append() 方法用于在列表末尾添加新的对象
    else:
        table.append(255)

for png in imgs:    
    img = Image.open('./source_data/target_images/'+png)
    trans = Transformer((352, 352))
    test_images = trans(img)
    
    model_unet.eval()# 在测试模型时，确保权值不变，不启用 BatchNormalization 和 Dropout
    # test_images = x_transforms(img)# 将图像转化为tensor
    image = test_images.unsqueeze(0)  
    label_image = model_unet(image).sigmoid()        
    label_image = torch.squeeze(label_image)
    
    show = ToPILImage()
    imgs = show((label_image +1)/2)   #转换的时候，会自动从0-1转换成0-256，所以0.5会变成127
    Img = imgs.convert('L')   
    img = Img.point(table, '1')        
    img.save('./source_data/target_output/'+png)



