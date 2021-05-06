import argparse
import os.path as osp
from Generator import generator
from Discriminator import FCDiscriminator
from dataloader import crackDataset
from dataloader import fractureDataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms


def get_arguments():
    parser = argparse.ArgumentParser(description = "Adversarial learning")
    parser.add_argument("--cpu", action = 'store_true', help="choose to use cpu device")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default = 2.5e-4, 
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning_rate_D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--snapshot_dir", type=str, default='./models_weights/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--iter_size", type=int, default=1,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    return parser.parse_args() 
args = get_arguments()


x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

y_transforms = transforms.ToTensor()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main(num_epochs = 20):
    device = torch.device("cuda" if not args.cpu else "cpu")     
    cudnn.enabled = True
    cudnn.benchmark = True
    
    
    model_unet = generator(1,1)
    model_unet.train()
    model_unet.to(device)
    
   
    model_D = FCDiscriminator(num_classes = 1)
    model_D.train()
    model_D.to(device)
    
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    
   
    optimizer_unet = optim.SGD(model_unet.parameters(),
                               lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))     
    optimizer_unet.zero_grad()
    optimizer_D.zero_grad()
    
    crack_dataset = crackDataset("source_data", "images_source", "labels", transform=x_transforms,target_transform=y_transforms)
    source_dataloaders = DataLoader(crack_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
       
    fracture_dataset = fractureDataset("source_data", "target_images", transform=x_transforms,target_transform=y_transforms)
    target_dataloaders = DataLoader(fracture_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
    targetloader_iter = enumerate(target_dataloaders)
        
    source_label = 0
    target_label = 1
    
    '''starting training'''
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*20)
        
        dt_size = len(source_dataloaders.dataset)
        steps = 0
               
        for images, labels in source_dataloaders:
            steps += 1
            try:            
                loss_seg_value = 0
                loss_adv_target_value1 = 0    
                loss_adv_value = 0
                
                optimizer_unet.zero_grad()
                optimizer_D.zero_grad()
               
                
                for sub_i in range(args.iter_size):
                    '''train G'''
                    for param in model_D.parameters():
                        param.requires_grad = False
                        
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    pred = model_unet(images)
                    
                    loss_seg = criterion(pred, labels)          
                    loss = loss_seg
                    loss = loss / args.iter_size     # proper normalization
                    loss.backward()
                    loss_seg_value += loss_seg.item() / args.iter_size
                    
                    _, batch = targetloader_iter.__next__() 
                                                
                    images = batch
                    images = images.to(device)
                    
                    target_output = model_unet(images)
                    pred_target = model_D(target_output)
                    
                    loss_adv = criterion(pred_target, torch.FloatTensor(pred_target.data.size()).fill_(source_label).to(device))
                    loss = loss_adv
                    loss = loss / args.iter_size
                    loss.backward()
                    loss_adv_target_value1 += loss_adv.item() / args.iter_size
                    
                    '''train D'''                  
                    for param in model_D.parameters():
                        param.requires_grad = True
                        
                    source_input = pred.detach()
                    D_out1 = model_D(source_input)
                    
                    loss_D = criterion(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
                    loss_D = loss_D / args.iter_size / 2
                    loss_D.backward()            
                    loss_adv_value += loss_D.item()
                    
                    # train with target
                    target_output = target_output.detach()
                    D_out2 = model_D(target_output)
                    loss_D = criterion(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))            
                    loss_D = loss_D / args.iter_size / 2            
                    loss_D.backward()            
                    loss_adv_value += loss_D.item()
                    
                optimizer_unet.step()
                optimizer_D.step()
                            
                print(
                'iter = {0:8d}/{1:8d}, loss_seg_value = {2:.3f}, loss_adv_target_value1 = {3:.3f}, loss_adv_value = {4:.3f}'.format(
                    steps, (dt_size - 1) // source_dataloaders.batch_size + 1, loss_seg_value, loss_adv_target_value1, loss_adv_value))    
                                       
            
            except StopIteration: # Since the number of target domains is much smaller than that of source domains, the target domain data needs to be reloaded when exiting 
                   del target_dataloaders
                   fracture_dataset = fractureDataset("source_data", "target_images", transform=x_transforms,target_transform=y_transforms)
                   target_dataloaders = DataLoader(fracture_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
                   targetloader_iter = enumerate(target_dataloaders)
                   
                   _, batch = targetloader_iter.__next__()              
                                                
                   images = batch
                   images = images.to(device)
                    
                   target_output = model_unet(images)
                   pred_target = model_D(target_output)
                    
                   loss_adv = criterion(pred_target, torch.FloatTensor(pred_target.data.size()).fill_(source_label).to(device))
                   loss = loss_adv
                   loss = loss / args.iter_size
                   loss.backward()
                   loss_adv_target_value1 += loss_adv.item() / args.iter_size
                    
                   '''train D'''
                   for param in model_D.parameters():
                        param.requires_grad = True
                        
                   source_input = pred.detach()
                   D_out1 = model_D(source_input)
                    
                   loss_D = criterion(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
                   loss_D = loss_D / args.iter_size / 2
                   loss_D.backward()            
                   loss_adv_value += loss_D.item()
                    
                    # train with target
                   target_output = target_output.detach()
                   D_out2 = model_D(target_output)
                   loss_D = criterion(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))            
                   loss_D = loss_D / args.iter_size / 2            
                   loss_D.backward()            
                   loss_adv_value += loss_D.item()
                    
                   optimizer_unet.step()
                   optimizer_D.step()
                            
                   print('exp = {}'.format(args.snapshot_dir))
                   print(
                    'iter = {0:8d}/{1:8d}, loss_seg_value = {2:.3f}, loss_adv_target_value1 = {3:.3f}, loss_adv_value = {4:.3f}'.format(
                    steps, (dt_size - 1) // source_dataloaders.batch_size + 1, loss_seg_value,loss_adv_target_value1, loss_adv_value))    
                                      
    print('save model ...')
    torch.save(model_unet.state_dict(), osp.join(args.snapshot_dir, 'Adversarial_learning_epoches_' + str(num_epochs) + '.pth'))
                   
if __name__ == '__main__':
   main(num_epochs = 100)
    
    