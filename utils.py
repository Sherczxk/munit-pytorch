import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

################################# Loss ###########################

def recon_loss(x_a,x_b):
    loss = nn.L1Loss(reduce=True, size_average=True)
    return loss(x_a,x_b)
    
def dis_loss(x_real,x_fake,gan_type='lsgan'):
    if gan_type == 'lsgan':
        loss = nn.MSELoss(reduce=True, size_average=True)
        dis_loss = loss(x_real,torch.ones(x_real.size()).cuda())+\
                    loss(x_fake,torch.zeros(x_fake.size()).cuda())
    elif gan_type == 'nsgan':
        loss = nn.BCELoss(reduce=True, size_average=True)
        dis_loss = loss(F.sigmoid(x_real),torch.ones(x_real.size()).cuda())+\
                    loss(F.sigmoid(x_fake),torch.zeros(x_fake.size()).cuda())
    else:
        raise NotImplementedError('GAN type [%s] is not Unsupported' % gan_type)
    return dis_loss
    
def gen_loss(x_fake,gan_type='lsgan'):
    if gan_type == 'lsgan':
        loss = nn.MSELoss(reduce=True, size_average=True)
        gen_loss = loss(x_fake,torch.ones(x_fake.size()).cuda())
    elif gan_type == 'nsgan':
        loss = nn.BCELoss(reduce=True, size_average=True)
        gen_loss = loss(F.sigmoid(x_fake),torch.ones(x_fake.size()).cuda())
    else:
        raise NotImplementedError('GAN type [%s] is not Unsupported' % gan_type)
    return gen_loss
    
#################################### utils #######################

def weight_init(model,init_type='gaussian'):
    classname = model.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)