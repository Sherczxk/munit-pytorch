import torch
import torch.nn as nn
import os
from network import Discriminator,Generator
from utils import recon_loss,gen_loss,dis_loss
from utils import weight_init

class MUNIT(nn.Module):
    def __init__(self,opt):
        super(MUNIT,self).__init__()
        
        # generators and discriminators
        self.gen_a = Generator(opt.ngf,opt.style_dim,opt.mlp_dim)
        self.gen_b = Generator(opt.ngf,opt.style_dim,opt.mlp_dim)
        self.dis_a = Discriminator(opt.ndf)
        self.dis_b = Discriminator(opt.ndf)
        #random style code
        self.s_a = torch.randn(opt.display_size,opt.style_dim,1,1,requires_grad=True).cuda()
        self.s_b = torch.randn(opt.display_size,opt.style_dim,1,1,requires_grad=True).cuda()
        
        #optimizers
        dis_params = list(self.dis_a.parameters())+list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters())+list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam(dis_params,lr=opt.lr,beta=opt.beta1,weight_delay=opt.weight_delay)
        self.gen_opt = torch.optim.Adam(gen_params,lr=opt.lr,beta=opt.beta1,weight_delay=opt.weight_delay)
        
        # nerwork weight initialization
        self.apply(weight_init('kaiming'))
        self.dis_a.apply(weight_init('gaussian'))
        self.dis_b.apply(weight_init('gaussian'))
        
    def forward(self,x_a,x_b):
        c_a,s_a_prime = self.gen_a.encode(x_a) 
        c_b,s_b_prime = self.gen_b.encode(x_b)
        x_a2b = self.gen_b.decode(c_a,self.s_b)
        x_b2a = self.gen_a.decode(c_b,self.s_a)
        return x_a2b,x_b2a
    
    def backward_G(self,x_a,x_b):
    
        #encoding and decoding
        s_a = torch.randn(x_a.size(0),opt.style_dim,1,1,requires_grad=True).cuda()
        s_b = torch.randn(x_b.size(0),opt.style_dim,1,1,requires_grad=True).cuda()
        c_a,s_a_prime = self.gen_a.encode(x_a) 
        c_b,s_b_prime = self.gen_b.encode(x_b)
        x_a_rec = self.gen_a.decode(c_a,s_a_prime)
        x_b_rec = self.gen_b.decode(c_b,s_b_prime)
        x_a2b = self.gen_b.decode(c_a,s_b)
        x_b2a = self.gen_a.decode(c_b,s_a)
        c_a_rec,s_b_rec = self.gen_b.encode(x_a2b)
        c_b_rec,s_a_rec = self.gen_a.encode(x_b2a)
        x_a_fake = self.dis_a(x_b2a)
        x_b_fake = self.dis_b(x_a2b)
            
        #loss function
        self.loss_xa = recon_loss(x_a_rec-x_a)
        self.loss_xb = recon_loss(x_b_rec-x_b)
        self.loss_ca = recon_loss(c_a_rec-c_a)
        self.loss_cb = recon_loss(c_b_rec-c_b)
        self.loss_sa = recon_loss(s_a_rec-s_a)
        self.loss_sb = recon_loss(s_b_rec-s_b)
        if opt.x_cyc_w>0:
            x_a2b2a = self.gen_b.decode(c_a_rec,s_a_prime)
            x_b2a2b = self.gen_a.decode(c_b_rec,s_b_prime)
            self.loss_cyc_a2b = recon_loss(x_a2b2a,x_a)
            self.loss_cyc_b2a = recon_loss(x_b2a2b,x_b)
        else:
            self.loss_cyc_a2b = 0
            self.loss_cyc_b2a = 0
        self.loss_gen_a = gen_loss(x_a_fake,opt.gan_type)
        self.loss_gen_b = gen_loss(x_b_fake,opt.gan_type)
        
        self.gen_total_loss = opt.gan_w*(self.loss_gen_a+self.loss_gen_b)+\
                              opt.x_w*(self.loss_xa+self.loss_xb)+\
                              opt.c_w*(self.loss_ca+self.loss_cb)+\
                              opt.s_w*(self.loss_sa+self.loss_sb)+\
                              opt.x_cyc_w*(self.loss_cyc_a2b+self.loss_cyc_b2a)
        self.gen_total_loss.backward()
        return self.gen_total_loss
        
    def barkward_D(self,x_a,x_b):
        s_a = torch.randn(x_a.size(0),opt.style_dim,1,1,requires_grad=True).cuda()
        s_b = torch.randn(x_b.size(0),opt.style_dim,1,1,requires_grad=True).cuda()
        c_a,_ = self.gen_a.encode(x_a.detach()) 
        c_b,_ = self.gen_b.encode(x_b.detach())
        x_a2b = self.gen_b.decode(c_a,s_b)
        x_b2a = self.gen_a.decode(c_b,s_a)
        x_a_real = self.dis_a(x_a.detach())
        x_a_fake = self.dis_a(x_b2a)
        x_b_real = self.dis_b(x_b.detach())
        x_b_fake = self.dis_b(x_a2b)
        
        self.loss_dis_a = dis_loss(x_a.detach(),x_b2a,opt.gan_type)
        self.loss_dis_b = dis_loss(x_b.detach(),x_a2b,opt.gan_type)
        self.dis_total_loss = opt.gan_w*(self.loss_dis_a+self.loss.dis_b)
        
        self.dis_total_loss.backward()
        return self.dis_total_loss
        
    def optimize_parameters(self,x_a,x_b):
        self.gen_opt.zero_grad()
        self.dis_opt.zero_grad()
        self.backward_G(x_a,x_b)
        self.barkward_D(x_a,x_b)
        self.gen_opt.step()
        self.dis_opt.step()
        
    def sample(self,x_a,x_b,num_style):
        x_a2b = []
        x_b2a = []
        for i in range(x_a.size(0)):
            s_a = torch.randn(num_style,opt.style_dim,1,1).cuda()
            s_b = torch.randn(num_style,opt.style_dim,1,1).cuda()
            c_a_i,_ = self.gen_a.encode(x_a[i].unsqueeze(0)) 
            c_b_i,_ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_i_a2b = []
            x_i_b2a = []
            for j in range(num_style):
                #[1,opt.style_dim,1,1]
                s_a_j = s_a[j].unsqueeze(0) 
                s_b_j = s_b[j].unsqueeze(0)
                x_i_a2b.append(self.gen_b.decode(c_a_i,s_b_j))
                x_i_b2a.append(self.gen_a.decode(c_b_i,s_a_j))
            #[num_style,c,h,w]
            x_i_a2b = torch.cat(x_i_a2b,dim=0) 
            x_i_b2a = torch.cat(x_i_b2a,dim=0)
            x_a2b.append(x_i_a2b)
            x_b2a.append(x_i_b2a)
        #[batch_size,num_style,c,h,w]
        x_a2b = torch.stack(x_a2b)
        x_b2a = torch.stack(x_b2a)
        #[batch_size*num_style,c,h,w]
        x_a2b = x_a2b.view(-1,x_a2b.size()[2:])
        x_b2a = x_b2a.view(-1,x_b2a.size()[2:])
        return x_a2b,x_b2a
        
        
    def test(self,input,direction,style):
        output = []
        if direction == 'a2b':
            encoder = self.gen_a.encode()
            decoder = self.gen_b.decode()
        else:
            encoder = self.gen_b.encode()
            decoder = self.gen_a.decode()
        content,_ = encode(input.unsqueeze(0))
        for i in range(style.size()):
            output.append(decoder(content,style[i].unsqueeze(0)))
        output = torch.cat(output,dim=0)
        return output
            
        
    def save(self,save_dir,iterations):
        save_gen = os.path.join(save_dir,'gen_%8d.pt'%(iteration+1))
        save_dis = os.path.join(save_dir,'dis_%8d.pt'%(iteration+1))
        save_opt = os.path.join(save_dir,'optimizer.pt')
        torch.save({'a':self.gen_a.state_dict(),'b':self.gen_b.state_dict()},save_gen)
        torch.save({'a':self.dis_a.state_dict(),'b':self.dis_b.state_dict()},save_dis)
        torch.save({'gen':self.gen_opt.state_dict(),'dis':self.dis_opt.state_dict()},save_opt)
        
    
