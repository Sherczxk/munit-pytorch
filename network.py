import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################
#  Discriminator
##############################################################

class Discriminator(nn.Module):
    def __init__(self,ndf):
        super(Discriminator,self).__init__()
        self.ndf = ndf
        self.norm = 'none'
        self.activ = 'lrelu'
        self.pad = 'reflect'
        self.num_scales = 3
        self.num_layer = 4
        self.downsample = nn.AvgPool2d(3,stride=2,padding=[1,1],count_include_pad=False)
        self.model = nn.ModuleList()
        for i in range(self.num_scales):
            self.model.append(self.scale_dis())
    
    def scale_dis(self):
        ndf = self.ndf
        dis_model = []
        dis_model += [Pad( self.pad,1),nn.Conv2d(3,ndf,4,2),Norm( norm_type='none',dim=ndf),Activ( self.activ)]
        for i in range(self.num_layer-1):
            dis_model += [Pad( self.pad_type,1),nn.Conv2d(ndf,ndf*2,4,2),Norm( self.norm,ndf*2),Activ( self.activ)]
            ndf *= 2
        dis_model += [nn.Conv2d(ndf,1,1,1,0)]
        dis_model = nn.Sequential(*dis_model)
        return dis_model
        
    def forward(self,x):
        outputs = []
        #discriminator coarse2fine
        for i in range(self.num_scales):
            outputs.append(self.scale_dis(x))
            x = self.downsample(x)
        return outputs
###############################################################
#  Generator
###############################################################

class Generator(nn.Module):
    def __init__(self,ngf,style_dim,mlp_dim):
        super(Generator,self).__init__()
        self.ngf = ngf
        self.style_dim = style_dim
        self.mlp_dim = mlp_dim
    
        self.Es = Style_Encoder(ngf,style_dim)
        self.Ec = Content_Encoder(ngf)
    
    def encode(self,x):
        content_code,_ = self.Ec(x)
        style_code = self.Es(x)
        return content_code,style_code
        
    def decode(self,content,style):
        content_dim = content.size(1)
        images = Decoder(self.style_dim,content_dim,self.mlp_dim,content,style)
        return images

    def forward(self,x):
        content,style = self.encode(x)
        x_rec = self.decode(content,style)
        return x_rec
        
#------------------------- Style Encoder -------------------------
class Style_Encoder(nn.Module):
    def __init__(self,ngf,style_dim):
        super(Style_Encoder,self).__init__()
        self.num_downsample_1 = 2
        self.num_downsample_2 = 2
        self.ngf = ngf
        self.style_dim = style_dim
        self.norm = 'none'
        self,activ = 'relu'
        self.pad = 'reflect'
        
        #  downsample->global_pooling->FC_layer
        self.model = []
        self.model += [Pad( self.pad,3),nn.Conv2d(3,ngf,7,1),Norm( self.norm,self.ngf),Activ( self.activ)]
        self.model += [self.downsampling_blocks()]
        self.model += [self.global_pooling()]
        self.model += [self.FC_layer()]
        self.model = nn.Sequential(*self.model)
    
    def forward(self,x):
        return self.model(x)
    
    def downsampling_blocks(self):
        ngf = self.ngf
        m = []
        for i in range(self.num_downsample_1):
            m += [Pad( self.pad,1),nn.Conv2d(ngf,ngf*2,4,2),Norm( self.norm,ngf*2),Activ( self.activ)]
            ngf *= 2
        for i in range(self.num_downsample_2):
            m += [Pad( self.pad,1),nn.Conv2d(ngf,ngf,4,2),Norm( self.norm,ngf),Activ( self.activ)]
        m = nn.Sequential(*m)
        return m
    
    def global_pooling(self):
        return nn.AdaptiveAvgPooling2d(1)
        
    def FC_layer(self):
        m = nn.Sequential([Pad( self.pad,0),nn.Conv2d(self.ngf,self.style_dim,1,1)])
        return m
    
#------------------------- Content Encoder -----------------------
class Content_Encoder(nn.Module):
    def __init__(self,ngf):
        super(Content_Encoder,self).__init__()
        self.num_downsample = 2
        self.num_res = 4
        self.ngf = ngf
        self.norm = 'in'
        self.activ = 'relu'
        self.pad = 'reflect'
        
        #  downsample->residual blocks
        self.model = []
        self.model += [Pad(self.pad,3),nn.Conv2d(3,ngf,7,1),Norm(self.norm,self.ngf),Activ(self.activ)]
        self.model += [self.downsampling_blocks()]
        self.model += [self.residual_blocks()]
        self.model = nn.Sequential(*self.model)
    
    def forward(self,x):
        return self.model(x)
        
    def downsampling_blocks(self):
        ngf = self.ngf
        m = []
        for i in range(self.num_downsample):
            m += [Pad( self.pad,1),nn.Conv2d(ngf,ngf*2,4,2),Norm( self.norm,ngf*2),Activ( self.activ)]
            ngf *= 2
        m = nn.Sequential(*m)
        return m
        
    def residual_blocks(self):
        ngf = self.ngf
        m = []
        for i in range(self.num_res):
            m += [residual_block(ngf,self.norm,self.activ,self.pad_type)]
        m = nn.Sequential(*m)
        return m
        
class residual_block(nn.Module):
    def __init__(self,dim,norm,activ,pad_type):
        super(residual_block,self).__init__()
        
        self.block = []
        self.block += [Pad(pad_type,1),nn.Conv2d(dim,dim,3,1),Norm(norm,dim),Activ(activ)]
        self.block += [Pad(pad_type,1),nn.Conv2d(dim,dim,3,1),Norm(norm),Activ(activ_type = 'none')]
        self.block = nn.Sequential(*self.block)
        
    def forward(self,x):
        input = x
        x = self.block(x)
        x += input
        return x
    
#------------------------- Decoder -------------------------------
class Decoder(nn.Module):
    def __init__(self,style_dim,content_dim,mlp_dim):
        super(Decoder,self).__init__()
        self.num_upsamle = 2
        self.num_res = 4
        self.content_dim = content_dim 
        self.activ = 'relu'
        self.pad = 'zero'
        self.upsample = nn.Upsample(scale_factor=2)
        self.num_adain_params = 2*2*content_dim*num_res
        self.mlp = MLP(style_dim,num_adain_params,mlp_dim)
    
        # AdIN_residual_blocks->upsampling
        
        self.AdainRes = self.AdIN_residual_block(self.content_dim,self.activ,self.pad_type)
        self.upsampling = self.upsampling_blocks()
        
    def forward(self,content,style):
        input_dim = self.content_dim
        adain_params = self.mlp(style)
        #AdIN_residual_blocks
        for i in range(self.num_res):
            adain_params_i = adain_params[:,input_dim*i:4*input_dim*(i+1)]
            content = self.AdainRes(adain_params_i,content)
        #upsampling
        content = self.upsampling(content)
        return content
        
    def upsampling_blocks(self):
        input_dim = self.content_dim
        m = []
        for i in range(self.num_upsamle):
            m += [self.upsample,Pad(self.pad,2),nn.Conv2d(input_dim,input_dim//2,5,1),Norm(norm_type='ln',dim=input_dim//2),Activ(self.activ)]
            input_dim //=2
        m += [Pad(pad_type='reflect',padding=3),nn.Conv2d(self.content_dim,3,7,1),Norm(norm_type='none',dim=3),Activ(activ_type='tanh')]
        m = nn.Sequential(*m)
        return m
        
class AdIN_residual_block(nn.Module):
    def __init__(self,dim,activ,pad_type):
        super(AdIN_residual_blocks,self).__init__()
        self.dim = dim
        self.norm_1 = AdainNorm2d(dim)
        self.norm_2 = AdainNorm2d(dim)
        
        self.model = []
        self.model += [Pad(pad_type,1),nn.Conv2d(dim,dim,3,1),self.norm_1,Activ(activ)]
        self.model += [Pad(pad_type,1),nn.Conv2d(dim,dim,3,1),self.norm_2,Activ(activ_type = 'none')]
        self.model = nn.Sequential(*self.model)
        
    def forward(self,adain_params,x):
        dim = self.dim
        self.norm_1.beta = adain_params[:,:dim]
        self.norm_1.gamma = adain_params[:,dim:2*dim]
        self.norm_2.beta = adain_params[:,2*dim:3*dim]
        self.norm_2.gamma = adain_params[:,3*dim:4*dim]
        input = x
        x = self.model(x)
        x += input
        return x
    
class MLP(nn.Module):
    '''
    MLP is to calculate the parameters (gamma,beta) of AdainNorm2d
    '''
    def __init__(self,input_dim,output_dim,mlp_dim):
        super(MLP,self).__init__()
        self.num_layer = 3
        self.norm = 'none'
        self.activ = 'relu'
        
        self.model = []
        self.model += [nn.Linear(input_dim,mlp_dim,bias=True),Norm(self.norm,mlp_dim),Activ(self.activ)]
        for i in range(self.num_layer-2):
            self.model += [nn.Linear(mlp_dim,mlp_dim,bias=True),Norm(self.norm,mlp_dim),Activ(self.activ)]
        self.model += [nn.Linear(mlp_dim,output_dim,bias=True),Norm(norm_type='none',dim=output_dim),Activ(activ_type='none')]
        self.model = nn.Sequential(*self.model)
    
    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.model(x)

class AdainNorm2d(nn.Module):
    def __init__(self,number_features,eps=1e-5,momentum=0.1):
        super(AdainNorm2d,self).__init__()
        self.number_features = number_features
        self.eps = eps
        self.momentum = momentum
        # gamma,beta will be  dynamically assigned by MLP
        self.gamma = None
        self.beta = None
    
    def forward(self,x):
        b,c,h,w = x.size()
        mean = torch.zeros(self.number_features*b)
        var = torch.ones(self.number_features*b)
        x_reshaped = x.contiguous().view(1,b*c,h,w)
        x_new = F.batch_norm(x_reshaped,mean,var,self.gamma,self.beta,True,self.momentum,self.eps)
        return x_new.view(b,c,h,w)
    
    
###########################  function ###############################

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
        
    def forward(self,x):
        return x
    
def Norm(norm_type='none',dim=64):
    
    if norm_type == 'in':
        norm_type = nn.InstanceNorm2d(dim)
    elif norm_type == 'bn':
        norm_type = nn.BatchNorm2d(dim)
    elif norm_type == 'ln':
        norm_type = nn.LayerNorm(dim)
    elif norm_type == 'none':
        norm_type = Identity()
    else:
        raise NotImplementedError('normalization method [%s] is not implemented' % norm_type)
    return norm_type
        
def Activ(activ_type='none'):
    if activ_type == 'relu':
        activ_type = nn.ReLU(inplace=True)
    elif activ_type == 'lrelu':
        activ_type = nn.LeakyReLU(0.2,inplace=True)
    elif activ_type == 'prelu':
        activ_type = nn.PReLU()
    elif activ_type == 'tanh':
        activ_type = nn.Tanh()
    elif activ_type == 'none':
        activ_type = Identity()
    else:
        raise NotImplementedError('activation method [%s] is not implemented' % activ_type)
    return activ_type
    
def Pad(pad_type='zero',padding=0):
    if pad_type == 'reflect':
        pad_type = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        pad_type = nn.RepicationPad2d(padding)
    elif pad_type == 'zero':
        pad_type = nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError('padding method [%s] is not implemented' % pad_type)
    return pad_type