import torch
import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
		self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--datadir',type=str,default='./datasets/cat2dog',help='path to the dataset')
        self.parser.add_argument('--exp_name',type=str,default='cat2dog',help='experiment name')
        self.parser.add_argument('--checkpoint_dir',type=str,default='./checkpoint/cat2dog',help='path to the dataset')
        self.parser.add_argument('new_size',tpye=int,default=256,help='first resize the shortest image side to this size')
        self.parser.add_argument('crop_size',type=int,default=256,help='random crop size ')
        self.parser.add_argument('gan_w',type=int,default=1,help='weight of adversarial loss')
        self.parser.add_argument('x_w',type=int,default=10,help='weight of image reconstruction loss')
        self.parser.add_argument('c_w',type=int,default=1,help='weight of content reconstruction loss')
        self.parser.add_argument('s_w',type=int,default=1,help='weight of style reconstrcution loss')
        self.parser.add_argument('x_cyc_w',type=int,default=10,help='weight of cycle consistency loss')
        self.parser.add_argument('ngf',type=int,default=64,help='number of  filters in generator')
        self.parser.add_argument('ndf',type=int,default=64,help='number of  filters in discriminator')
        self.parser.add_argument('mlp_dim',type=int,default=256,help='number of  filters in MLP')
        self.parser.add_argument('style_dim',type=int,default=8,help='length of style code')
        self.parser.add_argument('--gan_type',type=str,default='lsgan',help='gan type of lsgan or nsgan')
        
        self.initialized = True
        
    def parser():
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
    
        # save the parameter
        args = vars(self.opt)
        expr_dir = os.path.join(self.opt.checkpoint_dir, self.opt.exp_name)
        if not os.path.exists(expr_dir):
            os.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
            
class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialized(self)
        
        self.parser.add_argument('--output_path',type=str,default='.',help='path of the output')
        self.parser.add_argument('--sample_dir',type=str,default='./sample/cat2dog',help='path to the save the sample images')
        self.parser.add_argument('--image_save_iter',type=int,default=1000,help='how often to save output images during training')
        self.parser.add_argument('--image_display_iter',type=int,default=100,help='how often to display images during training')
        self.parser.add_argument('--model_save_iter',type=int,default=10000,help='how often to save model')
        self.parser.add_argument('--log_save_iter',type=int,default=1,help='how often to save training state')
        self.parser.add_argument('--max_iter',type=int,default=1000000,help='maximum number of training iterations')
        self.parser.add_argument('--batch_size',type=int,default=1,help='batch size')
        self.parser.add_argument('--num_workers',type=int,default=4,help='num_workers')
        self.parser.add_argument('--display_size',type=int,default=8,help='diaplay size')
        self.parser.add_argument('--num_style',type=int,default=2,help='diaplay style number')
        self.parser.add_argument('--weight_delay',type=int,default=0.0001,help='weight delay')
        self.parser.add_argument('--beta_1',type=int,default=0.5,help='Adam parameter')
        self.parser.add_argument('--lr',type=int,default=0.0001,help='learning rate')
        
        self.isTrain = True
        
class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialized(self)
        self.parser.add_argument('--output_dir', type=str, default='./result', help="output image path")
        self.parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
        self.parser.add_argument('--direction', type=str, default='a2b', help="translation direction: a2b or b2a")
        self.parser.add_argument('--style_type', type=str, default='random', help="random style code:random, or sample from style images: sample ")
        self.parser.add_argument('--seed', type=int, default=10, help="random seed")
        self.parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
        self.parser.add_argument('--nrow',type=int, default=8, help="nrow of makegrid of test image")
        
        self.isTrain = False
        