import time
from option import TrainOptions
from data import get_train_data,get_test_data,save_image
import munit_trainer
import tensorboardX
import os 

opt = TrainOptions().parse()

trainer = munit_trainer(opt)
trainer.cuda()

##################################################################
# loading data --------------------------------------------
trainA, trainB = get_train_data(opt.datadir,opt.batch_size,\
                 opt.num_workers,opt.new_size,opt.crop_size)
                 
testA, testB = get_test_data(opt.datadir,opt.batch_size,\
               opt.num_workers,opt.new_size,opt.crop_size)
               
train_display_A = torch.stack([trainA.dataset[i] for i in range(opt.display_size)],-1).cuda()
train_display_B = torch.stack([trainB.dataset[i] for i in range(opt.display_size)],-1).cuda()
test_display_A = torch.stack([testA.dataset[i] for i in range(opt.display_size)],-1).cuda()
test_display_B = torch.stack([testB.dataset[i] for i in range(opt.display_size)],-1).cuda()
               
               
# output dir ------------------------------------------------
log_path = os.path.join(opt.output_path,'/logs',opt.exp_name)
save_image_path = os.path.join(opt.output_path,'/output',opt.exp_name,'/image')
save_checkpoint_path = os.path.join(opt.output_path,'/output',opt.exp_name,'/checkpoint')

for path in [log_path,save_image_path,save_checkpoint_path]:
    if not os.path.exists(path):
        os.mkdir(path)
        
train_sum = tensorboardX.SummaryWriter(os.path)

# start training ---------------------------------------------
for iter in range(opt.max_iter):
    for it,(x_a,x_b) in enumerate(zip(trainA,trainB)):
        x_a,x_b = x_a.cuda().detach(),x_b.cuda().detach()
        
    trianer.optimize_parameters(x_a,x_b)
    
    # update and save log file
    if (iter+1) % opt.log_save_iter ==0:
        print('Iteration: %08d/%08d' % (iter+1,opt.max_iter))
        members = []
        for attr in dir (trainer):
            if not callable(getattr(trainer,attr)) and not attr.startswith("__") \
            and ('loss' in attr or 'grad' in attr or 'nwd' in attr):
                members.append(attr)
        for m in members:
            train_sum.add_scalar(m,getattr(trainer,m),iter+q)
        
    # save current model
    if (iter+1) % opt.model_save_iter == 0:
        trainer.save(save_checkpoint_path,iter)
        
    # save sample images
    if (iter+1) % opt.image_save_iter == 0:
        with torch.no_grad():
            # train_a2b = [num_style*display_size,c,h,w]
            train_a2b,train_b2a = trainer.sample(train_display_A,train_display_B,opt.num_style)
            test_a2b,test_b2a = trainer.sample(test_display_A,test_display_B,opt.num_style)
            # sample_train = [2*(1+num_style)*display_size,c,h,w]
            sample_train = torch.cat(tuple([train_display_A,train_a2b,train_display_B,train_b2a]))
            sample_test = torch.cat(tuple([test_display_A,test_a2b,test_display_B,test_b2a]))
            # save sample images
            save_image(sample_train,save_image_path+'/train_%08d.jpg'%(iter+1))
            save_image(sample_test,save_image_path+'/test_%08d.jpg'%(iter+1))
            
    # display current translate images
    if (iter+1) % opt.image_display_iter == 0:
        with torch.no_grad():
            train_a2b,train_b2a = trainer.sample(train_display_A,train_display_B,opt.num_style)
            sample_train = torch.cat(tuple([train_display_A,train_a2b,train_display_B,train_b2a]))
            save_image(sample_train,save_image_path+'/train_current.jpg')

