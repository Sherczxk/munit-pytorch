import time
from option import TestOptions
from data import get_train_data,get_test_data,save_image
import munit_trainer
import os

opt = TestOptions().parse()
trainer = munit_trainer(opt)

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
test_result_path = os.path.join(opt.output_dir,opt.exp_name,opt.direction,opt.style_tyle)
if not os.path.exists(test_result_path):
    os.makedirs(test_result_path)

state_dict = torch.load(opt.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()

testA, testB = get_test_data(opt.datadir,opt.batch_size,\
               opt.num_workers,opt.new_size,opt.crop_size)

if opt.direction == 'a2b':
    content_encode = self.gen_a.encode
    style_encode = self.gen_b.encode
    decode = self.gen_b.decode
    test_data_size = testA.size(0)
    test_data = testA
    style_sample_size = testB.size(0)
    style_sample = testB
else:
    content_encode = self.gen_a.encode
    style_encode = self.gen_b.encode
    decode = self.gen_b.decode
    test_data_size = testB.size(0)
    test_data = testB
    style_sample_size = testA.size(0)
    style_sample = testA
    
with torch.no_grad():
    for i in range(test_data_size):
        image = test_data.dataset[i]
        if opt.style_tyle == 'random':
            styles = torch.randn(opt.num_style, opt.style_dim, 1, 1).cuda()
        else:
            sample_numbers = [random.randint(i,style_sample_size) for i in range(opt.num_style)]
            style_image = torch.cat([style_sample.dataset[i].unsqueeze(0) for i in sample_numbers],dim=0)
            _, styles = style_encode(style_image)
        results = []
        for i in range(opt.num_style):
            style = styles[i]
            result = test(image,opt.direction,style)
            results.append(result)
        results = torch.cat(results,dim=0)
        display_result = torch.cat(tuple([input.unsqueeze(0),results]),dim=0)
        save_image(display_result,opt.nrow,test_result_path+'/   {:04d}.jpg'.format(i))

