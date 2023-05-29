import os
import torch
import numpy as np
import sys

from week_4.crowd_count import CrowdCounter
from week_4 import network, utils
from week_4.data_loader import ImageDataLoader
from week_4.timer import Timer
from week_4.evaluate_model import evaluate_model


def log_print(text, color=None, on_color=None, attrs=None):
    print(text)



method = 'mcnn'
dataset_name = 'shtechA'
output_dir = './saved_models/'

train_path = '/Users/iraklisalia/Desktop/epam_cv_course/epam_cv_course/week_4/formatted_data/shanghaitech_part_A_patches_9/train'
train_gt_path = '/Users/iraklisalia/Desktop/epam_cv_course/epam_cv_course/week_4/formatted_data/shanghaitech_part_A_patches_9/train_den'
val_path = '/Users/iraklisalia/Desktop/epam_cv_course/epam_cv_course/week_4/formatted_data/shanghaitech_part_A_patches_9/val'
val_gt_path = '/Users/iraklisalia/Desktop/epam_cv_course/epam_cv_course/week_4/formatted_data/shanghaitech_part_A_patches_9/val_den'


#training configuration
start_step = 0
end_step = 2000
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)


# load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01)
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
best_mae = sys.maxsize


for epoch in range(start_step, end_step+1):    
    step = -1
    train_loss = 0
    for blob in data_loader:                
        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % disp_interval == 0:            
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)    
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            utils.save_results(im_data,gt_data,density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                step, 1./fps, gt_count,et_count)
            print(log_text)
            re_cnt = True
    
       
        if re_cnt:
            t.tic()
            re_cnt = False

    if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, net)
        #calculate error on the validation dataset
        mae,mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
        print(log_text)
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
        print(log_text)
