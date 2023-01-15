# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import tqdm
import random
import os
import sys
from datetime import datetime
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

from tensorboardX import SummaryWriter

from pymic.train_infer.loss import *
from pymic.train_infer.get_optimizer import get_optimiser
from pymic.train_infer.net_factory import get_network
from pymic.util.parse_config import parse_config

torch.backends.cudnn.benchmark = True

from pymic.self_supervised_tasks.dataset.nifi_dataset_ssl import NiftyDataset_SSL_RPL_ROT_Exemp
from pymic.self_supervised_tasks.dataset.nifi_dataset_ssl import NiftyDataset_SSL_RPL_ROT, NiftyDataset_SSL_RPL_ROT2
from pymic.self_supervised_tasks.algorithms.relative_patch_location import RelativePatchLocationModel
from pymic.self_supervised_tasks.algorithms.rotation import RotationModel
from pymic.self_supervised_tasks.algorithms.rpl_rot_exemp import RPL_ROT_EXEMP_MODEL
from pymic.self_supervised_tasks.algorithms.rpl_rot import RPL_ROT_MODEL

NUM_CLASS = 2

ALGORITHM_CE = ['rotation','rpl','rpl-rot-exemp']

def make_derangement(indices):
    if len(indices) == 1:
        return indices
    for i in range(len(indices) - 1, 0, -1):
        j = random.randrange(i)  # 0 <= j <= i-1
        indices[j], indices[i] = indices[i], indices[j]
    return indices

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

NORMAL_TRAIN_DIR = "../../Dataset_Rad2/gtv_normal_processed_uni/small_scale"
class TrainSSLAgent():
    def __init__(self, config,exp='debug'):
        self.config = config

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        self.model_name = self.config['SSL']['encoder'].lower()
        chpt_prefx  = os.path.join(self.config['training']['checkpoint_prefix'],self.model_name)
        self.output_dir = f"{chpt_prefx}__{dt_string}"
        os.makedirs(self.output_dir,exist_ok=True)
        self.exp = exp
        if exp != 'debug':        
            os.makedirs(self.output_dir,exist_ok=True)
        else:
            debug='thrash'
            self.output_dir = f"{chpt_prefx}__{debug}"
            os.makedirs(self.output_dir,exist_ok=True)


    def __create_dataset(self):
        data_root_dir  = self.config['dataset']['root_dir_train']

        img_normal_paths = []

        train_img_dir = os.path.join(NORMAL_TRAIN_DIR,'data')
        for train_normal_img in os.listdir(train_img_dir):
            train_img_normal_path = os.path.join(train_img_dir,train_normal_img)
            img_normal_paths.append(train_img_normal_path)
        
        random.shuffle(img_normal_paths)
        
        split_idx = int(len(img_normal_paths)*.80)
        train_img_paths = img_normal_paths[:split_idx]
        test_img_paths = img_normal_paths[split_idx:]

        train_dataset = NiftyDataset_SSL_RPL_ROT2(
            root_dir=data_root_dir,
            data_paths=train_img_paths,
            )
        valid_dataset = NiftyDataset_SSL_RPL_ROT2(
            root_dir=data_root_dir,
            data_paths=test_img_paths,
            )

        batch_size = self.config['training']['batch_size']
        self.batch_size = batch_size

        self.train_loader = torch.utils.data.DataLoader(train_dataset,pin_memory=True,
            drop_last=True , batch_size = batch_size, shuffle=True, num_workers=8)

        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,pin_memory=True,
            drop_last=True, batch_size = 4, shuffle=False, num_workers=2)
        self.val_batch_sz = 4
        



    def __create_network(self):
        print(self.config['SSL']['algorithm'])
        if self.config['SSL']['algorithm'] == 'rpl':
            self.net = RelativePatchLocationModel(self.config)
        elif self.config['SSL']['algorithm'] == 'rotation':
            self.net = RotationModel(self.config)
        elif self.config['SSL']['algorithm'] == 'rpl-rot-exemp':
            self.net = RPL_ROT_EXEMP_MODEL(self.config)
            print("Create Network RPL-ROT-EXEMP Model")
        elif self.config['SSL']['algorithm'] == 'rpl-rot':
            self.net = RPL_ROT_MODEL(self.config)
            print("Create Network RPL-ROT Model")
        if self.config['network']['use_pretrain']:
            self.__use_pretrained_encoder()
            
    
    def __use_pretrained_encoder(self):
        encoder_model = get_network(self.config['network'])
        ssl_model_dict = self.net._encoder.state_dict()
        encoder_model_dict = encoder_model.state_dict()
        encoder_model_dict = {k: v for k, v in encoder_model_dict.items() if k in ssl_model_dict}

        ssl_model_dict.update(encoder_model_dict)
        self.net._encoder.load_state_dict(ssl_model_dict)
        print("\n\n\n*********************************")
        print("Using Pretrained Encoder From Checkpoint:{}".format(self.config['network']['pretrained_model_path']))
        print("*********************************\n\n\n")


    def __create_optimizer(self):

        self.optimizer = get_optimiser(self.config['training']['optimizer'],
                self.net.parameters(),
                self.config['training'])
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,self.config['training']['lr_milestones'],self.config['training']['lr_gamma'])

        if self.config['SSL']['algorithm'] in ALGORITHM_CE:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def __train(self,exp):
        if self.exp != 'debug':
            self.train_txt = open( os.path.join(self.output_dir,'output_{}.txt'.format(exp)) ,'a')
        else:
            self.train_txt = open('output_{}.txt'.format(exp) ,'w')
        print(self.net,file=self.train_txt)
        print("Total number of parameters:{}".format(count_parameters(self.net)),file=self.train_txt)
        print(self.net)
        print("Total number of parameters:{}".format(count_parameters(self.net)))
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)



        chpt_prefx  = self.config['training']['checkpoint_prefix']
        iter_start  = self.config['training']['iter_start']

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None

        self.__create_optimizer()
        
        with open(os.path.join(self.output_dir,'config_file.txt'),'w') as f:
            print("Config File",file=f)
            for section in self.config:
                print("\n\nKey-Val pairs under Section -----> {}".format(section),file=f)
                for key,val in self.config[section].items():
                    print("{} ------> {}".format(key,val),file=f)

        print("{0:} training start".format(str(datetime.now())[:-7]))
        print("{0:} training start".format(str(datetime.now())[:-7]),file=self.train_txt)
        EPOCH_SAVE = 500
        VAL_EPOCH = 50
        tqdm_gem = tqdm.tqdm(self.train_loader)
        total_epoch = self.config['training']['max_epoch']
        it = 0
        self.net.train()
        epoch = 0

        self._test_epoch(epoch)
        while epoch < total_epoch:
            train_loss  = 0
            correct_acc_epoch = 0

            for tr_idx,data in enumerate(tqdm_gem):
                images_rpl = data['rpl']['image'].to(device)
                labels_rpl = data['rpl']['label'].to(device)
                labels_rpl = torch.argmax(labels_rpl.long(),dim=1)


                images_rot = data['rot']['image'].to(device)
                labels_rot = data['rot']['label'].to(device)
                labels_rot = torch.argmax(labels_rot.long(),dim=1)



                '''
                For RPLs
                '''
                images_rpl = images_rpl.view(2*self.batch_size,1,16,48,48)
                self.optimizer.zero_grad()

                outputs_rpl = self.net.forward_rpl(images_rpl)
                loss_rpl = self.loss(outputs_rpl,labels_rpl)

                outputs_rot = self.net.forward_rot(images_rot)
                loss_rot = self.loss(outputs_rot,labels_rot)
                
                total_loss = loss_rpl + loss_rot 
                total_loss.backward()
                self.optimizer.step()

                it += 1


                _, predicted_rpl = torch.max(outputs_rpl,dim=1)
                correct_acc_rpl = (predicted_rpl==labels_rpl).sum().item()

                _, predicted_rot = torch.max(outputs_rot,dim=1)
                correct_acc_rot = (predicted_rot==labels_rot).sum().item()
                correct_acc = (correct_acc_rot+correct_acc_rpl)/2

                correct_acc_epoch += correct_acc
                acc_iter = correct_acc/self.batch_size
                


                print("Epoch-->{}, Iteration-->{}, Loss-Value--->{}, Accuracy-->{}".format(epoch,it,total_loss.item(),acc_iter))
                print("Epoch-->{}, Iteration-->{}, Loss-Value--->{}, Accuracy-->{}".format(epoch,it,total_loss.item(),acc_iter),file=self.train_txt)

                # evaluate performance on validation set
                train_loss += total_loss.item()


            train_avg_loss = train_loss / len(tqdm_gem)
            epoch_accuracy = correct_acc_epoch/(self.batch_size*len(tqdm_gem))

            print('\n\n')
            print('\n\n',file=self.train_txt)
            print('****'*8)
            print('****'*8,file=self.train_txt)
            print("{0:} it {1:}, loss {2:.4f}".format(
                str(datetime.now())[:-7], it + 1, train_avg_loss))
            print("{0:} it {1:}, loss {2:.4f}".format(
                str(datetime.now())[:-7], it + 1, train_avg_loss),file=self.train_txt)
            print("Epoch-Average Loss:{}, Accuracy:{}".format(train_avg_loss,epoch_accuracy))
            print("Epoch-Average Loss:{}, Accuracy:{}".format(train_avg_loss,epoch_accuracy),file=self.train_txt)
            print('****'*8)
            print('****'*8,file=self.train_txt)
            print('\n\n')
            print('\n\n',file=self.train_txt)
            epoch += 1
            self.schedule.step()

            if (epoch % VAL_EPOCH == 0):
            # if True:
                self._test_epoch(epoch)

            if (epoch % EPOCH_SAVE ==  0) and (epoch > 5000):
                print('\n\n\n\n\n')
                print('\n\n\n\n\n',file=self.train_txt)
                print("####"*5)
                print("####"*5,file=self.train_txt)
                print("Saving Model")
                print("Saving Model",file=self.train_txt)

                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                # save_name = "{0:}_{1:}.pt".format(self.model_name, it + 1)
                save_name = "{0:}_{1:}.pt".format(self.model_name, epoch)
                save_path = os.path.join(self.output_dir,save_name)
                torch.save(save_dict, save_path)
                print("Full Save Path:{}".format(save_path))
                print("Full Save Path:{}".format(save_path),file=self.train_txt)
                print("Saved")
                print("Saved",file=self.train_txt)
                print("####"*5)
                print("####"*5,file=self.train_txt)
                print('\n\n\n\n\n')
                print('\n\n\n\n\n',file=self.train_txt)
            tqdm_gem = tqdm.tqdm(self.train_loader)

    def _test_epoch(self,epoch):

        torch.cuda.empty_cache()
        print('\n\n\n\n\n')
        print('\n\n\n\n\n',file=self.train_txt)
        print("$$$$"*5)
        print("$$$$"*5,file=self.train_txt)
        # self.net.eval()
        print("Begin Test-Validation at Epoch-->{}".format(epoch))
        print("Begin Test at Epoch-->{}".format(epoch),file=self.train_txt)

        device = torch.device(self.config['training']['device_name'])
        self.net.eval()

        start_time = time.time()
        avg_val_acc = []
        avg_val_loss = []
        with torch.no_grad():
            for iter_test,data in enumerate(self.valid_loader):
                # images = data['rpl']['image']
                # labels = torch.argmax(data['rpl']['label'].long(),dim=1)
                # images, labels = images.to(device), labels.to(device)

                images_rpl = data['rpl']['image'].to(device)
                labels_rpl = data['rpl']['label'].to(device)
                labels_rpl = torch.argmax(labels_rpl.long(),dim=1)
                images_rpl = images_rpl.view(2*self.val_batch_sz,1,16,48,48)

                images_rot = data['rot']['image'].to(device)
                labels_rot = data['rot']['label'].to(device)
                labels_rot = torch.argmax(labels_rot.long(),dim=1)
            
                outputs_rpl = self.net.forward_val_rpl(images_rpl)
                outputs_rot = self.net.forward_val_rot(images_rot)                

                loss_rpl = self.loss(outputs_rpl,labels_rpl)
                _, predicted_rpl = torch.max(outputs_rpl,dim=1)
                correct_acc_rpl = (predicted_rpl==labels_rpl).sum().item()
                acc_iter_rpl = correct_acc_rpl/len(labels_rpl)

                loss_rot = self.loss(outputs_rot,labels_rot)
                _, predicted_rot = torch.max(outputs_rot,dim=1)
                correct_acc_rot = (predicted_rot==labels_rot).sum().item()
                acc_iter_rot = correct_acc_rot/len(labels_rot)
                
                loss = loss_rpl+loss_rot
                acc_iter = (acc_iter_rot+acc_iter_rpl)/2

                
                avg_val_acc.append(acc_iter)
                avg_val_loss.append(loss.item())

                print("Validation: Epoch-->{}, Iteration-->{}, Loss-Value--->{}, Accuracy-->{}".format(epoch,iter_test,loss.item(),acc_iter))
                print("Validation: Epoch-->{}, Iteration-->{}, Loss-Value--->{}, Accuracy-->{}".format(epoch,iter_test,loss.item(),acc_iter),file=self.train_txt)


        avg_val_loss = sum(avg_val_loss)/len(avg_val_loss)
        avg_val_acc = sum(avg_val_acc)/len(avg_val_acc)
        print("\n\n\n")
        print("\n\n\n",file=self.train_txt)
        print("*****"*10)
        print("*****"*10,file=self.train_txt)

        avg_time = (time.time() - start_time) / len(self.valid_loader)
        print("Final Test Result at Epoch-->{}".format(epoch))
        print("Average testing time {0:}".format(avg_time))
        print("Final Test Result at Epoch-->{}".format(epoch),file=self.train_txt)        
        print("Average testing time {0:}".format(avg_time),file=self.train_txt)
        print('Average Validation Loss:{} and Validation Accuracy:{}'.format(avg_val_loss,avg_val_acc))
        print('Average Validation Loss:{} and Validation Accuracy:{}'.format(avg_val_loss,avg_val_acc),file=self.train_txt)
        print("Done Validation")
        print('*****************************************\n\n\n')
        print('*****************************************',file=self.train_txt)
        print("Done Validation",file=self.train_txt)
        print('*****************************************\n\n\n',file=self.train_txt)

        self.net.train()
    

    def run(self,exp):
        agent.__create_dataset()
        agent.__create_network()
        self.__train(exp)

import random
import numpy as np
seed = 30121994
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# torch.set_deterministic(True)

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print(sys.argv)
        print('python train_infer.py train config.cfg exp')
        exit()
    cfg_file = str(sys.argv[1])
    exp = str(sys.argv[2])


    config   = parse_config(cfg_file)
    agent    = TrainSSLAgent(config,exp)
    agent.run(exp)