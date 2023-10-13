import torch
from torch import nn
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
from pandas import DataFrame
import random
import numpy as np
import os
from pathlib import Path
from tqdm.auto import tqdm

from monai.losses import DiceLoss
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    ScaleIntensityd,
    RandAffined,
    RandSpatialCropd,
    CenterSpatialCropd
)

from nnUNet.nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from nnUNet.nnunet.network_architecture.initialization import InitWeights_He

import torchmetrics


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0)) 


#===================================================================
program = "SFLV2 nnUNet on KiTS19"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     


#===================================================================
# No. of users
num_users = 6
epochs = 1
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV2
lr = 6e-4

#=====================================================================================================
#                           BASE Model definition
#=====================================================================================================
class Baseline(Generic_UNet):
    def __init__(self):
        pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        conv_kernel_sizes = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
        super(Baseline, self).__init__(
            input_channels=1,
            base_num_features=32,
            num_classes=3,
            num_pool=5,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0, "inplace": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=True,
            convolutional_upsampling=True,
            max_num_features=None,
            basic_block=ConvDropoutNormNonlin,
        )

def get_full_model():
    model = Baseline()
    return model

pretrained_model_path = '/data2/Shreyas/SPLIT_LEARNING/medical_split_learning/Datasets/kits19/models/pretrained/best_new_model.pth'

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side


class ClientSideModel(nn.Module):
    def __init__(self,input_channels=1,pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        full_model = get_full_model()
        model_state_dict = torch.load(pretrained_model_path)
        full_model.load_state_dict(model_state_dict)

        # conv_blocks_context, convolutional pooling in full model must be True
        self.front_contexts = full_model.conv_blocks_context[:2] # 1, 2
        
    def forward(self, x):
        self.skips = [] # reset skips every forward pass
        for stacked_conv_layer in self.front_contexts:
            x = stacked_conv_layer(x)
            self.skips.append(x)
        return x



net_glob_client = ClientSideModel()
net_glob_client.to(device)
print(net_glob_client)


#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side

class ServerSideModel(nn.Module):
    def __init__(self,pretrained=True,skips=[]):
        super().__init__()
        full_model = get_full_model()
        self.pretrained = pretrained
        full_model = get_full_model()
        model_state_dict = torch.load(pretrained_model_path)
        full_model.load_state_dict(model_state_dict)

        # conv_blocks_context, convolutional pooling in full model must be True
        self.contexts = full_model.conv_blocks_context[2:-1] # 3,4,5
        self.final_context = full_model.conv_blocks_context[-1] # 6
        self.tu = full_model.tu # all
        self.localizations = full_model.conv_blocks_localization # all
        self.final_seg_output = full_model.seg_outputs[-1]

        self.skips = []

        
    def forward(self, x):
        for stacked_conv_layer in self.contexts:
            x = stacked_conv_layer(x)
            self.skips.append(x)

        x = self.final_context(x)

        skips = skips[::-1]

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[u]), dim=1)
            x = self.localizations[u](x)
        
        x = self.final_seg_output(x)
        
        return x
    

net_glob_server = ServerSideModel()
net_glob_server.to(device)
print(net_glob_server)


#===================================================================================
# For Server Side Loss and Dice 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = DiceLoss(to_onehot_y=True,softmax=True)
count1 = 0
count2 = 0


#====================================================================================================
#                                  Server Side Programs
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# i'm keeping the function names same to reduce errors, calculate_accuracy will return DICE score
def calculate_accuracy(fx, y):
    dice = torchmetrics.functional.dice(
            preds=fx,
            target=y.long(),
            zero_division=1e-8,
            mdmc_average='samplewise', # multi-dim multi-class
            ignore_index=0, # ignore bg
            num_classes=3
        )
    return dice

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []


#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False


# Server-side function associated with Training 
def train_server(fx_client, y, skips, l_epoch_count, l_epoch, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user, lr
    
    net_glob_server.train()
    optimizer_server = torch.optim.AdamW(net_glob_server.parameters(), lr = lr)

    # set skips from client-side
    net_glob_server.skips = skips
    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_glob_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # server-side model net_glob_server is global so it is updated automatically in each pass to this function
    
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        prRed('Client{} Train => Local Epoch: {} \tDICE: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
                
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not
                       
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                        
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, skips, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    net_glob_server.skips = skips
    net_glob_server.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net_glob_server(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tDICE: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if all users are served for one round ----------                    
            if fed_check:
                fed_check = False
                                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                              
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg DICE {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg DICE {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
         
    return 


#==============================================================================================================
#                                       Clients Side Program
#==============================================================================================================


# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        #self.selected_clients = []

        self.ldr_train = DataLoader(dataset_train,batch_size=2,shuffle=True)
        self.ldr_test = DataLoader(dataset_test,batch_size=2,shuffle=False)
        

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.AdamW(net.parameters(), lr = self.lr) 
        
        for iter in tqdm(range(self.local_ep),desc='client train'):
            len_batch = len(self.ldr_train)
            for batch_idx, batch_data in tqdm(enumerate(self.ldr_train),total=len(self.ldr_train)):
                images, labels = batch_data['image'], batch_data['label']
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                skips = fx.skips
                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, skips, iter, self.local_ep, self.idx, len_batch)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, batch_data in tqdm(enumerate(self.ldr_test),total=len(self.ldr_test),desc='client eval'):
                images, labels = batch_data['image'], batch_data['label']
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                evaluate_server(fx, labels, self.idx, len_batch, ell)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return          


#=============================================================================
#                         Data loading 
#============================================================================= 

class KITSDataBuilder:
    def __init__(self,):
        self.thresholded_sites_path = Path('./data/kits19.csv').resolve()
        self.thresholded_sites = pd.read_csv(self.thresholded_sites_path)
        self.full_data_size = len(self.thresholded_sites)
        self.data_dir = Path('/data2/Shreyas/SPLIT_LEARNING/data/kits19/data').resolve()
        

    def get_client_cases(self,client_id, pool=False):
        if pool:
            client = self.thresholded_sites
        else:
            client = self.thresholded_sites.query(f'site_ids == {client_id}').reset_index(drop=True)
        main_cases = client.query(f"train_test_split == 'train'")['case_ids'].to_list()
        test_cases = client.query(f"train_test_split == 'test'")['case_ids'].to_list()
        return main_cases, test_cases
    
    def _make_dict(self,cases):
        return [
            {
                'image': self.data_dir/c/'imaging.nii.gz',
                'label': self.data_dir/c/'segmentation.nii.gz'
            } for c in cases
        ]
    
    def get_data_dict(self,client_id,pool,split_val=False):
        main_cases, test_cases = self.get_client_cases(client_id,pool)

        main_dict = self._make_dict(main_cases)
        test_dict = self._make_dict(test_cases)

        if split_val:
            train_dict, valid_dict = train_test_split(main_dict,test_size=0.1,shuffle=True)
            return train_dict, valid_dict, test_dict

        return main_dict, test_dict

    def get_data_transforms(
            self,
            voxel_spacing=(2.90,1.45,1.45),
            spatial_size=(96,96,96),
            is_dynamic=False
    ):
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], 
                     pixdim=voxel_spacing, 
                     mode=("bilinear", "nearest")
                    ),
            RandSpatialCropd(
                keys=["image","label"],
                roi_size=spatial_size
            ),
            ResizeWithPadOrCropd(keys=["image", "label"],
            spatial_size=spatial_size,
            mode='constant'),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=spatial_size,
                translate_range=(40, 40, 2),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            ),
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], 
                     pixdim=voxel_spacing, 
                     mode=("bilinear", "nearest")
                    ),
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=spatial_size
            ),
            ResizeWithPadOrCropd(keys=["image", "label"],
            spatial_size=spatial_size,
            mode='constant')
        ])

        return train_transforms, val_transforms
    
    def get_datasets(self,client_id,cache=False,cache_rate=1.0,pool=False,is_dynamic=False):
        train_files, test_files = self.get_data_dict(client_id,pool)
        train_tfms, val_tfms = self.get_data_transforms(is_dynamic=is_dynamic)
        
        train_ds = Dataset(data=train_files, transform=train_tfms)
        test_ds = Dataset(data=test_files, transform=val_tfms)

        return train_ds, test_ds
    
#=============================================================================
#                         Training & Testing
#=============================================================================
 

net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()

builder = KITSDataBuilder()

# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in tqdm(range(epochs)):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []
      
    for idx in tqdm(idxs_users):
        dataset_train, dataset_test = builder.get_datasets(idx)
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test)
        # Training ------------------
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        
        # Testing -------------------
        local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        
            
    # Ater serving all clients for its local epochs------------
    # Federation process at Client-Side------------------------
    print("------------------------------------------------------------")
    print("------ Fed Server: Federation process at Client-Side -------")
    print("------------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)   
    
    # Update client-side global model 
    net_glob_client.load_state_dict(w_glob_client)    
    
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'dice_train':acc_train_collect, 'dice_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "sflv2_kits19_test", index = False)     

#=============================================================================
#                         Program Completed
#=============================================================================
