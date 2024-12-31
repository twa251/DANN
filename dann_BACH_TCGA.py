"""
Implements PADA:
"""

"""
Updated by Tianyi Wang 
Dec-30-2024

"""
import argparse
import torch
from torch import nn
from tqdm import tqdm, trange
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
import data_loader_brca as data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from numpy.linalg import inv, eig
from torch.autograd import Function
import resnet
import alexnet
import torch.nn.functional as F
import time
import os,sys
#from finetune_office31 import load_model
# command setting
arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
arg_parser.add_argument('--batch-size', type=int, default=128) #update to 128
arg_parser.add_argument('--threshold', help='threshold for explained PCA variations', type=float, default=0.8)
arg_parser.add_argument('--epochs', type=int, default=20) #  run 20 epoch
arg_parser.add_argument('--k-disc', type=int, default=5)
arg_parser.add_argument('--k-clf', type=int, default=1)
arg_parser.add_argument('--model', type=str, help='model name', default='resnet')
arg_parser.add_argument('--cuda', type=int, help='cuda id', default=0) # org = 1
arg_parser.add_argument('--source', type=str, default='BACH')
arg_parser.add_argument('--target', type=str, default='TCGA')
arg_parser.add_argument('--beta', type=float, default=1.)
arg_parser.add_argument('--lr', type=float, default=1e-3)
arg_parser.add_argument('--decay', type=float, default=1e-4) # need try different lr
arg_parser.add_argument('--baseline', type=float, default=0.7, help='baseline accuracy for saving')

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
args = arg_parser.parse_args()
N_CLASS = 4
learning_rate = args.lr
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
softmax = nn.Softmax(dim=1)
MOMENTUM = 0.9
DECAY = args.decay
n_disc = N_CLASS
criterion = nn.BCEWithLogitsLoss()
cls_criterion = nn.CrossEntropyLoss()

# Not called in current code
def one_hot(batch,depth):
    ones = torch.sparse.torch.eye(depth).to(device)
    return ones.index_select(0,batch)

# Create directory
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output=grad_output.neg()*ctx.alpha
        return output, None

class disc(nn.Module):
    def __init__(self, model_name):
        super(disc, self).__init__()
        if model_name == 'alexnet': # we will not use this
            n_features = 256
        elif model_name == 'resnet':
            n_features = 2048
        self.f1 = nn.Linear(n_features, 1024)
        self.f2 = nn.Linear(1024,1024)
        self.logit = nn.Linear(1024, 1)
    def forward(self,x,alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = F.dropout(F.relu(self.f1(x)))
        x = F.dropout(F.relu(self.f2(x)))
        x = self.logit(x)
        return x




# default use ResNet model
def load_model(name='resnet'):
    if name == 'alexnet':
        model = alexnet.alexnet(pretrained=True)
        n_features = model.classifier[6].in_features
        fc = torch.nn.Linear(n_features, N_CLASS)
        model.classifier[6] = fc
    elif name == 'resnet':
        model = resnet.resnet50(pretrained=True)
        n_features = model.fc.in_features
        fc = torch.nn.Linear(n_features, N_CLASS)
        model.fc = fc
    return model




def main():

    model_name = str(args.model)
    disc_model = disc(model_name).to(device)
    model = load_model(model_name).to(device)
    ## define optimizer


    def get_optimizer(model_name):
        if model_name == 'alexnet':
            param_group = [{'params': model.features.parameters(), 'lr': learning_rate}]
            for i in range(6):
                param_group += [{'params': model.classifier[i].parameters(), 'lr': learning_rate}]
            param_group += [{'params': model.classifier[6].parameters(), 'lr': learning_rate * 10}]
        elif model_name == 'resnet':
            param_group = []
            for k, v in model.named_parameters():
                if not k.__contains__('fc'):
                    param_group += [{'params': v, 'lr': learning_rate}]
                else:
                    param_group += [{'params': v, 'lr': learning_rate * 10}]
        param_group += [{'params': disc_model.parameters(), 'lr': learning_rate * 10}]
        optimizer = optim.SGD(param_group, momentum=MOMENTUM, weight_decay=DECAY)
        # We may need to come back for optimizer later
        #optimizer = optim.Adam(param_group, weight_decay=DECAY)
        return optimizer

    # Schedule learning rate
    def lr_schedule(optimizer, epoch):
        def lr_decay(LR, n_epoch, e):
            return LR / (1 + 10 * e / n_epoch) ** 0.75
        for i in range(len(optimizer.param_groups)):
            if i < len(optimizer.param_groups) - 2:
                optimizer.param_groups[i]['lr'] = lr_decay(learning_rate, args.epochs, epoch)
            else:
                optimizer.param_groups[i]['lr'] = lr_decay(learning_rate, args.epochs, epoch) * 10

    ## load discriminator for multi-class classification
    half_batch = args.batch_size // 2
    root_dir = '/scratch/wang_lab/BRCA_project/Data'
    BATCH_SIZE = {'src': int(half_batch), 'tar': int(half_batch)}
    #domain = {'src': str(args.source), 'tar': str(args.target)}
    #dataloaders = {}
    # add validaiton loader
    #target_loader= data_loader.load_data(root_dir, domain['tar'], BATCH_SIZE['tar'], 'tar')
    #target_loader_test = data_loader.load_data(root_dir, domain['tar'], BATCH_SIZE['tar'], 'test')
    #source_loader = data_loader.load_data(root_dir, domain['src'], BATCH_SIZE['src'], 'src')
    train_loader, val_loader = data_loader.load_data(root_dir, args.source, args.batch_size,phase='src')  # UPDATED: Added validation loader
    target_loader = data_loader.load_data(root_dir, args.target, args.batch_size,phase='tar')  # UPDATED: Target data loader uses 80% split
    # print(target_loader)
    # print(source_loader)
    optimizer = get_optimizer(model_name)
    #best_target_acc = 0.
    #beta = args.beta
    # Save each epoch as a result
    checkpoint_dir = f'dann_brca/{args.source}_{args.target}/{args.decay}_{args.lr}_{args.beta}'
    ensure_dir(checkpoint_dir)

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(train_loader), loop_iterable(target_loader))
        disc_loss_v = 0
        cls_loss_v = 0
        correct = 0.
        lr_schedule(optimizer, epoch-1)

        len_dataloader = min(len(train_loader), len(target_loader))  #source_loader -> train_loader
        for l in trange(len_dataloader, leave=False):
            # Train discriminator
            p = float(l + epoch*len_dataloader)/args.epochs/len_dataloader
            alpha = 2./(1.+np.exp(-10*p)) - 1
            set_requires_grad(model, requires_grad=True)
            set_requires_grad(disc_model, requires_grad=True)
            model.train()
            (source_x, source_y), target_x = next(batch_iterator) # no target y needed
            source_x, target_x = source_x.to(device)
            source_y= source_y.to(device) # removed target y

            source_pred, source_feature = model(source_x)
            target_pred, target_feature = model(target_x)
            cls_loss = cls_criterion(source_pred, source_y)
            # discriminator
            discriminator_x = torch.cat([source_feature, target_feature]).squeeze()
            #print(discriminator_x.size())
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                         torch.zeros(target_x.shape[0], device=device)])
            disc_output = disc_model(discriminator_x, alpha).squeeze()
            disc_loss = criterion(disc_output, discriminator_y)
            #print(disc_loss.size())
            disc_accuracy = ((disc_output > 0).long() == discriminator_y.long()).float().mean().item()
            optimizer.zero_grad()
            loss = cls_loss + beta*disc_loss
            loss.backward()
            optimizer.step()
            #beta = beta * (2 / (1 + np.exp(-10. * epoch / args.epochs)) - 1)  ## calculate beta
            disc_loss_v += disc_loss.item()
            cls_loss_v += cls_loss.item()

        mean_disc_loss = disc_loss_v / len_dataloader
        mean_cls_loss = cls_loss_v / len_dataloader

        # Add validation loop
        model.eval()
        val_correct = 0.
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                preds = torch.max(preds, 1)[1]
                val_correct += torch.sum(preds == labels.data)
            val_acc = val_correct.double() / len(val_loader.dataset)
            print(f"Validation Accuracy: {val_acc:.4f}")

        ## evaluation the acc for target domain
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"{checkpoint_dir}/epoch_{epoch:02d}_valacc_{val_acc:.4f}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'disc_model_state_dict': disc_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'validation_accuracy': val_acc,
        'mean_disc_loss': mean_disc_loss,
        'mean_cls_loss': mean_cls_loss
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Save feature representations for debugging (NEW)
    feature_save_path = f"{checkpoint_dir}/features_epoch_{epoch:02d}.npz"
    features_est = np.concatenate([source_feature.detach().cpu().numpy(),
                                   target_feature.detach().cpu().numpy()], axis=0)
    np.savez(feature_save_path, features=features_est)
    print(f"Features saved at {feature_save_path}")


if __name__ == '__main__':

    main()
