"""
Implements PADA:
"""
import argparse
import torch
from torch import nn
from tqdm import tqdm, trange
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
# update data_loader
import data_loader_all_type as data_loader
import numpy as np
import torch
import torch.nn as nn # neural network layers, ex: nn.Linear, nn.Conv2d, nn.ReLU
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
arg_parser.add_argument('--batch-size', type=int, default=64)
arg_parser.add_argument('--threshold', help='threshold for explained PCA variations', type=float, default=0.8)
# TW: CHANGE EPOCH DEFAULT = 50 (ORG=500)
arg_parser.add_argument('--epochs', type=int, default=50)#iterations per epoch= num of samples in data / batch size 
arg_parser.add_argument('--k-disc', type=int, default=5) # discriminator update frequency
arg_parser.add_argument('--k-clf', type=int, default=1) # classifier update frequency (job: correct predict class labels for the input data)
arg_parser.add_argument('--model', type=str, help='model name', default='resnet')
#updated 2024-12-19 change default = 0 from 1
arg_parser.add_argument('--cuda', type=int, help='cuda id', default=0)
# update default
arg_parser.add_argument('--source', type=str, default='DANN_BRCA_img')
arg_parser.add_argument('--target', type=str, default='WSI_resize_crop')
arg_parser.add_argument('--beta', type=float, default=1.)
arg_parser.add_argument('--lr', type=float, default=1e-3)  #learning rate
arg_parser.add_argument('--decay', type=float, default=1e-4)  #weight
arg_parser.add_argument('--baseline', type=float, default=0.7, help='baseline accuracy for saving') #only model's accuracy over 70% will be saved

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
args = arg_parser.parse_args()
# 2024-12-20 May have to replace N_Class to 4? (cell stage)
N_CLASS = 4
learning_rate = args.lr
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
#TW:Gy(Gf(x):V,c)=softmax(Gf(x)+c)
softmax = nn.Softmax(dim=1)
MOMENTUM = 0.9 # common value 0.9, 0.99, here 90% of the previous gradient 's influence is retained 
# weight decay parameter
DECAY = args.decay
n_disc = N_CLASS
criterion = nn.BCEWithLogitsLoss() #discriminator is trained to differentiate between domains
cls_criterion = nn.CrossEntropyLoss() # training the classifier to predict class (ex 31 class in office; 4 cell stage) 

def one_hot(batch,depth):
    ones = torch.sparse.torch.eye(depth).to(device)
    return ones.index_select(0,batch)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha): # alpha is chosen between 10**-2 and 1 (in DANN paper choose among 9 values) 
        ctx.alpha = alpha # store scaling factor alpha for later use in backpropagation 
        return x.view_as(x) # Pass the input 'x' unchanged

    @staticmethod
    def backward(ctx, grad_output):
        output=grad_output.neg()*ctx.alpha 
        return output, None

class disc(nn.Module):
    def __init__(self, model_name):
        super(disc, self).__init__() # call the constructor for the parent class (nn.Modile)
        if model_name == 'alexnet':
            n_features = 256 #input feature size
        elif model_name == 'resnet':
            n_features = 2048
        self.f1 = nn.Linear(n_features, 1024) # hidden layer 1 input size: n_feature, output size: 1024
        self.f2 = nn.Linear(1024,1024) # hidden layer 2
        self.logit = nn.Linear(1024, 1) # output layer
    def forward(self,x,alpha):
        x = ReverseLayerF.apply(x, alpha) # GRL
        x = F.dropout(F.relu(self.f1(x))) # pervent overfitting
        x = F.dropout(F.relu(self.f2(x)))
        x = self.logit(x) # un-normalized score (binary) for domain classification 
        return x




# load pre-trained model
def load_model(name='alexnet'):
    if name == 'alexnet':
        model = alexnet.alexnet(pretrained=True) # pre-trained: they already learned general features ( e,g edge, textures, and shapes)
        n_features = model.classifier[6].in_features # get the number of input features of the last layer. 
        fc = torch.nn.Linear(n_features, N_CLASS) # fully connect layer defined: nn.Linear(in_features, out_features), now we replace the output feature to N_CLASS 
        model.classifier[6] = fc # now we updated the alexnet with new fc layer
    elif name == 'resnet':
        model = resnet.resnet50(pretrained=True)
        n_features = model.fc.in_features
        fc = torch.nn.Linear(n_features, N_CLASS)
        model.fc = fc
    # update: 24-12-25 (begin)
    # feature extractor
    class FeatureExtractor(nn.Module):
        def __init__(self, backbone):
            super(FeatureExtractor, self).__init__()
            self.backbone = nn.Sequential(*list(backbone.children())[:-1]) # Remobe FC layer
        def forwaed(self,x):
            x=self.backbone(x)
            return x.view(x.size(0),-1) # flatten features

    # end    
    return model, feature_extractor




def main():
    # torch.manual_seed(100)
    # torch.cuda.manual_seed(100)
    ## load model name
    model_name = str(args.model)
    ## load disc model
    disc_model = disc(model_name).to(device)
    #disc_model.apply(weights_init)
    # update 24-12-25
    #model = load_model(model_name).to(device)
    model, feature_extractor = load_model(model_name)
    model = model.to(device) # new
    feature_extractor = feature_extractor.to(device) # new
    ## define optimizer


    def get_optimizer(model_name):
        if model_name == 'alexnet':
            param_group = [{'params': model.features.parameters(), 'lr': learning_rate}]
            for i in range(6):
                param_group += [{'params': model.classifier[i].parameters(), 'lr': learning_rate}] # += similar as append, which add another item
            param_group += [{'params': model.classifier[6].parameters(), 'lr': learning_rate * 10}] # applied higher learning_rate to final layer
        elif model_name == 'resnet':
            param_group = []
            for k, v in model.named_parameters():
                if not k.__contains__('fc'):
                    param_group += [{'params': v, 'lr': learning_rate}]
                else:
                    param_group += [{'params': v, 'lr': learning_rate * 10}] # applied higher learning_rate to final layer
        param_group += [{'params': disc_model.parameters(), 'lr': learning_rate * 10}]
        optimizer = optim.SGD(param_group, momentum=MOMENTUM, weight_decay=DECAY)
        #optimizer = optim.Adam(param_group, weight_decay=DECAY)
        return optimizer

    # Schedule learning rate
    def lr_schedule(optimizer, epoch):
        def lr_decay(LR, n_epoch, e):
            return LR / (1 + 10 * e / n_epoch) ** 0.75 #new lr
        for i in range(len(optimizer.param_groups)):
            if i < len(optimizer.param_groups) - 2:
                optimizer.param_groups[i]['lr'] = lr_decay(learning_rate, args.epochs, epoch)
            else:
                optimizer.param_groups[i]['lr'] = lr_decay(learning_rate, args.epochs, epoch) * 10

    ## load discriminator for multi-class classification
    half_batch = args.batch_size // 2
    #root_dir = 'data/DANN_ALL
    root_dir_src = '/scratch/wang_lab/BRCA_project/DANN_BRCA/data'
    root_dir_tar = '/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_ALL'
    BATCH_SIZE = {'src': int(half_batch), 'tar': int(half_batch)}
    domain = {'src': str(args.source), 'tar': str(args.target)}
    dataloaders = {}
    
    # 24-12-25 update data_loader for npy file 
    #target_loader = data_loader.load_data(root_dir, domain['tar'], BATCH_SIZE['tar'], 'tar') # batch: subset
    #target_loader_test = data_loader.load_data(root_dir, domain['tar'], BATCH_SIZE['tar'], 'test')
    #source_loader = data_loader.load_data(root_dir, domain['src'], BATCH_SIZE['src'], 'src')
    # if file is npy: use_npy=True; o.w False
    target_loader = load_data(root_dir_tar, domain['tar'], BATCH_SIZE['tar'], 'tar', use_npy=True)
    target_loader_test = load_data(root_dir_tar, domain['tar'], BATCH_SIZE['tar'], 'test', use_npy=True)
    source_loader = load_data(root_dir_src, domain['src'], BATCH_SIZE['src'], 'src', use_npy=False) 

    
    # print(target_loader)
    # print(source_loader)
    optimizer = get_optimizer(model_name)
    best_target_acc = 0.
    beta = args.beta
    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
        disc_loss_v = 0
        cls_loss_v = 0
        correct = 0.
        lr_schedule(optimizer, epoch-1) # lr adjustment

        len_dataloader = min(len(source_loader), len(target_loader))
        for l in trange(len_dataloader, leave=False): #trange create process bar while iterating, l is current iteration index
            # Train discriminator
            p = float(l + epoch*len_dataloader)/args.epochs/len_dataloader
            alpha = 2./(1.+np.exp(-10*p)) - 1 #scaling para 
            set_requires_grad(model, requires_grad=True) # allow update during backpropagation
            set_requires_grad(disc_model, requires_grad=True)
            model.train()
                # change made: 2024-12-20
                # SOURCE_Y is a cancer stage
                # targe Y remove ( TCGE'S label gives a random number)
                # org: (source_x, source_y), (target_x, target_y) = next(batch_iterator)
            (source_x, source_y), (target_x, _) = next(batch_iterator)
            source_x, target_x = source_x.to(device), target_x.to(device)
                # org: source_y, target_y = source_y.to(device), target_y.to(device)
            source_y = source_y.to(device)
            source_pred, source_feature = model(source_x)
            target_pred, target_feature = model(target_x)
            cls_loss = cls_criterion(source_pred, source_y)
                # discriminator
            discriminator_x = torch.cat([source_feature, target_feature]).squeeze() #remove extra singleton dimension
            #print(discriminator_x.size())
                # create labels for source (1) and target (0) domain
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device), 
                                         torch.zeros(target_x.shape[0], device=device)])
            disc_output = disc_model(discriminator_x, alpha).squeeze() # value = 1: discriminator thinks the feature belongs to the source domain; 0 o.w
            disc_loss = criterion(disc_output, discriminator_y) #loss for disc
            #print(disc_loss.size())
            disc_accuracy = ((disc_output > 0).long() == discriminator_y.long()).float().mean().item()
            optimizer.zero_grad() #resets the gradients of all parameters to avoid accumulation from previous steps.
            loss = cls_loss + beta*disc_loss
            loss.backward() #computes the gradients for the combined loss (cls_loss + beta * disc_loss).
            optimizer.step() #updates the model and discriminator parameters using the gradients.
            #beta = beta * (2 / (1 + np.exp(-10. * epoch / args.epochs)) - 1)  ## calculate beta
            disc_loss_v += disc_loss.item()
            cls_loss_v += cls_loss.item()

        mean_disc_loss = disc_loss_v / len_dataloader
        mean_cls_loss = cls_loss_v / len_dataloader

        ## evaluation the acc for target domain
          ## update 24-12-25 
        features_est=[]
        for inputs, _ in target_loader_test:
            inputs = inputs.to(device)
            feature_extractor.eval()
            features=feature_extractor(inputs)
            features_est.append(features.detach().cpu().numpy())

        # save extracted features
        features_est = np.concatenate(features_est,axis=0)
        np.save("extracted_features_target.npy",features_est)
        
        """
        features_est = []
        #labels_est1, labels_est = [],[] #comment out 12-25
            # 2024-12-20 replace label to _ since target doesn't have label
            # org: for inputs, labels in target_loader_test:
        for inputs, _ in target_loader_test:
                # org: inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(device)
            model.eval()
            preds, features = model(inputs)            
            labels_est1.append(preds.detach().cpu().numpy())
            preds = torch.max(preds, 1)[1]
            correct += torch.sum(preds == labels.data)
            features_est.append(features.detach().cpu().numpy())
            labels_est.append(preds.detach().cpu().numpy())
        target_acc = correct.double() / len(target_loader.dataset)
        target_acc = target_acc.cpu().numpy()

        features_est = np.concatenate(features_est,axis=0)
        labels_est = np.concatenate(labels_est,axis=0)
        labels_est1 = np.concatenate(labels_est1,axis=0)

        """
        
        #print(target_acc)
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            if best_target_acc > args.baseline:
                timestampTime = time.strftime("%H%M%S")
                timestampDate = time.strftime("%d%m%Y")
                timestampLaunch = timestampDate + '-' + timestampTime
                if not os.path.exists('dann_office/{}_{}/'.format(args.source, args.target)):
                    os.makedirs('dann_office/{}_{}/'.format(args.source, args.target))
                if not os.path.exists('dann_office/{}_{}/{}_{}_{}'.format(args.source, args.target, args.decay, args.lr, args.beta)):
                    os.makedirs('dann_office/{}_{}/{}_{}_{}'.format(args.source, args.target, args.decay, args.lr, args.beta))                
                torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'disc_model_state_dict': disc_model.state_dict(),
                    'optimizer' : optimizer.state_dict()}, 'dann_office/{}_{}/{}_{}_{}/'.format(args.source, args.target, args.decay, args.lr, args.beta) + timestampLaunch + '_' + str(epoch) + '_' + str(round(best_target_acc.item(),3)) + '.pth.tar')
                np.savez('dann_office/{}_{}/{}_{}_{}/'.format(args.source, args.target, args.decay, args.lr, args.beta) + timestampLaunch + '_' + str(epoch) + '_' + str(round(best_target_acc.item(),3)) + '.npz', features_est, labels_est, labels_est1)
        if epoch == args.epochs:
            print('save the last epoch model')
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampLaunch = timestampDate + '-' + timestampTime
            if not os.path.exists('dann_office/{}_{}/'.format(args.source, args.target)):
                os.makedirs('dann_office/{}_{}/'.format(args.source, args.target))
            if not os.path.exists('dann_office/{}_{}/{}_{}_{}'.format(args.source, args.target, args.decay, args.lr, args.beta)):
                os.makedirs('dann_office/{}_{}/{}_{}_{}'.format(args.source, args.target, args.decay, args.lr, args.beta))                
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'disc_model_state_dict': disc_model.state_dict(),
                'optimizer' : optimizer.state_dict()}, 'dann_office/{}_{}/{}_{}_{}/'.format(args.source, args.target, args.decay, args.lr, args.beta) + timestampLaunch + '_' + str(epoch) + '_' + str(round(target_acc.item(),3)) + '.pth.tar')
            np.savez('dann_office/{}_{}/{}_{}_{}/'.format(args.source, args.target, args.decay, args.lr, args.beta) + timestampLaunch + '_' + str(epoch) + '_' + str(round(target_acc.item(),3)) + '.npz', features_est, labels_est, labels_est1)


        tqdm.write(f'EPOCH {epoch:03d}: test_acc={target_acc:.4f}, disc_loss={mean_disc_loss:.4f}, cls_loss={mean_cls_loss:.4f}, disc_acc={disc_accuracy:.4f}')

    with open('dann_office.o', 'w') as f:
        f.write('save')

if __name__ == '__main__':

    main()
