import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from torch.nn.utils import clip_grad_norm
import string

from dataset import TSNDataSet
from models import TSN, SimpleLinear, FinalLayer
from transforms import *
from opts import parser
from word_embedding import WordEmbedding

import pandas as pd

best_prec1 = 0


def main():
    global args, best_prec1, model_paths_by_type_dictionary
    model_paths_by_type_dictionary = define_ModelPaths()
    args = parser.parse_args()
    #args will contain filenames- trainsegment, testsegment, frames path, optical flow path
    #softmaxindex filename
    
    #Defining directories
    qa_directory = args.qa_directory
    data_directory = args.data_directory
    word_embedding_directory = args.word_embedding_directory
    
    train_QA = read_file(os.path.join(qa_directory, args.train_QA))
    test_QA = read_file(os.path.join(qa_directory, args.test_QA))
    SoftmaxIndex = read_file(os.path.join(qa_directory, args.SoftmaxIndex))
    #path files
    frames_path = os.path.join(data_directory, args.frames_path)
    optical_flow_path = os.path.join(data_directory, args.optical_flow_path)
    word_embedding_file_path = os.path.join(word_embedding_directory,args.word_embedding_file_path)
    test_train_combined_file = os.path.join(qa_directory,args.test_train_combined_file)
        
    #create 2 models, one for RGB, one for optical flow
    #optimizers for each model
    #model 3 will be tentatively for combining TSN vectors and bringing to 300 dimension to match Glove dimension
    #model 4, takes dot product and fully connected layers
    #optimizer 3 and optimizer 4 also needed.
    #all optimizers, and gradients must be checked.
    
    model_rgb, policies_rgb = get_model_components(modality = 'RGB')
    model_flow, policies_flow = get_model_components(modality = 'Flow')    
    simple_model = SimpleLinear()
    simple_model = convert_to_cuda(simple_model)
    final_layer_model = FinalLayer()
    final_layer_model = convert_to_cuda(final_layer_model)

    if args.resume:
        for model in [model_rgb, model_flow, simple_model, final_layer_model]:
            load_checkpoint(model)
            
    cudnn.benchmark = True

    # Data loading code
    #transforms creation done by function, stored in dictionary.
    #QA_segment for train and test to be loaded
    #Softmax Index to be loaded.
    transforms_train_dictionary = create_transforms_dictionary(model_rgb, model_flow, mode = 'train')
    transforms_test_dictionary = create_transforms_dictionary(model_rgb, model_flow, mode = 'test')
    
    print('Defining Data loader')
    train_loader = torch.utils.data.DataLoader(
    TSNDataSet( root_path = None, num_segments=3,
               frames_path = frames_path , optical_flow_path = optical_flow_path, 
               QA_Individual_segments = train_QA, SoftmaxIndex = SoftmaxIndex,
               transform= transforms_train_dictionary
               
               ),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
    TSNDataSet( root_path = None, num_segments=3,
               frames_path = frames_path , optical_flow_path = optical_flow_path, 
               QA_Individual_segments = test_QA, SoftmaxIndex = SoftmaxIndex,
               transform= transforms_test_dictionary
               
               ),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
    
    print('Models and Data loaders defined')
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    print('RGB model Policies')
    for group in policies_rgb:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    print('Flow model Policies')
    for group in policies_flow:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))


    #initializing all optimizers
    optimizer_rgb = torch.optim.SGD(policies_rgb,
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_flow = torch.optim.SGD(policies_flow,
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_simple = torch.optim.SGD(simple_model.parameters(),
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    optimizer_final_layer = torch.optim.SGD(final_layer_model.parameters(),
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print('Optimizers defined')    
    """Creating word embedding object"""
    WordEmbedding_object = create_word_embedding_object(test_train_combined_file, word_embedding_file_path)
    
    print('Word Embedding Object defined')
    if args.evaluate:
        validate(val_loader, model_rgb, model_flow, criterion, simple_model,
             final_layer_model, WordEmbedding_object, 0)
        return

    print('Start Training')
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch in progress')
        print(epoch)
        adjust_learning_rate(optimizer_rgb, epoch, args.lr_steps)
        adjust_learning_rate(optimizer_flow, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model_rgb, model_flow, criterion, 
              optimizer_rgb, optimizer_flow, simple_model,optimizer_simple, 
              final_layer_model, optimizer_final_layer, WordEmbedding_object, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            #prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
            prec1 = validate(val_loader, model_rgb, model_flow, criterion, simple_model,
             final_layer_model, WordEmbedding_object, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            
            """Modified checkpoint code, model.arch not relevant for simplemodel and finallayermodel"""
            for model in [model_rgb, model_flow, simple_model, final_layer_model]:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best
                , model.module.modality)

#Right now, we haven't yet computer feature embeddings. We will focus on that later, first we will compute losses.
def train(train_loader, model_rgb, model_flow, criterion, optimizer_rgb, 
          optimizer_flow, simple_model, optimizer_simple, final_layer_model, 
          optimizer_final_layer, WordEmbedding_object, epoch):
    #global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model_rgb.module.partialBN(False)
        model_flow.module.partialBN(False)
    else:
        model_rgb.module.partialBN(True)
        model_flow.module.partialBN(True)

    # switch to train mode
    model_rgb.train()
    model_flow.train()
    simple_model.train()
    final_layer_model.train()

    end = time.time()
    for i, (input_questions,input_rgb, input_flow, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.cuda(async=True)
        target = target.cuda()
        #Calling word embedding object to convert questions to glove vector tensors
        QuestionEmbeddings = WordEmbedding_object(input_questions)
        QuestionEmbeddings = QuestionEmbeddings.cuda()
        #Do autograd transformation
        autograd_transform(input_rgb)
        autograd_transform(input_flow)
        autograd_transform(target)

        # compute output from TSN
        output_TSN = compute_video_features(model_rgb, model_flow, input_rgb, input_flow)    
        output_TSN_reduced = simple_model(output_TSN)
        #output_TSN_reduced will be m*300 shape. We can run a trial on it and see.
        #Now we just take glove embeddings of questions
        #assume that we do have glove embeddings now. m*300. Thats the input_questions.
        #We compute dot product.
        output_TSN_reduced = output_TSN_reduced*QuestionEmbeddings
        #We have 156 softmax indices in total, from 0 to 155.
        output = final_layer_model(output_TSN_reduced)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data, input_rgb.size(0))
        top1.update(prec1.data, input_rgb.size(0))
        top5.update(prec5.data, input_rgb.size(0))


        # compute gradient and do SGD step
        #All optimizers zero grad.
        #Need to check if this function works.
        optimizers_zero_grad(optimizer_rgb, optimizer_flow, optimizer_simple, optimizer_final_layer)

        loss.backward()

        if args.clip_gradient is not None:
            for model in [model_rgb, model_flow, simple_model, final_layer_model]:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        
        #All optimizers step.
        #Need to check if this function works
        optimizers_step(optimizer_rgb, optimizer_flow, optimizer_simple, optimizer_final_layer)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer_rgb.param_groups[-1]['lr'])))


def validate(val_loader, model_rgb, model_flow, criterion, simple_model,
             final_layer_model, WordEmbedding_object,iter, logger=None):
    #global args
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_rgb.eval()
    model_flow.eval()
    simple_model.eval()
    final_layer_model.eval()

    end = time.time()
    for i, (input_questions,input_rgb, input_flow, target) in enumerate(val_loader):
        target = target.cuda()
        #target = target.cuda(async=True)
        QuestionEmbeddings = WordEmbedding_object(input_questions)
        QuestionEmbeddings = QuestionEmbeddings.cuda()
        #Do autograd transformation
        autograd_transform(input_rgb)
        autograd_transform(input_flow)
        autograd_transform(target)
        # compute output
        output_TSN = compute_video_features(model_rgb, model_flow, input_rgb, input_flow)    
        output_TSN_reduced = simple_model(output_TSN)
        output_TSN_reduced = output_TSN_reduced*QuestionEmbeddings
        
        output = final_layer_model(output_TSN_reduced)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data, input_rgb.size(0))
        top1.update(prec1.data, input_rgb.size(0))
        top5.update(prec5.data, input_rgb.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg

def define_ModelPaths():
    model_paths_by_type_dictionary = {
            'RGB':'./snapshots_model_RGB',
            'Flow':'./snapshots_model_Flow',
            'SimpleModel':'./snapshots_model_simple',
            'FinalLayer':'./snapshots_model_finalLayer'
            }
    return model_paths_by_type_dictionary

def save_checkpoint(state, is_best, modality, filename='checkpoint.pth.tar'):
    #global args
    #Making directory if not found
    for snapshot_path in list(model_paths_by_type_dictionary.values()):
        if not os.path.isdir(snapshot_path):
            os.makedirs(snapshot_path)
    
    FilePath = model_paths_by_type_dictionary[modality]        
    filename = '_'.join(('model', modality.lower(), filename))
    
    torch.save(state, os.path.join(FilePath, filename))
    if is_best:
        best_name = '_'.join(('model', modality.lower(), 'best.pth.tar'))
        shutil.copyfile(os.path.join(FilePath, filename), os.path.join(FilePath, best_name))

def load_checkpoint(model, filename='checkpoint.pth.tar'):
    global best_prec1
    modality = model.module.modality
    FilePath = model_paths_by_type_dictionary[modality]
    filename = '_'.join(('model', modality.lower(), filename))
    checkpoint = torch.load(os.path.join(FilePath, filename))
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    #global args
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def create_transforms_dictionary(*argv,mode = 'train'):
    #global args
    dictionary = {}
    for model in argv:
        crop_size = model.module.crop_size
        scale_size = model.module.scale_size
        input_mean = model.module.input_mean
        input_std = model.module.input_std
        #policies = model.get_optim_policies()
        train_augmentation = model.module.get_augmentation()     
    # Data loading code
        if model.module.modality != 'RGBDiff':
            normalize = GroupNormalize(input_mean, input_std)
        else:
            normalize = IdentityTransform()     
        if mode == 'train':
            transform=torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=model.module.arch == 'BNInception'),
                           ToTorchFormatTensor(div=model.module.arch != 'BNInception'),
                           normalize,
                       ])
        else:
            transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=model.module.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])
    
        dictionary[model.module.modality] = transform
    return dictionary

def convert_to_cuda(model):
    return torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

def get_model_components(modality):
    #global args
    model = TSN(num_segments = 3, modality = modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, partial_bn=not args.no_partialbn)
    
    policies = model.get_optim_policies()
    #model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()    
    model = convert_to_cuda(model)
    return model, policies

def read_file(filename):
    return pd.read_csv(filename)

def autograd_transform(input):
    input = torch.autograd.Variable(input)

def compute_video_features(model_rgb, model_flow, input_rgb, input_flow):
    output_rgb_model = model_rgb(input_rgb)
    output_flow_model = model_flow(input_flow)
    output = (output_rgb_model + output_flow_model) / 2
    return output    

def optimizers_zero_grad(*argv):
    for optimizer in argv:
        optimizer.zero_grad()

def optimizers_step(*argv):
    for optimizer in argv:
        optimizer.step()

def create_word_embedding_object(filename, embedding_file):
    dataframe = read_file(filename)
    series = dataframe['question']
    return WordEmbedding(series, embedding_file)
    

if __name__ == '__main__':
    main()
