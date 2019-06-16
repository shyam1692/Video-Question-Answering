import argparse

"""Let's decide on compulsary positional arguments and optional arguments"""
"""
Definite compulsary ones

1. train_QA (csv file)
2. test_QA (csv file)
3. SoftmaxIndex

6. word_embedding_file_path (txt file)
7. test_train_combined_file (for vocabulary for glove vectors loading)

Optional ones
1. QA_file_directory
2. data_directory
3. word_embedding_directory
4. frames_path
5. optical_flow_path

The ones to modify
1. resume

The ones we have to delete
1. modality
2. dataset
3. train_list
4. val_list
5. Dropout for now
6. Snapshot prefix
7. Flow prefix
"""

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('train_QA', type=str, help = 'train file name')
parser.add_argument('test_QA', type=str, help = 'test file name')
parser.add_argument('test_train_combined_file', type=str, help = 'test_train_combined_file file name')
parser.add_argument('SoftmaxIndex', type=str, help = 'SoftmaxIndex file name')
parser.add_argument('word_embedding_file_path', type=str, help = 'word_embedding_file_path')

# ========================= Model Configs ==========================
#defining the directories, keeping default values
parser.add_argument('--qa_directory', type=str, default="../QA_generation")
parser.add_argument('--data_directory', type=str, default="../Data")
parser.add_argument('--word_embedding_directory', type=str, default="../word_embedding")
parser.add_argument('--frames_path', type=str, default="Frames")
parser.add_argument('--optical_flow_path', type=str, default="Optical_Flow")
#end define directories
parser.add_argument('--arch', type=str, default="resnet18")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

#parser.add_argument('--dropout', '--do', default=0.5, type=float,
#                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    #help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', action = 'store_true',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)








