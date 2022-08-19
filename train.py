from __future__ import print_function
import argparse
import sys
import time
import collections
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
from cm import ClusterMemory, ImageMemory
from sklearn.cluster import DBSCAN
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData, new_RegDBData, new_SYSUData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from re_ranking import re_ranking
from loss import OriTripletLoss, TripletLoss_WRT
from faiss_rerank import compute_jaccard_distance
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=10, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/mnt/data/dataset/SYSU-MM01/SYSU-MM01'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/mnt/data/dataset/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
    # test_mode = [1, 2]  # thermal to visible
modal = ['', 'visible', 'thermal']
checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method=='agw':
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
    transforms.RandomGrayscale(p=0.5),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=modal[test_mode[1]])
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal=modal[test_mode[0]])

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local='off', gm_pool = 'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)
visible_image_memory = ImageMemory(2048, len(trainset.train_color_label), temp=args.temp,
                           momentum=args.momentum, use_hard=False).cuda()
thermal_image_memory = ImageMemory(2048, len(trainset.train_thermal_label), temp=args.temp,
                           momentum=args.momentum, use_hard=False).cuda()
cudnn.benchmark = True

# if len(args.resume) > 0:
if False:
    # model_path = '/mnt/data/stz/ideas/un-Cross-Modal-Re-ID-master2/save_model/sysu_base_p4_n8_lr_0.1_seed_0_epoch_60.t'#checkpoint_path + args.resume
    model_path = '/mnt/data/stz/ideas/JCCL-ReID-master/save_model/regdb_base_38.8-38.1.t'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
    return torch.exp(-kernel_input)  # (x_size, y_size)

def mmd_loss(x, y,reduction=None):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd
mmd_criterion = mmd_loss
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    # gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            # gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    # query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            # query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    query_feat = torch.from_numpy(query_feat)
    gall_feat = torch.from_numpy(gall_feat)
    # compute the similarity
    m, n = query_feat.shape[0], gall_feat.shape[0]
    distmat = torch.pow(query_feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gall_feat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, query_feat, gall_feat.t())
    distmat = distmat.cpu().numpy()
    # distmat = np.matmul(query_feat, np.transpose(gall_feat))
    print("Enter reranking")
    re_distmat = re_ranking(query_feat, gall_feat, k1=20, k2=6, lambda_value=0.3)
    # distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(re_distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(re_distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(60)
print(
         'result:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
             cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
# print(
#         'result:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
# training
print('==> Start Training...')
pool_dim = 2048
for epoch in range(start_epoch, 61 - start_epoch):
    eps = 0.6
    net.eval()
    with torch.no_grad():
        print('==> Create pseudo labels for unlabeled data')
        visible_train = TestData(trainset.train_color_image, trainset.train_color_label, transform=transform_test,
                                 img_size=(args.img_w, args.img_h))
        visible_train = data.DataLoader(visible_train, batch_size=loader_batch, \
                                        shuffle=False, num_workers=args.workers, drop_last=True)
        train_color = len(trainset.train_color_image)
        feat_p = np.zeros((train_color, pool_dim))
        # feat_fc = np.zeros((nquery, pool_dim))
        ptr = 0
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(visible_train):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, test_mode[1])
                feat_p[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                # feat_p[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        features = torch.from_numpy(feat_p).float()## torch.cat([feat_p[f].unsqueeze(0) for f in len(trainset.train_color_label)], 0)
        rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)
        # instance_memory.mem = features.cuda()
        if epoch == 0:
            # DBSCAN cluster
            visible_image_memory.features = features.cuda()
            print('Clustering criterion: eps: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=6, metric='precomputed', n_jobs=-1)

        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    # generate new dataset and calculate cluster centers
    @torch.no_grad()
    def generate_cluster_features(labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in centers.keys()
        ]
        centers = torch.stack(centers, dim=0)
        return centers

    visible_cluster_features = generate_cluster_features(pseudo_labels, features)
    del visible_train, feat_pool

    visible_pseudo_labeled = []
    visible_pseudo_labeled_dataset = []
    visible_pseudo_labeled_img = []
    for i, (img, label) in enumerate(zip(trainset.train_color_image, pseudo_labels.tolist())):
        visible_pseudo_labeled.append(label)
        # cam.append(cid)
        if label != -1:
            visible_pseudo_labeled_dataset.append(label)
            visible_pseudo_labeled_img.append([img, i])
    print(
        '==> visible Statistics for epoch {}: {} clusters, {} samples'.format(epoch, num_cluster, len(visible_pseudo_labeled_dataset)))
    visible_memory = ClusterMemory(2048, num_cluster, temp=args.temp,
                           momentum=args.momentum, use_hard=True).cuda()
    visible_memory.features = F.normalize(visible_cluster_features, dim=1).cuda()
    ############################################
    with torch.no_grad():
        print('==> Create pseudo labels for unlabeled data')
        thermal_train = TestData(trainset.train_thermal_image, trainset.train_thermal_label, transform=transform_test,
                                 img_size=(args.img_w, args.img_h))
        thermal_train = data.DataLoader(thermal_train, batch_size=loader_batch, \
                                        shuffle=False, num_workers=args.workers, drop_last=True)
        train_thermal = len(trainset.train_thermal_image)
        feat_p = np.zeros((train_thermal, pool_dim))
        # feat_fc = np.zeros((nquery, pool_dim))
        ptr = 0
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(thermal_train):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, test_mode[1])
                feat_p[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                # feat_p[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        features = torch.from_numpy(feat_p).float()## torch.cat([feat_p[f].unsqueeze(0) for f in len(trainset.train_color_label)], 0)
        rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)
        # instance_memory.mem = features.cuda()
        if epoch == 0:
            thermal_image_memory.features = features.cuda()
            # DBSCAN cluster
            print('Clustering criterion: eps: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=6, metric='precomputed', n_jobs=-1)

        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    # generate new dataset and calculate cluster centers
    @torch.no_grad()
    def generate_cluster_features(labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in centers.keys()
        ]
        centers = torch.stack(centers, dim=0)
        return centers

    thermal_cluster_features = generate_cluster_features(pseudo_labels, features)
    del thermal_train, feat_pool

    thermal_pseudo_labeled = []
    thermal_pseudo_labeled_dataset = []
    thermal_pseudo_labeled_img = []
    for i, (img, label) in enumerate(zip(trainset.train_thermal_image, pseudo_labels.tolist())):
        thermal_pseudo_labeled.append(label)
        # cam.append(cid)
        if label != -1:
            thermal_pseudo_labeled_dataset.append(label)
            thermal_pseudo_labeled_img.append([img, i])
    print(
        '==> infrared Statistics for epoch {}: {} clusters, {} samples'.format(epoch, num_cluster, len(thermal_pseudo_labeled_dataset)))
    thermal_memory = ClusterMemory(2048, num_cluster, temp=args.temp,
                                   momentum=args.momentum, use_hard=True).cuda()
    thermal_memory.features = F.normalize(thermal_cluster_features, dim=1).cuda()
    # identity sampler
    # train_loader, train_source = get_train_loader(pseudo_labeled_dataset, args.batch_size, args.workers, args.num_pos, 200,
    #                                               trainset=pseudo_labeled_dataset)
    color_pos, thermal_pos = GenIdx(visible_pseudo_labeled_dataset, thermal_pseudo_labeled_dataset)
    sampler = IdentitySampler(visible_pseudo_labeled_dataset, \
                              thermal_pseudo_labeled_dataset, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)
    print(epoch)

    loader_batch = args.batch_size * args.num_pos
    new_trainset = []
    new_trainset.append(visible_pseudo_labeled_img)
    new_trainset.append(visible_pseudo_labeled_dataset)
    new_trainset.append(thermal_pseudo_labeled_img)
    new_trainset.append(thermal_pseudo_labeled_dataset)
    new_trainset.append(sampler.index1)
    new_trainset.append(sampler.index2)
    new_trainset.append(transform_train)
    if dataset == 'sysu':
        new_trainset = new_SYSUData(new_trainset, transform=transform_train)
    elif dataset == 'regdb':
        new_trainset = new_RegDBData(new_trainset, args.trial, transform=transform_train)
    trainloader = data.DataLoader(new_trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 5 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print('result:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('re_ranking:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))
