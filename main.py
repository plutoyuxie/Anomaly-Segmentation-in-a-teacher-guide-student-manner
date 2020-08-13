import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pickle

import dataset.mvtec as mvtec


def parse_args():
    parser = argparse.ArgumentParser("vgg16_school")
    parser.add_argument("--dataset_path", type=str, default="D:/user/dataset/mvtec_anomaly_detection")
    parser.add_argument("--class_name", type=str, default="bottle")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--percent", type=int, default=99)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


class VGG(nn.Module):
    def __init__(self, pretrained_vgg):
        super(VGG, self).__init__()
        # 1st downsample: 224 -> 112      receptive field: 14
        self.stage1 = pretrained_vgg.features[:8]
        # 2nd downsample: 112 -> 56       receptive field: 40
        self.stage2 = pretrained_vgg.features[8:15]
        # 3rd downsample: 56 -> 28        receptive field: 92
        self.stage3 = pretrained_vgg.features[15:22]

    def forward(self, x):
        stage1_fea = self.stage1(x)
        stage2_fea = self.stage2(stage1_fea)
        stage3_fea = self.stage3(stage2_fea)
        
        return [stage1_fea, stage2_fea, stage3_fea]


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # data
    train_dataset = mvtec.MVTecDataset(args.dataset_path, args.class_name, is_train=True, resize=224, cropsize=224)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True)
    test_dataset = mvtec.MVTecDataset(args.dataset_path, args.class_name, is_train=False, resize=224, cropsize=224)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, pin_memory=True)
    # network
    pretrained_vgg = models.vgg16(pretrained=True)
    teacher = VGG(pretrained_vgg)
    teacher = teacher.to(device)
    teacher.eval()
    
    vgg = models.vgg16(pretrained=False)
    student = VGG(vgg)
    student = student.to(device)

    student_checkpoint = os.path.join(args.save_path, 'student_parameters_{}.pt'.format(args.class_name))
    if not os.path.exists(student_checkpoint):
        # train student
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, weight_decay=0.00001)

        for epoch in tqdm(range(args.max_epoch)):
            for img, _, _ in train_dataloader:
                img = img.to(device)
                
                with torch.no_grad():
                    surrogate_label = teacher(img)

                optimizer.zero_grad()
                pred = student(img)

                loss1 = criterion(pred[0], surrogate_label[0].detach())
                loss2 = criterion(pred[1], surrogate_label[1].detach())
                loss3 = criterion(pred[2], surrogate_label[2].detach())
                
                loss = 0.25*loss1 + 0.5*loss2 + loss3
                
                loss.backward()
                optimizer.step()

        torch.save(student.state_dict(), student_checkpoint)
    else:
        student.load_state_dict(torch.load(student_checkpoint))

    student.eval()

    def get_score_map_list(is_test, size=224):
        phase = 'test' if is_test else 'train'
        data_loader = test_dataloader if is_test else train_dataloader
        gt_mask_list = []  # pixel-level label
        test_imgs = []
        score_map_list = []
        for (x, _, mask) in tqdm(data_loader, '| feature extraction | %s | %s |' % (phase, args.class_name)):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            with torch.no_grad():
                surrogate_label = teacher(x.to(device))
                pred = student(x.to(device))
            
            score_maps = []
            for t, p in zip(surrogate_label, pred):
                score_map = torch.pow(t - p, 2).mean(1).unsqueeze(0)
                score_map = F.interpolate(score_map, size=size, mode='bilinear', align_corners=False)
                score_maps.append(score_map)
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)
            score_map = score_map.squeeze().cpu().detach().numpy()
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)
            if score_map.ndim < 3:
                score_map = np.expand_dims(score_map, axis=0)
            
            score_map_list.extend(score_map)

        return test_imgs, gt_mask_list, score_map_list

    ############################################################################################
    #                         get optimal threshold using ground_truth masks                   #
    ############################################################################################
    # flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
    # flatten_score_map_list = np.concatenate(score_map_list).ravel()

    # # calculate per-pixel level ROCAUC
    # fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
    # per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
    # # total_pixel_roc_auc.append(per_pixel_rocauc)
    # print('%s pixel ROCAUC: %.3f' % (args.class_name, per_pixel_rocauc))
    # # fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (args.class_name, per_pixel_rocauc))

    # # get optimal threshold
    # precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
    # a = 2 * precision * recall
    # b = precision + recall
    # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    # threshold = thresholds[np.argmax(f1)]

    ############################################################################################
    #                         estimate threshold based on positive samples                     #
    ############################################################################################
    if not args.threshold:
        threshold_path = os.path.join(args.save_path, 'threshold_{}.pt'.format(args.class_name))
        if not os.path.exists(threshold_path):
            _, _, score_map_list = get_score_map_list(is_test=False)
            thresholds = np.percentile(np.concatenate(score_map_list), list(range(101)))
            with open(threshold_path, 'wb') as threshold_file:
                pickle.dump(thresholds, threshold_file)
        else:    
            with open(threshold_path, 'rb') as threshold_file:
                thresholds = pickle.load(threshold_file)
        threshold = thresholds[args.percent]
        print('estimated threshold: %s' % threshold)
    else:
        threshold = args.threshold

    # testing
    test_imgs, gt_mask_list, score_map_list = get_score_map_list(is_test=True)
    # save images
    visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, args.class_name)


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, save_path, class_name):

    for t_idx in tqdm(range(len(test_imgs)), '| save images | test | %s |' % class_name):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images_{}'.format(class_name)), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images_{}'.format(class_name), '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == "__main__":
    main()
