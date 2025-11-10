
import argparse
import os
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
import json
from DADSN import DADSN
from util import get_random_index_patches, map_tooth_label


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=16000, help='point number [default: 16000]') ###
    parser.add_argument('--num_classes', type=int, default=17, help='Number of classes') ###
    parser.add_argument('--exp_name', type=str, default='debug', help='experiment root')  ###
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=32, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--expand_times', type=float, default=1)
    return parser.parse_args()


def calculate_dice(result, label, num_cls):
    dices = np.zeros(num_cls)
    nums = np.zeros(num_cls)

    for i in range(num_cls):
        gt = label == i
        pred = result == i

        gt_num = gt.sum()
        if gt_num > 0:
            intersection = (gt * pred).sum()
            unionset = gt.sum() + pred.sum()
            dice = (2 * intersection) / (unionset)
            dices[i] = dice
            nums[i] = gt_num

    # average_dice = dices.sum() / (nums > 0).sum()
    dices[nums == 0] = 1
    average_dice = dices.mean()
    return dices, nums, average_dice


def test(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'outputs/'+args.exp_name
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
    test_epoch = "best"

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = args.num_classes
    patch_point = args.npoint
    BATCH_SIZE = 4
    expand_times = args.expand_times

    with open(os.path.join(str(experiment_dir), "split_names.json")) as f:
        test_names = json.load(f)["val_names"]

    log_string("The number of test data is: %d" % len(test_names))

    '''MODEL LOADING'''
    classifier = DADSN(
        num_classes=NUM_CLASSES, k=args.k, emb_dims=args.emb_dims).cuda()

    if test_epoch == "best":
        checkpoint = torch.load(str(experiment_dir) + '/models/best_model.pth')
    else:
        checkpoint = torch.load(str(experiment_dir) + f'/models/model_epoch{test_epoch}.pth')
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        total_correct = 0
        total_seen = 0

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        total_dice = 0
        total_dice_values = 0
        total_gt_num = 0

        import time
        start = time.time()

        for i, name in tqdm(enumerate(test_names), total=len(test_names)):
            case_name = test_names[i]["npy_path"].split('/')[-1].split('.')[0]

            npy_path = name["npy_path"]
            label_path = name["label_path"]

            features = np.load(npy_path)  # Nx6
            vertices, normals = features[:, :3], features[:, 3:]

            label = json.load(open(label_path, 'r'))
            label = np.array(label['vertex_labels'])
            
            label = map_tooth_label(label)
            assert label.max() < NUM_CLASSES

            num_all = len(vertices)

            patch_indices = get_random_index_patches(num_all, patch_point, expand=expand_times)
            pred_prob = np.zeros((num_all, NUM_CLASSES), dtype=np.float32)
            pred_times = np.zeros(num_all, dtype=np.float32)

            points_list = []

            for selected_point_idxs in patch_indices:

                ori_points = vertices[selected_point_idxs]
                ori_normals = normals[selected_point_idxs]

                normalized_points = (ori_points - ori_points.mean(axis=0)) / ori_points.std(axis=0)
                points = np.concatenate([normalized_points, ori_normals], axis=1)

                points = torch.Tensor(points)
                points = points.float().transpose(1, 0)
                points_list.append(points)
            
            points = torch.stack(points_list, dim=0).cuda()
            seg_pred_all_list = []
            for j in range(0, len(points_list), BATCH_SIZE):
                batch_points = points[j:j+BATCH_SIZE]
                if len(batch_points) == 0:
                    continue
                seg_pred = classifier(batch_points)
                seg_pred_all_list.append(seg_pred)
            seg_pred_all = torch.cat(seg_pred_all_list, dim=0)
            seg_pred_all = seg_pred_all.contiguous().cpu().data.numpy()

            for j, selected_point_idxs in enumerate(patch_indices):
                pred_val = seg_pred_all[j]
                pred_prob[selected_point_idxs] += pred_val
                pred_times[selected_point_idxs] += 1

            pred_prob /= pred_times[:, None]
            output = np.argmax(pred_prob, axis=1)

            dice_values, gt_nums, average_dice = calculate_dice(output, label, NUM_CLASSES)

            total_dice += average_dice.astype(np.float32)
            valid_mask = (gt_nums > 0).astype(np.int32)
            total_dice_values += dice_values.astype(np.float32) * valid_mask
            total_gt_num += valid_mask

            correct = np.sum((output == label))
            total_correct += correct
            total_seen += num_all

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((label == l))
                total_correct_class[l] += np.sum((output == l) & (label == l))
                total_iou_deno_class[l] += np.sum(((output == l) | (label == l)))

        tsize = len(test_names)
        itime = (time.time() - start) / tsize
        log_string(f"Average time per case: {itime}, total size: {tsize}")

        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        log_string('eval point avg class IoU: %f' % (mIoU))
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (str(l), total_correct_class[l] / float(total_iou_deno_class[l]))

        log_string(iou_per_class_str)
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
        log_string(f"total dice: {total_dice / len(test_names)}")
        log_string(f"total dice values: {total_dice_values / total_gt_num}")
        log_string(f"total gt num: {total_gt_num}")

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    test(args)
