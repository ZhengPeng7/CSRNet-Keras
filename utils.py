import os
import cv2
import glob
import h5py
import numpy as np
from random import shuffle


def load_img(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img[:, :, 0]=(img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1]=(img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2]=(img[:, :, 2] - 0.406) / 0.225
    return img


def img_from_h5(path):
    gt_file = h5py.File(path, 'r')
    density_map = np.asarray(gt_file['density'])
    stride = 8
    density_map_quarter = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist())
    for r in range(density_map_quarter.shape[0]):
        for c in range(density_map_quarter.shape[1]):
            density_map_quarter[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
    return density_map_quarter


def gen_x_y(img_paths, train_val_test='train'):
    if train_val_test == 'train':
        shuffle(img_paths)
    x, y = [], []
    for i in img_paths:
        x_ = load_img(i)
        x.append(np.expand_dims(x_, axis=0))
        y_ = img_from_h5(i.replace('.jpg', '.h5').replace('images', 'ground'))
        y.append(np.expand_dims(np.expand_dims(y_, axis=0), axis=-1))
    return x, y, img_paths


def eval_loss(model, x, y):
    preds = []
    for i in x:
        preds.append(np.squeeze(model.predict(i)))
    labels = []
    for i in y:
        labels.append(np.squeeze(i))
    losses_DMD = []
    for i in range(len(preds)):
        losses_DMD.append(np.sum(np.abs(preds[i] - labels[i])))
    loss_DMD = np.mean(losses_DMD)
    losses_MAE = []
    for i in range(len(preds)):
        losses_MAE.append(np.abs(np.sum(preds[i]) - np.sum(labels[i])))
    loss_DMD = np.mean(losses_DMD)
    loss_MAE = np.mean(losses_MAE)
    return loss_DMD, loss_MAE


def gen_paths(path_file_root='data/paths_train_val_test', dataset='A'):
    path_file_root_curr = os.path.join(path_file_root, 'paths_'+dataset)
    img_paths = []
    for i in sorted([os.path.join(path_file_root_curr, p) for p in os.listdir(path_file_root_curr)]):
        with open(i, 'r') as fin:
            img_paths.append(eval(fin.read()))
    return img_paths    # img_paths_test, img_paths_train, img_paths_val


def eval_path_files(dataset="A", validation_split=0.2):
    root = 'data/ShanghaiTech/'
    paths_train = os.path.join(root, 'part_' + dataset, 'train_data', 'images')
    paths_test = os.path.join(root, 'part_' + dataset, 'test_data', 'images')

    img_paths_train = []
    for img_path in glob.glob(os.path.join(paths_train, '*.jpg')):
        img_paths_train.append(str(img_path))
    print("len(img_paths_train) =", len(img_paths_train))
    img_paths_test = []
    for img_path in glob.glob(os.path.join(paths_test, '*.jpg')):
        img_paths_test.append(str(img_path))
    print("len(img_paths_test) =", len(img_paths_test))

    from random import shuffle
    shuffle(img_paths_train)
    shuffle(img_paths_test)
    lst_to_write = [img_paths_train, img_paths_train[:int(len(img_paths_train)*validation_split)], img_paths_test]
    for idx, i in enumerate(['train', 'val', 'test']):
        with open('data/paths_train_val_test/paths_'+dataset+'/paths_'+i+'.txt', 'w') as fout:
            fout.write(str(lst_to_write[idx]))
            print('Writing to data/paths_train_val_test/paths_'+dataset+'/paths_'+i+'.txt')
