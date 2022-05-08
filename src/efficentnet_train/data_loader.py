import os
import random
import sys
import time

import numpy as np
from os.path import join as join_pth
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset


def generate_testing_data_set_frame(dataset_pth, a_neg_single_subset=True):
    """
    :param dataset_pth: path to faces dataset root where root has root/person[i]/img[i] ...
    :param a_neg_single_subset: get acnchor negative from people with one photo only
    :return: dataFrame of pairs
    """

    namesList = np.array(os.listdir(dataset_pth))
    np.random.shuffle(namesList)
    all_persons_imgs = []  # [name, [list of photos] ]
    multi_img_persons = []  # [name, [list of photos] ]
    single_img_persons = []  # [name, [one_photo] ]

    for name in namesList:
        imgs = []
        for img_name in os.listdir(join_pth(dataset_pth, name)):
            imgs.append(img_name)
        all_persons_imgs.append([name, imgs])
        if len(imgs) == 1:
            single_img_persons.append([name, imgs])
        else:
            multi_img_persons.append([name, imgs])

    datasetList = []
    # anchor negative is from single photos only
    for person_photos in multi_img_persons:
        name = person_photos[0]
        photos = person_photos[1]
        for i in range(len(photos)):
            # anchor
            img1 = '{}/{}'.format(name, photos[i])
            for j in range(i + 1, len(photos)):
                # row (img1,img2, distance=0)
                # anchor postive
                datasetList.append([img1, '{}/{}'.format(name, photos[j]), 0])
                # anchor negative
                if a_neg_single_subset:
                    rand_person = random.choice(single_img_persons)
                    rand_person_name = rand_person[0]
                    rand_photo = rand_person[1][0]
                    datasetList.append([img1, '{}/{}'.format(rand_person_name, rand_photo), 1])
                else:
                    rand_person = random.choice(all_persons_imgs)
                    while rand_person[0] == name:
                        rand_person = random.choice(all_persons_imgs)
                    rand_person_name = rand_person[0]
                    rand_person_photos = rand_person[1]
                    rand_photo = np.random.choice(rand_person_photos)
                    datasetList.append([img1, '{}/{}'.format(rand_person_name, rand_photo), 1])
    return datasetList


def get_imgs_dict(dataset_pth) -> dict:
    """
    load all images in the dataset in a dictionary ,key is the image path relative to dataset path("person/image_name")
     and value is the image in numpy
    :param dataset_pth: the root path of the dataset
    :return: dict {img_path:numpy_img}
    """
    ts = time.time()
    images_dict = {}
    all_names = os.listdir(dataset_pth)
    cnt = 0.0
    total = float(len(all_names))
    for name in all_names:
        photos = os.listdir(join_pth(dataset_pth, name))
        for photo in photos:
            img_pth = join_pth(dataset_pth, name, photo)
            try:
                img = np.array(Image.open(img_pth))
                images_dict['{}/{}'.format(name, photo)] = img
            except:
                print("error loading --> " + img_pth)
        cnt += 1.0
        sys.stdout.flush()
        sys.stdout.write("\r " + str((cnt * 100.0) / total)[:8] + " % of the folders processed")
    te = time.time()

    print(f" img dict loaded in {round((te - ts) / 60.0, 2)} m")
    return images_dict


def data_generator(batch_size, imgs_dict, dataset_pth=None, transforms=None):
    names_list = np.array(os.listdir(dataset_pth))
    np.random.shuffle(names_list)
    multi_img_persons = []  # [(name, [list of photos])... ]
    single_img_persons = []  # [(name, [one_photo])... ]
    for name in names_list:
        imgs = []
        for img_name in os.listdir(join_pth(dataset_pth, name)):
            imgs.append(img_name)
        if len(imgs) == 1:
            single_img_persons.append((name, imgs))
        else:
            multi_img_persons.append((name, imgs))
    while True:
        a = []  # anchor batch
        p = []  # positive batch
        n = []  # negative batch
        for _ in range(batch_size):
            rand_single_img_person = random.choice(single_img_persons)

            rand_multi_photo_person = random.choice(multi_img_persons)
            random_two_same_pics = random.sample(rand_multi_photo_person[1], 2)
            # personName/img_name
            a_img_name = '{}/{}'.format(rand_multi_photo_person[0], random_two_same_pics[0])
            p_img_name = '{}/{}'.format(rand_multi_photo_person[0], random_two_same_pics[1])
            n_img_name = '{}/{}'.format(rand_single_img_person[0], rand_single_img_person[1][0])

            if transforms is not None:
                a.append(transforms(imgs_dict[a_img_name]))
                p.append(transforms(imgs_dict[p_img_name]))
                n.append(transforms(imgs_dict[n_img_name]))
            else:
                a.append(imgs_dict[a_img_name])
                p.append(imgs_dict[p_img_name])
                n.append(imgs_dict[n_img_name])
        a, p, n = np.array(a), np.array(p), np.array(n)

        if transforms is not None:
            yield tensor(a), tensor(p), tensor(n)
        else:
            yield a, p, n


class FacesDataset(Dataset):
    def __init__(self, imgs_dict, dataset_path, no_of_rows, transform=None):

        names_list = np.array(os.listdir(dataset_path))
        np.random.shuffle(names_list)
        self.multi_img_persons = []  # [(name, [list of photos])... ]
        self.single_img_persons = []  # [(name, [one_photo])... ]
        for name in names_list:
            imgs = []
            for img_name in os.listdir(join_pth(dataset_path, name)):
                imgs.append(img_name)
            if len(imgs) == 1:
                self.single_img_persons.append((name, imgs))
            else:
                self.multi_img_persons.append((name, imgs))
        self.imgs_dict = imgs_dict
        self.no_of_rows = no_of_rows
        self.transform = transform

    def __getitem__(self, idx):

        rand_single_img_person = random.choice(self.single_img_persons)
        rand_multi_photo_person = random.choice(self.multi_img_persons)
        random_two_same_pics = random.sample(rand_multi_photo_person[1], 2)
        # personName/img_name
        a_img_name = '{}/{}'.format(rand_multi_photo_person[0], random_two_same_pics[0])
        p_img_name = '{}/{}'.format(rand_multi_photo_person[0], random_two_same_pics[1])
        n_img_name = '{}/{}'.format(rand_single_img_person[0], rand_single_img_person[1][0])
        a_img = self.imgs_dict[a_img_name]
        p_img = self.imgs_dict[p_img_name]
        n_img = self.imgs_dict[n_img_name]

        if self.transform is not None:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)
        if idx == self.no_of_rows:
            raise StopIteration
        return a_img, p_img, n_img

    def __len__(self):
        return self.no_of_rows
