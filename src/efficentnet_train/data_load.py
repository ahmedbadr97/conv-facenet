import os
import random
import sys
import time

import numpy as np
from os.path import join as join_pth
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.functional import normalize
from . import utils


def load_image(path, transform=None, expand=False):
    img = Image.open(path)
    if transform is not None:
        img = transform(img)
        if expand:
            img = img.unsqueeze(0)
    else:
        img = np.array(img)
    return img


def get_pic_features_dict(dataset_pth, model, transform=None, cuda=False):
    cnt = 0.0
    pic_features_dict = {}
    names = os.listdir(dataset_pth)
    model.eval()
    if cuda:
        model.cuda()
    total_time_taken = 0.0
    total_names = float(len(names))
    with torch.no_grad():
        for name in names:
            ts = time.time()
            folder_pth = join_pth(dataset_pth, name)
            pics_list = os.listdir(folder_pth)
            for pic_name in pics_list:
                features_vector = None
                try:
                    pic_path = join_pth(folder_pth, pic_name)
                    face = load_image(pic_path, transform, expand=True)

                    if cuda:
                        face = face.cuda()
                        features_vector = model(face)[0].cpu().tolist()
                    else:
                        features_vector = model(face)[0].tolist()
                except:
                    pic_features_dict[f'{name}/{pic_name}'] = None

                pic_features_dict[f'{name}/{pic_name}'] = features_vector
            cnt += 1.0
            te = time.time()
            total_time_taken += (te - ts)
            avg_time_per_name = total_time_taken / cnt
            finished_out_of_10 = int((cnt * 10.0) / total_names)
            remaining_out_of_10 = 10 - finished_out_of_10

            sys.stdout.flush()
            sys.stdout.write("\r data processed [" + str('=' * finished_out_of_10) + str(
                '.' * remaining_out_of_10) + "] time remaing=" + str(avg_time_per_name * (total_names - cnt) / 60.0)[
                                                                 0:5])

    model.train()
    return pic_features_dict


def get_imgs_dict(dataset_pth, all_names=None) -> dict:
    """
    load all images in the dataset in a dictionary ,key is the image path relative to dataset path("person/image_name")
     and value is the image in numpy
    :param dataset_pth: the root path of the dataset
    :return: dict {img_path:numpy_img}
    """
    ts = time.time()
    images_dict = {}
    if all_names is None:
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


class FacesDataset(IterableDataset):
    def __init__(self, dataset_path, no_of_rows, transform=None, load_imgs_from_dict=False, img_features_dict=None,
                 subset=None,
                 select_from_negative_cnt=0):
        """
        :param dataset_path: path that has folder of identities and each identity has it's photos
        :param no_of_rows: limit of no of triplet rows you wish to generate (anchor_img,postive_img,negative_img)
        :param subset:(percentage from 0.0 to 1.0) specify a percentage you wish to take from (identities) not all dataset
        :param transform: (torch.transforms)
        :param load_imgs_from_dict:(boolean) load all images from  {"identity/img_name":img} dictionary
        :param img_features_dict: dictionary of {"identity/img_name":img_features} to select hard negative photo
        :param select_from_negative_cnt: no of randomly chosen images you wish to have to select the hardest negative photo
        """
        names_list = os.listdir(dataset_path)
        random.shuffle(names_list)
        if subset is not None:
            names_list = random.sample(names_list, int(subset * len(names_list)))
        self.person_imgs_list = []  # [(name, [list of photos])... ]
        for name in names_list:
            imgs = []
            for img_name in os.listdir(join_pth(dataset_path, name)):
                imgs.append(img_name)
            self.person_imgs_list.append((name, imgs))
        self.no_of_rows = no_of_rows
        self.transform = transform
        self.dataset_path = dataset_path
        self.img_features_dict = img_features_dict
        self.select_from_negative_cnt = select_from_negative_cnt
        self.load_imgs_from_dict = load_imgs_from_dict
        if load_imgs_from_dict:
            self.images_dict = get_imgs_dict(dataset_path, names_list)

    def get_min_dist_face(self, anchor_name, anchor_img_name):
        """
        get negative image which is closer to the positive image
        :param anchor_name:
        :param anchor_img_name:
        :return: img_path for the chosen image
        """
        if self.select_from_negative_cnt == 0 or self.img_features_dict is None:
            negative_name, negative_imgs = random.choice(self.person_imgs_list)
            while negative_name == anchor_name:
                negative_name, negative_imgs = random.choice(self.person_imgs_list)
            return f"{negative_name}/{random.choice(negative_imgs)}"
        imgs_path = []
        anchor_vector = self.img_features_dict[f"{anchor_name}/{anchor_img_name}"]
        for i in range(self.select_from_negative_cnt):
            person_name, person_imgs = random.choice(self.person_imgs_list)
            while person_name == anchor_name:
                person_name, person_imgs = random.choice(self.person_imgs_list)
            rand_img = random.choice(person_imgs)
            imgs_path.append(f"{person_name}/{rand_img}")
        min_dist = np.Inf
        min_img_path = ""
        for img_path in imgs_path:
            negative_vector = self.img_features_dict[img_path]
            dist = utils.euclidean_distance(anchor_vector, negative_vector)
            if dist < min_dist:
                min_img_path = img_path
                min_dist = dist
        return min_img_path

    def load_imgs_dict(self):
        self.images_dict = get_imgs_dict(self.dataset_path, )

    def __iter__(self):
        for i in range(self.no_of_rows):
            # 1- select random anchor person
            anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            while len(anchor_imgs) < 2:
                anchor_per_name, anchor_imgs = random.choice(self.person_imgs_list)
            # 2- select two random pictures for the choosen person
            random_two_same_pics = random.sample(anchor_imgs, 2)

            a_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[0])
            p_img_name = '{}/{}'.format(anchor_per_name, random_two_same_pics[1])

            # 3- select random negative picture that it's feature close to the chosen person
            n_img_name = self.get_min_dist_face(anchor_per_name, anchor_img_name=random_two_same_pics[0])

            a_img = self.load_img(a_img_name)
            p_img = self.load_img(p_img_name)
            n_img = self.load_img(n_img_name)

            yield a_img, p_img, n_img

    def load_img(self, img_path):
        if not self.load_imgs_from_dict or img_path not in self.images_dict:
            img = np.array(Image.open(self.dataset_path + "/" + img_path))
        else:
            img = self.images_dict[img_path]

        if self.transform is not None:
            img=self.transform(img)
        return img

    def __len__(self):
        return self.no_of_rows
class FacesPairDataset(FacesDataset):
    def __init__(self, dataset_path, no_of_rows, transform=None, load_imgs_from_dict=False, img_features_dict=None,
                 subset=None,
                 select_from_negative_cnt=0):
        super().__init__(dataset_path, no_of_rows,transform,load_imgs_from_dict,img_features_dict,subset,select_from_negative_cnt)
    def __iter__(self):
        for i in range(self.no_of_rows):
            for anchor_name,anchor_imgs in self.person_imgs_list:
                for i in range(len(anchor_imgs)):
                    # anchor
                    anchor_img_path = f'{self.dataset_path}/{anchor_name}/{anchor_imgs[i]}'
                    anchor_img=self.load_img(anchor_img_path)

                    for j in range(i + 1, len(anchor_imgs)):
                        anchor_positive_img_path=f'{self.dataset_path}/{anchor_name}/{anchor_imgs[j]}'
                        anchor_positive_img=self.load_img(anchor_positive_img_path)

                        yield anchor_img,anchor_positive_img,1





class Normalize(torch.nn.Module):

    def forward(self, img):
        t_mean = torch.mean(img, dim=[1, 2])
        t_std = torch.std(img, dim=[1, 2])
        return normalize(img, t_mean.__array__(), t_std.__array__())

    def __init__(self):
        super().__init__()
