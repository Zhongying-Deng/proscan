import torch
import SimpleITK as sitk
import os
import numpy as np
import torch.utils.data
import pandas as pd

from monai.transforms import LoadImage, apply_transform

class NifitDataSetProstate(torch.utils.data.Dataset):
    def __init__(self, root_path,
                 transforms1=None, transforms2=None, shuffle_labels=False, phase='train', fold=3):

        # Init membership variables
        self.root = root_path
        self.folders = {'CT': 'images_ct', 'DWI': 'images_dwi', 'PET': 'images_pet', 'T2': 'images_t2'}

        self.CTList = []
        self.PETList = []
        self.DWIList = []
        self.T2List = []
        self.LabelList = []
        self.length = 0

        data_label = pd.read_excel(os.path.join(self.root, 'label_file.xlsx'), header=0)
        self.read_feasible_image(data_label)

        self.transforms1 = transforms1
        self.transforms2 = transforms2

        self.shuffle_labels = shuffle_labels
        self.loader = LoadImage(None, True, np.float32)
        self.bit = sitk.sitkFloat32
        

    def read_feasible_image(self, data_label):
        '''
        Store the imaging data path into a list, and non-imaging data features into a list. When call forward, we can just get the image directly according to these lists.
        '''
        # filter the patient id with both MRI and PET data
        count = 0
        file_names = data_label.iloc[:, 0].tolist()
        self.LabelList = data_label.iloc[:, 1].tolist()
        for fname in file_names:
            fname = str(fname)
            # store the information
            self.CTList.append(os.path.join(self.folders['CT'], fname + '.nii.gz'))
            self.PETList.append(os.path.join(self.folders['PET'], fname + '.nii.gz'))
            self.DWIList.append(os.path.join(self.folders['DWI'], fname + '.nii.gz'))
            self.T2List.append(os.path.join(self.folders['T2'], fname + '.nii'))

        print('Length of label list: {}, CT list: {}, PET list: {}'.format(
            len(self.LabelList), len(self.CTList), len(self.PETList)))
        print('Image number for class 0-1: {}, {}'.format(
            self.LabelList.count(0), self.LabelList.count(1)))

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def __getitem__(self, index):
        label = self.LabelList[index]
        # read image and label
        CT_path = os.path.join(self.root, self.CTList[index])
        PET_path = os.path.join(self.root, self.PETList[index])
        DWI_path =  os.path.join(self.root, self.DWIList[index])
        T2_path = os.path.join(self.root, self.T2List[index])

        image_CT = self.loader(CT_path)
        image_PET = self.loader(PET_path)
        image_DWI = self.loader(DWI_path)
        image_T2 = self.loader(T2_path)
        if self.transforms1 is not None:
            image_CT = apply_transform(self.transforms1, image_CT, map_items=False)
            image_PET = apply_transform(self.transforms1, image_PET, map_items=False)
        if self.transforms2 is not None:
            image_DWI = apply_transform(self.transforms2, image_DWI, map_items=False)
            image_T2 = apply_transform(self.transforms2, image_T2, map_items=False)

        return image_CT, image_PET, image_DWI, image_T2, label

    def __len__(self):
        return len(self.LabelList)
