import os
import random
import PIL.Image
import numpy as np
import torch
import torchvision

class FaceShifterDataset(torch.utils.data.TensorDataset):
    def __init__(self, root = "data", ratios=None):
        super(FaceShifterDataset, self).__init__()
        # transforms
        self.source_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomGrayscale(0.02),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)], 0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.target_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomGrayscale(0.01),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)], 0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #torchvision.transforms.RandomErasing(0.02, (0.02, 0.1), (0.5, 2)),
        ])
        self.basic_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set: occlusion
        self.root = root
        self.s_occlusion = list()
        self._walk(os.path.join(root, "shapenet_png"), self.s_occlusion)
        # image sets
        self.data_dict = dict()
        ## vanilla
        self.data_dict["vanilla"] = list()
        self._walk(os.path.join(root, "vanilla"), self.data_dict["vanilla"])
        ## edited target
        self.data_dict["edited_target"] = list()
        self._walk(os.path.join(root, "edited_target/original"), self.data_dict["edited_target"])
        ## source as gt
        self.data_dict["source_as_gt/beard"] = list()
        self._walk(os.path.join(root, "source_as_gt/beard"), self.data_dict["source_as_gt/beard"])
        self.data_dict["source_as_gt/fat"] = list()
        self._walk(os.path.join(root, "source_as_gt/fat"), self.data_dict["source_as_gt/fat"])
        ## target as gt
        self.data_dict["target_as_gt/bald"] = list()
        self._walk(os.path.join(root, "target_as_gt/bald"), self.data_dict["target_as_gt/bald"])
        self.data_dict["target_as_gt/grin"] = list()
        self._walk(os.path.join(root, "target_as_gt/grin"), self.data_dict["target_as_gt/grin"])
        self.data_dict["target_as_gt/forehead_hair_1"] = list()
        self._walk(os.path.join(root, "target_as_gt/forehead_hair_1"), self.data_dict["target_as_gt/forehead_hair_1"])
        self.data_dict["target_as_gt/forehead_hair_2"] = list()
        self._walk(os.path.join(root, "target_as_gt/forehead_hair_2"), self.data_dict["target_as_gt/forehead_hair_2"])
        self.data_dict["target_as_gt/glasses"] = list()
        self._walk(os.path.join(root, "target_as_gt/glasses"), self.data_dict["target_as_gt/glasses"])
        self.data_dict["target_as_gt/sun_glasses"] = list()
        self._walk(os.path.join(root, "target_as_gt/sun_glasses"), self.data_dict["target_as_gt/sun_glasses"])
        # probabilities
        if ratios is None:
            self.p_vanilla                      = 0.6
            self.p_vanilla_same                 = 0.112
            self.p_edited_target                = 0.06
            self.p_different_poses              = 0.0
            # X as gt
            self.p_source_as_gt_beard           = 0.024
            self.p_source_as_gt_fat             = 0.054
            self.p_target_as_gt_bald            = 0.012
            self.p_target_as_gt_grin            = 0.012
            self.p_target_as_gt_forehead_hair_1 = 0.03
            self.p_target_as_gt_forehead_hair_2 = 0.03
            self.p_target_as_gt_glasses         = 0.042
            self.p_target_as_gt_sun_glasses     = 0.024
        else:
            self.p_vanilla                      = ratios[0]
            self.p_vanilla_same                 = ratios[1]
            self.p_edited_target                = ratios[2]
            self.p_different_poses              = ratios[3]
            # X as gt
            self.p_source_as_gt_beard           = ratios[4]
            self.p_source_as_gt_fat             = ratios[5]
            self.p_target_as_gt_bald            = ratios[6]
            self.p_target_as_gt_grin            = ratios[7]
            self.p_target_as_gt_forehead_hair_1 = ratios[8]
            self.p_target_as_gt_forehead_hair_2 = ratios[9]
            self.p_target_as_gt_glasses         = ratios[10]
            self.p_target_as_gt_sun_glasses     = ratios[11]
        self.p_vanilla_occlusion = 0.0
        self.p_accumulated = np.cumsum([self.p_vanilla,
                                        self.p_vanilla_same,
                                        self.p_edited_target,
                                        self.p_different_poses,
                                        self.p_source_as_gt_beard,
                                        self.p_source_as_gt_fat,
                                        self.p_target_as_gt_bald,
                                        self.p_target_as_gt_grin,
                                        self.p_target_as_gt_forehead_hair_1,
                                        self.p_target_as_gt_forehead_hair_2,
                                        self.p_target_as_gt_glasses,
                                        self.p_target_as_gt_sun_glasses])

    def __getitem__(self, item):
        o = random.random()
        if   o < self.p_accumulated[0]:   # vanilla
            if random.random() < self.p_vanilla_occlusion:  # with occlusion
                return self._sample_vanilla_occlusion(self.data_dict["vanilla"])
            else:                                           # without occlusion
                return self._sample_vanilla(self.data_dict["vanilla"])
        elif o < self.p_accumulated[1]:   # vanilla_same
            return self._sample_vanilla_same(self.data_dict["vanilla"])
        elif o < self.p_accumulated[2]:   # edited_target
            return self._sample_edited_target(self.data_dict["edited_target"])
        elif o < self.p_accumulated[3]:   # different_poses
            # Not Implemented
            return self.__getitem__(item)
        elif o < self.p_accumulated[4]:   # source_as_gt/beard
            return self._sample_source_as_gt(self.data_dict["source_as_gt/beard"], True)
        elif o < self.p_accumulated[5]:   # source_as_gt/fat
            return self._sample_source_as_gt(self.data_dict["source_as_gt/fat"], True)
        elif o < self.p_accumulated[6]:   # target_as_gt/bald
            return self._sample_target_as_gt(self.data_dict["target_as_gt/bald"], False)
        elif o < self.p_accumulated[7]:   # target_as_gt/grin
            return self._sample_target_as_gt(self.data_dict["target_as_gt/grin"], False)
        elif o < self.p_accumulated[8]:   # target_as_gt/forehead_hair_1
            if random.random() < 0.5:
                # with forehead hair (target, edited) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/forehead_hair_1"], True)
            else:
                # no forehead hair (target, original) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/forehead_hair_1"], False)
        elif o < self.p_accumulated[9]:   # target_as_gt/forehead_hair_2
            if random.random() < 0.5:
                # with forehead hair (target, edited) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/forehead_hair_2"], True)
            else:
                # no forehead hair (target, original) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/forehead_hair_2"], False)
        elif o < self.p_accumulated[10]:  # target_as_gt/glasses
            if random.random() < 0.5:
                # with glasses (target, edited) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/glasses"], True)
            else:
                # no glasses (target, original) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/glasses"], False)
        elif o < self.p_accumulated[11]:  # target_as_gt/sun_glasses
            if random.random() < 0.5:
                # with glasses (target, edited) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/sun_glasses"], True)
            else:
                # no glasses (target, original) as ground truth
                return self._sample_target_as_gt(self.data_dict["target_as_gt/sun_glasses"], False)

    def __len__(self):
        return 16 * 4 * 10000

    def _sample_vanilla(self, image_paths):
        image_path_s, image_path_t = random.choices(image_paths, k=2)
        Xs = self.source_transforms(PIL.Image.open(image_path_s))
        Xt = self.target_transforms(PIL.Image.open(image_path_t))
        GT = -torch.ones([3, 256, 256], dtype=torch.float32)
        with_gt = torch.zeros([], dtype=torch.float32)
        src_as_true = True
        return Xs, Xt, GT, with_gt, src_as_true

    def _sample_vanilla_occlusion(self, image_paths):
        image_path_s, image_path_t = random.choices(image_paths, k=2)
        image_path_o = random.choice(self.s_occlusion)
        Xs = self.source_transforms(PIL.Image.open(image_path_s))
        pil_t, pil_o = self._get_occluded_target_gt(image_path_t, image_path_o)
        Xt = self.basic_transforms(pil_t)
        GT = self.basic_transforms(pil_o)
        # set to zero to distinguish from real GT
        # occlusion mask for real-GT and this one are different
        # with_gt is with-complete-gt
        with_gt = torch.zeros([], dtype=torch.float32)
        src_as_true = False
        return Xs, Xt, GT, with_gt, src_as_true

    def _sample_vanilla_same(self, image_paths):
        image_path_s = random.choice(image_paths)
        Xs = self.source_transforms(PIL.Image.open(image_path_s))
        Xt = self.target_transforms(PIL.Image.open(image_path_s))
        GT = Xt
        with_gt = torch.ones([], dtype=torch.float32)
        src_as_true = True
        return Xs, Xt, GT, with_gt, src_as_true

    def _sample_source_as_gt(self, image_paths, src_as_original):
        if src_as_original:
            image_path = random.choice(image_paths)
            if ".original." in image_path:
                image_path_s = image_path
                image_path_t = image_path.replace(".original.", ".edited.")
            elif ".edited." in image_path:
                image_path_t = image_path
                image_path_s = image_path.replace(".edited.", ".original.")
            else:
                raise NotImplemented
        else:
            image_path = random.choice(image_paths)
            if ".edited." in image_path:
                image_path_s = image_path
                image_path_t = image_path.replace(".edited.", ".original.")
            elif ".original." in image_path:
                image_path_t = image_path
                image_path_s = image_path.replace(".original.", ".edited.")
            else:
                raise NotImplemented
        src_as_true = True
        Xs = self.basic_transforms(PIL.Image.open(image_path_s))
        Xt = self.basic_transforms(PIL.Image.open(image_path_t))
        GT = Xs
        with_gt = torch.ones([], dtype=torch.float32)
        return Xs, Xt, GT, with_gt, src_as_true

    def _sample_target_as_gt(self, image_paths, src_as_original):
        image_path = random.choice(image_paths)
        if src_as_original:
            if ".original." in image_path:
                image_path_s = image_path
                image_path_t = image_path.replace(".original.", ".edited.")
            elif ".edited." in image_path:
                image_path_s = image_path.replace(".edited.", ".original.")
                image_path_t = image_path
            else:
                raise NotImplementedError
        else:
            if ".original." in image_path:
                image_path_s = image_path.replace(".original.", ".edited.")
                image_path_t = image_path
            elif ".edited." in image_path:
                image_path_s = image_path
                image_path_t = image_path.replace(".edited.", ".original.")
            else:
                raise NotImplementedError
        src_as_true = False
        Xs = self.source_transforms(PIL.Image.open(image_path_s))
        Xt = self.target_transforms(PIL.Image.open(image_path_t))
        GT = Xt
        with_gt = torch.ones([], dtype=torch.float32)
        return Xs, Xt, GT, with_gt, src_as_true

    def _sample_edited_target(self, image_paths):
        image_path_s, image_path_t, image_path_gt = self._get_s_t_gt_paths(image_paths)
        Xs = self.source_transforms(PIL.Image.open(image_path_s))
        Xt = self.basic_transforms(PIL.Image.open(image_path_t))
        GT = self.basic_transforms(PIL.Image.open(image_path_gt))
        with_gt = torch.ones([], dtype=torch.float32)
        src_as_true = True
        return Xs, Xt, GT, with_gt, src_as_true

    def _walk(self, curr, images):
        # curr: current directory during traverse
        for f in os.listdir(curr):
            if os.path.isfile(os.path.join(curr, f)):
                images.append(os.path.join(curr, f))
            else:
                self._walk(os.path.join(curr, f), images)

    def _get_correspond_processed_target(self, original_target_path):
        prefix, suffix = original_target_path.split("/original/")
        photo_set_prefix = suffix.split("/")[0]
        candidate_attribute_photo_sets = [m for m in
            os.listdir(os.path.join(self.root, "edited_target/edited"))
            if m.startswith(photo_set_prefix)]
        processed_target_path = os.path.join(os.path.join(self.root, "edited_target/edited"),
                                             random.choice(candidate_attribute_photo_sets),
                                             *(suffix.split("/")[1:]))
        return processed_target_path

    def _get_s_t_gt_paths(self, image_paths):
        selected_dir = os.path.dirname(random.choice(image_paths))
        try:
            source, groundtruth = random.sample(os.listdir(selected_dir), 2)
        except ValueError:
            source = random.sample(os.listdir(selected_dir), 1)[0]
            groundtruth = source
        source = os.path.join(selected_dir, source)
        groundtruth = os.path.join(selected_dir, groundtruth)
        processed_target = self._get_correspond_processed_target(groundtruth)
        if os.path.exists(source) and os.path.exists(processed_target) and os.path.exists(groundtruth):
            return source, processed_target, groundtruth
        else:
            return self._get_s_t_gt_paths(image_paths)

    def _get_occluded_target_gt(self, target_path, occlusion_path):
        l = random.randint(64, 96)
        image = np.asarray(PIL.Image.open(target_path).convert("RGB"))
        resized_occlusion = np.asarray(PIL.Image.open(occlusion_path).convert("RGB").resize((l, l)))
        rand_h = random.randint(0, 256 - l)
        rand_w = random.randint(0, 256 - l)
        occlusion = np.ones((256, 256, 3), dtype="uint8") * 255
        occlusion[rand_h:rand_h+l, rand_w:rand_w+l, :] = resized_occlusion
        mask = np.zeros((256, 256, 3), dtype="uint8")  # 0-> occlusion
        for i in range(256):
            for j in range(256):
                if occlusion[i, j, 0] == 255 and \
                        occlusion[i, j, 1] == 255 and \
                        occlusion[i, j, 2] == 255:
                    mask[i, j, 0] = 1
                    mask[i, j, 1] = 1
                    mask[i, j, 2] = 1
        occluded_image = mask * image + (1 - mask) * occlusion
        occluded_image = PIL.Image.fromarray(occluded_image)
        masked_occlusion = (1 - mask) * occlusion
        masked_occlusion = PIL.Image.fromarray(masked_occlusion)
        return occluded_image, masked_occlusion

