#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class CameraDataset(Dataset):
    def __init__(self, cameras_info, resolution_scale, args, is_nerf_synthetic, is_test):
        """
        Initializes the CameraDataset.
        :param cameras: List of camera information.
        :param resolution_scale: Scale of the resolution.
        :param args: Model parameters.
        :param is_nerf_synthetic: Boolean indicating if the dataset is NeRF synthetic.
        :param is_test: Boolean indicating if the dataset is for testing.
        """
        self.cameras = cameraList_from_camInfos(cameras_info, resolution_scale, args, is_nerf_synthetic, is_test)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        #cameras=loadCam(self.args, idx, self.cameras[idx], self.resolution_scale, self.is_nerf_synthetic, self.is_test_dataset)
        return self.cameras[idx]
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = CameraDataset(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
        #     print("Loading Test Cameras")
        #     self.test_cameras[resolution_scale] = CameraDataset(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            train_dataset = CameraDataset(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            self.train_cameras[resolution_scale] = DataLoader(
                train_dataset,
                batch_size=8,  # Set your desired batch size
                shuffle=False,
                num_workers=4,  # Number of worker processes for prefetching
                pin_memory=True,  # Optional: Speeds up data transfer to GPU
                prefetch_factor=2
            )

            print("Loading Test Cameras")
            test_dataset = CameraDataset(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
            self.test_cameras[resolution_scale] = DataLoader(
                test_dataset,
                batch_size=8,
                shuffle=False,  # Typically, test data is not shuffled
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2
            )
        self.scene_info = scene_info
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
