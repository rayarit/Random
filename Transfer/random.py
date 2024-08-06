import os
import copy
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

class CustomNuScenesDataset(Dataset):
    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, ann_file=None, pipeline=None, data_root=None, test_mode=False, modality=None):
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.ann_file = ann_file
        self.pipeline = pipeline
        self.data_root = data_root
        self.test_mode = test_mode
        self.modality = modality or {'use_camera': True}
        self.data_infos = self.load_annotations(self.ann_file)
        
    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        return data_infos

    def prepare_train_data(self, index):
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or not (example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = torch.stack(imgs_list)
        queue[-1]['img_metas'] = metas_map
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_test_data(self, idx):
        # Implement test data preparation if needed
        pass

    def pre_pipeline(self, input_dict):
        # Implement pre-pipeline processing if needed
        pass

    def pipeline(self, input_dict):
        # Implement pipeline processing if needed
        pass

    def filter_empty_gt(self):
        # Implement filter for empty ground truth if needed
        pass

    def get_ann_info(self, idx):
        # Implement annotation information retrieval if needed
        pass

    def _rand_another(self, idx):
        # Implement random another index if needed
        pass



##=====================================================


# BEvFormer-tiny consumes at lease 6700M GPU memory
# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN -> R50
# smaller BEV: 200*200 -> 50*50
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)
from projects.mmdet3d_plugin.custom_utils import select_best_indices


_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3 # each sequence contains `queue_length` frames.

# Export Flags
export_model = False # Set to True for model export.
export_until_encoder = False # Set to True to export the model up to the encoder only.
export_batch_size = 1; assert export_batch_size == 1, "Only batch size 1 is supported for export." # Set the batch size for onnx export. (default = 1)
precomputed_canbus_lidar2img = False # set to True to export the model with frozen canbus and lidar2img.
precomputed_canbus_path = "./precomputed_canbus_lidar2img/can_bus.npy" # Path to Precomputed Can Bus for ONNX Export.
precomputed_lidar2img_path = "./precomputed_canbus_lidar2img/lidar2img.npy" # Path to Precomputed Lidar2Img for ONNX Export.
num_cams = 6 # Number of cameras to use for ONNX Export. (default = 6)

# ONNXRT Flags
onnx_runtime = False # Set to True to run eval in ONNX Runtime.
onnx_paths_for_onnxrt = dict(with_prev_bev="./onnx/BEVFormer_Tiny_with_PrevBev_ONNXRT_Compatible.onnx", 
                        without_prev_bev="./onnx/BEVFormer_Tiny_without_PrevBev_ONNXRT_Compatible.onnx") # ONNX paths for ONNX Runtime compatibility.
onnx_runtime_using_single_thread = True # Disable ONNX Runtime's multithreading by setting this to True to prevent erratic ScatterND node outputs, ensuring reliable model performance.

# Optimization Flags
original_code = False # Toggles between original (baseline) and optimized code; if True, 'use_optimized_linear' should be False.
use_optimized_linear=True # Switch from Linear to Optimized Linear.
split_heads = True # Determines MSHA (Multi-Single-Head Attention) or MHA (Multi-Head Attention) mechanism.
grid_sample_mode = "nearest" # Switch from Nearest to Bilinear

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    export_model=export_model,
    precomputed_canbus_lidar2img = precomputed_canbus_lidar2img,
    precomputed_canbus_path = precomputed_canbus_path,
    precomputed_lidar2img_path = precomputed_lidar2img_path,
    onnx_runtime = onnx_runtime,
    onnxruntime_with_prev_bev_path = onnx_paths_for_onnxrt['with_prev_bev'],
    onnxruntime_without_prev_bev_path = onnx_paths_for_onnxrt['without_prev_bev'],
    export_until_encoder = export_until_encoder,
    onnx_runtime_using_single_thread = onnx_runtime_using_single_thread,
    lidar_to_img_tf_matrix_grid_size = (4, 4), # Grid size of the transformation matrix from LiDAR to image coordinates.
    export_batch_size = export_batch_size,
    num_cams=num_cams,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        export_batch_size = export_batch_size,
        use_optimized_linear = use_optimized_linear,
        original_code = original_code,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            original_code = original_code,
            num_cams=num_cams,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            rotate_center = [bev_h_//2, bev_w_//2], # Bug Fix: setting it to center of the BEV grid instead of incorrect hardcoded value in the baseline.
            export_batch_size = export_batch_size,
            encoder=dict(
                type='BEVFormerEncoder',
                original_code = original_code,
                use_optimized_linear=use_optimized_linear,
                num_layers=3,
                bev_h=bev_h_,
                bev_w=bev_w_,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            use_optimized_linear=use_optimized_linear,
                            original_code = original_code,
                            split_heads = split_heads,
                            num_heads=4,
                            dim_per_head=12,
                            num_levels=1,
                            embed_dims=_dim_,
                            attn_cfg=dict(
                                type='MultiScaleDeformableAttention_TSA_Optimized',
                                use_optimized_linear=use_optimized_linear,
                                embed_dims = _dim_, 
                                num_bev_queue = 2, 
                                num_heads = 4, 
                                dim_per_head = 12,
                                num_levels = 1, 
                                num_points = 4,
                                im2col_step = 64,
                                grid_sample_mode = grid_sample_mode
                                ),
                            ),
                        dict(
                            type='SpatialCrossAttention',
                            use_optimized_linear=use_optimized_linear,
                            original_code = original_code,
                            split_heads = split_heads,
                            pc_range=point_cloud_range,
                            num_cams=num_cams,
                            max_len=select_best_indices(bev_h_, bev_w_),
                            export_batch_size = export_batch_size,
                            cross_attn=dict(
                                type='MSDeformableAttention3D_SCA_Optimized',
                                use_optimized_linear=use_optimized_linear,
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                                num_heads = 4,
                                dim_per_head = 12,
                                grid_sample_mode=grid_sample_mode
                                ),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    use_optimized_linear=use_optimized_linear,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                use_optimized_linear = use_optimized_linear,
                original_code = original_code,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='Custom_MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            use_optimized_linear = use_optimized_linear,
                            split_heads = split_heads,
                            original_code = original_code,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention_Decoder_Optimized',
                            use_optimized_linear = use_optimized_linear,
                            original_code = original_code,
                            embed_dims=_dim_,
                            grid_sample_mode=grid_sample_mode,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = '/local/mnt2/workspace/datasets/nuscenes'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/nuscenes_infos_temporal_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + "/nuscenes_infos_temporal_val.pkl",
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + "/nuscenes_infos_temporal_val.pkl",
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 72
evaluation = dict(interval=2, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=4)
