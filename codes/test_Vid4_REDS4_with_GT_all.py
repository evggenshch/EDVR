
'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import copy
import json

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch

import argparse

class metrics_file:

    def __init__(self, name):
        self.name = name
        self.gt_ssim = []
        self.aposterior_ssim = []
        self.psnr = []

    def add_gt_ssim(self, gt_ssim):
        self.gt_ssim.append(gt_ssim)

    def add_aposterior_ssim(self, aposterior_ssim):
        self.aposterior_ssim.append(aposterior_ssim)

    def add_psnr(self, psnr):
        self.psnr.append(psnr)



def main():
    ####################
    # arguments parser #
    ####################
    #  [format] dataset(vid4, REDS4) N(number of frames)



   # data_mode = str(args.dataset)
   # N_in = int(args.n_frames)
   # metrics = str(args.metrics)
   # output_format = str(args.output_format)


    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #data_mode = 'Vid4'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).


    # STAGE Vid4
    # Collecting results for Vid4

    model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False

    predeblur, HR_in = False, False
    back_RBs = 40

    N_model_default = 7
    data_mode = 'Vid4'

   # vid4_dir_map = {"calendar": 0, "city": 1, "foliage": 2, "walk": 3}
    vid4_results = {"calendar": {}, "city": {}, "foliage": {}, "walk": {}}

    #vid4_results = 4 * [[]]

    for N_in in range(1, N_model_default + 1):
        raw_model = EDVR_arch.EDVR(128, N_model_default, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
        model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

        test_dataset_folder = '../datasets/Vid4/BIx4'
        GT_dataset_folder = '../datasets/Vid4/GT'
        aposterior_GT_dataset_folder = '../datasets/Vid4/GT_7'

        crop_border = 0
        border_frame = N_in // 2  # border frames when evaluate
        padding = 'new_info'

        save_imgs = False

        raw_model.load_state_dict(torch.load(model_path), strict=True)

        model.nf = raw_model.nf
        model.center = N_in // 2  # if center is None else center
        model.is_predeblur = raw_model.is_predeblur
        model.HR_in = raw_model.HR_in
        model.w_TSA = raw_model.w_TSA

        if model.is_predeblur:
            model.pre_deblur = raw_model.pre_deblur  # Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            model.conv_1x1 = raw_model.conv_1x1  # nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if model.HR_in:
                model.conv_first_1 = raw_model.conv_first_1  # nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                model.conv_first_2 = raw_model.conv_first_2  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                model.conv_first_3 = raw_model.conv_first_3  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                model.conv_first = raw_model.conv_first  # nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        model.feature_extraction = raw_model.feature_extraction  # arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        model.fea_L2_conv1 = raw_model.fea_L2_conv1  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        model.fea_L2_conv2 = raw_model.fea_L2_conv2  # nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        model.fea_L3_conv1 = raw_model.fea_L3_conv1  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        model.fea_L3_conv2 = raw_model.fea_L3_conv2  # nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        model.pcd_align = raw_model.pcd_align  # PCD_Align(nf=nf, groups=groups)

        model.tsa_fusion.center = model.center

        model.tsa_fusion.tAtt_1 = raw_model.tsa_fusion.tAtt_1
        model.tsa_fusion.tAtt_2 = raw_model.tsa_fusion.tAtt_2

        model.tsa_fusion.fea_fusion = copy.deepcopy(raw_model.tsa_fusion.fea_fusion)
        model.tsa_fusion.fea_fusion.weight = copy.deepcopy(torch.nn.Parameter(raw_model.tsa_fusion.fea_fusion.weight[:, 0:N_in * 128, :, :]))

        model.tsa_fusion.sAtt_1 = copy.deepcopy(raw_model.tsa_fusion.sAtt_1)
        model.tsa_fusion.sAtt_1.weight = copy.deepcopy(torch.nn.Parameter(raw_model.tsa_fusion.sAtt_1.weight[:, 0:N_in * 128, :, :]))

        model.tsa_fusion.maxpool = raw_model.tsa_fusion.maxpool
        model.tsa_fusion.avgpool = raw_model.tsa_fusion.avgpool
        model.tsa_fusion.sAtt_2 = raw_model.tsa_fusion.sAtt_2
        model.tsa_fusion.sAtt_3 = raw_model.tsa_fusion.sAtt_3
        model.tsa_fusion.sAtt_4 = raw_model.tsa_fusion.sAtt_4
        model.tsa_fusion.sAtt_5 = raw_model.tsa_fusion.sAtt_5
        model.tsa_fusion.sAtt_L1 = raw_model.tsa_fusion.sAtt_L1
        model.tsa_fusion.sAtt_L2 = raw_model.tsa_fusion.sAtt_L2
        model.tsa_fusion.sAtt_L3 = raw_model.tsa_fusion.sAtt_L3
        model.tsa_fusion.sAtt_add_1 = raw_model.tsa_fusion.sAtt_add_1
        model.tsa_fusion.sAtt_add_2 = raw_model.tsa_fusion.sAtt_add_2

        model.tsa_fusion.lrelu = raw_model.tsa_fusion.lrelu

        model.recon_trunk = raw_model.recon_trunk

        model.upconv1 = raw_model.upconv1
        model.upconv2 = raw_model.upconv2
        model.pixel_shuffle = raw_model.pixel_shuffle
        model.HRconv = raw_model.HRconv
        model.conv_last = raw_model.conv_last

        model.lrelu = raw_model.lrelu

    #####################################################

        model.eval()
        model = model.to(device)

        #avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
        subfolder_name_l = []

        subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
        subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))

        subfolder_GT_a_l = sorted(glob.glob(osp.join(aposterior_GT_dataset_folder, "*")))
    # for each subfolder
        for subfolder, subfolder_GT, subfolder_GT_a in zip(subfolder_l, subfolder_GT_l, subfolder_GT_a_l):
            subfolder_name = osp.basename(subfolder)
            subfolder_name_l.append(subfolder_name)

            img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
            max_idx = len(img_path_l)

            print("MAX_IDX: ", max_idx)


            #### read LQ and GT images
            imgs_LQ = data_util.read_img_seq(subfolder)
            img_GT_l = []
            for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
                img_GT_l.append(data_util.read_img(None, img_GT_path))

            img_GT_a = []
            for img_GT_a_path in sorted(glob.glob(osp.join(subfolder_GT_a, '*'))):
                img_GT_a.append(data_util.read_img(None, img_GT_a_path))
            #avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

            # process each image
            for img_idx, img_path in enumerate(img_path_l):
                img_name = osp.splitext(osp.basename(img_path))[0]
                select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)

                imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

                if flip_test:
                    output = util.flipx4_forward(model, imgs_in)
                else:
                    print("IMGS_IN SHAPE: ", imgs_in.shape)
                    output = util.single_forward(model, imgs_in)
                output = util.tensor2img(output.squeeze(0))

                # calculate PSNR
                output = output / 255.
                GT = np.copy(img_GT_l[img_idx])
                # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
                #if data_mode == 'Vid4':  # bgr2y, [0, 1]

                GT = data_util.bgr2ycbcr(GT, only_y=True)
                output = data_util.bgr2ycbcr(output, only_y=True)
                GT_a = np.copy(img_GT_a[img_idx])
                GT_a = data_util.bgr2ycbcr(GT_a, only_y=True)
                output_a = copy.deepcopy(output)

                output, GT = util.crop_border([output, GT], crop_border)
                crt_psnr = util.calculate_psnr(output * 255, GT * 255)
                crt_ssim = util.calculate_ssim(output * 255, GT * 255)

                output_a, GT_a = util.crop_border([output_a, GT_a], crop_border)

                crt_aposterior = util.calculate_ssim(output_a * 255, GT_a * 255)  # CHANGE


                t = vid4_results[subfolder_name].get(str(img_name))

                if t != None:
                    vid4_results[subfolder_name][img_name].add_psnr(crt_psnr)
                    vid4_results[subfolder_name][img_name].add_gt_ssim(crt_ssim)
                    vid4_results[subfolder_name][img_name].add_aposterior_ssim(crt_aposterior)
                else:
                    vid4_results[subfolder_name].update({img_name: metrics_file(img_name)})
                    vid4_results[subfolder_name][img_name].add_psnr(crt_psnr)
                    vid4_results[subfolder_name][img_name].add_gt_ssim(crt_ssim)
                    vid4_results[subfolder_name][img_name].add_aposterior_ssim(crt_aposterior)


    ############################################################################
    #### model



#### writing vid4  results


    util.mkdirs('../results/calendar')
    util.mkdirs('../results/city')
    util.mkdirs('../results/foliage')
    util.mkdirs('../results/walk')
    save_folder = '../results/'

    for i, dir_name in enumerate(["calendar", "city", "foliage", "walk"]):
        save_subfolder = osp.join(save_folder, dir_name)
        for j, value in vid4_results[dir_name].items():
         #   cur_result = json.dumps(_)
            with open(osp.join(save_subfolder, '{}.json'.format(value.name)), 'w') as outfile:
                json.dump(value.__dict__, outfile, ensure_ascii=False, indent=4)
                #json.dump(cur_result, outfile)

            #cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)



###################################################################################





    # STAGE REDS

    reds4_results = {"000": {}, "011": {}, "015": {}, "020": {}}
    data_mode = 'sharp_bicubic'

    N_model_default = 5

    for N_in in range(1, N_model_default + 1):
        for stage in range(1,3):

            flip_test = False

            if data_mode == 'sharp_bicubic':
                if stage == 1:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_SR_L.pth'
                else:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'
            elif data_mode == 'blur_bicubic':
                if stage == 1:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth'
                else:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_Stage2.pth'
            elif data_mode == 'blur':
                if stage == 1:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth'
                else:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_Stage2.pth'
            elif data_mode == 'blur_comp':
                if stage == 1:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth'
                else:
                    model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_Stage2.pth'
            else:
                raise NotImplementedError

            predeblur, HR_in = False, False
            back_RBs = 40
            if data_mode == 'blur_bicubic':
                predeblur = True
            if data_mode == 'blur' or data_mode == 'blur_comp':
                predeblur, HR_in = True, True
            if stage == 2:
                HR_in = True
                back_RBs = 20

            if stage == 1:
                test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
            else:
                test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
                print('You should modify the test_dataset_folder path for stage 2')
            GT_dataset_folder = '../datasets/REDS4/GT'

            raw_model = EDVR_arch.EDVR(128, N_model_default, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
            model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

            crop_border = 0
            border_frame = N_in // 2  # border frames when evaluate
            # temporal padding mode
            if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
                padding = 'new_info'
            else:
                padding = 'replicate'
            save_imgs = True

            data_mode_t = copy.deepcopy(data_mode)
            if stage == 1 and data_mode_t != 'Vid4':
                data_mode = 'REDS-EDVR_REDS_SR_L_flipx4'
            save_folder = '../results/{}'.format(data_mode)
            data_mode = copy.deepcopy(data_mode_t)
            util.mkdirs(save_folder)
            util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)


            aposterior_GT_dataset_folder = '../datasets/REDS4/GT_5'

            crop_border = 0
            border_frame = N_in // 2  # border frames when evaluate

            raw_model.load_state_dict(torch.load(model_path), strict=True)

            model.nf = raw_model.nf
            model.center = N_in // 2  # if center is None else center
            model.is_predeblur = raw_model.is_predeblur
            model.HR_in = raw_model.HR_in
            model.w_TSA = raw_model.w_TSA

            if model.is_predeblur:
                model.pre_deblur = raw_model.pre_deblur  # Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
                model.conv_1x1 = raw_model.conv_1x1  # nn.Conv2d(nf, nf, 1, 1, bias=True)
            else:
                if model.HR_in:
                    model.conv_first_1 = raw_model.conv_first_1  # nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                    model.conv_first_2 = raw_model.conv_first_2  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                    model.conv_first_3 = raw_model.conv_first_3  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                else:
                    model.conv_first = raw_model.conv_first  # nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            model.feature_extraction = raw_model.feature_extraction  # arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
            model.fea_L2_conv1 = raw_model.fea_L2_conv1  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            model.fea_L2_conv2 = raw_model.fea_L2_conv2  # nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            model.fea_L3_conv1 = raw_model.fea_L3_conv1  # nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            model.fea_L3_conv2 = raw_model.fea_L3_conv2  # nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            model.pcd_align = raw_model.pcd_align  # PCD_Align(nf=nf, groups=groups)

            model.tsa_fusion.center = model.center

            model.tsa_fusion.tAtt_1 = raw_model.tsa_fusion.tAtt_1
            model.tsa_fusion.tAtt_2 = raw_model.tsa_fusion.tAtt_2

            model.tsa_fusion.fea_fusion = copy.deepcopy(raw_model.tsa_fusion.fea_fusion)
            model.tsa_fusion.fea_fusion.weight = copy.deepcopy(torch.nn.Parameter(raw_model.tsa_fusion.fea_fusion.weight[:, 0:N_in * 128, :, :]))

            model.tsa_fusion.sAtt_1 = copy.deepcopy(raw_model.tsa_fusion.sAtt_1)
            model.tsa_fusion.sAtt_1.weight = copy.deepcopy(torch.nn.Parameter(raw_model.tsa_fusion.sAtt_1.weight[:, 0:N_in * 128, :, :]))

            model.tsa_fusion.maxpool = raw_model.tsa_fusion.maxpool
            model.tsa_fusion.avgpool = raw_model.tsa_fusion.avgpool
            model.tsa_fusion.sAtt_2 = raw_model.tsa_fusion.sAtt_2
            model.tsa_fusion.sAtt_3 = raw_model.tsa_fusion.sAtt_3
            model.tsa_fusion.sAtt_4 = raw_model.tsa_fusion.sAtt_4
            model.tsa_fusion.sAtt_5 = raw_model.tsa_fusion.sAtt_5
            model.tsa_fusion.sAtt_L1 = raw_model.tsa_fusion.sAtt_L1
            model.tsa_fusion.sAtt_L2 = raw_model.tsa_fusion.sAtt_L2
            model.tsa_fusion.sAtt_L3 = raw_model.tsa_fusion.sAtt_L3
            model.tsa_fusion.sAtt_add_1 = raw_model.tsa_fusion.sAtt_add_1
            model.tsa_fusion.sAtt_add_2 = raw_model.tsa_fusion.sAtt_add_2

            model.tsa_fusion.lrelu = raw_model.tsa_fusion.lrelu

            model.recon_trunk = raw_model.recon_trunk

            model.upconv1 = raw_model.upconv1
            model.upconv2 = raw_model.upconv2
            model.pixel_shuffle = raw_model.pixel_shuffle
            model.HRconv = raw_model.HRconv
            model.conv_last = raw_model.conv_last

            model.lrelu = raw_model.lrelu

    #####################################################

            model.eval()
            model = model.to(device)

            #avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
            subfolder_name_l = []

            subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
            subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))

            subfolder_GT_a_l = sorted(glob.glob(osp.join(aposterior_GT_dataset_folder, "*")))
    # for each subfolder
            for subfolder, subfolder_GT, subfolder_GT_a in zip(subfolder_l, subfolder_GT_l, subfolder_GT_a_l):

                subfolder_name = osp.basename(subfolder)
                subfolder_name_l.append(subfolder_name)
                save_subfolder = osp.join(save_folder, subfolder_name)

                img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
                max_idx = len(img_path_l)

                print("MAX_IDX: ", max_idx)

                print("SAVE FOLDER::::::", save_folder)

                if save_imgs:
                    util.mkdirs(save_subfolder)


            #### read LQ and GT images
                imgs_LQ = data_util.read_img_seq(subfolder)
                img_GT_l = []
                for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
                    img_GT_l.append(data_util.read_img(None, img_GT_path))

                img_GT_a = []
                for img_GT_a_path in sorted(glob.glob(osp.join(subfolder_GT_a, '*'))):
                    img_GT_a.append(data_util.read_img(None, img_GT_a_path))
                #avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

            # process each image
                for img_idx, img_path in enumerate(img_path_l):
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)

                    imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

                    if flip_test:
                        output = util.flipx4_forward(model, imgs_in)
                    else:
                        print("IMGS_IN SHAPE: ", imgs_in.shape)
                        output = util.single_forward(model, imgs_in)
                    output = util.tensor2img(output.squeeze(0))

                    if save_imgs and stage == 1:
                        cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)
                    # calculate PSNR
                    if stage == 2:

                        output = output / 255.
                        GT = np.copy(img_GT_l[img_idx])
                        # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
                        #if data_mode == 'Vid4':  # bgr2y, [0, 1]

                        GT_a = np.copy(img_GT_a[img_idx])
                        output_a = copy.deepcopy(output)

                        output, GT = util.crop_border([output, GT], crop_border)
                        crt_psnr = util.calculate_psnr(output * 255, GT * 255)
                        crt_ssim = util.calculate_ssim(output * 255, GT * 255)

                        output_a, GT_a = util.crop_border([output_a, GT_a], crop_border)

                        crt_aposterior = util.calculate_ssim(output_a * 255, GT_a * 255)  # CHANGE


                        t = reds4_results[subfolder_name].get(str(img_name))

                        if t != None:
                            reds4_results[subfolder_name][img_name].add_psnr(crt_psnr)
                            reds4_results[subfolder_name][img_name].add_gt_ssim(crt_ssim)
                            reds4_results[subfolder_name][img_name].add_aposterior_ssim(crt_aposterior)
                        else:
                            reds4_results[subfolder_name].update({img_name: metrics_file(img_name)})
                            reds4_results[subfolder_name][img_name].add_psnr(crt_psnr)
                            reds4_results[subfolder_name][img_name].add_gt_ssim(crt_ssim)
                            reds4_results[subfolder_name][img_name].add_aposterior_ssim(crt_aposterior)



    ############################################################################
    #### model



#### writing reds4  results

    util.mkdirs('../results/000')
    util.mkdirs('../results/011')
    util.mkdirs('../results/015')
    util.mkdirs('../results/020')
    save_folder = '../results/'

    for i, dir_name in enumerate(["000", "011", "015", "020"]):     #   +
        save_subfolder = osp.join(save_folder, dir_name)
        for j, value in reds4_results[dir_name].items():
           # cur_result = json.dumps(value.__dict__)
            with open(osp.join(save_subfolder, '{}.json'.format(value.name)), 'w') as outfile:
                json.dump(value.__dict__, outfile, ensure_ascii=False, indent=4)





if __name__ == '__main__':
    main()
