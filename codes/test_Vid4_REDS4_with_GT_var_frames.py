
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

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch

import argparse

def main():
    ####################
    # arguments parser #
    ####################
    #  [format] dataset(vid4, REDS4) N(number of frames)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset')
    parser.add_argument('n_frames')
    parser.add_argument('stage')

    args = parser.parse_args()

    data_mode = str(args.dataset)
    N_in = int(args.n_frames)
    stage = int(args.stage)

    #if args.command == 'start':
    #    start(int(args.params[0]))
    #elif args.command == 'stop':
    #    stop(args.params[0], int(args.params[1]))
    #elif args.command == 'stop_all':
    #    stop_all(args.params[0])

    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #data_mode = 'Vid4'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).
    #stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False
    ############################################################################
    #### model
    if data_mode == 'Vid4':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
        else:
            raise ValueError('Vid4 does not support stage 2.')
    elif data_mode == 'sharp_bicubic':
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


    #### dataset
    if data_mode == 'Vid4':
        N_model_default = 7
        test_dataset_folder = '../datasets/Vid4/BIx4'
        GT_dataset_folder = '../datasets/Vid4/GT'
    else:
        N_model_default = 5
        if stage == 1:
            test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
        else:
            test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
            print('You should modify the test_dataset_folder path for stage 2')
        GT_dataset_folder = '../datasets/REDS4/GT'

    raw_model = EDVR_arch.EDVR(128, N_model_default, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### evaluation
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
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    print([a for a in dir(model) if not callable(getattr(model, a))])  # not a.startswith('__') and

    #model.load_state_dict(torch.load(model_path), strict=True)
    raw_model.load_state_dict(torch.load(model_path), strict=True)

    #   model.load_state_dict(torch.load(model_path), strict=True)

    #### change model so it can work with less input

    model.nf = raw_model.nf
    model.center = N_in // 2 #  if center is None else center
    model.is_predeblur = raw_model.is_predeblur
    model.HR_in = raw_model.HR_in
    model.w_TSA = raw_model.w_TSA
    #ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

    #### extract features (for each frame)
    if model.is_predeblur:
        model.pre_deblur = raw_model.pre_deblur #Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
        model.conv_1x1 = raw_model.conv_1x1 #nn.Conv2d(nf, nf, 1, 1, bias=True)
    else:
        if model.HR_in:
            model.conv_first_1 = raw_model.conv_first_1 #nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            model.conv_first_2 = raw_model.conv_first_2 #nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            model.conv_first_3 = raw_model.conv_first_3 #nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            model.conv_first = raw_model.conv_first # nn.Conv2d(3, nf, 3, 1, 1, bias=True)
    model.feature_extraction = raw_model.feature_extraction #  arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
    model.fea_L2_conv1 = raw_model.fea_L2_conv1 #nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
    model.fea_L2_conv2 = raw_model.fea_L2_conv2 #nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
    model.fea_L3_conv1 = raw_model.fea_L3_conv1 #nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
    model.fea_L3_conv2 = raw_model.fea_L3_conv2 #nn.Conv2d(nf, nf, 3, 1, 1, bias=True)


    model.pcd_align = raw_model.pcd_align #PCD_Align(nf=nf, groups=groups)


    ######## Resize TSA

    model.tsa_fusion.center = model.center
    # temporal attention (before fusion conv)
    model.tsa_fusion.tAtt_1 = raw_model.tsa_fusion.tAtt_1
    model.tsa_fusion.tAtt_2 = raw_model.tsa_fusion.tAtt_2

    # fusion conv: using 1x1 to save parameters and computation

    #print(raw_model.tsa_fusion.fea_fusion.weight.shape)

    #print(raw_model.tsa_fusion.fea_fusion.weight.shape)
    #print(raw_model.tsa_fusion.fea_fusion.weight[127][639].shape)
    #print("MAIN SHAPE(FEA): ", raw_model.tsa_fusion.fea_fusion.weight.shape)

    model.tsa_fusion.fea_fusion = copy.deepcopy(raw_model.tsa_fusion.fea_fusion)
    model.tsa_fusion.fea_fusion.weight = copy.deepcopy(torch.nn.Parameter(raw_model.tsa_fusion.fea_fusion.weight[:, 0:N_in * 128, :, :]))
    #[:][] #nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
    #model.tsa_fusion.fea_fusion.bias = raw_model.tsa_fusion.fea_fusion.bias

    # spatial attention (after fusion conv)
    model.tsa_fusion.sAtt_1 = copy.deepcopy(raw_model.tsa_fusion.sAtt_1)
    model.tsa_fusion.sAtt_1.weight = copy.deepcopy(torch.nn.Parameter(raw_model.tsa_fusion.sAtt_1.weight[:, 0:N_in * 128, :, :]))
    #[:][] #nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
    #model.tsa_fusion.sAtt_1.bias = raw_model.tsa_fusion.sAtt_1.bias

    #print(N_in * 128)
    #print(raw_model.tsa_fusion.fea_fusion.weight[:, 0:N_in * 128, :, :].shape)
    print("MODEL TSA SHAPE: ", model.tsa_fusion.fea_fusion.weight.shape)

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


    #if model.w_TSA:
    #    model.tsa_fusion = raw_model.tsa_fusion[:][:128 * N_in][:][:] #TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
    #else:
    #    model.tsa_fusion = raw_model.tsa_fusion[:][:128 * N_in][:][:] #nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

    #   print(self.tsa_fusion)

    #### reconstruction
    model.recon_trunk = raw_model.recon_trunk # arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
    #### upsampling
    model.upconv1 = raw_model.upconv1 #nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
    model.upconv2 = raw_model.upconv2 #nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
    model.pixel_shuffle = raw_model.pixel_shuffle # nn.PixelShuffle(2)
    model.HRconv = raw_model.HRconv
    model.conv_last = raw_model.conv_last

    #### activation function
    model.lrelu = raw_model.lrelu



    #####################################################


    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
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

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)    #  HERE GOTCHA
            print("SELECT IDX: ", select_idx)

            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                print("IMGS_IN SHAPE: ", imgs_in.shape)         # check this
                output = util.single_forward(model, imgs_in)    # error here 1
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            if data_mode == 'Vid4':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT, only_y=True)
                output = data_util.bgr2ycbcr(output, only_y=True)

            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
                                                     psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
