from __future__ import print_function

import os
import time
import random
import torch
import torch.nn as nn

import numpy as np

from utils import *
import torch.nn.functional as F
from same_conv2d import Conv2d
from same_conv2d import conv2d_same_padding
import torchvision
from torchvision import transforms


def concat(layers):
    return torch.cat(layers, dim=1)


device = torch.device('cuda:0')


class DecomNet(nn.Module):
    def __init__(self, in_channels, layer_num, out_channels=64, kernel_size=3):
        super().__init__()

        self.model = nn.Sequential()

        self.model.add_module('shallow_feature_extraction',
                              Conv2d(in_channels, out_channels, kernel_size * 3, stride=1))

        for idx in range(layer_num):
            self.model.add_module('activated_layer_%d' % idx,
                                  Conv2d(out_channels, out_channels, kernel_size, stride=1))
            self.model.add_module('relu' + str(idx), nn.ReLU())
        self.model.add_module('recon_layer', Conv2d(out_channels, 4, kernel_size, stride=1))

    def forward(self, input_im):
        out = self.model(input_im)
        R = F.sigmoid(out[:, 0:3, :, :])
        L = F.sigmoid(out[:, 3:4, :, :])
        return R, L


class Relight(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super().__init__()
        self.model0 = Conv2d(4, channel, kernel_size, stride=1)
        self.downSample = nn.Sequential(Conv2d(channel, channel, kernel_size, stride=2), nn.ReLU())
        self.upSample = nn.Sequential(Conv2d(channel, channel, kernel_size, stride=1), nn.ReLU())
        self.converge = Conv2d(channel * 3, channel, 1, stride=1)
        self.out = Conv2d(channel, 1, 3, stride=1)

    def forward(self, input_L, input_R):
        # 合并反射层和光照层个
        input_im = concat([input_R, input_L])
        # 扩大感受野，提取特征，将通道数扩展为channel个，此时不降低像素点
        conv0 = self.model0(input_im)

        # 经过三次下采样，通道数不变，减少像素点，进一步扩张感受野，提取图片特征
        conv1 = self.downSample(conv0)
        conv2 = self.downSample(conv1)
        conv3 = self.downSample(conv2)

        # 通过三次图像最近邻插值进行上采样，每次插值过后通过一次卷积提取图片特征并与上一步下采样得到的图像矩阵进行残差连接，
        # 逐渐将图片维度恢复成输入时的尺寸
        up1 = F.interpolate(conv3, (conv2.shape[2], conv2.shape[3]), mode='nearest')
        deconv1 = self.upSample(up1) + conv2

        up2 = F.interpolate(conv2, (conv1.shape[2], conv1.shape[3]), mode='nearest')
        deconv2 = self.upSample(up2) + conv1

        up3 = F.interpolate(conv1, (conv0.shape[2], conv0.shape[3]), mode='nearest')
        deconv3 = self.upSample(up3) + conv0

        # 对deconv1和deconv2进行最近领插值，将其扩张成与deconv3，也就是跟原始图像一样的尺寸
        deconv1_resize = F.interpolate(deconv1, (deconv3.shape[2], deconv3.shape[3]), mode='nearest')
        deconv2_resize = F.interpolate(deconv2, (deconv3.shape[2], deconv3.shape[3]), mode='nearest')

        # 将其从通道数那一维度汇聚起来，此时feature_gather的通道数会变成channel*3个
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])

        # 通过卷积将汇聚后的通道数恢复成channel个
        feature_fusion = self.converge(feature_gather)

        # 通过1*1卷积将通道数变成1，最终输出的也就是光照分量
        output = self.out(feature_fusion)
        return output


class L2_loss:

    def getDecomNetLoss(self, R_low, I_low, R_high, I_high, batch_input_low, batch_input_high):
        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - batch_input_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - batch_input_high))
        recon_loss_multiply_low = torch.mean(torch.abs(R_high * I_low_3 - batch_input_low))
        recon_loss_multiply_high = torch.mean(torch.abs(R_low * I_high_3 - batch_input_high))
        smooth_loss_low = self.smooth(I_low, R_low)
        smooth_loss_high = self.smooth(I_high, R_high)
        equal_R_loss = torch.mean(torch.abs(R_low - R_high))
        loss = recon_loss_low + recon_loss_high + 0.001 * recon_loss_multiply_low + 0.001 * recon_loss_multiply_high + 0.1 * smooth_loss_low + 0.1 * smooth_loss_high + 0.01 * equal_R_loss
        return loss

    def getRelightLoss(self, R_low, batch_input_high, I_delta):

        I_delta_3 = concat([I_delta, I_delta, I_delta])

        smooth_loss_delta = self.smooth(I_delta, R_low)

        relight_loss = torch.mean(torch.abs(R_low * I_delta_3 - batch_input_high))
        loss = relight_loss + 3 * smooth_loss_delta
        return loss

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32, requires_grad=False).reshape(1, 1, 2,
                                                                                                            2).to(
            device)
        smooth_kernel_y = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32, requires_grad=False).transpose(0,
                                                                                                              1).reshape(
            1, 1, 2, 2).to(
            device)

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        return torch.abs(conv2d_same_padding(input=input_tensor, weight=kernel, stride=(1, 1)))

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):

        input_R = torchvision.transforms.Grayscale()(input_R).to(device)

        gradient_x = self.gradient(input_I, "x")
        ave_gradient_x = self.ave_gradient(input_R, "x")
        #
        gradient_y = self.gradient(input_I, "y")
        ave_gradient_y = self.ave_gradient(input_R, "y")

        return torch.mean(
            gradient_x * torch.exp(-10 * ave_gradient_x) + gradient_y * torch.exp(-10 * ave_gradient_y))

        # return torch.mean(
        #     self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I,"y") * torch.exp(
        #         -10 * self.ave_gradient(input_R, "y")))


class lowlight_enhance(object):
    def __init__(self, in_channels, channel=64, kernel_size=3):
        self.decomNetPthPath = './weight/decomNet.pth'
        self.relightNetPthPath = './weight/relightNet.pth'
        self.DecomNet_layer_num = 5
        self.decomNet = DecomNet(in_channels=in_channels, layer_num=self.DecomNet_layer_num, out_channels=channel,
                                 kernel_size=kernel_size).to(device)
        self.relightNet = Relight(channel=channel, kernel_size=kernel_size).to(device)
        if os.path.exists(self.decomNetPthPath):
            self.decomNet = torch.load(self.decomNetPthPath)
        if os.path.exists(self.relightNetPthPath):
            self.relightNet = torch.load(self.relightNetPthPath)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, sample_dir, lr,
              ckpt_dir, eval_every_epoch, train_phase):

        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)
        iter_num = 0
        start_epoch = 0
        start_step = 0
        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (
            train_phase, start_epoch, iter_num))
        image_id = 0

        optimizer = None

        loss = L2_loss()

        if train_phase == "Decom":
            optimizer = torch.optim.Adam(self.decomNet.parameters(), lr)
        elif train_phase == "Relight":

            optimizer = torch.optim.Adam(self.relightNet.parameters(), lr)

        start_time = time.time()
        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype='float32')
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype='float32')
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)

                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(
                        train_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(
                        train_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data = zip(*tmp)
                batch_input_low = batch_input_low.transpose(0, 3, 1, 2)
                batch_input_high = batch_input_high.transpose(0, 3, 1, 2)
                batch_input_low = torch.tensor(batch_input_low)
                batch_input_high = torch.tensor(batch_input_high)
                batch_input_low = batch_input_low.to(device)
                batch_input_high = batch_input_high.to(device)
                optimizer.zero_grad()
                R_low, I_low = self.decomNet(batch_input_low)

                if train_phase == "Decom":
                    R_high, I_high = self.decomNet(batch_input_high)
                    result_loss = loss.getDecomNetLoss(R_low, I_low, R_high, I_high, batch_input_low, batch_input_high)

                else:
                    I_delta = self.relightNet(I_low, R_low)
                    result_loss = loss.getRelightLoss(R_low, batch_input_high, I_delta)

                result_loss.backward()
                optimizer.step()
                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, result_loss))
                iter_num += 1

        if train_phase == "Decom":
            torch.save(self.decomNet, self.decomNetPthPath)

        else:
            torch.save(self.relightNet, self.relightNetPthPath)

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):

        print("[*] Reading checkpoint...")

        print("[*] Load weights successfully...")

        print("[*] Testing...")
        self.decomNet.eval()
        self.relightNet.eval()
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            input_low_test = input_low_test.transpose(0, 3, 1, 2)
            input_low_test = torch.tensor(input_low_test,requires_grad=False).to(device)
            R_low, I_low = self.decomNet(input_low_test)
            I_delta = self.relightNet(I_low, R_low)
            I_delta_3 = concat([I_delta, I_delta, I_delta])
            S = R_low * I_delta_3

            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S." + suffix), S)
            torch.cuda.empty_cache()

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low],
                                                   feed_dict={self.input_low: input_low_eval})
            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta],
                                                   feed_dict={self.input_low: input_low_eval})

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1,
                        result_2)
