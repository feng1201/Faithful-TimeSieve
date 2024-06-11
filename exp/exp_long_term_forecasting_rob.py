from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')
from collections import OrderedDict
from utils import PGD


class Exp_Long_Term_Forecast_rob(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_rob, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate/10)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test_zao(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            #stddev = 0.2
                            #nosiy = torch.randn_like(batch_x) * stddev
                            #batch_x = batch_x + nosiy
                            stddev = 0.2
                            batch_x = batch_x + torch.normal(mean=0.0, std=stddev, size=batch_x.size()).to(self.device)
                            outputs, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, pretrain_path=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        pretrain_path = os.path.join(path, 'checkpoint.pth')
        #save_path = os.path.join('rob_'+path + '_0', 'checkpoint.pth')
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        # os.makedirs(save_path)

        if os.path.exists(pretrain_path):
            print(f"Loading pretrained model from {pretrain_path}")
            self.model.load_state_dict(torch.load(pretrain_path, map_location=self.device))
        else:
            print(f"Pretrained model not found at {pretrain_path}. Ensure the path is correct.")
            return
        teacher_model = copy.deepcopy(self.model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        loss_ce = nn.L1Loss()
        loss_rob = nn.L1Loss()
        sim_loss = nn.L1Loss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(30):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                pgd_attack = PGD(self.model, eps=0.15, alpha=0.01, steps=20, random_start=True, device=self.device, pred_len = self.args.pred_len)

                perturbed_batch_x = pgd_attack(batch_x, batch_y)
                train_steps = len(train_loader)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs_PGD, IB_loss_PGD,feature_PGD = self.model(perturbed_batch_x, batch_y, dec_inp, batch_y_mark)
                            outputs_nos, IB_loss_nos,feature_nos = self.model(batch_x, batch_y, dec_inp, batch_y_mark)
                            outputs_th, IB_loss_th,feature_th = teacher_model(batch_x, batch_y, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(perturbed_batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs_PGD = outputs_PGD[:, -self.args.pred_len:, f_dim:]
                        outputs_nos = outputs_nos[:, -self.args.pred_len:, f_dim:]
                        outputs_th= outputs_th[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 希望PGD前后输出接近
                        adv_loss = loss_rob(outputs_PGD,outputs_nos)*0.05
                        # 常规的回归任务loss，希望预测和groundtruth差不多
                        loss_cls = criterion(outputs_PGD, batch_y)+ IB_loss_PGD
                        # 希望teacher的输出和原本的输出距离接近
                        sim_loss = loss_rob(outputs_th, outputs_nos)*1 # 0.5时损失为0.03  现在调整为1
                        # IB输出特征间的loss
                        loss_feature = loss_rob(feature_PGD, feature_nos )*0.05
                        # 整体loss
                        loss = adv_loss+loss_cls+sim_loss+loss_feature
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(perturbed_batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(perturbed_batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = 0
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.test_zao(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
          #  early_stopping(vali_loss, self.model, path)
           # if early_stopping.early_stop:
           #     print("Early stopping")
          #      break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'rob_checkpoint.pth')  # Adding 'rob_' prefix
        torch.save(self.model.state_dict(), best_model_path)  # Save the model with 'rob_' prefix

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))  # Load the model with 'rob_' prefix
        #'rob_./checkpoints/long_term_forecast_2024_6_3_TimeSieve_custom_ftM_sl192_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_0/checkpoint.pth/checkpoint.pth'

        return self.model

    #def test(self, setting, test=0):
    def test(self, setting, model_path=None, test=1):

        test_data, test_loader = self._get_data(flag='test')

        if model_path is None:
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth') #'D:/BaiduNetdiskDownload/xiaoro-实验/checkpoints/textrob/checkpoint.pth'
#os.path.join('./checkpoints/' + setting, 'checkpoint.pth')

        if test:
            if os.path.exists(model_path):
                print('Loading model from:', model_path)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                print('Model file not found:', model_path)
                return  # 如果模型文件不存在，返回
        '''
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        # test_data, test_loader = self._get_data(flag='test')
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        # 检查是否需要从文件加载模型
        if test:
            if os.path.exists(model_path):
                print('Loading model from:', model_path)
                self.model.load_state_dict(torch.load(model_path))
            else:
                 print('Model file not found:', model_path)
                 return  # 如果模型文件不存在，返回
        '''
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, _,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
