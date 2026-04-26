import os
import torch
import csv
import tqdm
import torch.optim as optim
import pypose as pp
import torch.utils.data as Data
from utils import move_to, plot_imu_bias, plot_imu_noise
from datasets import SeqeuncesDataset

class PreIntegrator():
    def __init__(self, conf):
        self.conf = conf
        self.dtype = torch.float64
        self.scale = 1000
        self.integrator = pp.module.IMUPreintegrator(prop_cov=True, reset=True)

    def get_preintegator_items(self, input_data, init_state, label, bias_gyro = None, bias_acc = None, noise_gyro = None, noise_acc = None):
        B, F = input_data["dt"].shape[:2]

        if bias_gyro is None:
            bias_gyro = torch.zeros_like(input_data['gyro'], dtype=self.dtype, device=self.conf.device)
        else:
            bias_gyro = bias_gyro.unsqueeze(1).repeat(1, F, 1).to(self.conf.device)

        if bias_acc is None:
            bias_acc = torch.zeros_like(input_data['acc'], dtype=self.dtype, device=self.conf.device)
        else:
            bias_acc = bias_acc.unsqueeze(1).repeat(1, F, 1).to(self.conf.device)

        if noise_gyro is None:
            noise_gyro = torch.zeros_like(input_data['gyro'], dtype=self.dtype, device=self.conf.device)

        if noise_acc is None:
            noise_acc = torch.zeros_like(input_data['acc'], dtype=self.dtype, device=self.conf.device)

        preintegator_items = {}
        dr = pp.so3((input_data['gyro'] - bias_gyro - noise_gyro)*input_data["dt"]).Exp()
        dr = torch.cat([pp.identity_SO3(B, 1).to(self.conf.device), dr], dim=1)
        preintegator_items['delta_Q'] = dr[:, 1:, :]
        gamma_items = pp.cumprod(dr, dim = 1, left=False)
        
        dv = torch.zeros(B, 1, 3, dtype=self.dtype, device=self.conf.device)
        dv = torch.cat([dv, gamma_items[:, :F, :] @ (input_data['acc'] - bias_acc - noise_acc) * input_data["dt"]], dim=1)
        preintegator_items['delta_V'] = (input_data['acc'] - bias_acc - noise_acc) * input_data["dt"]
        beta_items = torch.cumsum(dv, dim=1)

        dp = torch.zeros(B, 1, 3, dtype=self.dtype, device=self.conf.device)
        dp = torch.cat([dp, beta_items[:, :F, :] * input_data["dt"] + gamma_items[:, :F, :] @ (input_data['acc'] - bias_acc - noise_acc) * 0.5 * input_data["dt"] ** 2], dim=1)
        preintegator_items['delta_P'] = (input_data['acc'] - bias_acc - noise_acc) * 0.5 * input_data["dt"] ** 2
        alpha_items = torch.cumsum(dp, dim=1)

        # check
        if False:
            gravity = torch.tensor([0., 0., 9.81007], dtype=self.dtype, device=self.conf.device)
            gt_delta_dt = torch.cumsum(input_data["dt"], dim = 1)
            offset_V_gravity = torch.cumsum(gravity * input_data['dt'], dim = 1)
            offset_P_gravity = torch.cumsum(0.5 * gravity * input_data['dt'] ** 2, dim = 1) + \
                torch.cumsum(torch.cat((torch.zeros(B, 1, 3, dtype=torch.float64, device=self.conf.device), offset_V_gravity), dim=1)[:, :-1, :] * input_data['dt'], dim = 1)

            predict = {
                'rot': init_state['rot'][:, 0:1, :] * gamma_items[:, 1:, :],
                'vel': init_state['vel'][:, 0:1, :] + init_state['rot'][:, 0:1, :] * beta_items[:, 1:, :] - offset_V_gravity,
                'pos': init_state['pos'][:, 0:1, :] + init_state['rot'][:, 0:1, :] * alpha_items[:, 1:, :] + init_state['vel'][:, 0:1, :] * gt_delta_dt \
                      - offset_P_gravity,
            }
            print("pos: ", predict['pos'][0:4, -1:, :], label['gt_pos'][0:4, -1:, :])
            print("rot: ", predict['rot'][0:4, -1:, :], label['gt_rot'][0:4, -1:, :])
            print("vel: ", predict['vel'][0:4, -1:, :], label['gt_vel'][0:4, -1:, :])

        preintegator_items['alpha_item'] = alpha_items[:, -1, :]
        preintegator_items['beta_item'] = beta_items[:, -1, :]
        preintegator_items['gamma_item'] = gamma_items[:, -1, :]

        return preintegator_items

    def print_preintegator_items_loss(self, preintegator_items, label):
        delta_P_loss = torch.nn.functional.mse_loss(preintegator_items['alpha_item'], label['gt_alpha_item'], reduction='mean')
        delta_V_loss = torch.nn.functional.mse_loss(preintegator_items['beta_item'], label['gt_beta_item'], reduction='mean')
        delta_Q_loss = (preintegator_items['gamma_item'].Inv() * label['gt_gamma_item']).Log().norm(dim=-1).mean()

        if True:
            print("delta_P_loss =", delta_P_loss.item() * self.scale)
            print("delta_V_loss =", delta_V_loss.item() * self.scale)
            print("delta_Q_loss =", delta_Q_loss.item() * self.scale)

    def get_imu_bias(self, input_data, label, preintegator_items):
        B, F = input_data["dt"].shape[:2]
        
        w_x = 0.5 * (input_data["gyro_mid"][:, 0:-1, :] + input_data["gyro_mid"][:, 1:, :])
        a_0_x = input_data["acc_mid"][:, 0:-1, :]
        a_1_x = input_data["acc_mid"][:, 1:, :]

        R_w_x = torch.zeros(B, F, 3, 3, dtype=self.dtype, device=self.conf.device)
        R_w_x[..., 0, 1] = -w_x[..., 2]
        R_w_x[..., 0, 2] = w_x[..., 1]
        R_w_x[..., 1, 0] = w_x[..., 2]
        R_w_x[..., 1, 2] = -w_x[..., 0]
        R_w_x[..., 2, 0] = -w_x[..., 1]
        R_w_x[..., 2, 1] = w_x[..., 0]

        R_a_0_x = torch.zeros(B, F, 3, 3, dtype=self.dtype, device=self.conf.device)
        R_a_0_x[..., 0, 1] = -a_0_x[..., 2]
        R_a_0_x[..., 0, 2] = a_0_x[..., 1]
        R_a_0_x[..., 1, 0] = a_0_x[..., 2]
        R_a_0_x[..., 1, 2] = -a_0_x[..., 0]
        R_a_0_x[..., 2, 0] = -a_0_x[..., 1]
        R_a_0_x[..., 2, 1] = a_0_x[..., 0]

        R_a_1_x = torch.zeros(B, F, 3, 3, dtype=self.dtype, device=self.conf.device)
        R_a_1_x[..., 0, 1] = -a_1_x[..., 2]
        R_a_1_x[..., 0, 2] = a_1_x[..., 1]
        R_a_1_x[..., 1, 0] = a_1_x[..., 2]
        R_a_1_x[..., 1, 2] = -a_1_x[..., 0]
        R_a_1_x[..., 2, 0] = -a_1_x[..., 1]
        R_a_1_x[..., 2, 1] = a_1_x[..., 0]        

        # F
        _variable_tmp = torch.einsum('...xy,...t -> ...xy', R_w_x, input_data['dt'])
        delta_Q_integration = torch.cat((pp.identity_SO3(B, 1).to(self.conf.device), preintegator_items['delta_Q']), dim=1)    # [128, 101, 4]
        unit_matrix = torch.eye(3).repeat([B, F, 1, 1]).to(self.conf.device)
        F_matrix = torch.eye(15).unsqueeze(0).repeat([B, F+1, 1, 1])                    # [128, 101, 15, 15]
        F_matrix[:, 1:, 0:3, 3:6] = torch.einsum('...xy,...t -> ...xy', -0.25 * delta_Q_integration[:, 0:-1, :].matrix() @ R_a_0_x , input_data['dt'] ** 2) + \
                                torch.einsum('...xy,...t -> ...xy', -0.25 * delta_Q_integration[:, 1:, :].matrix() @ R_a_1_x @ (unit_matrix - _variable_tmp) , input_data['dt'] ** 2)
        F_matrix[:, 1:, 0:3, 6:9] = torch.einsum('...xy,...t -> ...xy', unit_matrix, input_data['dt'])
        F_matrix[:, 1:, 0:3, 9:12] = torch.einsum('...xy,...t -> ...xy', -0.25 * (delta_Q_integration[:, 0:-1, :].matrix() + delta_Q_integration[:, 1:, :].matrix()), input_data['dt'] ** 2)
        F_matrix[:, 1:, 0:3, 12:15] = torch.einsum('...xy,...t -> ...xy', 0.25 * delta_Q_integration[:, 1:, :].matrix() @ R_a_1_x, input_data['dt'] ** 3) # 注意符号
        F_matrix[:, 1:, 3:6, 3:6] = unit_matrix - _variable_tmp
        F_matrix[:, 1:, 3:6, 12:15] = torch.einsum('...xy,...t -> ...xy', -unit_matrix, input_data['dt'])
        F_matrix[:, 1:, 6:9, 3:6] = torch.einsum('...xy,...t -> ...xy', -0.5 * delta_Q_integration[:, 0:-1, :].matrix() @ R_a_0_x, input_data['dt']) + \
                                torch.einsum('...xy,...t -> ...xy', -0.5 * delta_Q_integration[:, 1:, :].matrix() @ R_a_1_x @ (unit_matrix - _variable_tmp) , input_data['dt'])
        F_matrix[:, 1:, 6:9, 9:12] = torch.einsum('...xy,...t -> ...xy', -0.5 * (delta_Q_integration[:, 0:-1, :].matrix() + delta_Q_integration[:, 1:, :].matrix()), input_data['dt'])
        F_matrix[:, 1:, 6:9, 12:15] = torch.einsum('...xy,...t -> ...xy', 0.5 * delta_Q_integration[:, 1:, :].matrix() @ R_a_1_x, input_data['dt'] ** 2) # 注意符号
        F_matrix = pp.cumprod(F_matrix, dim=1)
        
        # J
        J_ba_alpha = F_matrix[:, -1, 0:3, 9:12]
        J_bw_alpha = F_matrix[:, -1, 0:3, 12:15]
        J_ba_beta = F_matrix[:, -1, 6:9, 9:12]
        J_bw_beta = F_matrix[:, -1, 6:9, 12:15]
        J_bw_gamma = F_matrix[:, -1, 3:6, 12:15]

        alpha_diff = label['gt_alpha_item'] - preintegator_items['alpha_item']
        beta_diff = label['gt_beta_item'] - preintegator_items['beta_item']
        gamma_diff = (preintegator_items['gamma_item'].Inv() * label['gt_gamma_item']).Log() # Log 映射将其转换为 so(3) 空间的矢量

        delta_bw = torch.zeros(B, 3, 1, requires_grad=True)
        optimizer = optim.Adam([delta_bw], lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

        total_iterations = 15000
        losses = 0.0
        t_range = tqdm.tqdm(range(total_iterations), disable = False, total=total_iterations)
        for i, _ in enumerate(t_range):
            gamma_approx = (torch.matmul(J_bw_gamma, delta_bw)).squeeze(dim=-1)
            # gamma_approx = torch.cat((gamma_approx, torch.ones(B, 1)), dim=1)
            # gamma_approx = pp.SO3(gamma_approx)
            # gamma_approx = pp.so3(gamma_approx).Exp()
            gamma_approx = move_to(gamma_approx, self.conf.device)

            loss = torch.nn.functional.l1_loss(gamma_diff, gamma_approx)
            losses += loss

            t_range.set_description('Solve Gyro Bias. epoch: %03d, losses: %.06f'%(i, losses / (i + 1)))
            t_range.refresh()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        optimized_delta_bw = delta_bw.detach()
        bias_gyro = optimized_delta_bw.squeeze(-1).mean(dim=0)
        bias_gyro = bias_gyro.repeat(B, 1)
        print("\nbias_gyro =", bias_gyro[0,:])

        bias_gyro_tmp = bias_gyro.unsqueeze(-1).cpu()
        delta_ba = torch.zeros(B, 3, 1, requires_grad=True)
        optimizer = optim.Adam([delta_ba], lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

        total_iterations = 15000
        losses = 0.0
        t_range = tqdm.tqdm(range(total_iterations), disable = False, total=total_iterations)
        for i, _ in enumerate(t_range):
            alpha_approx = (torch.matmul(J_ba_alpha, delta_ba) + torch.matmul(J_bw_alpha, bias_gyro_tmp)).squeeze(dim=-1)
            beta_approx = (torch.matmul(J_ba_beta, delta_ba) + torch.matmul(J_bw_beta, bias_gyro_tmp)).squeeze(dim=-1)
            alpha_approx, beta_approx = move_to([alpha_approx, beta_approx], self.conf.device)

            loss_alpha = torch.nn.functional.l1_loss(alpha_diff, alpha_approx)
            loss_beta = torch.nn.functional.l1_loss(beta_diff, beta_approx)
            
            loss = loss_beta
            losses += loss

            t_range.set_description('Solve Acc Bias. epoch: %03d, losses: %.06f'%(i, losses / (i + 1)))
            t_range.refresh()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        optimized_delta_ba = delta_ba.detach()
        bias_acc = optimized_delta_ba.squeeze(-1).mean(dim=0)
        bias_acc = bias_acc.repeat(B, 1)
        print("\nbias_acc =", bias_acc[0,:])
        
        return bias_acc, bias_gyro

    def get_imu_noise(self, input_data, label, preintegator_items):
        B, F = input_data['dt'].shape[:2]

        gamma_diff = (preintegator_items['delta_Q'].Inv() * label['gt_delta_Q']).Log()
        # gamma_diff = (label['gt_delta_Q'] * preintegator_items['delta_Q'].Inv()).Log()
        noise_gyro = -gamma_diff / input_data["dt"]
        
        beta_diff = label['gt_delta_V'] - preintegator_items['delta_V']
        noise_acc = -beta_diff / input_data["dt"]

        print("noise_acc =", noise_acc[0, :, :].mean(dim=0))
        print("noise_gyro =", noise_gyro[0, :, :].mean(dim=0))

        return noise_acc, noise_gyro

def cal_imu_bias(conf):
    dataset_config=conf.dataset.inference
    os.makedirs(os.path.join(conf.output_path, "label"), exist_ok=True)
    for data_list in dataset_config.data_list:
        for sub_dataset_name in data_list.data_drive:
            os.makedirs(os.path.join(conf.output_path, f"label/{sub_dataset_name}"), exist_ok=True)
            csv_output_path = os.path.join(conf.output_path, f"label/{sub_dataset_name}/imu_label_data.csv")
            if os.path.exists(csv_output_path):
                continue
            with open(csv_output_path, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                                'fake_gyro_bias_x', 'fake_gyro_bias_y', 'fake_gyro_bias_z',
                                'fake_acc_bias_x', 'fake_acc_bias_y', 'fake_acc_bias_z',
                                'fake_gyro_noise_x', 'gyro_noise_y', 'gyro_noise_z',
                                'fake_acc_noise_x', 'fake_acc_noise_y', 'fake_acc_noise_z'])
                file.close()
            file = open(csv_output_path, mode='a', newline='')
            writer = csv.writer(file)
            
            sub_dataset = SeqeuncesDataset(data_set_config=dataset_config, data_root = data_list["data_root"], data_name = sub_dataset_name)
            loader = Data.DataLoader(dataset=sub_dataset, batch_size=sub_dataset.__len__(), shuffle=False, collate_fn=SeqeuncesDataset.pad)
            print(f">> Processing: {sub_dataset_name}, len:", sub_dataset.__len__())
            
            preintegrator = PreIntegrator(conf)
            t_range = tqdm.tqdm(loader, dynamic_ncols=True)
            for i, (input_data, init_state, label) in enumerate(t_range):
                input_data, init_state, label = move_to([input_data, init_state, label], conf.device)

                _preintegator_items = preintegrator.get_preintegator_items(input_data, init_state, label)
                preintegrator.print_preintegator_items_loss(_preintegator_items, label)

                _bias_acc, _bias_gyro = preintegrator.get_imu_bias(input_data, label, _preintegator_items)
                _preintegator_items = preintegrator.get_preintegator_items(input_data, init_state, label, bias_gyro = _bias_gyro, bias_acc = _bias_acc)
                preintegrator.print_preintegator_items_loss(_preintegator_items, label)

                _noise_acc, _noise_gyro = preintegrator.get_imu_noise(input_data, label, _preintegator_items)
                _preintegator_items = preintegrator.get_preintegator_items(input_data, init_state, label, bias_gyro = _bias_gyro, bias_acc = _bias_acc, noise_gyro = _noise_gyro, noise_acc = _noise_acc)
                preintegrator.print_preintegator_items_loss(_preintegator_items, label)

                if True:
                    plot_imu_bias(conf, sub_dataset_name, _bias_gyro, _bias_acc, label)
                if True:
                    plot_imu_noise(conf, sub_dataset_name, _noise_acc, _noise_gyro)

                if True:
                    B, F = input_data["dt"].shape[:2]
                    for i in range(B):
                        if (i == 0) :
                            for j in range(F):
                                angular_velocity_vals = input_data["gyro"].detach().cpu().numpy()[i,j,0:3]
                                linear_acceleration_vals = input_data["acc"].detach().cpu().numpy()[i,j,0:3]
                                gyro_bias_vals = _bias_gyro.detach().cpu().numpy()[i,0:3]
                                acc_bias_vals = _bias_acc.detach().cpu().numpy()[i,0:3]
                                gyro_noise_vals = [0.0, 0.0, 0.0]
                                acc_noise_vals = [0.0, 0.0, 0.0]

                                writer.writerow(list(angular_velocity_vals) + list(linear_acceleration_vals) +
                                    list(gyro_bias_vals) + list(acc_bias_vals) +
                                    list(gyro_noise_vals) + list(acc_noise_vals))
                        else:
                            angular_velocity_vals = input_data["gyro"].detach().cpu().numpy()[i,-1,0:3]
                            linear_acceleration_vals = input_data["acc"].detach().cpu().numpy()[i,-1,0:3]
                            gyro_bias_vals = _bias_gyro.detach().cpu().numpy()[i,0:3]
                            acc_bias_vals = _bias_acc.detach().cpu().numpy()[i,0:3]
                            gyro_noise_vals = _noise_gyro.detach().cpu().numpy()[i,-1,0:3]
                            acc_noise_vals = _noise_acc.detach().cpu().numpy()[i,-1,0:3]

                            writer.writerow(list(angular_velocity_vals) + list(linear_acceleration_vals) +
                                    list(gyro_bias_vals) + list(acc_bias_vals) +
                                    list(gyro_noise_vals) + list(acc_noise_vals))
            file.close()    
            print(f">> {sub_dataset_name}/imu_label_data.csv: done.")