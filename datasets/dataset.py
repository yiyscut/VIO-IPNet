import os
import torch
import torch.utils.data as Data
import pypose as pp
from abc import ABC

class Sequence(ABC):
    subclasses = {}
    print(">> Sequence init...")
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

class SeqeuncesDataset(Data.Dataset):
    def __init__(self, data_set_config, data_root = None, data_name = None):
        if data_name is None and data_root is None:
            print(">> SeqeuncesDataset init [%s]..."%data_set_config.mode)
        else:
            print("\n>> SeqeuncesDataset is used to calculate the labels...")
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.gt_gyro_bias,
            self.gt_acc_bias,
            self.fake_gyro_bias,
            self.fake_acc_bias,
            self.fake_gyro_noise,
            self.fake_acc_noise,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], 0)
        self.conf = data_set_config
        self.DataClass = Sequence.subclasses

        if data_name is None and data_root is None:
            for conf in data_set_config.data_list:
                for path in conf.data_drive:
                    self.construct_index_map(conf, conf["data_root"], path, self.seq_idx, conf.window_size, conf.step_size)
                    self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_name, self.seq_idx, 20, 1)
            self.seq_idx += 1

    def load_data(self, seq, label_path, start_frame, end_frame):
        if "time" in seq.data.keys():
            self.ts.append(seq.data["time"][start_frame:end_frame])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        if os.path.exists(os.path.join(label_path, f"label/label_done")):
            self.fake_acc_bias.append(seq.data["fake_acc_bias"][start_frame:end_frame])
            self.fake_gyro_bias.append(seq.data["fake_gyro_bias"][start_frame:end_frame])
            self.fake_acc_noise.append(seq.data["fake_acc_noise"][start_frame:end_frame])
            self.fake_gyro_noise.append(seq.data["fake_gyro_noise"][start_frame:end_frame])
        else:
            placeholder = torch.zeros_like(seq.data["acc"])
            self.fake_acc_bias.append(placeholder)
            self.fake_gyro_bias.append(placeholder)
            self.fake_acc_noise.append(placeholder)
            self.fake_gyro_noise.append(placeholder)
        self.dt.append(seq.data["dt"][start_frame:end_frame+1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame:end_frame+1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame:end_frame+1])
        self.gt_velo.append(seq.data["gt_velocity"][start_frame:end_frame+1])

        if "gt_b_gyro" in seq.data.keys():
            self.gt_gyro_bias.append(seq.data["gt_b_gyro"][start_frame:end_frame+1])
        else:
            placeholder = torch.zeros_like(seq.data["acc"])
            self.gt_gyro_bias.append(placeholder)

        if "gt_b_acc" in seq.data.keys():
            self.gt_acc_bias.append(seq.data["gt_b_acc"][start_frame:end_frame+1])
        else:
            placeholder = torch.zeros_like(seq.data["acc"])
            self.gt_acc_bias.append(placeholder)

    def construct_index_map(self, conf, data_root, data_name, seq_id, window_size, step_size):
        seq = self.DataClass[conf.name](conf.label_path, data_root, data_name, intepolate = True, **self.conf)   # Euroc
        seq_len = seq.get_length() - 1
        start_frame, end_frame = 0, seq_len
        _duration = end_frame - start_frame
        self.index_map += [
            [seq_id, j, j+window_size] for j in range(0, _duration - window_size, step_size)
        ]
        self.load_data(seq, conf.label_path, start_frame, end_frame)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        # 0, 0, 100->0, 10, 110->0, 20, 120
        seq_id, frame_id, end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]

        return {
            'dt': self.dt[seq_id][frame_id: end_frame_id],
            'acc': self.acc[seq_id][frame_id: end_frame_id],
            'gyro': self.gyro[seq_id][frame_id: end_frame_id],
            'fake_acc_bias': self.fake_acc_bias[seq_id][frame_id: end_frame_id],
            'fake_gyro_bias': self.fake_gyro_bias[seq_id][frame_id: end_frame_id],
            'fake_acc_noise': self.fake_acc_noise[seq_id][frame_id: end_frame_id],
            'fake_gyro_noise': self.fake_gyro_noise[seq_id][frame_id: end_frame_id],
            'acc_mid': self.acc[seq_id][frame_id: end_frame_id+1],
            'gyro_mid': self.gyro[seq_id][frame_id: end_frame_id+1],
            'gt_pos': self.gt_pos[seq_id][frame_id: end_frame_id+1],
            'gt_rot': self.gt_ori[seq_id][frame_id: end_frame_id+1],
            'gt_vel': self.gt_velo[seq_id][frame_id: end_frame_id+1],
            'gt_b_gyro': self.gt_gyro_bias[seq_id][frame_id: end_frame_id+1],
            'gt_b_acc': self.gt_acc_bias[seq_id][frame_id: end_frame_id+1],
            'init_rot': self.gt_ori[seq_id][frame_id : end_frame_id],
            'init_pos': self.gt_pos[seq_id][frame_id][None, ...],
            'init_vel': self.gt_velo[seq_id][frame_id][None, ...],
        }

    def get_dtype(self):
        return self.acc[0].dtype    # torch.float64

    @classmethod
    def pad(cls, batch):
        # data:                                 init_state:                             label: 
        #   dt: torch.Size([128, 100, 1])          pos: torch.Size([128, 1, 3])            gt_pos: torch.Size([128, 101, 3])
        #   acc: torch.Size([128, 100, 3])         vel: torch.Size([128, 1, 3])            gt_vel: torch.Size([128, 101, 3])
        #   gyro: torch.Size([128, 100, 3])        rot: torch.Size([128, 100, 4])          gt_rot: torch.Size([128, 101, 4])
        #   acc_mid: torch.Size([128, 101, 3])     ----------------------------            gt_b_gyro: torch.Size([128, 101, 3])
        #   gyro_mid: torch.Size([128, 101, 3])    ----------------------------            gt_b_acc: torch.Size([128, 101, 3])
        #   ----------------------------------     ----------------------------            gt_delta_P: torch.Size([128, 100, 3])
        #   ----------------------------------     ----------------------------            gt_delta_V: torch.Size([128, 100, 3])
        #   ----------------------------------     ----------------------------            gt_delta_Q: torch.Size([128, 100, 4])
        #   ----------------------------------     ----------------------------            gt_alpha_item: torch.Size([128, 3])
        #   ----------------------------------     ----------------------------            gt_beta_item: torch.Size([128, 3])
        #   ----------------------------------     ----------------------------            gt_gamma_item: torch.Size([128, 4])

        dt = torch.stack([d['dt'] for d in batch])
        acc = torch.stack([d['acc'] for d in batch])
        gyro = torch.stack([d['gyro'] for d in batch])
        acc_mid = torch.stack([d['acc_mid'] for d in batch])
        gyro_mid = torch.stack([d['gyro_mid'] for d in batch])
        gt_pos = torch.stack([d['gt_pos'] for d in batch])
        gt_rot = torch.stack([d['gt_rot'] for d in batch])
        gt_vel = torch.stack([d['gt_vel'] for d in batch])
        gt_b_gyro = torch.stack([d['gt_b_gyro'] for d in batch])
        gt_b_acc = torch.stack([d['gt_b_acc'] for d in batch])
        init_pos = torch.stack([d['init_pos'] for d in batch])
        init_rot = torch.stack([d['init_rot'] for d in batch])
        init_vel = torch.stack([d['init_vel'] for d in batch])
        fake_acc_bias = torch.stack([d['fake_acc_bias'] for d in batch])
        fake_gyro_bias = torch.stack([d['fake_gyro_bias'] for d in batch])
        fake_acc_noise = torch.stack([d['fake_acc_noise'] for d in batch])
        fake_gyro_noise = torch.stack([d['fake_gyro_noise'] for d in batch])
        
        input_data, init_state, label = {'dt': dt, 'acc': acc, 'gyro': gyro, 'acc_mid': acc_mid, 'gyro_mid': gyro_mid}, \
            {'pos': init_pos, 'vel': init_vel, 'rot': init_rot,}, \
            {'gt_pos': gt_pos, 'gt_vel': gt_vel, 'gt_rot': gt_rot, 'gt_b_gyro': gt_b_gyro, 'gt_b_acc': gt_b_acc, \
             'fake_acc_bias': fake_acc_bias, 'fake_gyro_bias': fake_gyro_bias, 'fake_acc_noise': fake_acc_noise, 'fake_gyro_noise': fake_gyro_noise,}

        # print("Shapes in input_data:")
        # for key, value in input_data.items():
        #     print(f"{key}: {value.shape}")

        # print("\nShapes in init_state:")
        # for key, value in init_state.items():
        #     print(f"{key}: {value.shape}")

        # print("\nShapes in label:")
        # for key, value in label.items():
        #     print(f"{key}: {value.shape}")
        
        gravity = torch.tensor([0., 0., 9.81007], dtype=torch.float64)
        gt_delta_Q = label['gt_rot'][:, 0:-1, :].Inv() * label['gt_rot'][:, 1:, :]
        gt_delta_V = label['gt_rot'][:, 0:-1, :].Inv() * (label['gt_vel'][:, 1:, :] - label['gt_vel'][:, 0:-1, :] + gravity * input_data['dt'][:, :, :])
        gt_delta_P = label['gt_rot'][:, 0:-1, :].Inv() * (label['gt_pos'][:, 1:, :] - label['gt_pos'][:, 0:-1, :] - label['gt_vel'][:, 0:-1, :] * input_data['dt'][:, :, :] + 0.5 * gravity * input_data['dt'][:, :, :] * input_data['dt'][:, :, :])
        
        label['gt_delta_P'] = gt_delta_P
        label['gt_delta_V'] = gt_delta_V
        label['gt_delta_Q'] = gt_delta_Q

        B, F = input_data["dt"].shape[:2]
        gt_delta_Q = torch.cat([pp.identity_SO3(B, 1), gt_delta_Q], dim=1)
        gt_delta_Q = pp.cumprod(gt_delta_Q, dim = 1, left=False)

        gt_delta_V = gt_delta_Q[:, :F, :] * gt_delta_V 
        gt_delta_V = torch.cumsum(gt_delta_V, dim = 1)
        gt_delta_V_tmp = torch.cat((torch.zeros(B, 1, 3, dtype=torch.float64), gt_delta_V), dim=1)

        gt_delta_P = gt_delta_Q[:, :F, :] * gt_delta_P + gt_delta_V_tmp[:, :-1, :] * input_data['dt']
        gt_delta_P = torch.cumsum(gt_delta_P, dim = 1)

        gt_delta_dt = torch.cumsum(input_data["dt"], dim = 1)

        label['gt_alpha_item'] = gt_delta_P[:, -1, :]
        label['gt_beta_item'] = gt_delta_V[:, -1, :]
        label['gt_gamma_item'] = gt_delta_Q[:, -1, :]

        # check
        if False:
            offset_V_gravity = torch.cumsum(gravity * input_data['dt'], dim = 1)

            offset_P_gravity = torch.cumsum(0.5 * gravity * input_data['dt'] ** 2, dim = 1) + \
                torch.cumsum(torch.cat((torch.zeros(B, 1, 3, dtype=torch.float64), offset_V_gravity), dim=1)[:, :-1, :] * input_data['dt'], dim = 1)

            predict = {
                'rot': init_state['rot'][:, 0:1, :] * gt_delta_Q,
                'vel': init_state['vel'][:, 0:1, :] + init_state['rot'][:, 0:1, :] * gt_delta_V - offset_V_gravity,
                'pos': init_state['pos'][:, 0:1, :] + init_state['rot'][:, 0:1, :] * gt_delta_P + init_state['vel'][:, 0:1, :] * gt_delta_dt \
                      - offset_P_gravity,
            }
            print("pos: ", predict['pos'][0:4, -1:, :], label['gt_pos'][0:4, -1:, :])
            print("rot: ", predict['rot'][0:4, -1:, :], label['gt_rot'][0:4, -1:, :])
            print("vel: ", predict['vel'][0:4, -1:, :], label['gt_vel'][0:4, -1:, :])
        
        return  input_data, init_state, label
