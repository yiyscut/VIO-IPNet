import os
import torch
import numpy as np
import pypose as pp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from .dataset import Sequence

class AR_TABLE(Sequence):
    def __init__(self, label_path, data_root, data_name, intepolate = True, glob_coord=False, **kwargs):
        print(">> AR_TABLE", data_name, "init...")
        super(AR_TABLE, self).__init__()
        (   
            self.data_root, 
            self.data_name,
            self.data,
        ) = (data_root, data_name, dict())
        
        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path)
        self.load_gt(data_path)

        if os.path.exists(os.path.join(label_path, f"label/label_done")):
            fake_label_data_path = os.path.join(label_path, f"label/{data_name}/imu_label_data.csv")
            self.load_imu_label(fake_label_data_path)
        
        if intepolate:
            t_start = np.max([self.data['gt_time'][0], self.data['time'][0]])   # 选大的
            t_end = np.min([self.data['gt_time'][-1], self.data['time'][-1]])   # 选小的

            idx_start_imu = np.searchsorted(self.data['time'], t_start)
            idx_start_gt = np.searchsorted(self.data['gt_time'], t_start)

            idx_end_imu = np.searchsorted(self.data['time'], t_end, 'right')
            idx_end_gt = np.searchsorted(self.data['gt_time'], t_end, 'right')

            for k in ['gt_time', 'gt_pos', 'gt_quat', 'gt_velocity', 'gt_b_acc', 'gt_b_gyro']:
                self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

            for k in ['time', 'acc', 'gyro']:
                self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

            self.data["gt_orientation"] = self.interp_rot(self.data['time'], self.data['gt_time'], self.data['gt_quat'])
            self.data["gt_translation"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['gt_pos'])
            self.data["gt_b_acc"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["gt_b_acc"])
            self.data["gt_b_gyro"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["gt_b_gyro"])
            self.data["gt_velocity"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["gt_velocity"])
        
        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data['dt'] = (self.data["time"][1:] - self.data["time"][:-1])[:,None]
        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])

        if os.path.exists(os.path.join(label_path, f"label/label_done")):
            self.data["fake_gyro_bias"] = torch.tensor(self.data["fake_gyro_bias"]) 
            self.data["fake_acc_bias"] = torch.tensor(self.data["fake_acc_bias"])
            self.data["fake_gyro_noise"] = torch.tensor(self.data["fake_gyro_noise"])
            self.data["fake_acc_noise"] = torch.tensor(self.data["fake_acc_noise"])
            # print(self.data["gyro"][0, ...], torch.tensor(self.data["fake_gyro_test"][0, ...]))
            # print(self.data["acc"][0, ...], torch.tensor(self.data["fake_acc_test"][0, ...]))
        
        # change the acc and gyro scope into the global coordinate.  
        if glob_coord:
            self.data['gyro'] = self.data["gt_orientation"] * self.data['gyro']
            self.data['acc'] = self.data["gt_orientation"] * self.data['acc']

    def get_length(self):
        return self.data['time'].shape[0]

    def load_imu(self, folder):
        imu_data = np.loadtxt(os.path.join(folder, "raw_data.csv"), dtype=float, delimiter=',', skiprows=1)
        self.data["time"] = imu_data[:,0] / 1e9
        self.data["gyro"] = imu_data[:,1:4]         # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = imu_data[:,4:]           # acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self, folder):
        gt_data = np.loadtxt(os.path.join(folder, "gt_data.csv"), dtype=float, delimiter=',')
        self.data["gt_time"] = gt_data[:,0] / 1e9
        self.data["gt_pos"] = gt_data[:,1:4]
        self.data['gt_quat'] = gt_data[:,4:8] # w, x, y, z
        self.data["gt_velocity"] = gt_data[:,-9:-6]
        self.data["gt_b_acc"] = gt_data[:,-3:]
        self.data["gt_b_gyro"] = gt_data[:,-6:-3]
    
    def load_imu_label(self, folder):
        imu_label_data = np.loadtxt(os.path.join(folder), dtype=float, delimiter=',', skiprows=1)
        self.data["fake_gyro_test"] = imu_label_data[:,0:3]
        self.data["fake_acc_test"] = imu_label_data[:,3:6]
        self.data["fake_gyro_bias"] = imu_label_data[:,6:9]
        self.data["fake_acc_bias"] = imu_label_data[:,9:12]
        self.data["fake_gyro_noise"] = imu_label_data[:,12:15]
        self.data["fake_acc_noise"] = imu_label_data[:,15:18]

    def interp_rot(self, time, opt_time, quat):
        # interpolation in the log space
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat = torch.tensor(quat)
        quat = self.__qinterp(quat, gt_dt, imu_dt).double()
        self.data['rot_wxyz'] = quat
        rot = torch.zeros_like(quat)
        rot[:,3] = quat[:,0]
        rot[:,:3] = quat[:,1:]
        return pp.SO3(rot)

    def interp_xyz(self, time, opt_time, xyz):
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def __qinterp(self, qs, t, t_int):
        qs = R.from_quat(qs.numpy())
        slerp = Slerp(t, qs)
        interp_rot = slerp(t_int).as_quat()
        return torch.tensor(interp_rot)
