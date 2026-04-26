import rosbag
import argparse
import os
import rospy
import csv
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from pyhocon import ConfigFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/euroc.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")

    args = parser.parse_args()
    print(">> args = ", args)

    conf = ConfigFactory.parse_file(args.config)
    print(">> conf = ", conf)

    os.makedirs(os.path.join(conf.output_path, "debug"), exist_ok=True)
    for data_list in conf.dataset.inference.data_list:
        for path in data_list.data_drive:
            dataset_name = path

            raw_gyro_x, raw_gyro_y, raw_gyro_z = [], [], []
            raw_acc_x, raw_acc_y, raw_acc_z = [], [], []
            fake_gyro_bias_x, fake_gyro_bias_y, fake_gyro_bias_z = [], [], []
            fake_acc_bias_x, fake_acc_bias_y, fake_acc_bias_z = [], [], []
            print(data_list.label_path, {path})
            if os.path.exists(os.path.join(data_list.label_path, f"label/{path}/imu_label_data.csv")):
                imu_label_data = np.loadtxt(os.path.join(data_list.label_path, f"label/{path}/imu_label_data.csv"), dtype=float, delimiter=',', skiprows=1)
                raw_gyro_x = imu_label_data[:,0:1]
                raw_gyro_y = imu_label_data[:,1:2]
                raw_gyro_z = imu_label_data[:,2:3]

                raw_acc_x = imu_label_data[:,3:4]
                raw_acc_y = imu_label_data[:,4:5]
                raw_acc_z = imu_label_data[:,5:6]
                
                fake_gyro_bias_x = imu_label_data[:,6:7]
                fake_gyro_bias_y = imu_label_data[:,7:8]
                fake_gyro_bias_z = imu_label_data[:,8:9]

                fake_acc_bias_x = imu_label_data[:,9:10]
                fake_acc_bias_y = imu_label_data[:,10:11]
                fake_acc_bias_z = imu_label_data[:,11:12]

            gt_gyro_bias_x, gt_gyro_bias_y, gt_gyro_bias_z = [], [], []
            gt_acc_bias_x, gt_acc_bias_y, gt_acc_bias_z = [], [], []
            imu_label_data = np.loadtxt(os.path.join(data_list.gt_root, f"{path}/mav0/state_groundtruth_estimate0/data.csv"), dtype=float, delimiter=',')
            # imu_label_data = np.loadtxt(os.path.join(data_list.gt_root, f"{path}gt_data.csv"), dtype=float, delimiter=',')
            gt_gyro_bias_x = imu_label_data[:,-6:-5]
            gt_gyro_bias_y = imu_label_data[:,-5:-4]
            gt_gyro_bias_z = imu_label_data[:,-4:-3]

            gt_acc_bias_x = imu_label_data[:,-3:-2]
            gt_acc_bias_y = imu_label_data[:,-2:-1]
            gt_acc_bias_z = imu_label_data[:,-1:]

            csv_file_path = conf.output_path + f"bag/{path}/{path}.csv"

            gyro_bias_x, gyro_bias_y, gyro_bias_z= [], [], []
            gyro_noise_x, gyro_noise_y, gyro_noise_z = [], [], []
            acc_bias_x, acc_bias_y, acc_bias_z = [], [], []
            acc_noise_x, acc_noise_y, acc_noise_z = [], [], []
            time = []

            if os.path.exists(csv_file_path):
                with open(csv_file_path, mode='r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    header = next(csv_reader)       # 跳过 CSV 文件的表头

                    total_lines = sum(1 for _ in open(csv_file_path))
                    for row in tqdm.tqdm(csv_reader, total=total_lines, desc="Reading CSV file"):
                        time.append(int(row[0]) / 1e9)

                        gyro_bias_x.append(float(row[2]))
                        gyro_bias_y.append(float(row[3]))
                        gyro_bias_z.append(float(row[4]))

                        gyro_noise_x.append(float(row[5]))
                        gyro_noise_y.append(float(row[6]))
                        gyro_noise_z.append(float(row[7]))

                        acc_bias_x.append(float(row[8]))
                        acc_bias_y.append(float(row[9]))
                        acc_bias_z.append(float(row[10]))

                        acc_noise_x.append(float(row[11]))
                        acc_noise_y.append(float(row[12]))
                        acc_noise_z.append(float(row[13]))

                    plt.figure(figsize=(14, 10))

                    # Gyro Bias X, Y, Z
                    plt.subplot(4, 2, 1)
                    plt.plot(gyro_bias_x[:], color="#6699CC", label='IPNet_Output(gyro_bias_x)')
                    plt.plot(fake_gyro_bias_x[:], color="#66C1A4", label='Ours_Label')
                    plt.plot(gt_gyro_bias_x[:], color="#C166A4", label='Dataset_Label')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Gyro Bias X')
                    plt.legend()

                    plt.subplot(4, 2, 3)
                    plt.plot(gyro_bias_y[:], color="#6699CC", label='IPNet_Output(gyro_bias_y)')
                    plt.plot(fake_gyro_bias_y[:], color="#66C1A4", label='Ours_Label')
                    plt.plot(gt_gyro_bias_y[:], color="#C166A4", label='Dataset_Label')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Gyro Bias Y')
                    plt.legend()

                    plt.subplot(4, 2, 5)
                    plt.plot(gyro_bias_z[:], color="#6699CC", label='IPNet_Output(gyro_bias_z)')
                    plt.plot(fake_gyro_bias_z[:], color="#66C1A4", label='Ours_Label')
                    plt.plot(gt_gyro_bias_z[:], color="#C166A4", label='Dataset_Label')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Gyro Bias Z')
                    plt.legend()

                    # Acc Bias X, Y, Z
                    plt.subplot(4, 2, 2)
                    plt.plot(acc_bias_x[:], color="#6699CC", label='IPNet_Output(acc_bias_x)')
                    plt.plot(fake_acc_bias_x[:], color="#66C1A4", label='Ours_Label')
                    plt.plot(gt_acc_bias_x[:], color="#C166A4", label='Dataset_Label')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Acc Bias X')
                    plt.legend()

                    plt.subplot(4, 2, 4)
                    plt.plot(acc_bias_y[:], color="#6699CC", label='IPNet_Output(acc_bias_y)')
                    plt.plot(fake_acc_bias_y[:], color="#66C1A4", label='Ours_Label')
                    plt.plot(gt_acc_bias_y[:], color="#C166A4", label='Dataset_Label')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Acc Bias Y')
                    plt.legend()

                    plt.subplot(4, 2, 6)
                    plt.plot(acc_bias_z[:], color="#6699CC", label='IPNet_Output(acc_bias_z)')
                    plt.plot(fake_acc_bias_z[:], color="#66C1A4", label='Ours_Label')
                    plt.plot(gt_acc_bias_z[:], color="#C166A4", label='Dataset_Label')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Acc Bias Z')
                    plt.legend()

                    # Raw Gyro
                    plt.subplot(4, 2, 7)
                    plt.plot(raw_gyro_x[:], color="#6699CC", label='raw_gyro_x')
                    plt.plot(raw_gyro_y[:], color="#CC0033", label='raw_gyro_y')
                    plt.plot(raw_gyro_z[:], color="#66C1A4", label='raw_gyro_z')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Raw-Gyro (X, Y, Z)')
                    plt.legend()

                    # Raw Acc
                    plt.subplot(4, 2, 8)
                    plt.plot(raw_acc_x[:], color="#6699CC", label='raw_acc_x')
                    plt.plot(raw_acc_y[:], color="#CC0033", label='raw_acc_y')
                    plt.plot(raw_acc_z[:], color="#66C1A4", label='raw_acc_z')
                    plt.margins(y=0.3)
                    plt.autoscale(enable=True, axis='y', tight=False)
                    plt.title('Raw-Acc (X, Y, Z)')
                    plt.legend()

                    plt.suptitle(f'Comparison of IMU bias data in X, Y, and Z axes on {dataset_name}', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(conf.output_path + f'debug/{dataset_name}.png')
            else:
                print(f">> {csv_file_path} does not exist.")
