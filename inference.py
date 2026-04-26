import os
import torch
import argparse
import rospy
import queue
import numpy as np
import csv
import rosbag
import tqdm

from pyhocon import ConfigFactory
from model import IPNet
from sensor_msgs.msg import Imu
from collections import deque
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/euroc.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")

    args = parser.parse_args()
    print(">> args = ", args)

    conf = ConfigFactory.parse_file(args.config)
    print(">> conf = ", conf)

    model = IPNet(conf).to(conf.device)
    ckpt_path = os.path.join(conf.output_path, "ckpt/best_model.ckpt") # best_model_test_append,best_model_test
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location=torch.device(conf.device))
    print(f"\nloaded state dict {ckpt_path} in epoch {checkpoint['epoch']}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(os.path.join(conf.output_path, "bag"), exist_ok=True)
    for data_list in conf.dataset.inference.data_list:
        for path in data_list.data_drive:

            os.makedirs(os.path.join(conf.output_path, f"bag/{path}"), exist_ok=True)
            csv_output_path = os.path.join(conf.output_path, f"bag/{path}/{path}.csv")
            if os.path.exists(csv_output_path):
                continue
            with open(csv_output_path, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'seq',
                                'gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z',
                                'gyro_noise_x', 'gyro_noise_y', 'gyro_noise_z',
                                'acc_bias_x', 'acc_bias_y', 'acc_bias_z',
                                'acc_noise_x', 'acc_noise_y', 'acc_noise_z',
                                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])
                file.close()
            file = open(csv_output_path, mode='a', newline='')
            writer = csv.writer(file)

            data_path = os.path.join(data_list.data_root, "bag/" + path) + ".bag"
            bag = rosbag.Bag(data_path)
            topic_name = data_list.topic_name
            total_messages = bag.get_message_count(topic_filters=[topic_name])
            print("\n>> Processing file: " + data_path)

            TARGET_SIZE = data_list.window_size
            timestamps_deque = deque(maxlen=TARGET_SIZE)
            angular_velocity_deque = deque(maxlen=TARGET_SIZE)
            linear_acceleration_deque = deque(maxlen=TARGET_SIZE)
            init_flag = False
            seq = 0
            
            for topic, msg, t in tqdm.tqdm(bag.read_messages(topics=[topic_name]), total=total_messages, desc="Reading IMU messages"):
                t = msg.header.stamp.to_sec()
                angular_velocity = msg.angular_velocity
                linear_acceleration = msg.linear_acceleration

                timestamps_deque.append([t])
                angular_velocity_deque.append([angular_velocity.x, angular_velocity.y, angular_velocity.z])
                linear_acceleration_deque.append([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

                if len(timestamps_deque) == TARGET_SIZE:
                    timestamps_array = np.array(timestamps_deque, dtype=np.float64)
                    angular_velocity_array = np.array(angular_velocity_deque, dtype=np.float64)
                    linear_acceleration_array = np.array(linear_acceleration_deque, dtype=np.float64)

                    timestamps_tensor = torch.tensor(timestamps_array, dtype=torch.float64).unsqueeze(dim=0).to(conf.device)
                    angular_velocity_tensor = torch.tensor(angular_velocity_array, dtype=torch.float64).unsqueeze(dim=0).to(conf.device)
                    linear_acceleration_tensor = torch.tensor(linear_acceleration_array, dtype=torch.float64).unsqueeze(dim=0).to(conf.device)

                    input_data = {'dt': timestamps_tensor, 'acc': linear_acceleration_tensor, 'gyro': angular_velocity_tensor}

                    start_time = time.time()
                    gyro_bias, gyro_noise, acc_bias, acc_noise = model(input_data)
                    end_time = time.time()
                    total_time = end_time - start_time
                    # with open('time_log.txt', 'a') as file:
                    #     file.write(f'{total_time}\n')

                    if init_flag == False:
                        init_flag = True
                        for i in range(TARGET_SIZE):
                            timestamp = rospy.Time.from_sec(timestamps_array[i][0])
                            gyro_bias_vals = torch.mean(gyro_bias, dim=1)[0].detach().cpu().numpy()[0:3]
                            gyro_noise_vals = [0.0, 0.0, 0.0]
                            acc_bias_vals = torch.mean(acc_bias, dim=1)[0].detach().cpu().numpy()[0:3]
                            acc_noise_vals = [0.0, 0.0, 0.0]
                            angular_velocity_vals = angular_velocity_array[i][0:3]
                            linear_acceleration_vals = linear_acceleration_array[i][0:3]

                            writer.writerow([timestamp] + [seq] +
                                        list(gyro_bias_vals) + list(gyro_noise_vals) +
                                        list(acc_bias_vals) + list(acc_noise_vals) +
                                        list(angular_velocity_vals) + list(linear_acceleration_vals))
                            seq = seq + 1
                    else:
                        timestamp = rospy.Time.from_sec(timestamps_array[i][0])
                        gyro_bias_vals = torch.mean(gyro_bias, dim=1)[0].detach().cpu().numpy()[0:3]
                        # gyro_noise_vals = gyro_noise[0][-1].detach().cpu().numpy()[0:3]
                        acc_bias_vals = torch.mean(acc_bias, dim=1)[0].detach().cpu().numpy()[0:3]
                        # acc_noise_vals = acc_noise[0][-1].detach().cpu().numpy()[0:3]
                        angular_velocity_vals = angular_velocity_array[i][0:3]
                        linear_acceleration_vals = linear_acceleration_array[i][0:3]

                        writer.writerow([timestamp] + [seq] +
                                        list(gyro_bias_vals) + list(gyro_noise_vals) +
                                        list(acc_bias_vals) + list(acc_noise_vals) +
                                        list(angular_velocity_vals) + list(linear_acceleration_vals))
                        seq = seq + 1

                    # print(f"\nt = {timestamp}")
                    # print(f"gyro = {angular_velocity_array[-1]}")
                    # print(f"acc = {linear_acceleration_array[-1]}")
                    # print(f"gyro_bias = {gyro_bias[0]}")
                    # print(f"acc_bias = {acc_bias[0]}")
                    # print(f"gyro_noise = {gyro_noise[0][-1]}")
                    # print(f"acc_noise = {acc_noise[0][-1]}")
            bag.close()
            file.close()
            file_name = conf.output_path + f"bag/{path}/{path}.csv"
            print(">> Getting new files: " + file_name)
