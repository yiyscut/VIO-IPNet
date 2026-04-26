import os, argparse
import numpy as np
import rosbag
import csv
import sys
from sensor_msgs.msg import Imu

from pyhocon import ConfigFactory

IMU_TOPIC = '/d455/imu'

def write_imu_to_csv(bag_file, output_csv):
    bag = rosbag.Bag(bag_file)
    with open(output_csv, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['timestamp',
                            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                            'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])

        for topic, msg, t in bag.read_messages(topics=[IMU_TOPIC]):
            timestamp = str(msg.header.stamp)
            angular_velocity = msg.angular_velocity
            linear_acceleration = msg.linear_acceleration

            csvwriter.writerow([timestamp, 
                                angular_velocity.x, angular_velocity.y, angular_velocity.z,
                                linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

    bag.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ar_table.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")
    args = parser.parse_args()
    print(">> args = ", args)

    conf = ConfigFactory.parse_file(args.config)
    print(">> conf = ", conf)

    for data_list in conf.dataset.inference.data_list:
        for path in data_list.data_drive:
            os.makedirs(os.path.join(data_list.data_root, path), exist_ok=True)
            bag_file = os.path.join(data_list.data_root, path, path + ".bag")
            output_csv = os.path.join(data_list.data_root, path, "raw_data.csv")

            write_imu_to_csv(bag_file, output_csv)
            print(">> Getting new files: " + output_csv)