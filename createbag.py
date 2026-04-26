import rosbag
import argparse
import os
import rospy
import csv
import tqdm

from sensor_msgs.msg import Imu
from pyhocon import ConfigFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/euroc.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")

    args = parser.parse_args()
    print(">> args = ", args)

    conf = ConfigFactory.parse_file(args.config)
    print(">> conf = ", conf)

    os.makedirs(os.path.join(conf.output_path, "bag"), exist_ok=True)
    for data_list in conf.dataset.inference.data_list:
        for path in data_list.data_drive:
            input_bag_path = os.path.join(data_list.data_root, "bag/" + path) + ".bag"
            input_bag = rosbag.Bag(input_bag_path, 'r')
            print("\n>> Processing file: " + input_bag_path)

            output_bag_path = os.path.join(conf.output_path, f"bag/{path}/{path}.bag")
            # if os.path.exists(output_bag_path):
            #     continue
            output_bag = rosbag.Bag(output_bag_path, 'w')

            for topic, msg, t in input_bag.read_messages():
                if topic != '/imu0':
                    output_bag.write(topic, msg, t)

            csv_file_path = os.path.join(conf.output_path, f"bag/{path}/{path}.csv")
            if os.path.exists(csv_file_path):
                with open(csv_file_path, mode='r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    header = next(csv_reader)

                    total_lines = sum(1 for _ in open(csv_file_path))
                    for row in tqdm.tqdm(csv_reader, total=total_lines, desc="Reading CSV file"):
                        imu_msg = Imu()

                        timestamp_in_ns = int(row[0])
                        timestamp_in_seconds = timestamp_in_ns / 1e9
                        imu_msg.header.stamp = rospy.Time.from_sec(timestamp_in_seconds)
                        imu_msg.header.seq = int(row[1])

                        imu_msg.angular_velocity.x = float(row[14])  # angular_velocity_x
                        imu_msg.angular_velocity.y = float(row[15])  # angular_velocity_y
                        imu_msg.angular_velocity.z = float(row[16])  # angular_velocity_z

                        imu_msg.linear_acceleration.x = float(row[17])  # linear_acceleration_x
                        imu_msg.linear_acceleration.y = float(row[18])  # linear_acceleration_y
                        imu_msg.linear_acceleration.z = float(row[19])  # linear_acceleration_z

                        imu_msg.angular_velocity_covariance[0] = float(row[2])  # gyro_bias_x
                        imu_msg.angular_velocity_covariance[1] = float(row[3])  # gyro_bias_y
                        imu_msg.angular_velocity_covariance[2] = float(row[4])  # gyro_bias_z
                        imu_msg.angular_velocity_covariance[3] = float(row[5])  # gyro_noise_x
                        imu_msg.angular_velocity_covariance[4] = float(row[6])  # gyro_noise_y
                        imu_msg.angular_velocity_covariance[5] = float(row[7])  # gyro_noise_z

                        imu_msg.linear_acceleration_covariance[0] = float(row[8])  # acc_bias_x
                        imu_msg.linear_acceleration_covariance[1] = float(row[9])  # acc_bias_y
                        imu_msg.linear_acceleration_covariance[2] = float(row[10])  # acc_bias_z
                        imu_msg.linear_acceleration_covariance[3] = float(row[11])  # acc_noise_x
                        imu_msg.linear_acceleration_covariance[4] = float(row[12])  # acc_noise_y
                        imu_msg.linear_acceleration_covariance[5] = float(row[13])  # acc_noise_z

                        output_bag.write('/imu_output', imu_msg, imu_msg.header.stamp)
            else:
                print(f">> {csv_file_path} does not exist.")

            input_bag.close()
            output_bag.close()

            print(">> Getting new files: " + output_bag_path)