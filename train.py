import os
import torch
import argparse
import numpy as np
import tqdm
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert
from datasets import SeqeuncesDataset
from model import IPNet
from utils import move_to
from model.preintegrator import cal_imu_bias
from torch.optim.lr_scheduler import StepLR

def train(model, loader, criterion, epoch, optimizer, conf):
    model.train()
    gyro_bias_losses, acc_bias_losses, gyro_noise_losses, acc_noise_losses = 0.0, 0.0, 0.0, 0.0
    t_range = tqdm.tqdm(loader, dynamic_ncols=True)
    t_range.set_description('\n\n>>>>>> training epoch: %03d, gyro_losses: np.inf, acc_losses: np.inf'%(epoch))
    for i, (input_data, init_state, label) in enumerate(t_range):
        input_data, init_state, label = move_to([input_data, init_state, label], conf.device)

        gyro_bias, gyro_noise, acc_bias, acc_noise = model(input_data)
        gyro_bias_loss = criterion(gyro_bias, label["fake_gyro_bias"][:, 19::20, :])
        acc_bias_loss = criterion(acc_bias, label["fake_acc_bias"][:, 19::20, :])
        # gyro_noise_loss = criterion(gyro_noise, label["fake_gyro_noise"])
        # acc_noise_loss = criterion(acc_noise, label["fake_acc_noise"])

        gyro_bias_losses += gyro_bias_loss.item()
        acc_bias_losses += acc_bias_loss.item()
        # gyro_noise_losses += gyro_noise_loss.item()
        # acc_noise_losses += acc_noise_loss.item()

        loss = gyro_bias_loss + acc_bias_loss # + gyro_noise_loss + acc_noise_loss

        t_range.set_description('>>>> training, gyro_bias_losses: %.06f, acc_bias_losses: %.06f, gyro_noise_losses: %.06f, acc_noise_losses: %.06f'% \
            (gyro_bias_losses/(i + 1), acc_bias_losses/(i + 1), gyro_noise_losses/(i + 1), acc_noise_losses/(i + 1)))
        t_range.refresh()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"gyro_bias_losses": (gyro_bias_losses/(i + 1)), "acc_bias_losses": (acc_bias_losses/(i + 1)), "gyro_noise_losses": (gyro_noise_losses/(i + 1)), "acc_noise_losses": (acc_noise_losses/(i + 1))}

def test(model, loader, criterion, conf):
    model.eval()
    gyro_bias_losses, acc_bias_losses, gyro_noise_losses, acc_noise_losses = 0.0, 0.0, 0.0, 0.0
    t_range = tqdm.tqdm(loader, dynamic_ncols=True) 
    t_range.set_description('\n\n>>>>>> testing, gyro_losses: np.inf, acc_losses: np.inf')
    for i, (input_data, init_state, label) in enumerate(t_range):
        input_data, init_state, label = move_to([input_data, init_state, label], conf.device)

        gyro_bias, gyro_noise, acc_bias, acc_noise = model(input_data)
        gyro_bias_loss = criterion(gyro_bias, label["fake_gyro_bias"][:, 19::20, :])
        acc_bias_loss = criterion(acc_bias, label["fake_acc_bias"][:, 19::20, :])
        # gyro_noise_loss = criterion(gyro_noise, label["fake_gyro_noise"])
        # acc_noise_loss = criterion(acc_noise, label["fake_acc_noise"])

        gyro_bias_losses += gyro_bias_loss.item()
        acc_bias_losses += acc_bias_loss.item()
        # gyro_noise_losses += gyro_noise_loss.item()
        # acc_noise_losses += acc_noise_loss.item()

        loss = gyro_bias_loss + acc_bias_loss # + gyro_noise_loss + acc_noise_loss

        t_range.set_description('>>>> testing, gyro_bias_losses: %.06f, acc_bias_losses: %.06f, gyro_noise_losses: %.06f, acc_noise_losses: %.06f'% \
            (gyro_bias_losses/(i + 1), acc_bias_losses/(i + 1), gyro_noise_losses/(i + 1), acc_noise_losses/(i + 1)))
        t_range.refresh()
    return {"gyro_bias_losses": (gyro_bias_losses/(i + 1)), "acc_bias_losses": (acc_bias_losses/(i + 1)), "gyro_noise_losses": (gyro_noise_losses/(i + 1)), "acc_noise_losses": (acc_noise_losses/(i + 1))}

def save_ckpt(model, optimizer, scheduler, epoch_i, test_loss, conf, save_best = False):
    # if epoch_i%conf.result_save_freq==conf.result_save_freq-1:  # 5
    #     torch.save(
    #         {
    #             'epoch': epoch_i,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #             'best_loss': test_loss,
    #         }, os.path.join(conf.output_path, "ckpt/%04d.ckpt"%epoch_i)
    #     )

    if save_best:
        print("saving the best model", test_loss)
        torch.save(
            {
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': test_loss,
            }, os.path.join(conf.output_path, "ckpt/best_model.ckpt")
        )
    
    torch.save(
            {
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': test_loss,
            }, os.path.join(conf.output_path, "ckpt/newest.ckpt")
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/euroc.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")

    parser.add_argument('--pretrained', type=str, default=None, help='path to pretrained checkpoint')
    args = parser.parse_args()
    print(">> args = ", args)

    conf = ConfigFactory.parse_file(args.config)
    print(">> conf = ", conf)

    os.makedirs(os.path.join(conf.output_path, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.output_path, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))
    
    if not os.path.exists(os.path.join(conf.output_path, f"label/label_done")):
        cal_imu_bias(conf)
        os.makedirs(os.path.join(conf.output_path, f"label/label_done"), exist_ok=True)
        print("\n>> Please run the program again...")
        exit()
        
    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesDataset(data_set_config=conf.dataset.test)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=SeqeuncesDataset.pad)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=SeqeuncesDataset.pad)

    model = IPNet(conf).to(conf.device, dtype = train_dataset.get_dtype())

    if args.pretrained is not None:
        print(f">> Loading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=conf.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        current_lr = 1e-7 
        print(f">> Fine-tuning mode: Setting learning rate to {current_lr}")
    else:
        print(">> Training from scratch...")
        current_lr = 1e-6

    criterion = torch.nn.functional.l1_loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-6, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_list, train_gyro_bias_loss_list, valid_gyro_bias_loss_list, train_acc_bias_loss_list, valid_acc_bias_loss_list = [], [], [], [], []
    train_gyro_noise_loss_list, valid_gyro_noise_loss_list, train_acc_noise_loss_list, valid_acc_noise_loss_list = [], [], [], []
    best_loss = np.inf
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 18))
    for epoch_i in range(0, conf.num_epochs):
        train_loss = train(model, train_loader, criterion, epoch_i, optimizer, conf)
        valid_loss = test(model, test_loader, criterion, conf)

        total_loss = valid_loss["gyro_bias_losses"] + valid_loss["acc_bias_losses"] + valid_loss["gyro_noise_losses"] + valid_loss["acc_noise_losses"]
        scheduler.step()
        if total_loss < best_loss:
            best_loss = total_loss
            save_best = True
        else:
            save_best = False
        save_ckpt(model, optimizer, scheduler, epoch_i, best_loss, conf, save_best=save_best)
