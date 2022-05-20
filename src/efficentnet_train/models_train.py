import os
import sys
import time
from datetime import datetime

import pandas as pd
from torch import optim, no_grad
from torch.nn import TripletMarginLoss
import torch


def triplet_loss_test(model, test_loader, loss_function, cuda):
    batch_size = test_loader.batch_size
    no_batches = len(test_loader)
    dataset_size = float(len(test_loader.dataset))
    model.eval()
    if cuda:
        model.cuda()
    loss_sum = 0.0
    cnt = 0.0
    time_sum = 0.0
    with no_grad():
        for anchor_img, positive_img, negative_img in test_loader:
            ts = time.time()
            if cuda:
                anchor_img, positive_img, negative_img = anchor_img.cuda(), positive_img.cuda(), negative_img.cuda()
            anchor_vector = model(anchor_img)
            positive_vector = model(positive_img)
            negative_vector = model(negative_img)
            loss = loss_function(anchor_vector, positive_vector, negative_vector)
            loss_sum += loss.item() * batch_size

            cnt += 1.0
            finished = int((cnt * 10) / no_batches)
            remaining = 10 - finished
            te = time.time()
            time_sum += (te - ts)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)
            sys.stdout.write("\r Testing  [" + str(
                "=" * finished + str("." * remaining) + "] time remaining = " + str(
                    time_remaing / 60.0)[:8]+ " Avg Test_Loss=" + str(loss_sum / (cnt * batch_size))[:8]))

            test_loss = loss_sum / dataset_size

        return test_loss


def triplet_loss_train(model, epochs, learn_rate, train_loader, test_loader, cuda=False, weight_saving_path=None,
                       epoch_data_saving_path=None, notes=None
                       ):
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_function = TripletMarginLoss()
    batch_size = train_loader.batch_size
    no_batches = len(train_loader)
    dataset_size = float(len(train_loader.dataset))
    min_train_loss = 800000
    min_test_loss = 800000
    model.train()
    if cuda:
        model.cuda()

    train_losses = []
    test_losses = []
    for e in range(epochs):
        epoch_start_time = time.time()
        loss_sum = 0.0

        cnt = 0.0
        time_sum = 0.0
        for anchor_img, positive_img, negative_img in train_loader:
            batch_start_t = time.time()
            optimizer.zero_grad()
            anchor_img.requires_grad = True
            positive_img.requires_grad = True
            negative_img.requires_grad = True

            if cuda:
                anchor_img, positive_img, negative_img = anchor_img.cuda(), positive_img.cuda(), negative_img.cuda()

            anchor_vector = model(anchor_img)
            positive_vector = model(positive_img)
            negative_vector = model(negative_img)

            loss = loss_function(anchor_vector, positive_vector, negative_vector)
            loss.backward()

            optimizer.step()
            loss_sum += loss.item() * batch_size

            cnt += 1.0
            finished = int((cnt * 10) / no_batches)
            remaining = 10 - finished
            batch_end_time = time.time()
            time_sum += (batch_end_time - batch_start_t)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)
            sys.stdout.write("\r epoch " + str(e + 1) + " [" + str(
                "=" * int((cnt * 10) / no_batches) + str("." * remaining) + "] time remaining = " + str(
                    time_remaing / 60.0)[:8]) + " Avg Train_Loss=" + str(loss_sum / (cnt * batch_size))[:8])
        print()
        train_loss = loss_sum / dataset_size
        train_losses.append(train_loss)
        test_loss = triplet_loss_test(model, test_loader, loss_function, cuda)
        test_losses.append(test_loss)
        epoch_end_time = time.time()

        print()
        print(f" epoch {e + 1} train_loss ={train_loss} test_loss={test_loss}")
        if  test_loss < min_test_loss:
            print(
                f"new minimum test loss {str(test_loss)[:8]} ", end=" ")
            if weight_saving_path is not None:
                save_train_weights(model, train_loss, test_loss, weight_saving_path)
                print("achieved, model weights saved", end=" ")
            print()

            min_train_loss = train_loss
            min_test_loss = test_loss

        if train_loss < test_loss:
            print("!!!Warning Overfitting!!!")
        epoch_time_taken = round((epoch_end_time - epoch_start_time) / 60, 1)
        save_epochs_to_csv(epoch_data_saving_path, train_loss, len(train_loader.dataset), test_loss,
                           len(test_loader.dataset), epoch_time_taken, notes)
    return train_losses, test_losses


def save_train_weights(model, train_loss, test_loss, saving_path):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]}) Test_({str(test_loss)[:8]}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path


def save_epochs_to_csv(csv_save_path, train_loss, no_train_rows, test_loss, no_test_rows, time_taken, notes=None):
    if notes is None:
        notes = ""
    date_now = datetime.now()
    if len(csv_save_path) == 0:
        full_path = "train_data.csv"
    else:
        full_path = f"{csv_save_path}/train_data.csv"
    row = [[train_loss, no_train_rows, test_loss, no_test_rows, time_taken, notes, date_now.strftime('%d/%m/%Y'),
            date_now.strftime('%H:%M:00')]]
    df = pd.DataFrame(row,
                      columns=["Train Loss", "no train rows", "Test Loss", "No test rows", "Time taken (M)", "Notes",
                               "Date", "Time"])

    if not os.path.exists(full_path):
        df.to_csv(full_path, index=False)
    else:
        df.to_csv(full_path, mode='a', header=False, index=False)


def full_classification_train(face_descriptor, face_identifier, epochs, learn_rate, train_loader, test_loader,
                              cuda=False, weight_saving_path=None,
                              epoch_data_saving_path=None, notes=None):
    pass
