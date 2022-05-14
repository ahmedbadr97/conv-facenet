import sys
import time
from datetime import datetime

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
                    time_remaing / 60.0)[:8]))

            test_loss = loss_sum / dataset_size

        return test_loss


def triplet_loss_train(model, epochs, learn_rate, train_loader, test_loader, weight_saving_path, cuda):
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
        loss_sum = 0.0

        cnt = 0.0
        time_sum = 0.0
        for anchor_img, positive_img, negative_img in train_loader:
            ts = time.time()
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
            te = time.time()
            time_sum += (te - ts)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)
            sys.stdout.write("\r epoch " + str(e + 1) + " [" + str(
                "=" * int((cnt * 10) / no_batches) + str("." * remaining) + "] time remaining = " + str(
                    time_remaing / 60.0)[:8]))
        print()
        train_loss = loss_sum / dataset_size
        train_losses.append(train_loss)
        test_loss = triplet_loss_test(model, test_loader, loss_function, cuda)
        test_losses.append(test_loss)
        print()
        print(f" epoch {e + 1} train_loss ={train_loss} test_loss={test_loss}")
        if train_loss < min_train_loss and test_loss < min_test_loss:
            save_train_weights(model, train_loss, test_loss, weight_saving_path)
            print(
                f"new minimum test loss {str(train_loss)[:8]} and train loss {str(test_loss)[:8]} achieved, model weights saved")
            min_train_loss = train_loss
            min_test_loss = test_loss

        if train_loss < test_loss:
            print("!!!Warning Overfitting!!!")
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
