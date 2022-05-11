import sys
import time

from torch import optim
from torch.nn import TripletMarginLoss


def train(model, epochs, learn_rate, train_loader, cuda):

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_function = TripletMarginLoss()
    batch_size = train_loader.batch_size
    no_batches = len(train_loader)
    dataset_size = float(len(train_loader.dataset))

    if cuda:
        model.cuda()

    train_losses = []
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

        epoch_loss = loss_sum / dataset_size
        train_losses.append(epoch_loss)
        print(f" epoch {e + 1} loss ={epoch_loss}")
    return train_losses
