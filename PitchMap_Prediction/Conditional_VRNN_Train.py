import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Conditional_VRNN import VRNN
from VRNN_loader_train import load_data_train
from VRNN_loader_val import load_data_val
from torch.utils.data import DataLoader
import numpy as np

model_name = 'ssim_condition_new'

# hyperparameters
num_workers = 1

seq_len = 6
batch_size = 64

h_dim = 1024
z_dim = 256
n_layers = 1
n_epochs = 100
# clip = 10
learning_rate = 1e-4
# seed = 128
# print_every = 10  # batches
save_every = 10 # epochs


def kl_anneal_function(anneal_function, step):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-0.0025 * (step - 2500))))
    elif anneal_function == 'linear':
        return min(1, step / 2500)


def train(epoch, step):
    train_kld_batch = []
    train_rec_batch = []
    train_los_batch = []

    kld_loss_all, rec_loss_all, train_loss = 0, 0, 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # transforming data
        data = data.to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        # print(data.shape)

        kld_loss, rec_loss = model(data)

        kld_loss = kld_loss / seq_len
        rec_loss = rec_loss / seq_len

        # print(kl_anneal_function('logistic', step))
        # loss = kl_anneal_function('logistic', step)*kld_loss + rec_loss

        # if epoch <= 20:
        #     loss = beta_kld * kld_loss + rec_loss
        # else:
        #     loss = beta_kld_2 * kld_loss + rec_loss

        if step <= 15000:
            beta = 1 / (1 + np.exp(-0.0005 * (step - 25000)))
        else:
            beta = 1 / (1 + np.exp(-0.0005 * (15000 - 25000)))

        # if step <= 8000:
        #     beta = 1 / (1 + np.exp(-0.00020 * (step - 25000)))
        # else:
        #     beta = 1 / (1 + np.exp(-0.00020 * (8000 - 25000)))

        loss = beta * kld_loss + rec_loss

        loss.backward()
        optimizer.step()

        # grad norm clipping, only in pytorch version >= 1.10
        # nn.utils.clip_grad_norm_(model.parameters(), clip)

        kld_loss_all += kld_loss
        rec_loss_all += rec_loss
        train_loss += loss

        train_kld_batch.append(kld_loss.cpu().detach().numpy().item())
        train_rec_batch.append(rec_loss.cpu().detach().numpy().item())
        train_los_batch.append(loss.cpu().detach().numpy().item())

        step += 1

    print('STEP------------', step)

    print('====> Epoch: {} Average kld loss: {:.4f}'.format(
        epoch, kld_loss_all / len(train_loader.dataset) * batch_size))
    print('====> Epoch: {} Average rec loss: {:.4f}'.format(
        epoch, rec_loss_all / len(train_loader.dataset) * batch_size))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset) * batch_size))

    return train_loss / len(train_loader.dataset) * batch_size, kld_loss_all / len(
        train_loader.dataset) * batch_size, rec_loss_all / len(train_loader.dataset) * batch_size, step, \
        train_kld_batch, train_rec_batch, train_los_batch


def test(epoch, step):
    """uses test data to evaluate
    likelihood of the model"""
    test_kld_batch = []
    test_rec_batch = []
    test_los_batch = []
    mean_kld_loss, mean_rec_loss, test_loss = 0, 0, 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            kld_loss, rec_loss = model(data)
            kld_loss = kld_loss / seq_len
            rec_loss = rec_loss / seq_len

            # if epoch <= 20:
            #     loss = beta_kld * kld_loss + rec_loss
            # else:
            #     loss = beta_kld_2 * kld_loss + rec_loss

            if step <= 15000:
                beta = 1 / (1 + np.exp(-0.0005 * (step - 25000)))
            else:
                beta = 1 / (1 + np.exp(-0.0005 * (15000 - 25000)))

            # if step <= 8000:
            #     beta = 1 / (1 + np.exp(-0.00020 * (step - 25000)))
            # else:
            #     beta = 1 / (1 + np.exp(-0.00020 * (8000 - 25000)))

            loss = beta * kld_loss + rec_loss

            mean_kld_loss += kld_loss
            mean_rec_loss += rec_loss
            test_loss += loss

            test_kld_batch.append(kld_loss.cpu().numpy().item())
            test_rec_batch.append(rec_loss.cpu().numpy().item())
            test_los_batch.append(loss.cpu().numpy().item())

    mean_kld_loss /= len(test_loader.dataset) / batch_size
    mean_rec_loss /= len(test_loader.dataset) / batch_size
    test_loss /= len(test_loader.dataset) / batch_size

    print('====> Test set loss: KLD Loss = {:.4f},  REC Loss = {:.4f} '.format(
        mean_kld_loss, mean_rec_loss))

    return test_loss, mean_kld_loss, mean_rec_loss, test_kld_batch, test_rec_batch, test_los_batch


# changing device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# init model + optimizer + datasets
train_dataset = load_data_train()
test_dataset = load_data_val()

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

model = VRNN(h_dim, z_dim, n_layers)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_los_log = []
train_kld_log = []
train_rec_log = []
test_los_log = []
test_kld_log = []
test_rec_log = []

train_los_batch_log = []
train_kld_batch_log = []
train_rec_batch_log = []
test_los_batch_log = []
test_kld_batch_log = []
test_rec_batch_log = []

step = 0

for epoch in range(1, n_epochs + 1):

    # training + testing
    train_los, train_kld, train_rec, step, train_kld_batch, train_rec_batch, train_los_batch = train(epoch, step)
    test_los, test_kld, test_rec, test_kld_batch, test_rec_batch, test_los_batch = test(epoch, step - 1)
    with torch.no_grad():
        train_los_log.append(train_los.cpu())
        train_kld_log.append(train_kld.cpu())
        train_rec_log.append(train_rec.cpu())

        test_los_log.append(test_los.cpu())
        test_kld_log.append(test_kld.cpu())
        test_rec_log.append(test_rec.cpu())

        train_los_batch_log += train_los_batch
        train_kld_batch_log += train_kld_batch
        train_rec_batch_log += train_rec_batch
        test_los_batch_log += test_los_batch
        test_kld_batch_log += test_kld_batch
        test_rec_batch_log += test_rec_batch

    # saving model
    if epoch % save_every == 0:
        fn = '' + str(model_name) + '/model/' + str(epoch) + '.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to ' + fn)

