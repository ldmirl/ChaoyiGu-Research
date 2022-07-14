from classifier_loader_train import load_data_train
from classifier_loader_test import load_data_test
from classifier_loader_unlabel import load_data_unlabel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from PitchSeq_Classifier import classifier
import numpy as np

batch_size = 4
num_workers = 1
seq_len = 2
h_dim = 128
z_dim = 256
n_layers = 1
n_epochs = 100
# clip = 10
learning_rate = 1e-4
# seed = 128
# print_every = 10  # batches
save_every = 10  # epochs

train_dataset = load_data_train()
unlabel_dataset = load_data_unlabel()
test_dataset = load_data_test()

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

unlabel_loader = DataLoader(
    unlabel_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

# changing device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

model = classifier(h_dim, z_dim, n_layers)
model = model.to(device)
# Try RMSprop if possible
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def cross_entropy(y_prediction, y):
    prediction_loss = y * torch.log(1e-7 + y_prediction) + (1 - y) * torch.log(1e-7 + 1 - y_prediction)
    return -torch.mean(prediction_loss, dim=[1, 2])


def train(step, epoch):
    train_loss = 0
    train_loss_log = []
    # for batch_idx, (item1, item2) in enumerate(zip(train_loader, unlabel_loader)):

    for batch_idx, (data, label, idx) in enumerate(train_loader):
        # data, label, idx = item1
        # unlabel_data, idx = item2
        data = data.to(device)
        # unlabel_data = unlabel_data.to(device)
        out = model(data)
        loss = torch.mean(torch.nn.CrossEntropyLoss()(out, label.long()))
        # if step > 500000:
        #     pseudo_label = model(unlabel_data)
        #     pseudo_label = torch.nn.functional.softmax(pseudo_label,dim=1)
        #     pseudo_label = torch.argmax(pseudo_label, dim=1)
        #     out_unlabel =  model(unlabel_data)

        # loss = 0
        # for i in range(seq_len):
        #     variable = out[:,i,:]
        #     loss += F.cross_entropy(variable, label.long())

        # if step <= 500000:
        #     loss = torch.mean(torch.nn.CrossEntropyLoss()(out, label.long()))
        # else:
        #     alpha = (step-5000)/(30000-5000)
        #     loss = torch.mean(torch.nn.CrossEntropyLoss()(out, label.long())) + \
        #            alpha*torch.mean(torch.nn.CrossEntropyLoss()(out_unlabel, pseudo_label))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

        train_loss_log.append(loss)

        step += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset) * batch_size))
    return train_loss_log, step


def test(epoch):
    test_loss = 0
    test_loss_log = []
    with torch.no_grad():
        for batch_idx, (data, label, idx) in enumerate(test_loader):
            # transforming data
            data = data.to(device)

            out = model(data)
            # loss = 0
            # for i in range(seq_len):
            #     variable = out[:,i,:]
            #     loss += F.cross_entropy(variable, label.long())

            loss = torch.mean(torch.nn.CrossEntropyLoss()(out, label.long()))

            test_loss += loss

            test_loss_log.append(loss)
        print('====> Test set Average loss: {:.4f}'.format(test_loss/ len(test_loader.dataset) * batch_size))

    return test_loss_log


step = 0

train_log = []
test_log = []
for epoch in range(1, n_epochs + 1):
    train_loss, step = train(step, epoch)
    test_loss = test(epoch)
    with torch.no_grad():
        train_log.append(np.array([i.cpu() for i in train_loss]))
        test_log.append(np.array([i.cpu() for i in test_loss]))
    # saving model
    if epoch % save_every == 0:
        fn_weights = '/home/aarongu/anaconda3/envs/VRNN_Torch/classifier_0/model_weights/' + str(epoch) + '.pth'
        fn_model = '/home/aarongu/anaconda3/envs/VRNN_Torch/classifier_0/model/' + str(epoch) + '.pth'
        torch.save(model.state_dict(), fn_weights)
        torch.save(model, fn_model)
        print('Saved model to ' + fn_model)
    print(step)
np.save('/home/aarongu/anaconda3/envs/VRNN_Torch/classifier_0/loss/train_loss.npy', np.array(train_log))
np.save('/home/aarongu/anaconda3/envs/VRNN_Torch/classifier_0/loss/test_loss.npy', np.array(test_log))

# use cross-validation to train