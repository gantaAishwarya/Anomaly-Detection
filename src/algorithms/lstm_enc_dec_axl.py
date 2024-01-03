import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from .algorithm_utils import Algorithm, PyTorchUtils
from torch.optim.lr_scheduler import StepLR

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 40, batch_size: int = 10, lr: float = 1e-3,
                 hidden_size: int =10, sequence_length: int = 84, step: int = 10,
                 train_gaussian_percentage: float = 0.30,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name     , seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.window = step
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed = None
        self.proto_input_space_ind = None
        self.hidden_and_prototype_as_df = pd.DataFrame(columns=['hidden_and_prototype_sequences', 'indicator'])
        self.mean, self.cov, self.epoch_loss = None, None, None

    def loss_e(self, prototype, enc_hidden):
        k_list = []
        for k in range(prototype.shape[0]):
            b_list = []
            for batch in range(enc_hidden.shape[0]):
                l = torch.sum(torch.mul(prototype[k] - enc_hidden[batch], prototype[k] - enc_hidden[batch]))
                b_list.append(l)
            b_list = torch.stack(b_list)
            min_b = torch.min(b_list)
            k_list.append(min_b)
        k_list = torch.stack(k_list)
        return torch.sum(k_list)

    def loss_d(self, prototype, d_min=2.0):
        sum = torch.tensor(0)
        for i in range(prototype.shape[0]):
            for j in range(i + 1, prototype.shape[0]):
                sum = sum + torch.square(torch.max(torch.tensor(0), d_min - torch.sqrt_(torch.sum(torch.mul(
                    prototype[i] - prototype[j], prototype[i] - prototype[j]), 0))))
        return sum

    def loss_c(self, prototype, enc_hidden):
        b_list = []
        for batch in range(enc_hidden.shape[0]):
            k_list = []
            for k in range(prototype.shape[0]):
                l = torch.sum(torch.mul(prototype[k] - enc_hidden[batch], prototype[k] - enc_hidden[batch]))
                k_list.append(l)
            k_list = torch.stack(k_list)
            min_e = torch.min(k_list)
            b_list.append(min_e)
        b_list = torch.stack(b_list)
        return torch.sum(b_list)


    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in
                     range(0, data.shape[0] - self.sequence_length + 1, self.window)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        print('Number of sequences:' + str(len(sequences)))
        print('Batch size:' + str(self.batch_size))
        print('Sequence Length:' + str(self.sequence_length))
        print('hidden size:' + str(self.hidden_size))
        print('Epochs:' + str(self.num_epochs))

        # train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
        #                           sampler=SubsetRandomSampler(indices), pin_memory=True)
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.lstmed.parameters(), lr=self.lr)
        #scheduler = StepLR(optimizer,step_size=1,gamma=0.85)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1, verbose=False)

        self.lstmed.train()

        epochloss = []

        #es = EarlyStopping(patience=20)
        for epoch in trange(self.num_epochs):
            #if epoch > 10:
             #scheduler.step()

            logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
            l = []
            loss_cum = 0
            for ts_batch in train_gaussian_loader:
                output, enc_hidden,a = self.lstmed(self.to_var(ts_batch),return_latent = True)
                # optimizer.zero_grad()
                #loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                loss_c = self.loss_c(self.lstmed.prototype_layer.prototype, enc_hidden)
                loss_d = self.loss_d(self.lstmed.prototype_layer.prototype, d_min=3)
                loss_e = self.loss_e(self.lstmed.prototype_layer.prototype, enc_hidden)
                #loss_w = torch.sum(torch.abs(self.lstmed.hidden2output.weight))
                # 0.2 * loss_w
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float())) + \
                       1.0 * loss_e + 1 * loss_d + 1.0 * loss_c
                print(loss_c, loss_d, loss_e, nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float())))
                loss_cum += loss.item()
                l.append(loss)
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

            #scheduler.step()
            epochloss.append(sum(l))
        #a_min = torch.zeros(len(sequences),self.lstmed.prototype_layer.k)
        self.proto_input_space_ind = torch.full((len(sequences),self.lstmed.prototype_layer.k),-1.)

        for i in range(len(sequences)):
            output, enc_hidden,a = self.lstmed(self.to_var(
                torch.Tensor(sequences[i]).expand(1,sequences[i].shape[0],sequences[i].shape[1])),
                return_latent =True)
            self.hidden_and_prototype_as_df.loc[len(self.hidden_and_prototype_as_df)] = [enc_hidden[0].
                                                                                            detach().numpy().tolist(),0]
            self.proto_input_space_ind[i] = a
            #for k in range(a.shape[1]):
            #    if a_min[k] > a[torch.argmin(a[:,k]),k]:
            #        a_min[k] = a[torch.argmin(a[:,k]),k]
            #        self.proto_input_space_ind[k,:] = torch.Tensor([i,i+self.sequence_length])

        for k in range(self.lstmed.prototype_layer.prototype.shape[0]):
            self.hidden_and_prototype_as_df.loc[len(self.hidden_and_prototype_as_df)] = [
                self.lstmed.prototype_layer.
                    prototype[k].detach().numpy().tolist(), 1]
        #print(self.proto_input_space_ind)
        #print(self.hidden_and_prototype_as_df)
            #Early stop
            #es(loss_cum/len(train_loader))

            #if es.early_stop:
            #    break
            #print(f'Epoch {epoch + 1} \t\t Training Loss: {l / len(train_loader)} ')

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_loader:
            # for ts_batch intrain_loader train_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)
        self.epoch_loss = epochloss


    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in
                     range(0, data.shape[0] - self.sequence_length + 1, self.window)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, (i * self.window): (i * self.window) + self.sequence_length] = score

        scores = np.nanmean(lattice, axis=0)
        scores = np.nan_to_num(scores)
          # added

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, (i * self.window): (i * self.window) + self.sequence_length,
                :] = output
            self.prediction_details.update(
                {'reconstructions_mean': np.nan_to_num(np.nanmean(lattice, axis=0).T)})  # added

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, (i * self.window): (i * self.window) + self.sequence_length,
                :] = error
            self.prediction_details.update({'errors_mean': np.nan_to_num(np.nanmean(lattice, axis=0).T)})  # added

        return scores

class prototype_layer(nn.Module, PyTorchUtils):
    def __init__(self, hidden_size: int, seed:int, gpu:int, k=2):
        super().__init__()
        PyTorchUtils.__init__(self,seed,gpu)
        self.hidden_size = hidden_size
        self.k = k
        self.prototype_size = torch.Tensor(k,hidden_size)
        self.init_values = nn.init.uniform(self.prototype_size,a=0.0,b=1.0)
        self.prototype = nn.Parameter(self.init_values)
        self.similarity2output = nn.Linear(self.k,2)
    def forward(self,x,batch_size):
        a = torch.zeros((batch_size,self.k))
        d = x[0].unsqueeze(1) - self.prototype.unsqueeze(1)
        for i in range(self.k):
            a[:,i] = torch.exp(-torch.sum(torch.mul(d[0,i],d[0,i]),1))
        return a

class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                              num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)

        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                              num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.prototype_layer = prototype_layer(self.hidden_size, seed, gpu, k=2)
        self.to_device(self.prototype_layer)

        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        #return self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_())
         return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                 self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        a = self.prototype_layer(enc_hidden, batch_size)
        return (output, enc_hidden[0][-1],a) if return_latent else output

