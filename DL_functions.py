
import numpy as np
import torch
import torch.nn as nn


def data_split(z_data, r_data, m_data, target_data, ratio, split):
    '''
    split data
    :param z_data: characteristics
    :param r_data: stock return
    :param m_data: benchmark model
    :param target_data: target portfolio
    :param ratio: train/test ratio for split c
    :param split: if "future", split data into past/future periods using "ratio",
                  if integer "t", select test data every t periods
    :return: train and test data
    '''

    ff_n = m_data.shape[1]  # factor number
    port_n = target_data.shape[1]  # target (portfolio) number
    [t, n] = r_data.shape  # time length and stock number
    p = int(z_data.shape[1] / n)  # characteristics number
    #z_data = z_data.reshape((t, p, n)).transpose((0, 2, 1))   # dim: (t,n,p)
    z_data = z_data.reshape((t, p, n)).permute(0, 2, 1)  # dim: (t, n, p)


    # train sample and test sample
    if split == 'future':
        test_idx = torch.arange(int(t * ratio), t)
    else:
        test_idx = torch.arange(0, t, split)

    
    train_idx = torch.tensor([i for i in range(t) if i not in test_idx.tolist()])

# Extract training and test data using the indices
    z_train = z_data[train_idx]
    z_test = z_data[test_idx]
    r_train = r_data[train_idx]
    r_test = r_data[test_idx]
    target_train = target_data[train_idx]
    target_test = target_data[test_idx]
    m_train = m_data[train_idx]
    m_test = m_data[test_idx]

    # Getting the shape
    t_train = len(train_idx)

    return z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p



def get_batch(total, batch_number):
    '''
    create batches
    :param total: number of data points
    :param batch_number: number of batches
    :return: batches
    '''
    sample = np.arange(total)
    np.random.shuffle(sample)
    batch = np.array_split(sample, batch_number)
   
    batch = [batch[i][:total//batch_number] for i in range (0, batch_number)]
    return batch


class DeepFactorModel(nn.Module):
    def __init__(self, input_dim, layer_size, activation, ff_n, port_n):
        super(DeepFactorModel, self).__init__()
        layers = []
        self.activation = activation
        self.layer_number_tanh = len(layer_size)
        for i in range(len(layer_size)):
            layers.append(nn.Linear(input_dim if i == 0 else layer_size[i - 1], layer_size[i]))
            layers.append(activation())
        self.network = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.randn(layer_size[-1], port_n))
        self.gamma = nn.Parameter(torch.randn(ff_n, port_n))
    
    def forward(self, z, r, m, target, para, epoch):
        weights_l1 = 0.0
        x = z
        #print(x.shape)
        # Construct the network (prior to sorting)
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if i < self.layer_number_tanh - 2:  # No L1 regularization on the last layer
                    weights_l1 += torch.sum(torch.abs(layer.weight)) - torch.sum(torch.abs(torch.diag(layer.weight)))
            x = layer(x)
        #print(x.shape)
        # Softmax for factor weights
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        normalized_char = (x[-1] - mean) / (torch.sqrt(var) + 1e-5)
        #print(normalized_char.shape)
        transformed_char_a = -500 * torch.exp(-50 * normalized_char)
        transformed_char_b = -500 * torch.exp(50 * normalized_char)
        w_tilde = torch.softmax(-transformed_char_a + transformed_char_b, dim=1) 
        w_tilde = torch.softmax(-transformed_char_a, dim=1) - torch.softmax(transformed_char_b, dim=1)
        w_tilde = normalized_char
        # Construct factors
        r_tensor = r.view(r.shape[0], r.shape[1], 1)  # Reshape to [nobs, n, 1]
       # print("w_tilde shape:", w_tilde.shape)
        #print("r shape:", r.shape)
        w_tilde = w_tilde.transpose(1, 2)
        r_tensor = r.unsqueeze(-1)  # 从 [137, 1000] 转换为 [137, 1000, 1]

        f_tensor = torch.bmm(w_tilde, r_tensor)  # Matrix multiplication
        f = f_tensor.view(f_tensor.shape[0], -1)  # Reshape to [nobs, fsort_number]
        #if epoch < 0:
         #   print("normalized_char: ", normalized_char)
        # Forecast return and alpha
        target_hat = f @ self.beta + m @ self.gamma
        alpha = (target - target_hat).mean(dim=0)
        #print("f", f.shape)
        #if epoch < 0:
         #   print("target_hat", target_hat)
          #  print("alpha", alpha)
        # Define loss
        zero = torch.zeros_like(alpha)
        loss1 = nn.MSELoss()(target, target_hat)
        loss2 = nn.MSELoss()(alpha, zero)
        loss = loss1 + para['Lambda'] * loss2 + para['Lambda2'] * weights_l1
        #if epoch < 0:
         #   print("loss", loss)
        return loss, f,  normalized_char
    """

    def forward(self, z, r, m, target, para):
        weights_l1 = 0.0
        x = z

        # Construct the network (prior to sorting)
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if i < self.layer_number_tanh - 2:  # No L1 regularization on the last layer
                    weights_l1 += torch.sum(torch.abs(layer.weight)) - torch.sum(torch.abs(torch.diag(layer.weight)))
            x = layer(x)
        
        
        x = x.squeeze()

        # Softmax for factor weights
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        normalized_char = (x - mean) / (torch.sqrt(var) + 1e-5)

        transformed_char_a = -50 * torch.exp(-5 * normalized_char)
        transformed_char_b = -50 * torch.exp(5 * normalized_char)
        w_tilde = torch.softmax(transformed_char_a, dim=1) - torch.softmax(transformed_char_b, dim=1)

        
        r_tensor = r.unsqueeze(2)  
        
       
        if w_tilde.dim() == 2:
            w_tilde = w_tilde.unsqueeze(0).repeat(r_tensor.shape[0], 1, 1)

        
        f_tensor = torch.bmm(w_tilde, r_tensor)  # Matrix multiplication
        f = f_tensor.squeeze()  

        # Forecast return and alpha
        target_hat = f @ self.beta + m @ self.gamma
        alpha = (target - target_hat).mean(dim=0)

        # Define loss
        zero = torch.zeros_like(alpha)
        loss1 = nn.MSELoss()(target, target_hat)
        loss2 = nn.MSELoss()(alpha, zero)
        loss = loss1 + para['Lambda'] * loss2 + para['Lambda2'] * weights_l1

        return loss, target_hat, alpha
"""
def dl_alpha(data, layer_size, para):
    '''
    construct deep factors
    :param data: a dict of input data
    :param layer_size: a list of neural layer sizes (from bottom to top)
    :param para: training and tuning parameters
    :return: constructed deep factors and deep characteristics
    '''
    print(torch.__version__)

    # split data to training sample and test sample
    z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p = \
        data_split(data['characteristics'], data['stock_return'], data['factor'], data['target_return'],
                   para['train_ratio'], para['split'])

    n_train = r_train.shape[0]
    print(n_train)
    # sample size of training data

    # the last element of layer_size is the number of deep factors
    fsort_number = layer_size[-1]

    model = DeepFactorModel(input_dim=p, layer_size=layer_size, activation=nn.ReLU, ff_n=ff_n, port_n=port_n)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=para['learning_rate'])

    # Training loop
    loss_path = []
    early_stopping = 10
    stop_flag = 0
    thresh = 0.000005

    # Training
    last_f = None
    last_char = None
    last_loss = torch.inf
    for epoch in range(para['epoch']):
        model.train()
        batch_number = int(t_train / para['batch_size'])
        #print("t", t_train)
        #print("num", batch_number)
        batch = torch.tensor(get_batch(t_train, batch_number), dtype=torch.long)
        for idx in range(batch_number):
            # Assuming 'get_batch' function returns the indices for the current batch
            

            z_batch = z_train[batch[idx]]
            r_batch = r_train[batch[idx]]
            target_batch = target_train[batch[idx]]
            m_batch = m_train[batch[idx]]

            optimizer.zero_grad()
            
            # Forward pass
            loss, target_hat, alpha = model(z_batch, r_batch, m_batch, target_batch, para, epoch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate current loss
        with torch.no_grad():
            z_full = z_train
            r_full = r_train
            target_full = target_train
            m_full = m_train

            loss, f, char = model(z_full, r_full, m_full, target_full, para, epoch)
            if not torch.isnan(loss) and loss < last_loss:
                #print(epoch)
                #print(loss, f, char)
                last_loss = loss
                last_f = f
                last_char = char
            loss_path.append(loss.item())

            print(f"Epoch {epoch}, Loss: {loss.item()}")

            if loss.item() < thresh:
                stop_flag += 1
            else:
                stop_flag = 0

            if stop_flag >= early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break

    model.eval()
    """z_full, r_full, m_full, target_full, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p = \
        data_split(data['characteristics'], data['stock_return'], data['factor'], data['target_return'],
                   1, para['split'])
    n_full = r_full.shape[0]
    with torch.no_grad():
        _, factor, deep_char =  model(z_batch, r_batch, m_batch, target_batch, para, epoch)
        return factor, deep_char"""

    return last_f, last_char