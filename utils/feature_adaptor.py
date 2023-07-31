import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def std_loss(output, target_std):
    '''
    to enhance the compactness of the embedding space
    '''
    # target_std = np.std(embeddings, axis=0) * factor 
    std_output = torch.std(output, dim=0)
    std_output = torch.mean(std_output)
    # target_std = torch.mean(target_std) * factor # done outside now
    res = torch.mean(torch.pow(std_output - target_std, 2))
    return res
    # return torch.mean(torch.pow(torch.from_numpy(np.std(output.detach().numpy(), axis=0) - target_std, 2)))

def l2_norm_of_weigth(model, target_size=384, use_cuda=True):
    # L2 regularization term
    l1_reg = torch.tensor(0.0)#.cuda()
    if use_cuda:
        l1_reg = l1_reg.cuda()
    for param in model.parameters():
        l1_reg += torch.norm(param, p=1)
    l2_reg = torch.abs(l1_reg - target_size)
    return l2_reg

# Define the custom loss function with L2 regularization
def custom_loss(output, model, target_std, taget_size_of_weights=384, use_cuda=True):
    std_loss_ = std_loss(output, target_std)
    # print(std_loss_)
    l2_norm_of_weigth_ = l2_norm_of_weigth(model, taget_size_of_weights, use_cuda)
    # print(l2_norm_of_weigth_)
    l1_distance_ = torch.mean(torch.abs(output))
    # print(l1_distance_)
    
    std_loss_factor = 1
    l2_norm_of_weigth_factor = 0.1 # because of order of magnitude
    l1_distance_factor = 1
    
    sum_of_factors = std_loss_factor + l2_norm_of_weigth_factor + l1_distance_factor
    std_loss_factor /= sum_of_factors
    l2_norm_of_weigth_factor /= sum_of_factors
    l1_distance_factor /= sum_of_factors
        
    return std_loss_factor * std_loss_ + l2_norm_of_weigth_factor * l2_norm_of_weigth_ + l1_distance_factor * l1_distance_, std_loss_, l2_norm_of_weigth_, l1_distance_

class CustomDataset(Dataset):
    def __init__(self, embeddings, target_length):
        self.embeddings = embeddings
        self.targets = np.zeros(shape=(embeddings.shape[0], target_length))
        # print(self.targets.shape)
        # print(self.embeddings.shape)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

class FeatureAdaptor(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureAdaptor, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.fc(x)
    
def train_one_epoch(model, optimizer, data_loader, target_std, target_size_of_weights=384*384*0.2, use_cuda=True):
    model.train()
    total_loss = 0.0
    std_loss = 0.0
    weight_loss = 0.0
    mean_loss = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
            # target_std = target_std.cuda()
        output = model(data)
        loss, std_loss, weight_loss, mean_loss  = custom_loss(output, model, target_std, target_size_of_weights, use_cuda)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        std_loss += std_loss.item()
        weight_loss += weight_loss.item()
        mean_loss += mean_loss.item()
    return total_loss / len(data_loader), std_loss / len(data_loader), weight_loss / len(data_loader), mean_loss / len(data_loader)


def get_feature_adaptor(embeddings, shrinking_factor=0.5, std_factor = 0.1, batch_size=32, num_workers=12, lr=0.001, epochs=12, use_cuda=True):
    # embeddings = np.random.rand(1000, 384)
    # target_length = 384
    print('\nFeauture Adaptor training...')
    use_cuda = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    target_length = int(embeddings.shape[1] * shrinking_factor)
    dataset = CustomDataset(embeddings, target_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = FeatureAdaptor(embeddings.shape[1], target_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target_std = np.mean(np.std(embeddings, axis=0)) * std_factor * shrinking_factor
    target_size_of_weights = embeddings.shape[1] * target_length # now it's 1! #0.5 # mean value of the weights should be 0.5
    for epoch in range(epochs):
        loss, std_loss, weight_loss, mean_loss = train_one_epoch(model, optimizer, data_loader, target_std, target_size_of_weights, use_cuda)
        print('Epoch: {} of {}, Loss: {}, std_loss: {}, weight_loss: {}, mean_loss: {}'.format(epoch+1, epochs, loss, std_loss, weight_loss, mean_loss))
    return model
