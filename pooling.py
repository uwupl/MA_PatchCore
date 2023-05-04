import torch


def adaptive_pooling(feature, pooling_strategy):
    '''
    depending on input size and strategy, different pooling methods are applied for each layer.
    '''
    spatial_dim = feature.shape[3]
    
    if pooling_strategy.__contains__('default'):
        pool = torch.nn.AvgPool2d(3, 1, 1)
    
    elif pooling_strategy.__contains__('first_trial'):
        # everything to 7x7 with 224 input size
        if spatial_dim == 56:  # TODO --> adapt depeding in input size of pic
            pool = torch.nn.AvgPool2d(kernel_size=8, stride=4, padding=4)
        elif spatial_dim == 28:
            pool = torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=2)
        elif spatial_dim == 14:
            pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        elif spatial_dim == 7:
            pool = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=1)#
    elif pooling_strategy.__contains__('second_trial'):
        if spatial_dim == 56:  # TODO --> adapt depeding in input size of pic
            pool = torch.nn.AvgPool2d(kernel_size=8, stride=4, padding=2)
        elif spatial_dim == 28:
            pool = torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        elif spatial_dim == 14:
            pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        elif spatial_dim == 7:
            pool = torch.nn.Identity()#
        
    elif pooling_strategy.__contains__('max_1'):
        # everything to 7x7 with 224 input size
        if spatial_dim == 56:  # TODO --> adapt depeding in input size of pic
            pool = torch.nn.MaxPool2d(kernel_size=8, stride=4, padding=4)
        elif spatial_dim == 28:
            pool = torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=2)
        elif spatial_dim == 14:
            pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        elif spatial_dim == 7:
            pool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=1)#
    return pool(feature) 