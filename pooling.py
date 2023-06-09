import torch


def adaptive_pooling(feature, pooling_strategy):
    '''
    depending on input size and strategy, different pooling methods are applied for each layer.
    '''
    spatial_dim = feature.shape[3]

    
    if pooling_strategy.__contains__('default'):
        pool = torch.nn.AvgPool2d(3, 1, 1)
    elif pooling_strategy.__contains__('avg110'):
        pool = torch.nn.AvgPool2d(1, 1, 0)
    elif pooling_strategy.__contains__('avg311'): # 28x28 --> 28x28 --> mehr samples 1000*28*28  * 0.01 = (7840,128)
        pool = torch.nn.AvgPool2d(3, 1, 1)
    elif pooling_strategy.__contains__('avg321'): # 28x28 --> 14x14 --> weniger samples 1000*14*14  * 0.01 = (1960,128)
        pool = torch.nn.AvgPool2d(3, 2, 1)
    elif pooling_strategy.__contains__('avg331'):
        pool = torch.nn.AvgPool2d(3, 3, 1)
    elif pooling_strategy.__contains__('avg512'):
        pool = torch.nn.AvgPool2d(5, 1, 2)
    elif pooling_strategy.__contains__('avg522'):
        pool = torch.nn.AvgPool2d(5, 2, 2)
    elif pooling_strategy.__contains__('avg532'):
        pool = torch.nn.AvgPool2d(5, 3, 2)
    elif pooling_strategy.__contains__('avg713'):
        pool = torch.nn.AvgPool2d(7, 1, 3)
    elif pooling_strategy.__contains__('avg723'):
        pool = torch.nn.AvgPool2d(7, 2, 3)
    elif pooling_strategy.__contains__('avg733'):
        pool = torch.nn.AvgPool2d(7, 3, 3)
    elif pooling_strategy.__contains__('avg914'):
        pool = torch.nn.AvgPool2d(9, 1, 4)
    elif pooling_strategy.__contains__('avg924'):
        pool = torch.nn.AvgPool2d(9, 2, 4)
    elif pooling_strategy.__contains__('avg934'):
        pool = torch.nn.AvgPool2d(9, 3, 4)
    
    elif pooling_strategy.__contains__('max110'):
        pool = torch.nn.MaxPool2d(1, 1, 0)
    elif pooling_strategy.__contains__('max311'):
        pool = torch.nn.MaxPool2d(3, 1, 1)
    elif pooling_strategy.__contains__('max321'):
        pool = torch.nn.MaxPool2d(3, 2, 1)
    elif pooling_strategy.__contains__('max331'):
        pool = torch.nn.MaxPool2d(3, 3, 1)
    elif pooling_strategy.__contains__('max512'):
        pool = torch.nn.MaxPool2d(5, 1, 2)
    elif pooling_strategy.__contains__('max522'):
        pool = torch.nn.MaxPool2d(5, 2, 2)
    elif pooling_strategy.__contains__('max532'):
        pool = torch.nn.MaxPool2d(5, 3, 2)
    elif pooling_strategy.__contains__('max713'):
        pool = torch.nn.MaxPool2d(7, 1, 3)
    elif pooling_strategy.__contains__('max723'):
        pool = torch.nn.MaxPool2d(7, 2, 3)
    elif pooling_strategy.__contains__('max733'):
        pool = torch.nn.MaxPool2d(7, 3, 3)
    elif pooling_strategy.__contains__('max914'):
        pool = torch.nn.MaxPool2d(9, 1, 4)
    elif pooling_strategy.__contains__('max924'):
        pool = torch.nn.MaxPool2d(9, 2, 4)
    elif pooling_strategy.__contains__('max934'):
        pool = torch.nn.MaxPool2d(9, 3, 4)
        
    
    elif pooling_strategy.__contains__('first_trial'):
        # everything to 7x7 with 224 input size
        if spatial_dim == 56: #layer 1 # TODO --> adapt depeding in input size of pic
            pool = torch.nn.AvgPool2d(kernel_size=8, stride=4, padding=4)
        elif spatial_dim == 28: #layer 2
            pool = torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=2)
        elif spatial_dim == 14: #layer 3
            pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        elif spatial_dim == 7: #layer 4
            pool = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)#
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