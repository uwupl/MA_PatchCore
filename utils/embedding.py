import torch
import torch.nn.functional as F
import numpy as np
import numba as nb

def embedding_concat_frame(embeddings, cuda_active):
    '''
    framework for concatenating more than two features or less than two
    '''
    no_of_embeddings = len(embeddings)
    if no_of_embeddings == int(1):
        embeddings_result = embeddings[0].cpu()
    elif no_of_embeddings == int(2):
        embeddings_result = embedding_concat(embeddings[0], embeddings[1])
    elif no_of_embeddings > int(2):
        for k in range(no_of_embeddings - 1):
            if k == int(0):
                embeddings_result = embedding_concat(embeddings[0], embeddings[1]) # default
                pass
            else:
                if torch.cuda.is_available() and cuda_active:
                    embeddings_result = embedding_concat(embeddings_result.cuda(), embeddings[k+1])
                else:
                    embeddings_result = embedding_concat(embeddings_result, embeddings[k+1].cpu())
    return embeddings_result

def embedding_concat(x, y):
    '''
    alligns dimensions
    
    TODO: numba version plus lightweight version
    
    from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    '''
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def reshape_embedding_old(embedding):
    '''
    flattens spatial dimensions and concatenates channels. Results in 1D-Vector
    
    TODO: numba or numpy version! 
    '''
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return np.array(embedding_list)

@nb.njit
def reshape_embedding(embedding):
    '''
    flattens spatial dimensions and concatenates channels. Results in 1D-Vector
    '''
    # embeddings = np.empty((embedding.shape[0]*embedding.shape[2]*embedding.shape[3], embedding.shape[1]))
    # out = np.reshape(embedding, (embedding.shape[0]*embedding.shape[2]*embedding.shape[3], embedding.shape[1]))
    out = np.empty(shape=(embedding.shape[0]*embedding.shape[2]*embedding.shape[3], embedding.shape[1]), dtype=np.float32) # TODO: dtype?
    counter = int(0)
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                out[counter, :] = embedding[k, :, i, j]
                counter += 1
    return out