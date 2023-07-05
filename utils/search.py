import torch
from utils.utils import distance_matrix
from scipy.spatial.distance import cdist
class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=None, metric='euclidean'):
        self.k = k
        self.p = p
        # self.metrices = { 
        #             0:'euclidean', # 0.88
        #             1:'minkowski', # nur mit p spannend
        #             2:'cityblock', # manhattan
        #             3:'chebyshev',
        #             4:'cosine',
        #             5:'correlation',
        #             6:'hamming',
        #             7:'jaccard',
        #             8:'braycurtis',
        #             9:'canberra',
        #             10:'jensenshannon',
        #             # 11:'matching', # sysnonym for hamming
        #             11:'dice',
        #             12:'kulczynski1',
        #             13:'rogerstanimoto',
        #             14:'russellrao',
        #             15:'sokalmichener',
        #             16:'sokalsneath',
        #             # 18:'wminkowski',
        #             17:'mahalanobis',
        #             18:'seuclidean',
        #             19:'sqeuclidean',
        #             }
        # self.metrices_dict = {metric: i for i, metric in enumerate(metrices)}
        self.metric = metric
        print(f"\nUsing metric: {self.metric}\n")
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if self.p is None:
            dist = torch.from_numpy(cdist(x, self.train_pts, metric=self.metric))
        else:
            dist = torch.from_numpy(cdist(x, self.train_pts, metric=self.metric, p=self.p))
        knn = dist.topk(self.k, largest=False)
        return knn
