import torch
from torch import nn
import torch.nn.functional as F

class Memory_bank(nn.Module):

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2):
        super(Memory_bank, self).__init__()

        self.register_buffer("queue_anchor", torch.randn(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.knn = knn
        self.n_class = n_class

    def enqueue(self, anchors, labels):

        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        self.queue_anchor[ptr:ptr + bs, :] = anchors
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs
    
    # anchor set加载完后计算类中心分布
    def compute_class_center(self):
        # (n_class, 50274)
        self.class_center = torch.zeros(self.n_class, self.queue_anchor.shape[1])
        for i in range(self.n_class):
            self.class_center[i] = torch.mean(self.queue_anchor[self.queue_label == i], dim=0)
        
    def central_moment_discrepancy(self, X, Y, p=2):
        mean_X = torch.mean(X[:,None,:], dim=2)
        mean_Y = torch.mean(Y[:,None,:], dim=2)
        
        moment_X = torch.pow(X - mean_X, p)
        moment_Y = torch.pow(Y - mean_Y, p)
        
        cmd = torch.sum(torch.abs(moment_X - moment_Y),dim=1).reshape(1,-1)
        return cmd

    
    def compute_score(self, distance, t):
        result = torch.exp(-distance/t)
        result[result>1.] = -1.
        return result
    def get_instance_score(self, a, a_label, knn, num_class, t=0.1):
        """
        a: 
        a_label: 
        knn: 
        num_class: 
        """
        assert len(a_label)==len(a)
        assert knn*num_class <= len(a_label)
        n_anchor = len(a_label)
        instance_score_matrix = torch.zeros((n_anchor,num_class)).to(a.dtype)
        for i in range(num_class):
            instance_score_matrix[:, i] = torch.tensor([a[j] if a_label[j]==i else -1. for j in range(len(a_label))])
        instance_score_matrix = self.compute_score(instance_score_matrix, t)
        top_k_score, _ = torch.topk(instance_score_matrix, knn, dim=0) #
        top_k_score = torch.nn.Softmax(dim=1)(top_k_score)
        instance_score = torch.mean(top_k_score, dim=0)
        return instance_score
    def get_instance_distance(self, a, a_label, knn, num_class):
        """
        a: 
        a_label: 
        knn: 
        num_class: 
        return: instance_distance[num_class]\n
        """
        instance_distance_matrix = torch.zeros((len(a_label),num_class)).to(a.dtype)
        inf_distance = float('inf')
        for i in range(num_class):
            instance_distance_matrix[:, i] = torch.tensor([a[0][j] if a_label[j]==i else inf_distance for j in range(len(a_label))])
        top_k_distance, _ = torch.topk(instance_distance_matrix, knn, dim=0, largest=False) #
        instance_distance = torch.mean(top_k_distance, dim=0)
        return instance_distance
    def knn_infer(self, query, args):
        """
        query: [batch_size, dim]
        """
        # query.shape = [batch_size, dim] [1,50257]
        # kl_dis.shape = [1, len(self.queue_anchor)] [1,24]
        if args.cmd_p in range(1,5):
            kl_distance = self.central_moment_discrepancy(self.queue_anchor, query, args.cmd_p)
            # class_center_distance = self.central_moment_discrepancy(self.class_center, query, args.cmd_p)
        elif args.cmd_p > 4:
            # Combining different cmd's of different orders and the weighting needs to be normalized between them.
            kl_distance = torch.stack([self.central_moment_discrepancy(self.queue_anchor, query, p) for p in range(1, 5)], dim=2)
            kl_distance = F.normalize(kl_distance, p=1, dim=1)
            kl_distance = torch.sum(kl_distance, dim=2)
            # compute class center distance
            # class_center_distance = torch.stack([self.central_moment_discrepancy(self.class_center, query, p) for p in range(1, 5)], dim=2)
            # class_center_distance = F.normalize(class_center_distance, p=1, dim=1)
            # class_center_distance = torch.sum(class_center_distance, dim=2)
                       
        elif args.cmd_p == -1:
            kl_distance = torch.mean(self.queue_anchor[:, None, :] * (self.queue_anchor[:, None, :].log() - query.log()), dim=2).transpose(1, 0)
        else:
            kl_distance = 1 - torch.cosine_similarity(self.queue_anchor, query, dim=1).reshape(1,-1)

        class_center_distance = torch.mean(self.class_center[:, None, :] * (self.class_center[:, None, :].log() - query.log()), dim=2)
        
        # shape：class_center_distance[n_class,1], kl_distance[1, n_anchor], queue_label[n_anchor], queue_anchor[n_anchor, dim]
        # instance_score[num_class]

        # instance_score = self.get_instance_score(kl_distance.transpose(0,1), self.queue_label.reshape(-1,1), args.knn, self.n_class, args.t)
        # class_score = self.compute_score(class_center_distance, args.t).squeeze(-1)
        # class_score = torch.nn.Softmax(dim=0)(class_score)
        instance_score = self.get_instance_distance(kl_distance, self.queue_label, args.knn, self.n_class)
        if args.use_class_center == 1:
            score = class_center_distance.squeeze(-1) * args.alpha + instance_score * (1 - args.alpha)
            kl_class = torch.tensor([ class_center_distance[self.queue_label[idx]] for idx in range(len(self.queue_label))])
            kl_distance = kl_distance*(1-args.alpha) + kl_class*args.alpha
                
        else:
            score = instance_score
        
        # print("class_center_distance:", class_center_distance)
        # print("instance_score:", instance_score)
        # print("score:", score)
        # print("kl_distance:", kl_distance)
        score = F.normalize(score, p=1, dim=0)
                    
        if args.use_knn:
            if self.knn == 1:
                # directly return the nearest neighbor
                return self.queue_label[kl_distance.argmin(dim=1)].tolist(), score.tolist()
            else:
                values, indices = torch.topk(kl_distance, self.knn, dim=1, largest=False)
                # count for each category within k nearest neighbors, and return the dominant category
                # knn_cnt.shape = [1, self.n_class]
                knn_cnt = torch.zeros((query.shape[0], self.n_class))
                for i in range(self.n_class):
                    knn_cnt[:, i] = (self.queue_label[indices] == i).sum(dim=1)
                return knn_cnt.argmax(dim=1).tolist(), score.tolist()
        else:
            return torch.tensor([torch.argmin(score)]).tolist(), score.tolist()
