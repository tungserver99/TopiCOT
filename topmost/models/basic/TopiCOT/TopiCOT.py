import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_kmeans
from .ECR import ECR
from .DCR import DCR
from .TCR import TCR


class TopiCOT(nn.Module):
    def __init__(self, vocab_size, doc_embedding, num_groups=20, num_data=None,
                 num_topics=50, en_units=200,
                 dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2,
                 weight_loss_ECR=250.0, alpha_ECR=20.0,
                 weight_loss_DCR=250.0, alpha_DCR=20.0,
                 weight_loss_TCR=250.0, alpha_TCR=20.0,
                 sinkhorn_max_iter=1000, weight_loss_InfoNCE=10.0, batch_size=200):
        super().__init__()

        self.num_topics = num_topics
        self.num_data = num_data
        self.batch_size = batch_size
        self.num_groups = num_groups
        self.beta_temp = beta_temp

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty(
            (num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(
            F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)

        doc_embedding = torch.Tensor(doc_embedding)
        
        # use kmean clustering to determine cluster embeddings
        kmean_model = torch_kmeans.KMeans(
            n_clusters=self.num_groups, max_iter=1000, seed=0, verbose=False,
            normalize='unit')
        cluster_result = kmean_model.fit(doc_embedding[None, :, :])
        doc_centroids = cluster_result._result.centers.squeeze(0)
        self.cluster_emb = doc_centroids
        labels_flat = cluster_result._result.labels.flatten()

        # Calculate the number of unique clusters
        num_of_clusters = labels_flat.max().item() + 1  # Assumes cluster indices are 0-based

        # Count the occurrences of each cluster
        cluster_counts = torch.bincount(labels_flat, minlength=num_of_clusters)

        # Calculate the ratio of elements in each cluster to the total number of elements
        total_elements = labels_flat.numel()
        ratios = cluster_counts / total_elements

        cluster_mass = nn.Parameter(ratios.unsqueeze(1), requires_grad=False)
        
        # set up DCR
        # self.group = torch.nn.functional.one_hot(
        #     cluster_result._result.labels.squeeze(0), self.num_groups).to(float)
        # self.group[self.group==0.0] = 0.01
        # self.group /= self.group.sum(axis=0, keepdim=True)

        # self.DCR = DCR(weight_loss_DCR, doc_centroids)
        # self.theta_prj = nn.Sequential(nn.Linear(self.num_topics, 384),
        #                                nn.Dropout(dropout))
        self.theta_prj = nn.Sequential(nn.Linear(self.num_topics, 384),
                                       nn.Dropout(dropout))
        self.doc_weights = nn.Parameter((torch.ones(self.batch_size) / self.batch_size).unsqueeze(1))
        self.DCR = TCR(doc_centroids, weight_loss_DCR, alpha_TCR, sinkhorn_max_iter, init_a_dist = self.doc_weights, init_b_dist=cluster_mass)

        # set up TCR
        self.topic_emb_prj = nn.Sequential(nn.Linear(embed_size, 384),
                                       nn.Dropout(dropout))
        self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))
        self.TCR = TCR(doc_centroids, weight_loss_TCR, alpha_TCR, sinkhorn_max_iter, init_a_dist = self.topic_weights, init_b_dist=cluster_mass)

        # for InfoNCE
        self.prj_rep = nn.Sequential(nn.Linear(self.num_topics, 384),
                                     nn.Dropout(dropout))
        self.prj_bert = nn.Sequential()
        self.weight_loss_InfoNCE = weight_loss_InfoNCE

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_KL(mu, logvar)

        return theta, loss_KL

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    # def get_loss_DCR(self, theta, bert):
    #     theta_prj = self.theta_prj(theta)
    #     loss_DCR = self.DCR(theta_prj, bert)
    #     return loss_DCR
    def get_loss_DCR(self, theta):
        theta_prj = self.theta_prj(theta)
        loss_DCR = self.DCR(theta_prj)
        return loss_DCR

    def get_loss_TCR(self, topic_emb):
        topic_prj = self.topic_emb_prj(topic_emb)
        loss_TCR = self.TCR(topic_prj)
        return loss_TCR

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    #Loss InfoNCE
    def sim(self, rep, bert):
        prep = self.prj_rep(rep)
        pbert = self.prj_bert(bert)
        return torch.exp(F.cosine_similarity(prep, pbert))

    def csim(self, bow, bert):
        pbow = self.prj_rep(bow)
        pbert = self.prj_bert(bert)
        csim_matrix = (pbow@pbert.T) / (pbow.norm(keepdim=True,
                                                  dim=-1)@pbert.norm(keepdim=True, dim=-1).T)
        csim_matrix = torch.exp(csim_matrix)
        csim_matrix = csim_matrix / csim_matrix.sum(dim=1, keepdim=True)
        return -csim_matrix.log()

    def compute_loss_InfoNCE(self, rep, contextual_emb):
        if self.weight_loss_InfoNCE <= 1e-6:
            return 0.
        else:
            sim_matrix = self.csim(rep, contextual_emb)
            return sim_matrix.diag().mean() * self.weight_loss_InfoNCE

    def forward(self, input, epoch_id=None, batch_idx=None):
        bert_emb = input['contextual_embed']
        input = input['data']
        theta, loss_KL = self.encode(input)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(input * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        loss_DCR = self.get_loss_DCR(theta, bert_emb)
        #loss_TCR = self.get_loss_TCR(self.topic_embeddings)
        if epoch_id > 10:
            loss_TCR =  self.get_loss_TCR(self.topic_embeddings)
        else:
            loss_TCR = 0
        loss_InfoNCE = self.compute_loss_InfoNCE(theta, bert_emb)
        loss = loss_TM + loss_ECR + loss_DCR + loss_TCR + loss_InfoNCE

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'loss_DCR': loss_DCR,
            'loss_TCR': loss_TCR,
            'loss_InfoNCE': loss_InfoNCE
        }

        return rst_dict
