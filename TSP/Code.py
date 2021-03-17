import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions
from collections import namedtuple
from itertools import count

device = "cuda:1"
floattype = torch.float

batchsize = 512
nsamples = 8
npoints = 5
emsize = 512


class Graph_Transformer(nn.Module):
    def __init__(self, emsize = 64, nhead = 8, nhid = 1024, nlayers = 3, ndecoderlayers = 0, dropout = 0.3):
        super().__init__()
        self.emsize = emsize
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout = dropout)
        decoder_layers = TransformerDecoderLayer(emsize, nhead, nhid, dropout = dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, ndecoderlayers)
        self.encoder = nn.Linear(2, emsize)
        self.outputattention_query = nn.Linear(emsize, emsize, bias = False)
        self.outputattention_key = nn.Linear(emsize, emsize, bias = False)
        self.start_token = nn.Parameter(torch.randn([emsize], device = device))
    
    def generate_subsequent_mask(self, sz): #last dimension will be softmaxed over when adding to attention logits, if boolean the ones turn into -inf
        #mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        mask = torch.triu(torch.ones([sz, sz], dtype = torch.bool, device = device), diagonal = 1)
        return mask
    
    def encode(self, src): #src must be [batchsize * nsamples, npoints, 3]
        src = self.encoder(src).transpose(0, 1)
        output = self.transformer_encoder(src)
        return output #[npoints, batchsize * nsamples, emsize]
    
    def decode_next(self, memory, tgt, route_mask): #route mask is [batchsize * nsamples, npoints], both memory and tgt must have batchsize and nsamples in same dimension (the 1th one)
        npoints = memory.size(0)
        batchsize = tgt.size(1)
        """if I really wanted this to be efficient I'd only recompute the decoder for the last tgt, and just remebering what the others looked like from before (won't change due to mask)"""
        """have the option to freeze the autograd on all but the last part of tgt, although at the moment this is a very natural way to say: initial choices matter more"""
        tgt_mask = self.generate_subsequent_mask(tgt.size(0))
        output = self.transformer_decoder(tgt, memory, tgt_mask) #[tgt, batchsize * nsamples, emsize]
        output_query = self.outputattention_query(memory).transpose(0, 1) #[batchsize * nsamples, npoints, emsize]
        output_key = self.outputattention_key(output[-1]) #[batchsize * nsamples, emsize]
        output_attention = torch.matmul(output_query * self.emsize ** -0.5, output_key.unsqueeze(-1)).squeeze(-1) #[batchsize * nsamples, npoints], technically don't need to scale attention as we divide by variance next anyway
        output_attention_tanh = output_attention.tanh() #[batchsize * nsamples, npoints]
        
        #we clone the route_mask incase we want to backprop using it (else it was modified by inplace opporations)
        output_attention = output_attention.masked_fill(route_mask.clone(), float('-inf')) #[batchsize * nsamples, npoints]
        output_attention_tanh = output_attention_tanh.masked_fill(route_mask.clone(), float('-inf')) #[batchsize * nsamples, npoints]
        
        return output_attention_tanh, output_attention #[batchsize * nsamples, npoints]
    
    def calculate_logprob(self, memory, routes): #memory is [npoints, batchsize * nsamples, emsize], routes is [batchsize * nsamples, npoints - 4], rather than backproping the entire loop, this saves vram (and computation)
        npoints = memory.size(0)
        ninternalpoints = routes.size(1)
        bigbatchsize = memory.size(1)
        memory_ = memory.gather(0, routes.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.emsize)) #[npoints - 4, batchsize * nsamples, emsize] reorder memory into order of routes
        tgt = torch.cat([self.start_token.unsqueeze(0).unsqueeze(1).expand(1, bigbatchsize, -1), memory_[:-1]]) #[npoints - 4, batchsize * nroutes, emsize], want to go from memory to tgt
        tgt_mask = self.generate_subsequent_mask(ninternalpoints)
        output = self.transformer_decoder(tgt, memory, tgt_mask) #[npoints - 4, batchsize * nsamples, emsize]
        """want probability of going from key to query, but first need to normalise (softmax with mask)"""
        output_query = self.outputattention_query(memory_).transpose(0, 1) #[batchsize * nsamples, npoints - 4, emsize]
        output_key = self.outputattention_key(output).transpose(0, 1) #[batchsize * nsamples, npoints - 4, emsize]
        attention_mask = torch.full([ninternalpoints, ninternalpoints], True, device = device).triu(1) #[npoints - 4, npoints - 4], True for i < j
        output_attention = torch.matmul(output_query * self.emsize ** -0.5, output_key.transpose(-1, -2))
        """quick fix to stop divergence"""
        output_attention_tanh = output_attention.tanh()
        
        output_attention_tanh = output_attention_tanh.masked_fill(attention_mask, float('-inf'))
        output_attention_tanh = output_attention_tanh - output_attention_tanh.logsumexp(-2, keepdim = True) #[batchsize * nsamples, npoints - 4, npoints - 4]
        
        output_attention = output_attention.masked_fill(attention_mask, float('-inf'))
        output_attention = output_attention - output_attention.logsumexp(-2, keepdim = True) #[batchsize * nsamples, npoints - 4, npoints - 4]
        
        """infact I'm almost tempted to not mask choosing a previous point, so it's forced to learn it and somehow incorporate it into its computation, but without much impact on reinforcing good examples"""
        logprob_tanh = output_attention_tanh.diagonal(dim1 = -1, dim2 = -2).sum(-1) #[batchsize * nsamples]
        logprob = output_attention.diagonal(dim1 = -1, dim2 = -2).sum(-1) #[batchsize * nsamples]
        return logprob_tanh, logprob #[batchsize * nsamples]

NN = Graph_Transformer().to(device)
optimizer = optim.Adam(NN.parameters())


class environment:    
    def reset(self, npoints, batchsize, nsamples=1):
        self.batchsize = (
            batchsize * nsamples
        )  # so that I don't have to rewrite all this code, we store these two dimensions together
        self.nsamples = nsamples
        self.npoints = npoints
        self.points = (
            torch.rand([batchsize, npoints, 2], dtype = floattype, device=device)
            .unsqueeze(1)
            .expand(-1, nsamples, -1, -1)
            .reshape(self.batchsize, npoints, 2)
        )
        
        self.distance_matrix = (self.points.unsqueeze(1) - self.points.unsqueeze(2)).square().sum(-1).sqrt() # [batchsize * nsamples, npoints, npoints]
        
        self.previous_point = None
        
        self.points_mask = torch.zeros(
                    [self.batchsize, npoints], dtype=torch.bool, device=device
                )
        self.points_sequence = torch.empty(
            [self.batchsize, 0], dtype=torch.long, device=device
        )
        
        self.cost = torch.zeros([self.batchsize], dtype = floattype, device=device)

        self.logprob = torch.zeros([self.batchsize], dtype = floattype, device=device, requires_grad=True)

    def update(self, point_index):  # point_index is [batchsize]
        
        assert list(point_index.size()) == [self.batchsize]
        assert str(point_index.device) == device
        assert self.points_mask.gather(1, point_index.unsqueeze(1)).sum() == 0
        
        if self.previous_point != None:
            self.cost += self.distance_matrix.gather(2, self.previous_point.unsqueeze(1).unsqueeze(2).expand(-1, self.npoints, 1)).squeeze(2).gather(1, point_index.unsqueeze(1)).squeeze(1)
        
        self.previous_point = point_index
        self.points_mask.scatter_(1, point_index.unsqueeze(1), True)
        self.points_sequence = torch.cat([self.points_sequence, point_index.unsqueeze(1)], dim = 1)
        
        return
    
    def sample_point(self, logits): #logits must be [batchsize * nsamples, npoints]
        probs = torch.distributions.categorical.Categorical(logits = logits)
        next_point = probs.sample() #size is [batchsize * nsamples]
        self.update(next_point)
        self.logprob = self.logprob + probs.log_prob(next_point)
        return next_point #[batchsize * nsamples]
    
    def sampleandgreedy_point(self, logits): #logits must be [batchsize * nsamples, npoints], last sample will be the greedy choice (but we still need to keep track of its logits)
        logits_sample = logits.view(-1, self.nsamples, self.npoints)[:, :-1, :]
        probs = torch.distributions.categorical.Categorical(logits = logits_sample)
        
        sample_point = probs.sample() #[batchsize, (nsamples - 1)]
        greedy_point = logits.view(-1, self.nsamples, self.npoints)[:, -1, :].max(-1, keepdim = True)[1] #[batchsize, 1]
        next_point = torch.cat([sample_point, greedy_point], dim = 1).view(-1)
        self.update(next_point)
        self.logprob = self.logprob + torch.cat([probs.log_prob(sample_point), torch.zeros([sample_point.size(0), 1], device = device)], dim = 1).view(-1)
        return next_point
    
    def laststep(self):
        
        assert self.points_sequence.size(1) == self.npoints
        
        self.cost += self.distance_matrix.gather(2, self.points_sequence[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, self.npoints, 1)).squeeze(2).gather(1, self.points_sequence[:, -1].unsqueeze(1)).squeeze(1)
    

env = environment()


def evaluate(epochs = 10, npoints = 10, batchsize = 100, nsamples = 2):
    NN.eval()
    for i in range(epochs):
        env.reset(npoints, batchsize, nsamples)
        """include the boundary points, kinda makes sense that they should contribute (atm only in the encoder, difficult to see how in the decoder)"""
        memory = NN.encode(env.points) #[npoints, batchsize * nsamples, emsize]
        #### #### #### remember to include tgt.detach() when reinstate with torch.no_grad()
        tgt = NN.start_token.unsqueeze(0).unsqueeze(1).expand(1, batchsize * nsamples, -1).detach() #[1, batchsize * nsamples, emsize]
        #with torch.no_grad(): #to speed up computation, selecting routes is done without gradient
        with torch.no_grad():
            for j in range(0, npoints):
                #### #### #### remember to include memory.detach() when reinstate with torch.no_grad()
                _, logits = NN.decode_next(memory.detach(), tgt, env.points_mask)
                next_point = env.sampleandgreedy_point(logits)
                """
                for inputing the previous embedding into decoder
                """
                tgt = torch.cat([tgt, memory.gather(0, next_point.unsqueeze(0).unsqueeze(2).expand(1, -1, memory.size(2)))]) #[nsofar, batchsize * nsamples, emsize]
                """
                for inputing the previous decoder output into the decoder (allows for an evolving strategy, but doesn't allow for fast training
                """
                ############

        env.laststep()
        
        _, logprob = NN.calculate_logprob(memory, env.points_sequence) #[batchsize * nsamples]
        NN.train()
        """
        clip logprob so doesn't reinforce things it already knows
        TBH WANT SOMETHING DIFFERENT ... want to massively increase training if find something unexpected and otherwise not
        """
        greedy_baseline = env.cost.view(batchsize, nsamples)[:, -1] #[batchsize], greedy sample
        
        print(greedy_baseline.mean().item(), logprob.view(batchsize, nsamples)[:, -1].mean().item())


def train(epochs = 30000, npoints = 10, batchsize = 100, nsamples = 8):
    NN.train()
    for i in range(epochs):
        env.reset(npoints, batchsize, nsamples)
        """include the boundary points, kinda makes sense that they should contribute (atm only in the encoder, difficult to see how in the decoder)"""
        memory = NN.encode(env.points) #[npoints, batchsize * nsamples, emsize]
        #### #### #### remember to include tgt.detach() when reinstate with torch.no_grad()
        tgt = NN.start_token.unsqueeze(0).unsqueeze(1).expand(1, batchsize * nsamples, -1).detach() #[1, batchsize * nsamples, emsize]
        #with torch.no_grad(): #to speed up computation, selecting routes is done without gradient
        with torch.no_grad():
            for j in range(0, npoints):
                #### #### #### remember to include memory.detach() when reinstate with torch.no_grad()
                _, logits = NN.decode_next(memory.detach(), tgt, env.points_mask)
                next_point = env.sampleandgreedy_point(logits)
                """
                for inputing the previous embedding into decoder
                """
                tgt = torch.cat([tgt, memory.gather(0, next_point.unsqueeze(0).unsqueeze(2).expand(1, -1, memory.size(2)))]) #[nsofar, batchsize * nsamples, emsize]
                """
                for inputing the previous decoder output into the decoder (allows for an evolving strategy, but doesn't allow for fast training
                """
                ############

        env.laststep()
        
        NN.eval()
        _, logprob = NN.calculate_logprob(memory, env.points_sequence) #[batchsize * nsamples]
        NN.train()
        """
        clip logprob so doesn't reinforce things it already knows
        TBH WANT SOMETHING DIFFERENT ... want to massively increase training if find something unexpected and otherwise not
        """
        greedy_prob = logprob.view(batchsize, nsamples)[:, -1].detach() #[batchsize]
        greedy_baseline = env.cost.view(batchsize, nsamples)[:, -1] #[batchsize], greedy sample
        fixed_baseline = 0.5 * torch.ones([1], device = device)
        min_baseline = env.cost.view(batchsize, nsamples)[:, :-1].min(-1)[0] #[batchsize], minimum cost
        baseline = greedy_baseline
        positive_reinforcement = - F.relu( - (env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1))) #don't scale positive reinforcement
        negative_reinforcement = F.relu(env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1))
        positive_reinforcement_binary = env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1) <= -0.05
        negative_reinforcement_binary = env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1) > 1
        """
        binary positive reinforcement
        """
        #loss = - ((logprob.view(batchsize, nsamples)[:, :-1] < -0.2) * logprob.view(batchsize, nsamples)[:, :-1] * positive_reinforcement_binary).mean() #+ (logprob.view(batchsize, nsamples)[:, :-1] > -1) * logprob.view(batchsize, nsamples)[:, :-1] * negative_reinforcement_binary
        """
        clipped binary reinforcement
        """
        loss = ( 
                - logprob.view(batchsize, nsamples)[:, :-1] 
                * (logprob.view(batchsize, nsamples)[:, :-1] < 0) 
                * positive_reinforcement_binary 
                + logprob.view(batchsize, nsamples)[:, :-1] 
                * (logprob.view(batchsize, nsamples)[:, :-1] > greedy_prob.unsqueeze(1) - 80) 
                * negative_reinforcement_binary 
        ).mean()
        """
        clipped binary postive, clipped weighted negative
        """
        #loss = ( - logprob.view(batchsize, nsamples)[:, :-1] * (logprob.view(batchsize, nsamples)[:, :-1] < -0.2) * positive_reinforcement_binary + logprob.view(batchsize, nsamples)[:, :-1] * (logprob.view(batchsize, nsamples)[:, :-1] > -2) * negative_reinforcement ).mean()
        """
        clipped reinforcement without rescaling
        """
        #loss = ((logprob.view(batchsize, nsamples)[:, :-1] < -0.7) * logprob.view(batchsize, nsamples)[:, :-1] * positive_reinforcement + (logprob.view(batchsize, nsamples)[:, :-1] > -5) * logprob.view(batchsize, nsamples)[:, :-1] * negative_reinforcement).mean()
        """
        clipped reinforcement
        """
        #loss = (logprob.view(batchsize, nsamples)[:, :-1] * positive_reinforcement / (positive_reinforcement.var() + 0.001).sqrt() + (logprob.view(batchsize, nsamples)[:, :-1] > -3) * logprob.view(batchsize, nsamples)[:, :-1] * negative_reinforcement / (negative_reinforcement.var() + 0.001).sqrt()).mean()
        """
        balanced reinforcement
        """
        #loss = (logprob.view(batchsize, nsamples)[:, :-1] * (positive_reinforcement / (positive_reinforcement.var() + 0.001).sqrt() + negative_reinforcement / (negative_reinforcement.var() + 0.001).sqrt())).mean()
        """
        regular loss
        """
        #loss = (logprob.view(batchsize, nsamples)[:, :-1] * (positive_reinforcement + negative_reinforcement)).mean()
        optimizer.zero_grad()
        loss.backward()
        #print(NN.encoder.weight.grad)
        optimizer.step()
        #print(greedy_baseline.mean().item())
        print(greedy_baseline.mean().item(), logprob.view(batchsize, nsamples)[:, -1].mean().item(), logprob.view(batchsize, nsamples)[:, :-1].mean().item(), logprob[batchsize - 1].item(), logprob[0].item(), env.logprob[0].item())
        
   



