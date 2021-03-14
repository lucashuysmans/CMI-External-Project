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

device = "cuda:0"
floattype = torch.float

batchsize = 512
nsamples = 8
npoints = 5
emsize = 512


class Graph_Transformer(nn.Module):
    def __init__(self, emsize = 64, nhead = 8, nhid = 1024, nlayers = 4, ndecoderlayers = 2, dropout = 0.1):
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
        
        self.lineartest0 = nn.Linear(2, emsize)
        self.lineartest1 = nn.Linear(2, emsize)
        self.lineartest2 = nn.Linear(2, emsize)
        self.lineartest3 = nn.Linear(2 * emsize, emsize)
    
    def generate_subsequent_mask(self, sz): #last dimension will be softmaxed over when adding to attention logits, if boolean the ones turn into -inf
        #mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        mask = torch.triu(torch.ones([sz, sz], dtype = torch.bool, device = device), diagonal = 1)
        return mask
    
    def encode(self, src): #src must be [batchsize * nsamples, npoints, 2]
        src = self.encoder(src).transpose(0, 1)
        output = self.transformer_encoder(src)
        return output #[npoints, batchsize * nsamples, emsize]
    
    def test_encode(self, src): #src must be [batchsize * nsamples, npoints = 5, 2]
        point_1 = self.lineartest1(src[:, 3, :] - 0.5) #want mean 0!!!!
        point_2 = self.lineartest1(src[:, 4, :] - 0.5)
        remaining = self.lineartest0(src[:, :3, :]) #[batchsize * nsamples, 3, emsize]
        point_1_message = self.lineartest1(src[:, 3, :] - 0.5).squeeze(1)
        point_2_message = self.lineartest2(src[:, 4, :] - 0.5).squeeze(1)
        point_1 = F.relu(torch.cat([point_1, point_2_message], dim = 1))
        point_2 = F.relu(torch.cat([point_2, point_1_message], dim = 1))
        point_1 = self.lineartest3(point_1)
        point_2 = self.lineartest3(point_2)
        src = torch.cat([torch.zeros_like(remaining.transpose(0, 1)), point_1.unsqueeze(0), point_2.unsqueeze(0)])
        return src #[npoints = 5, batchsize * nsamples, emsize]
    
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
    
    def calculate_logprob(self, memory, routes): #memory is [npoints, batchsize * nsamples, emsize], routes is [batchsize * nsamples, npoints - 3], rather than backproping the entire loop, this saves vram (and computation)
        npoints = memory.size(0)
        ninternalpoints = routes.size(1)
        bigbatchsize = memory.size(1)
        memory_ = memory.gather(0, routes.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.emsize)) #[npoints - 3, batchsize * nsamples, emsize] reorder memory into order of routes
        tgt = torch.cat([self.start_token.unsqueeze(0).unsqueeze(1).expand(1, bigbatchsize, -1), memory_[:-1]]) #[npoints - 3, batchsize * nroutes, emsize], want to go from memory to tgt
        tgt_mask = self.generate_subsequent_mask(ninternalpoints)
        output = self.transformer_decoder(tgt, memory, tgt_mask) #[npoints - 3, batchsize * nsamples, emsize]
        """want probability of going from key to query, but first need to normalise (softmax with mask)"""
        output_query = self.outputattention_query(memory_).transpose(0, 1) #[batchsize * nsamples, npoints - 3, emsize]
        output_key = self.outputattention_key(output).transpose(0, 1) #[batchsize * nsamples, npoints - 3, emsize]
        attention_mask = torch.full([ninternalpoints, ninternalpoints], True, device = device).triu(1) #[npoints - 3, npoints - 3], True for i < j
        output_attention = torch.matmul(output_query * self.emsize ** -0.5, output_key.transpose(-1, -2))
        """quick fix to stop divergence"""
        output_attention_tanh = output_attention.tanh()
        
        output_attention_tanh = output_attention_tanh.masked_fill(attention_mask, float('-inf'))
        output_attention_tanh = output_attention_tanh - output_attention_tanh.logsumexp(-2, keepdim = True) #[batchsize * nsamples, npoints - 3, npoints - 3]
        
        output_attention = output_attention.masked_fill(attention_mask, float('-inf'))
        output_attention = output_attention - output_attention.logsumexp(-2, keepdim = True) #[batchsize * nsamples, npoints - 3, npoints - 3]
        
        """infact I'm almost tempted to not mask choosing a previous point, so it's forced to learn it and somehow incorporate it into its computation, but without much impact on reinforcing good examples"""
        logprob_tanh = output_attention_tanh.diagonal(dim1 = -1, dim2 = -2).sum(-1) #[batchsize * nsamples]
        logprob = output_attention.diagonal(dim1 = -1, dim2 = -2).sum(-1) #[batchsize * nsamples]
        return logprob_tanh, logprob #[batchsize * nsamples]

NN = Graph_Transformer().to(device)
optimizer = optim.Adam(NN.parameters())


class environment:    
    def reset(self, npoints, batchsize, nsamples=1, corner_points = None, initial_triangulation = None):
        """
        corner_points, etc., shoudn't include a batch dimension
        """
        if corner_points == None:
            ncornerpoints = 4
        else:
            ncornerpoints = corner_points.size(0)
        if npoints <= ncornerpoints:
            print("Error: not enough points for valid problem instance")
            return
        self.batchsize = (
            batchsize * nsamples
        )  # so that I don't have to rewrite all this code, we store these two dimensions together
        self.nsamples = nsamples
        self.npoints = npoints
        self.points = (
            torch.rand([batchsize, npoints - ncornerpoints, 3], dtype = floattype, device=device)
            .unsqueeze(1)
            .expand(-1, nsamples, -1, -1)
            .reshape(self.batchsize, npoints - ncornerpoints, 3)
        )
        if corner_points == None:
            self.corner_points = torch.tensor(
                [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype = floattype, device=device
            )
        else:
            self.corner_points = corner_points
        self.points = torch.cat(
            [
                self.corner_points.unsqueeze(0).expand(self.batchsize, -1, -1),
                self.points,
            ],
            dim=-2,
        )  # [batchsize * nsamples, npoints, 3]
        self.points_mask = torch.cat(
            [
                torch.ones([self.batchsize, ncornerpoints], dtype=torch.bool, device=device),
                torch.zeros(
                    [self.batchsize, npoints - ncornerpoints], dtype=torch.bool, device=device
                ),
            ],
            dim=1,
        )
        self.points_sequence = torch.empty(
            [self.batchsize, 0], dtype=torch.long, device=device
        )

        """
        points are now triples
        triangles are now quadruples
        edges are now still just indices, but there are four of them per 'triangle', and they correspond to triples of points, not pairs
        we use  0,2,1  0,3,2  0,1,3  1,2,3  as the order of the four 'edges'/faces
        opposite face is always ordered such that the last two indices are swapped
        faces are always read ANTICLOCKWISE
        
        first three points of tetrahedron MUST be read clockwise (from the outside) to get correct sign on incircle test
        
        new point will be inserted in zeroth position, so if corresponding face of REMOVED tetrahedron is [x,y,z] (being read anticlockwise from outside in) new tetrahedron is [p, x, y, z]
        """
        
        """
        number of tetrahedra is not the same for each batch (in 3D), so store as a big list, and remember batch index that it comes from
        """
        if corner_points == None:
            initial_triangulation = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
        
        self.partial_delaunay_triangles = initial_triangulation.unsqueeze(0).expand(self.batchsize, -1, -1).reshape(-1, 4)
        self.batch_index = torch.arange(self.batchsize, dtype = torch.long, device = device).unsqueeze(1).expand(-1, initial_triangulation.size(0)).reshape(-1)
        
        self.batch_triangles = self.partial_delaunay_triangles.size(0) #[0]
        self.ntriangles = torch.full([self.batchsize], initial_triangulation.size(0), dtype = torch.long, device = device) #[self.batchsize]
        
        self.cost = torch.zeros([self.batchsize], dtype = floattype, device=device)

        self.logprob = torch.zeros([self.batchsize], dtype = floattype, device=device, requires_grad=True)

    def update(self, point_index):  # point_index is [batchsize]
        
        assert point_index.size(0) == self.batchsize
        assert str(point_index.device) == device
        assert self.points_mask.gather(1, point_index.unsqueeze(1)).sum() == 0
        
        triangles_coordinates = self.points[self.batch_index.unsqueeze(1), self.partial_delaunay_triangles] # [batch_triangles, 4, 3]
        
        newpoint = self.points[self.batch_index, point_index[self.batch_index]] # [batch_triangles, 3]

        incircle_matrix = torch.cat(
            [
                newpoint.unsqueeze(1),
                triangles_coordinates,
            ],
            dim=-2,
        )  # [batch_triangles, 5, 3]
        incircle_matrix = torch.cat(
            [
                (incircle_matrix * incircle_matrix).sum(-1, keepdim=True),
                incircle_matrix,
                torch.ones([self.batch_triangles, 5, 1], dtype = floattype, device=device),
            ],
            dim=-1,
        )  # [batch_triangles, 5, 5]
        assert incircle_matrix.dtype == floattype
        assert str(incircle_matrix.device) == device
        
        incircle_test = (
            incircle_matrix.det() > 0
        )  # [batch_triangles], is True if inside incircle
        
        conflicts = incircle_test.sum()
        
        conflicting_triangles = self.partial_delaunay_triangles[incircle_test] # [conflicts, 4]
        
        conflicting_edges_index0 = torch.empty_like(conflicting_triangles)
        indices = torch.LongTensor([0, 0, 0, 1])
        conflicting_edges_index0 = conflicting_triangles[:, indices] # [conflicts, 4]
        
        conflicting_edges_index1 = torch.empty_like(conflicting_triangles)
        indices = torch.LongTensor([2, 3, 1, 2])
        conflicting_edges_index1 = conflicting_triangles[:, indices] # [conflicts, 4]
        
        conflicting_edges_index2 = torch.empty_like(conflicting_triangles)
        indices = torch.LongTensor([1, 2, 3, 3])
        conflicting_edges_index2 = conflicting_triangles[:, indices] # [conflicts, 4]
        
        conflicting_edges = torch.cat([conflicting_edges_index0.view(-1).unsqueeze(-1), conflicting_edges_index1.view(-1).unsqueeze(-1), conflicting_edges_index2.view(-1).unsqueeze(-1)], dim = -1).reshape(-1, 3) # [conflicts * 4, 3]
        
        edge_batch_index = self.batch_index[incircle_test].unsqueeze(1).expand(-1, 4).reshape(-1) # [conflicts * 4]
        
        indices = torch.LongTensor([0, 2, 1])
        comparison_edges = conflicting_edges[:, indices] # [conflicts * 4, 3]        
        
        unravel_nomatch_mask = torch.ones([conflicts * 4], dtype = torch.bool, device = device) # [conflicts * 4]
        i = 1
        while True:
            
            todo_mask = unravel_nomatch_mask[:-i].logical_and(edge_batch_index[:-i] == edge_batch_index[i:])
            if i % 4 == 0:
                if todo_mask.sum() == 0:
                    break
            
            match_mask = todo_mask.clone()
            match_mask[todo_mask] = (conflicting_edges[:-i][todo_mask] != comparison_edges[i:][todo_mask]).sum(-1).logical_not()
            
            unravel_nomatch_mask[:-i][match_mask] = False
            unravel_nomatch_mask[i:][match_mask] = False
            
            i += 1
        
        batch_newtriangles = unravel_nomatch_mask.sum()
        
        nomatch_edges = conflicting_edges[unravel_nomatch_mask] # [batch_newtriangles, 3], already in correct order to insert into 1,2,3 (since already anticlockwise from outside in)
        assert list(nomatch_edges.size()) == [batch_newtriangles, 3]
        nomatch_batch_index = edge_batch_index[unravel_nomatch_mask] # [batch_newtriangles]
        
        nomatch_newpoint = point_index[nomatch_batch_index] # [batch_newtriangles]
        
        newtriangles = torch.cat([nomatch_newpoint.unsqueeze(1), nomatch_edges], dim = -1) # [batch_newtriangles, 4]
        
        
        nremoved_triangles = torch.zeros([self.batchsize], dtype = torch.long, device = device)
        nnew_triangles = torch.zeros([self.batchsize], dtype = torch.long, device = device)
        
        indices = self.batch_index[incircle_test]
        nremoved_triangles.put_(indices, torch.ones_like(indices, dtype = torch.long), accumulate = True) # [batchsize]
        
        indices = edge_batch_index[unravel_nomatch_mask]
        nnew_triangles.put_(indices, torch.ones_like(indices, dtype = torch.long), accumulate = True) # [batchsize]
        
        assert (nnew_triangles <= 2 * nremoved_triangles + 2).logical_not().sum().logical_not()
        
        """
        NOTE:
        I THINK it's possible for nnew_triangles to be less than nremoved_triangles (or my code is just buggy...)
        """
        
        assert nnew_triangles.sum() == batch_newtriangles
        assert nremoved_triangles.sum() == incircle_test.sum()
        
        nadditional_triangles = nnew_triangles - nremoved_triangles # [batchsize]
        ntriangles = self.ntriangles + nadditional_triangles # [batchsize]
        
        partial_delaunay_triangles = torch.empty([ntriangles.sum(), 4], dtype = torch.long, device = device)
        batch_index = torch.empty([ntriangles.sum()], dtype = torch.long, device = device)
        
        cumulative_triangles = torch.cat([torch.zeros([1], dtype = torch.long, device = device), nnew_triangles.cumsum(0)[:-1]]) # [batchsize], cumulative sum starts at zero
        
        """
        since may actually have LESS triangles than previous round, we insert all that survive into the first slots (in that batch)
        """
        good_triangle_indices = torch.arange(incircle_test.logical_not().sum(), dtype = torch.long, device = device)
        good_triangle_indices += cumulative_triangles[self.batch_index[incircle_test.logical_not()]]
        bad_triangle_indices_mask = torch.ones([ntriangles.sum(0)], dtype = torch.bool, device = device)
        bad_triangle_indices_mask.scatter_(0, good_triangle_indices, False)
        
        assert good_triangle_indices.size(0) == incircle_test.logical_not().sum()
        assert bad_triangle_indices_mask.sum() == batch_newtriangles
        
        partial_delaunay_triangles[good_triangle_indices] = self.partial_delaunay_triangles[~incircle_test]
        batch_index[good_triangle_indices] = self.batch_index[~incircle_test]
        
        partial_delaunay_triangles[bad_triangle_indices_mask] = newtriangles
        batch_index[bad_triangle_indices_mask] = nomatch_batch_index
        
        self.partial_delaunay_triangles = partial_delaunay_triangles
        self.batch_index = batch_index
        
        self.ntriangles = ntriangles
        self.batch_triangles = self.partial_delaunay_triangles.size(0)
        
        self.points_mask.scatter_(
            1, point_index.unsqueeze(1).expand(-1, self.npoints), True
        )
        self.points_sequence = torch.cat(
            [self.points_sequence, point_index.unsqueeze(1)], dim=1
        )
        
        self.cost += nremoved_triangles
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
    

env = environment()


def train(epochs = 30000, npoints = 14, batchsize = 100, nsamples = 8):
    NN.train()
    for i in range(epochs):
        env.reset(npoints, batchsize, nsamples)
        """include the boundary points, kinda makes sense that they should contribute (atm only in the encoder, difficult to see how in the decoder)"""
        memory = NN.encode(env.points) #[npoints, batchsize * nsamples, emsize]
        #### #### #### remember to include tgt.detach() when reinstate with torch.no_grad()
        tgt = NN.start_token.unsqueeze(0).unsqueeze(1).expand(1, batchsize * nsamples, -1).detach() #[1, batchsize * nsamples, emsize]
        #with torch.no_grad(): #to speed up computation, selecting routes is done without gradient
        with torch.no_grad():
            for j in range(4, npoints):
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

        
        NN.eval()
        _, logprob = NN.calculate_logprob(memory, env.points_sequence) #[batchsize * nsamples]
        NN.train()
        """
        clip logprob so doesn't reinforce things it already knows
        TBH WANT SOMETHING DIFFERENT ... want to massively increase training if find something unexpected and otherwise not
        """
        greedy_baseline = env.cost.view(batchsize, nsamples)[:, -1] #[batchsize], greedy sample
        fixed_baseline = 0.5 * torch.ones([1], device = device)
        min_baseline = env.cost.view(batchsize, nsamples)[:, :-1].min(-1)[0] #[batchsize], minimum cost
        baseline = greedy_baseline
        positive_reinforcement = - F.relu( - (env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1))) #don't scale positive reinforcement
        negative_reinforcement = F.relu(env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1))
        positive_reinforcement_binary = env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1) <= 0
        negative_reinforcement_binary = env.cost.view(batchsize, nsamples)[:, :-1] - baseline.unsqueeze(1) > 0
        """
        binary positive reinforcement
        """
        #loss = - ((logprob.view(batchsize, nsamples)[:, :-1] < -0.2) * logprob.view(batchsize, nsamples)[:, :-1] * positive_reinforcement_binary).mean() #+ (logprob.view(batchsize, nsamples)[:, :-1] > -1) * logprob.view(batchsize, nsamples)[:, :-1] * negative_reinforcement_binary
        """
        clipped binary reinforcement
        """
        #loss = ( - logprob.view(batchsize, nsamples)[:, :-1] * (logprob.view(batchsize, nsamples)[:, :-1] < -0.2) * positive_reinforcement_binary + logprob.view(batchsize, nsamples)[:, :-1] * (logprob.view(batchsize, nsamples)[:, :-1] > -2) * negative_reinforcement_binary ).mean()
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
        print(greedy_baseline.mean().item(), logprob.view(batchsize, nsamples)[:, -1].mean().item(), logprob.view(batchsize, nsamples)[:, :-1].mean().item(), logprob[0].item(), env.logprob[0].item())