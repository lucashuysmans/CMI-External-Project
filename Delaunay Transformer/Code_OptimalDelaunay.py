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

device = "cuda"


class environment():
    def reset(self, npoints, batchsize, nsamples):
        if npoints <= 3:
            print("Error: not enough points for valid problem instance")
            return
        self.batchsize = batchsize * nsamples #so that I don't have to rewrite all this code, we store these two dimensions together
        self.nsamples = nsamples
        self.npoints = npoints
        self.points = torch.rand([batchsize, npoints - 3, 2], device = device).unsqueeze(1).expand(-1, nsamples, -1, -1).reshape(self.batchsize, npoints - 3, 2)
        self.corner_points = torch.tensor([[0, 0], [2, 0], [0, 2]], dtype = torch.float, device = device)
        self.points = torch.cat([self.corner_points.unsqueeze(0).expand(self.batchsize, -1, -1), self.points], dim = -2) #[batchsize * nsamples, npoints, 2]
        self.points_mask = torch.cat([torch.ones([self.batchsize, 3], dtype = torch.bool, device = device), torch.zeros([self.batchsize, npoints - 3], dtype = torch.bool, device = device)], dim = 1)
        self.points_sequence = torch.empty([self.batchsize, 0], dtype = torch.long, device = device)
        
        """use a trick, for the purpose of an 'external' triangle that is always left untouched, which means we don't have to deal with boundary edges as being different. external triangle is [0, 1, 2] traversed clockwise..."""
        self.partial_delaunay_triangles = torch.tensor([[0, 2, 1], [0, 1, 2]], dtype = torch.int64, device = device).unsqueeze(0).expand(self.batchsize, -1, -1).contiguous() #[batchsize, ntriangles, 3] contains index of points, always anticlockwise
        self.partial_delaunay_edges = torch.tensor([5, 4, 3, 2, 1, 0], dtype = torch.int64, device = device).unsqueeze(0).expand(self.batchsize, -1).contiguous() #[batchsize, ntriangles * 3] contains location of corresponding edge (edges go in order 01, 12, 20). Edges will always flip since triangles are stored anticlockwise.
        
        self.ntriangles = 2 #can store as scalar, since will always be the same
        self.cost = torch.zeros([self.batchsize], device = device)
    
    
    def reset_copy(self, points, nsamples): #points is size [batchsize, npoints, 2]
        self.batchsize = points.size(0) * nsamples
        self.nsamples = nsamples
        self.npoints = points.size(1) + 3
        self.corner_points = torch.tensor([[0, 0], [2, 0], [0, 2]], dtype = torch.float, device = device)
        self.points = torch.cat([self.corner_points.unsqueeze(0).expand(self.batchsize, -1, -1), points.unsqueeze(1).expand(-1, nsamples, -1, -1).reshape(-1, self.npoints - 3, 2)], dim = -2) #[batchsize * nsamples, npoints, 2]
        self.points_mask = torch.cat([torch.ones([self.batchsize, 3], dtype = torch.bool, device = device), torch.zeros([self.batchsize, self.npoints - 3], dtype = torch.bool, device = device)], dim = 1)
        self.points_sequence = torch.empty([self.batchsize, 0], dtype = torch.long, device = device)
        
        self.partial_delaunay_triangles = torch.tensor([[0, 2, 1], [0, 1, 2]], dtype = torch.int64, device = device).unsqueeze(0).expand(self.batchsize, -1, -1).contiguous() #[batchsize, ntriangles, 3] contains index of points, always anticlockwise
        self.partial_delaunay_edges = torch.tensor([5, 4, 3, 2, 1, 0], dtype = torch.int64, device = device).unsqueeze(0).expand(self.batchsize, -1).contiguous()
        
        self.ntriangles = 2
        self.cost = torch.zeros([self.batchsize], device = device)
        
    
    def update(self, point_index): #point_index is [batchsize]
        if point_index.size(0) != self.batchsize:
            print("Error: point_index.size() doesn't match expected size, should be [batchsize]")
            return
        if self.points_mask.gather(1, point_index.unsqueeze(1)).sum():
            print("Error: some points already added")
            return
        triangles_coordinates = self.points.gather(1, self.partial_delaunay_triangles.view(self.batchsize, self.ntriangles * 3).unsqueeze(2).expand(-1, -1, 2)).view(self.batchsize, self.ntriangles, 3, 2) #[batchsize, ntriangles, 3, 2]
        newpoint = self.points.gather(1, point_index.unsqueeze(1).unsqueeze(2).expand(self.batchsize, 1, 2)).squeeze(1) #[batchsize, 2]
        
        incircle_matrix = torch.cat([triangles_coordinates, newpoint.unsqueeze(1).unsqueeze(2).expand(-1, self.ntriangles, 1, -1)], dim = -2) #[batchsize, ntriangles, 4, 2]
        incircle_matrix = torch.cat([incircle_matrix, (incircle_matrix * incircle_matrix).sum(-1, keepdim = True), torch.ones([self.batchsize, self.ntriangles, 4, 1], device = device)], dim = -1) #[batchsize, ntriangles, 4, 4]
        incircle_test = incircle_matrix.det() > 0 #[batchsize, ntriangles], is True if inside incircle
        removed_edge_mask = incircle_test.unsqueeze(2).expand(-1, -1, 3).reshape(-1) #[batchsize * ntriangles * 3]
        
        edges = (self.partial_delaunay_edges + self.ntriangles * 3 * torch.arange(self.batchsize, device = device).unsqueeze(1)).view(-1) #[batchsize * ntriangles * 3]
        neighbouring_edge = edges.masked_select(removed_edge_mask)
        neighbouring_edge_mask = torch.zeros([self.batchsize * self.ntriangles * 3], device = device, dtype = torch.bool)
        neighbouring_edge_mask[neighbouring_edge] = True
        neighbouring_edge_mask = (neighbouring_edge_mask * removed_edge_mask.logical_not()) #[batchsize * ntriangles * 3]
        
        n_new_triangles = neighbouring_edge_mask.view(self.batchsize, -1).sum(-1) #[batchsize]
        
        new_point = point_index.unsqueeze(1).expand(-1, self.ntriangles * 3).masked_select(neighbouring_edge_mask.view(self.batchsize, -1))
        
        second_point_mask = neighbouring_edge_mask.view(self.batchsize, -1, 3) #[batchsize, ntriangles 3]
        (first_point_indices0, first_point_indices1, first_point_indices2) = second_point_mask.nonzero(as_tuple = True)
        first_point_indices2 = (first_point_indices2 != 2) * (first_point_indices2 + 1)
        
        first_point = self.partial_delaunay_triangles[first_point_indices0, first_point_indices1, first_point_indices2] #[?]
        second_point = self.partial_delaunay_triangles.masked_select(second_point_mask) #[?]
        
        new_triangles_mask = torch.cat([incircle_test, torch.ones([self.batchsize, 2], dtype = torch.bool, device = device)], dim = 1) #[batchsize, ntriangles + 2]
        
        new_neighbouring_edges = 3 * new_triangles_mask.nonzero(as_tuple = True)[1] #[?], 3* since is the 01 edge of new triangles (see later)
        self.partial_delaunay_edges.masked_scatter_(neighbouring_edge_mask.view(self.batchsize, -1), new_neighbouring_edges) #still [batchsize, ntriangles * 3] for now
        
        self.partial_delaunay_triangles = torch.cat([self.partial_delaunay_triangles, torch.empty([self.batchsize, 2, 3], dtype = torch.long, device = device)], dim = 1)
        self.partial_delaunay_edges = torch.cat([self.partial_delaunay_edges, torch.empty([self.batchsize, 6], dtype = torch.long, device = device)], dim = 1)
        new_triangles = torch.stack([first_point, second_point, new_point], dim = 1) #[?, 3], edge here is flipped compared to edge in neighbouring triangle (so first_point is the second point in neighbouring edge)
        self.partial_delaunay_triangles.masked_scatter_(new_triangles_mask.unsqueeze(2).expand(-1, -1, 3), new_triangles) #[batchsize, ntriangles + 2, 3]
        
        new_edge01 = neighbouring_edge_mask.view(self.batchsize, -1).nonzero(as_tuple = True)[1] #[?]
        
        """we are currently storing which triangles have to be inserted, via the edges along the perimeter of the delaunay cavity, we need to compute which edge is to the 'left'/'right' of each edge"""
        """don't have the memory to do a batchsize * n * n boolean search, don't have the speed to do a batchsize^2 search (as would be the case for sparse matrix or similar)"""
        """best alternative: rotate the edge around right point, repeat until hit edge in mask (will never go to an edge of a removed triangle before we hit edge in mask) should basically be order 1!!!!!"""
        
        neighbouring_edge_index = neighbouring_edge_mask.nonzero(as_tuple = True)[0] #[?]
        next_neighbouring_edge_index = torch.empty_like(neighbouring_edge_index) #[?]
        
        rotating_flipped_neighbouring_edge_index = neighbouring_edge_mask.nonzero(as_tuple = True)[0] #[?], initialise
        todo_mask = torch.ones_like(next_neighbouring_edge_index, dtype = torch.bool) #[?]
        while todo_mask.sum():
            rotating_neighbouring_edge_index = rotating_flipped_neighbouring_edge_index + 1 - 3 * (rotating_flipped_neighbouring_edge_index % 3 == 2) #[todo_mask.sum()], gets smaller until nothing left EFFICIENCY (this may be seriously stupid, as it requires making a bunch of copies when I could be doing stuff inplace)
            
            update_mask = neighbouring_edge_mask[rotating_neighbouring_edge_index] #[todo_mask.sum()]
            update_mask_unravel = torch.zeros_like(todo_mask).masked_scatter(todo_mask, update_mask) #[?]
            
            next_neighbouring_edge_index.masked_scatter_(update_mask_unravel, rotating_neighbouring_edge_index.masked_select(update_mask)) #[?]
            
            todo_mask.masked_fill_(update_mask_unravel, False) #[?]
            rotating_flipped_neighbouring_edge_index = edges[rotating_neighbouring_edge_index.masked_select(update_mask.logical_not())] #[todo_mask.sum()]
        triangle_index = new_triangles_mask.view(-1).nonzero(as_tuple = True)[0] #[?], index goes up to batchsize * (ntriangles + 2), this is needed for when we invert the permutation by scattering (won't scatter same number of triangles per batch)
        
        next_triangle_index = torch.empty_like(edges).masked_scatter_(neighbouring_edge_mask, triangle_index)[next_neighbouring_edge_index] #[?], index goes up to batchsize * (ntriangles + 2)
        next_edge = 3 * next_triangle_index + 1 #[?]
        
        invert_permutation = torch.empty_like(new_triangles_mask.view(-1), dtype=torch.long) #[batchsize * (ntriangles + 2)]
        invert_permutation[next_triangle_index] = triangle_index #[batchsize * (ntriangles + 2)]
        previous_triangle_index = invert_permutation.masked_select(new_triangles_mask.view(-1)) #[?]
        previous_edge = 3 * previous_triangle_index + 2 #[?]
        
        """in the above we rotated around 'first_point' in our new triangles"""
        new_edge20 = next_edge % ((self.ntriangles + 2) * 3) #[?]
        new_edge12 = previous_edge % ((self.ntriangles + 2) * 3) #[?]
        
        new_edges = torch.stack([new_edge01, new_edge12, new_edge20], dim = 1) #[?, 3]
        self.partial_delaunay_edges.masked_scatter_(new_triangles_mask.unsqueeze(2).expand(-1, -1, 3).reshape(self.batchsize, -1), new_edges) #[batchsize, (ntriangles + 2) * 3]
        
        self.ntriangles += 2
        """currently only count the extra triangles you replace (not the on you have to remove because you're located there, and not the ones you make because you have to create two more"""
        self.cost += (n_new_triangles - 3)
        self.points_mask.scatter_(1, point_index.unsqueeze(1).expand(-1, self.npoints), True)
        self.points_sequence = torch.cat([self.points_sequence, point_index.unsqueeze(1)], dim = 1)
    
    def allindices(self): #generate all orders of point insertion
        npoints = self.npoints - 3
        allroutes = torch.empty([1, 0], dtype = torch.long, device = device)
        for i in range(npoints):
            nroutes = allroutes.size(0)
            remaining_mask = torch.ones([nroutes], dtype = torch.bool, device = device).unsqueeze(1).expand(-1, npoints).clone().scatter_(-1, allroutes, False)
            remaining_indices = remaining_mask.nonzero(as_tuple = True)[1]
            allroutes = allroutes.unsqueeze(1).expand(-1, remaining_mask[0, :].sum(), -1)
            allroutes = torch.cat([allroutes, remaining_indices.view(nroutes, -1).unsqueeze(2)], dim = -1).view(-1, allroutes.size(-1) + 1)
        return allroutes #[npoints!, npoints]


env = environment()
npoints = 10
minibatchsize = 1
nbatches = 10

for k in range(epochs = 30):
    points = torch.rand([1, npoints, 2], device = device)
    env.reset_copy(points)
    allroutes = env.allindices()
    for j in range(nbatches):
        env.reset_copy(points)
        
env.reset(npoints + 3, batchsize, math.factorial(npoints))
allroutes = env.allindices() + 3
allroutes = allroutes.unsqueeze(0).expand(batchsize, -1, -1).reshape(-1, npoints)
for j in range(10):
    for i in range(npoints):
        env.update(allroutes[:, i])
    print(env.cost.view(batchsize, -1).min(-1)[0].mean().item(), env.cost.mean().item())
    env.reset(npoints + 3, batchsize, math.factorial(npoints))