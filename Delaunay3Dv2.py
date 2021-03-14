"""
REDO THIS WITHOUT KEEPING TRACK OF EDGES

Idea: among removed triangles, pair up faces that both apear, left with faces that don't - the boundary, from which we construct new triangles

have two lists, faces left to check, and faces to check against (these will be all 3 anticlockwise versions of each face)
keep track of the batch you came from, and the index against which you are currently checking
increase index by one each time until either: find a match, or: no longer checking against same batch
at which point we remove FROM THE FIRST LIST
repeat until all removed
when find a match, mark it in second list
removed all marked faces
somehow find number remaining in each batch, and make sure to copy that many 'new points' into a long list
construct new triangles from the above information
"""



import math
import random
import numpy as np
import torch


device = "cuda:0"
floattype = torch.float


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
        assert list(triangles_coordinates.size()) == [self.batch_triangles, 4, 3]
        
        newpoint = self.points[self.batch_index, point_index[self.batch_index]] # [batch_triangles, 3]
        assert list(newpoint.size()) == [self.batch_triangles, 3]

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
        assert list(incircle_test.size()) == [self.batch_triangles]
        
        conflicts = incircle_test.sum()
        
        conflicting_triangles = self.partial_delaunay_triangles[incircle_test] # [conflicts, 4]
        assert list(conflicting_triangles.size()) == [conflicts, 4]
        
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
        assert list(conflicting_edges.size()) == [conflicts * 4, 3]
        
        conflicting_edges_copy = conflicting_edges.clone() # [conflicts * 4, 3], for later use to construct the new triangles
        
        edge_batch_index = self.batch_index[incircle_test].unsqueeze(1).expand(-1, 4).reshape(-1) # [conflicts * 4]
        assert list(edge_batch_index.size()) == [conflicts * 4]
        assert str(edge_batch_index.device) == device
        
        indices = torch.LongTensor([0, 2, 1])
        comparison_edges = conflicting_edges[:, indices] # [conflicts * 4, 3]
        assert list(comparison_edges.size()) == [conflicts * 4, 3]
        
        unravel_nomatch_mask = torch.ones([conflicts * 4], dtype = torch.bool, device = device) # [conflicts * 4]
        
        edge_batch_index_extended = torch.cat([edge_batch_index, torch.tensor([-1], dtype = torch.long, device = device)])
        
        starting_index = torch.arange(conflicts * 4, dtype = torch.long, device = device) # [conflicts * 4]
        current_comparison_index = starting_index.clone() # [conflicts * 4]
        batch_index_todo = edge_batch_index.clone()
        todo_mask = torch.ones([conflicts * 4], dtype = torch.bool, device = device) # [conflicts * 4]
        while todo_mask.sum():
            
            if todo_mask.sum() < todo_mask.size(0) // 2:
                conflicting_edges = conflicting_edges[todo_mask]
                current_comparison_index = current_comparison_index[todo_mask]
                starting_index = starting_index[todo_mask]
                batch_index_todo = batch_index_todo[todo_mask]
                todo_mask = torch.ones([todo_mask.sum()], dtype = torch.bool, device = device)
            
            assert conflicting_edges.size(0) == todo_mask.size(0)
            assert current_comparison_index.size() == todo_mask.size()
            assert batch_index_todo.size() == todo_mask.size()
            
            nomatch_mask = (conflicting_edges != comparison_edges[current_comparison_index]).sum(-1).bool()#.logical_or(batch_index_todo != edge_batch_index_extended[current_comparison_index])
            
            match_mask = nomatch_mask.logical_not().logical_and(todo_mask)
            match_index0 = starting_index[match_mask]
            match_index1 = current_comparison_index[match_mask]
            assert unravel_nomatch_mask[match_index0].logical_not().sum().logical_not()
            assert unravel_nomatch_mask[match_index1].logical_not().sum().logical_not()
            unravel_nomatch_mask[match_index0] = False
            unravel_nomatch_mask[match_index1] = False
            
            todo_mask = todo_mask.logical_and(nomatch_mask).logical_and(batch_index_todo == edge_batch_index_extended[current_comparison_index + 1])
            
            current_comparison_index = (current_comparison_index + 1) * todo_mask
        
        batch_newtriangles = unravel_nomatch_mask.sum()
        
        nomatch_edges = conflicting_edges_copy[unravel_nomatch_mask] # [batch_newtriangles, 3], already in correct order to insert into 1,2,3 (since already anticlockwise from outside in)
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
        
        if (nnew_triangles > 2 * nremoved_triangles + 2).sum():
            print(nnew_triangles, nremoved_triangles)
        
        assert (nnew_triangles <= 2 * nremoved_triangles + 2).logical_not().sum().logical_not()
        """
        should always be <=
        can sometimes not be equal
        if equal, this would mean the total removed triangles is determined by the UNIQUE final delaunay triangulation ... and the same would probably be true in any number of dimensions greater than 2 ...
        """
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
        
        cumulative_triangles = torch.cat([torch.zeros([1], dtype = torch.long, device = device), nnew_triangles.cumsum(0)[:-1]]) # [batchsize], cummulative sum starts at zero
        
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
        assert list(self.partial_delaunay_triangles.size()) == [self.ntriangles.sum(), 4]
        self.batch_triangles = self.partial_delaunay_triangles.size(0)
        
        self.points_mask.scatter_(
            1, point_index.unsqueeze(1).expand(-1, self.npoints), True
        )
        self.points_sequence = torch.cat(
            [self.points_sequence, point_index.unsqueeze(1)], dim=1
        )
        
        self.cost += nnew_triangles


# turn integer x,y coords (in nxn grid) into position d (0 to n^2-1) along the Hilbert curve.
def xy2d(n, x, y):
    [x, y] = [math.floor(x), math.floor(y)]
    [rx, ry, s, d] = [0, 0, 0, 0]
    s = n / 2
    s = math.floor(s)
    while s > 0:
        rx = (x & s) > 0  # bitwise and, and then boolean is it greater than 0?
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        [x, y] = rot(n, x, y, rx, ry)
        s = s / 2
        s = math.floor(s)
    return d


def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # Swap x and y
        t = x
        x = y
        y = t
    return x, y


def order(
    n, points
):  # turns tensor of points into integer distances along hilbert curve of itteration n
    grid = n * points.to("cpu")
    x = torch.empty([grid.size(0)])
    for i in range(points.size(0)):
        x[i] = xy2d(n, grid[i, 0], grid[i, 1])
    return x


def hilbert_insertion(npoints=103, batchsize=200):
    env.reset(npoints, batchsize)
    points = env.points[:, 3:]  # [batchsize, npoints - 3]
    insertion_order = torch.full([batchsize, npoints], float("inf"), device=device)
    for i in range(batchsize):
        insertion_order[i, 3:] = order(
            2 ** 6, points[i]
        )  # number of possible positions is n ** 2
    for i in range(npoints - 3):
        next_index = insertion_order.min(-1)[1]
        env.update(next_index)
        insertion_order.scatter_(1, next_index.unsqueeze(1), float("inf"))
    print(env.cost.mean().item(), env.cost.var().sqrt().item())


def random_insertion(npoints=104, batchsize=200):
    env.reset(npoints, batchsize)
    for i in range(npoints - 4):
        env.update(torch.full([batchsize], i + 4, dtype=torch.long, device=device))
    print(env.cost.mean().item(), env.cost.var().sqrt().item())

    """
    UNDER CONSTRUCTION
    """
def kdtree_insertion(npoints=103, batchsize=200):
    env.reset(npoints, batchsize)
    points = env.points[:, 3:]  # [batchsize, npoints - 3]