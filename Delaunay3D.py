import math
import random
import numpy as np
import torch


"""
Install pytorch first, see https://pytorch.org/get-started/locally/ for instructions
"""


"""
If a GPU is available set device to "cuda" below, and this will speed up the code significantly (be careful not to overflow the GPU memory)
To find out if a GPU is available you can use the command
    torch.cuda.is_available()
"""
device = "cpu"
floattype = torch.double


"""
To just compute the cost of the Hilbert Curve alogrithm use the function
    hilbert_insertion(npoints, batchsize)
npoints is the total number of points (including the 3 original corner points!!!!)
batchsize is the number of problem instances we triangulate (say 200, then we would have 200 sets of npoints)

This function will just print the average number of 'additional' triangles deleted, and the standard deviation of the number of 'additional' triangles deleted
"""


"""
To just compute the cost of the random insertion alogrithm use the function
    random_insertion(npoints, batchsize)
npoints is the total number of points (including the 3 original corner points!!!!)
batchsize is the number of problem instances we triangulate (say 200, then we would have 200 sets of npoints)

this function will just print the average number of 'additional' triangles deleted, and the standard deviation of the number of 'additional' triangles deleted
"""


"""
This class (below) is the actual delaunay algorithm

Initiate by
    env = environment()
To generate new random points use
    env.reset(npoints, batchsize)
(don't worry about what nsamples does in the actual code below, it's used for training NNs using reinforcement learning)

npoints is the total number of points to be generated, the first 3 are the 3 corners of the big triangle, all the rest are randomly chosen in the unit square
batchsize is how many problems you want to simultaneously create (lets say 200), then you would have 200 sets of npoints, currently all 200 are untriangulated

The current point locations can be found by
    env.points
The current (partial) delaunay triangulations can be found by
    env.triangles
These are stored as triples of indices, so the triangle [0, 2426, 564] refers to the 0th point, 2426th point, and 564th point stored in env.points
Note: the big outside triangle has two copies, a clockwise ([2, 1, 0]) and anticlockwise ([0, 1, 2]) one, the clockwise one will always stay and never get removed - it's used for some trickery for when the delaunay cavity includes one of the external edges, don't worry about it

To add a point to the current (partial) delaunay triangulation use
    env.update(points)
where points is a torch.tensor (see the pytorch website for how to generate and use tensors) of size [batchsize], containing the index of the points you want to add FOR EACH PROBLEM INSTANCE (remember there might be batchsize = 200 problem instances, so you would have to specify 200 points to add, one for each problem instance)
This will update the current (partial) delaunay triangulation, env.triangles

To see the current cost (the 'additional' triangles that had to be removed) use
    env.cost
This will output a torch.tensor of size [batchsize], i.e. a 'list' of costs, corresponding to each of the (say 200) problem instances
"""


class environment:
    def given(self, internal_points, corner_points, initial_triangulation, initial_triangulation_edges, nsamples = 1):
        """
        internal_points, external_points are [batchsize, npoints, 2]
        initial_triangulation is [batchsize, ntriangles, 3]
        """
        assert nsamples == 1
        assert internal_points.size(0) == corner_points.size(0)
        assert internal_points.size(0) == initial_triangulation.size(0)
        self.batchsize = internal_points.size(0) * nsamples
        self.nsamples = 1
        self.npoints = internal_points.size(1) + corner_points.size(1)
        self.points = torch.cat([corner_points.to(device), internal_points.to(device)], dim = 1)
        self.points_mask = torch.cat([torch.ones([self.batchsize, corner_points.size(1)], dtype = torch.bool, device = device), torch.zeros([self.batchsize, internal_points.size(1)], dtype = torch.bool, device = device)], dim = 1)
        self.points_sequence = torch.empty([self.batchsize, 0], dtype = torch.long, device = device)
        
        self.partial_delaunay_triangles = initial_triangulation.to(device)
        self.partial_delaunay_edges = initial_triangulation_edges.to(device)
        
        self.ntriangles = initial_triangulation.size(1)  # can store as scalar, since will always be the same
        self.cost = torch.zeros([self.batchsize], dtype = floattype, device = device)

        self.logprob = torch.zeros([self.batchsize], dtype = floattype, device = device, requires_grad = True)
    
    def reset(self, npoints, batchsize, nsamples=1, corner_points = None, initial_triangulation = None, initial_triangulation_edges = None):
        """
        corner_points, etc., shoudn't include a batch dimension
        """
        if corner_points == None:
            ncornerpoints = 3
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
            torch.rand([batchsize, npoints - ncornerpoints, 2], dtype = floattype, device=device)
            .unsqueeze(1)
            .expand(-1, nsamples, -1, -1)
            .reshape(self.batchsize, npoints - ncornerpoints, 2)
        )
        if corner_points == None:
            self.corner_points = torch.tensor(
                [[0, 0], [2, 0], [0, 2]], dtype = floattype, device=device
            )
        else:
            self.corner_points = corner_points
        self.points = torch.cat(
            [
                self.corner_points.unsqueeze(0).expand(self.batchsize, -1, -1),
                self.points,
            ],
            dim=-2,
        )  # [batchsize * nsamples, npoints, 2]
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

        """use a trick, for the purpose of an 'external' triangle that is always left untouched, which means we don't have to deal with boundary edges as being different. external triangle is [0, 1, 2] traversed clockwise..."""
        if corner_points == None:
            self.partial_delaunay_triangles = (
                torch.tensor([[0, 2, 1], [0, 1, 2]], dtype=torch.int64, device=device)
                .unsqueeze(0)
                .expand(self.batchsize, -1, -1)
                .contiguous()
            )  # [batchsize, ntriangles, 3] contains index of points, always anticlockwise
            self.partial_delaunay_edges = (
                torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.int64, device=device)
                .unsqueeze(0)
                .expand(self.batchsize, -1)
                .contiguous()
            )  # [batchsize, ntriangles * 3] contains location of corresponding edge (edges go in order 01, 12, 20). Edges will always flip since triangles are stored anticlockwise.
        else:
            self.partial_delaunay_triangles = initial_triangulation.unsqueeze(0).expand(self.batchsize, -1, -1).contiguous()
            self.partial_delaunay_edges = initial_triangulation_edges.unsqueeze(0).expand(self.batchsize, -1).contiguous()
        self.ntriangles = self.partial_delaunay_triangles.size(1)  # can store as scalar, since will always be the same
        self.cost = torch.zeros([self.batchsize], dtype = floattype, device=device)

        self.logprob = torch.zeros([self.batchsize], dtype = floattype, device=device, requires_grad=True)

    def update(self, point_index):  # point_index is [batchsize]
        if point_index.size(0) != self.batchsize:
            print(
                "Error: point_index.size() doesn't match expected size, should be [batchsize]"
            )
            return
        if self.points_mask.gather(1, point_index.unsqueeze(1)).sum():
            print("Error: some points already added")
            return
        triangles_coordinates = self.points.gather(
            1,
            self.partial_delaunay_triangles.view(self.batchsize, self.ntriangles * 3)
            .unsqueeze(2)
            .expand(-1, -1, 2),
        ).view(
            self.batchsize, self.ntriangles, 3, 2
        )  # [batchsize, ntriangles, 3, 2]
        newpoint = self.points.gather(
            1, point_index.unsqueeze(1).unsqueeze(2).expand(self.batchsize, 1, 2)
        ).squeeze(
            1
        )  # [batchsize, 2]

        incircle_matrix = torch.cat(
            [
                triangles_coordinates,
                newpoint.unsqueeze(1).unsqueeze(2).expand(-1, self.ntriangles, 1, -1),
            ],
            dim=-2,
        )  # [batchsize, ntriangles, 4, 2]
        incircle_matrix = torch.cat(
            [
                incircle_matrix,
                (incircle_matrix * incircle_matrix).sum(-1, keepdim=True),
                torch.ones([self.batchsize, self.ntriangles, 4, 1], dtype = floattype, device=device),
            ],
            dim=-1,
        )  # [batchsize, ntriangles, 4, 4]
        assert incircle_matrix.dtype == floattype
        incircle_test = (
            incircle_matrix.det() > 0
        )  # [batchsize, ntriangles], is True if inside incircle
        removed_edge_mask = (
            incircle_test.unsqueeze(2).expand(-1, -1, 3).reshape(-1)
        )  # [batchsize * ntriangles * 3]

        edges = (
            self.partial_delaunay_edges
            + self.ntriangles
            * 3
            * torch.arange(self.batchsize, device=device).unsqueeze(1)
        ).view(
            -1
        )  # [batchsize * ntriangles * 3]
        neighbouring_edge = edges.masked_select(removed_edge_mask)
        neighbouring_edge_mask = torch.zeros(
            [self.batchsize * self.ntriangles * 3], device=device, dtype=torch.bool
        )
        neighbouring_edge_mask[neighbouring_edge] = True
        neighbouring_edge_mask = (
            neighbouring_edge_mask * removed_edge_mask.logical_not()
        )  # [batchsize * ntriangles * 3]

        n_new_triangles = neighbouring_edge_mask.view(self.batchsize, -1).sum(
            -1
        )  # [batchsize]

        new_point = (
            point_index.unsqueeze(1)
            .expand(-1, self.ntriangles * 3)
            .masked_select(neighbouring_edge_mask.view(self.batchsize, -1))
        )

        second_point_mask = neighbouring_edge_mask.view(
            self.batchsize, -1, 3
        )  # [batchsize, ntriangles 3]
        (
            first_point_indices0,
            first_point_indices1,
            first_point_indices2,
        ) = second_point_mask.nonzero(as_tuple=True)
        first_point_indices2 = (first_point_indices2 != 2) * (first_point_indices2 + 1)

        first_point = self.partial_delaunay_triangles[
            first_point_indices0, first_point_indices1, first_point_indices2
        ]  # [?]
        second_point = self.partial_delaunay_triangles.masked_select(
            second_point_mask
        )  # [?]

        new_triangles_mask = torch.cat(
            [
                incircle_test,
                torch.ones([self.batchsize, 2], dtype=torch.bool, device=device),
            ],
            dim=1,
        )  # [batchsize, ntriangles + 2]

        new_neighbouring_edges = (
            3 * new_triangles_mask.nonzero(as_tuple=True)[1]
        )  # [?], 3* since is the 01 edge of new triangles (see later)
        self.partial_delaunay_edges.masked_scatter_(
            neighbouring_edge_mask.reshape(self.batchsize, -1), new_neighbouring_edges
        )  # still [batchsize, ntriangles * 3] for now

        self.partial_delaunay_triangles = torch.cat(
            [
                self.partial_delaunay_triangles,
                torch.empty([self.batchsize, 2, 3], dtype=torch.long, device=device),
            ],
            dim=1,
        )
        self.partial_delaunay_edges = torch.cat(
            [
                self.partial_delaunay_edges,
                torch.empty([self.batchsize, 6], dtype=torch.long, device=device),
            ],
            dim=1,
        )
        new_triangles = torch.stack(
            [first_point, second_point, new_point], dim=1
        )  # [?, 3], edge here is flipped compared to edge in neighbouring triangle (so first_point is the second point in neighbouring edge)
        self.partial_delaunay_triangles.masked_scatter_(
            new_triangles_mask.unsqueeze(2).expand(-1, -1, 3), new_triangles
        )  # [batchsize, ntriangles + 2, 3]

        new_edge01 = neighbouring_edge_mask.view(self.batchsize, -1).nonzero(
            as_tuple=True
        )[
            1
        ]  # [?]

        """we are currently storing which triangles have to be inserted, via the edges along the perimeter of the delaunay cavity, we need to compute which edge is to the 'left'/'right' of each edge"""
        """don't have the memory to do a batchsize * n * n boolean search, don't have the speed to do a batchsize^2 search (as would be the case for sparse matrix or similar)"""
        """best alternative: rotate the edge around right point, repeat until hit edge in mask (will never go to an edge of a removed triangle before we hit edge in mask) should basically be order 1!"""

        neighbouring_edge_index = neighbouring_edge_mask.nonzero(as_tuple=True)[
            0
        ]  # [?]
        next_neighbouring_edge_index = torch.empty_like(neighbouring_edge_index)  # [?]

        rotating_flipped_neighbouring_edge_index = neighbouring_edge_mask.nonzero(
            as_tuple=True
        )[
            0
        ]  # [?], initialise
        todo_mask = torch.ones_like(
            next_neighbouring_edge_index, dtype=torch.bool
        )  # [?]
        while todo_mask.sum():
            rotating_neighbouring_edge_index = (
                rotating_flipped_neighbouring_edge_index
                + 1
                - 3 * (rotating_flipped_neighbouring_edge_index % 3 == 2)
            )  # [todo_mask.sum()], gets smaller until nothing left EFFICIENCY (this may be seriously stupid, as it requires making a bunch of copies when I could be doing stuff inplace)

            update_mask = neighbouring_edge_mask[
                rotating_neighbouring_edge_index
            ]  # [todo_mask.sum()]
            update_mask_unravel = torch.zeros_like(todo_mask).masked_scatter(
                todo_mask, update_mask
            )  # [?]

            next_neighbouring_edge_index.masked_scatter_(
                update_mask_unravel,
                rotating_neighbouring_edge_index.masked_select(update_mask),
            )  # [?]

            todo_mask.masked_fill_(update_mask_unravel, False)  # [?]
            rotating_flipped_neighbouring_edge_index = edges[
                rotating_neighbouring_edge_index.masked_select(
                    update_mask.logical_not()
                )
            ]  # [todo_mask.sum()]
        triangle_index = new_triangles_mask.view(-1).nonzero(as_tuple=True)[
            0
        ]  # [?], index goes up to batchsize * (ntriangles + 2), this is needed for when we invert the permutation by scattering (won't scatter same number of triangles per batch)

        next_triangle_index = torch.empty_like(edges).masked_scatter_(
            neighbouring_edge_mask, triangle_index
        )[
            next_neighbouring_edge_index
        ]  # [?], index goes up to batchsize * (ntriangles + 2)
        next_edge = 3 * next_triangle_index + 1  # [?]

        invert_permutation = torch.empty_like(
            new_triangles_mask.view(-1), dtype=torch.long
        )  # [batchsize * (ntriangles + 2)]
        invert_permutation[
            next_triangle_index
        ] = triangle_index  # [batchsize * (ntriangles + 2)]
        previous_triangle_index = invert_permutation.masked_select(
            new_triangles_mask.view(-1)
        )  # [?]
        previous_edge = 3 * previous_triangle_index + 2  # [?]

        """in the above we rotated around 'first_point' in our new triangles"""
        new_edge20 = next_edge % ((self.ntriangles + 2) * 3)  # [?]
        new_edge12 = previous_edge % ((self.ntriangles + 2) * 3)  # [?]

        new_edges = torch.stack([new_edge01, new_edge12, new_edge20], dim=1)  # [?, 3]
        self.partial_delaunay_edges.masked_scatter_(
            new_triangles_mask.unsqueeze(2)
            .expand(-1, -1, 3)
            .reshape(self.batchsize, -1),
            new_edges,
        )  # [batchsize, (ntriangles + 2) * 3]

        self.ntriangles += 2
        """currently only count the extra triangles you replace (not the one you have to remove because you're located there, and not the ones you make because you have to create two more"""
        self.cost += n_new_triangles - 3
        self.points_mask.scatter_(
            1, point_index.unsqueeze(1).expand(-1, self.npoints), True
        )
        self.points_sequence = torch.cat(
            [self.points_sequence, point_index.unsqueeze(1)], dim=1
        )


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


def random_insertion(npoints=103, batchsize=200):
    env.reset(npoints, batchsize)
    for i in range(npoints - 3):
        env.update(torch.full([batchsize], i + 3, dtype=torch.long, device=device))
    print(env.cost.mean().item(), env.cost.var().sqrt().item())

    """
    UNDER CONSTRUCTION
    """
def kdtree_insertion(npoints=103, batchsize=200):
    env.reset(npoints, batchsize)
    points = env.points[:, 3:]  # [batchsize, npoints - 3]