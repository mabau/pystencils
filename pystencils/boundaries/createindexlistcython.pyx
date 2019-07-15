# distutils: language=c
# Workaround for cython bug
# see https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
WORKAROUND = "Something"

import cython

ctypedef fused IntegerType:
    short
    int
    long
    long long
    unsigned short
    unsigned int
    unsigned long

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def create_boundary_neighbor_index_list_2d(object[IntegerType, ndim=2] flag_field,
                                           int nr_of_ghost_layers, IntegerType boundary_mask, IntegerType fluid_mask,
                                           object[int, ndim=2] stencil, int single_link):
    cdef int xs, ys, x, y
    cdef int dirIdx, num_directions, dx, dy

    xs, ys = flag_field.shape
    boundary_index_list = []
    num_directions = stencil.shape[0]

    for y in range(nr_of_ghost_layers, ys - nr_of_ghost_layers):
        for x in range(nr_of_ghost_layers, xs - nr_of_ghost_layers):
            if flag_field[x, y] & fluid_mask:
                for dirIdx in range(num_directions):
                    dx = stencil[dirIdx,0]
                    dy = stencil[dirIdx,1]
                    if flag_field[x + dx, y + dy] & boundary_mask:
                        boundary_index_list.append((x,y, dirIdx))
                        if single_link:
                            break
    return boundary_index_list


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def create_boundary_neighbor_index_list_3d(object[IntegerType, ndim=3] flag_field,
                                           int nr_of_ghost_layers, IntegerType boundary_mask, IntegerType fluid_mask,
                                           object[int, ndim=2] stencil, int single_link):
    cdef int xs, ys, zs, x, y, z
    cdef int dirIdx, num_directions, dx, dy, dz

    xs, ys, zs = flag_field.shape
    boundary_index_list = []
    num_directions = stencil.shape[0]

    for z in range(nr_of_ghost_layers, zs - nr_of_ghost_layers):
        for y in range(nr_of_ghost_layers, ys - nr_of_ghost_layers):
            for x in range(nr_of_ghost_layers, xs - nr_of_ghost_layers):
                if flag_field[x, y, z] & fluid_mask:
                    for dirIdx in range(num_directions):
                        dx = stencil[dirIdx,0]
                        dy = stencil[dirIdx,1]
                        dz = stencil[dirIdx,2]
                        if flag_field[x + dx, y + dy, z + dz] & boundary_mask:
                            boundary_index_list.append((x,y,z, dirIdx))
                            if single_link:
                                break
    return boundary_index_list



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def create_boundary_cell_index_list_2d(object[IntegerType, ndim=2] flag_field,
                                       IntegerType boundary_mask, IntegerType fluid_mask,
                                       object[int, ndim=2] stencil, int single_link):
    cdef int xs, ys, x, y
    cdef int dirIdx, num_directions, dx, dy

    xs, ys = flag_field.shape
    boundary_index_list = []
    num_directions = stencil.shape[0]

    for y in range(0, ys):
        for x in range(0, xs):
            if flag_field[x, y] & boundary_mask:
                for dirIdx in range(num_directions):
                    dx = stencil[dirIdx,0]
                    dy = stencil[dirIdx,1]
                    if 0 <= x + dx < xs and 0 <= y + dy < ys:
                        if flag_field[x + dx, y + dy] & fluid_mask:
                            boundary_index_list.append((x,y, dirIdx))
                            if single_link:
                                break
    return boundary_index_list


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def create_boundary_cell_index_list_3d(object[IntegerType, ndim=3] flag_field,
                                       IntegerType boundary_mask, IntegerType fluid_mask,
                                       object[int, ndim=2] stencil, int single_link):
    cdef int xs, ys, zs, x, y, z
    cdef int dirIdx, num_directions, dx, dy, dz

    xs, ys, zs = flag_field.shape
    boundary_index_list = []
    num_directions = stencil.shape[0]

    for z in range(0, zs):
        for y in range(0, ys):
            for x in range(0, xs):
                if flag_field[x, y, z] & boundary_mask:
                    for dirIdx in range(num_directions):
                        dx = stencil[dirIdx,0]
                        dy = stencil[dirIdx,1]
                        dz = stencil[dirIdx,2]
                        if 0 <= x + dx < xs and 0 <= y + dy < ys and 0 <= z + dz < zs:
                            if flag_field[x + dx, y + dy, z + dz] & fluid_mask:
                                boundary_index_list.append((x,y,z, dirIdx))
                                if single_link:
                                    break
    return boundary_index_list