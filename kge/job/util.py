import torch


def get_batch_sp_po_coords(
    batch, num_entities, sp_index: dict, po_index: dict
) -> torch.LongTensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entities columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    num_ones = 0
    NOTHING = torch.zeros([0], dtype=torch.long)
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
        num_ones += len(sp_index.get((s, p), NOTHING))
        num_ones += len(po_index.get((p, o), NOTHING))

    coords = torch.zeros([num_ones, 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()

        objects = sp_index.get((s, p), NOTHING)
        coords[current_index : (current_index + len(objects)), 0] = i
        coords[current_index : (current_index + len(objects)), 1] = objects
        current_index += len(objects)

        subjects = po_index.get((p, o), NOTHING) + num_entities
        coords[current_index : (current_index + len(subjects)), 0] = i
        coords[current_index : (current_index + len(subjects)), 1] = subjects
        current_index += len(subjects)

    return coords


def get_batch_so_coords(batch, p_index: dict) -> torch.LongTensor:
    """Given a set of relations, lookup matches for (?,p,?) in given index

    Each row in batch holds a relation p. Returns the non-zero coordinates
    of a 3-way binary tensor of shape (batch_size, num_entities, num_entities)
    for each relation in batch.

    Returns coords tensor of shape (num_ones, 2) where num_ones is
    number of positive triples for all relations in batch.
    """
    num_ones = 0
    NOTHING = torch.zeros([0], dtype=torch.long)
    for i, relation in enumerate(batch):
        num_ones += len(p_index.get(relation.item(), NOTHING))

    coords = torch.zeros([num_ones, 3], dtype=torch.long)
    current_index = 0
    for i, relation in enumerate(batch):
        tuples = p_index.get(relation.item(), NOTHING)
        # Skip relations with no positive triples in given index
        if not tuples.size(0):
            continue
        coords[current_index : (current_index + len(tuples)), 0] = i
        coords[current_index : (current_index + len(tuples)), 1:] = torch.LongTensor(tuples)
        current_index += len(tuples)

    return coords


def coord_to_sparse_tensor(nrows, ncols, coords, device, value=1.0, depth=None):
    # Set size of sparse tensor
    if depth:
        size_ = torch.Size([depth, nrows, ncols])
    else:
        size_ = torch.Size([nrows, ncols])

    # Create sparse tensor
    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            size_,
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            size_,
            device=device,
        )

    return labels


def convert_1d_indices_to_2d(indices_1d, width, column_offset):
    """Returns tensor of shape (len(indices_1d), 2) with
    corresponding 2D coordinates for matrix with given width"""

    indices_2d = []
    for index_1d in indices_1d[0,:]:
        row = int(index_1d.item() / width)
        column = int((index_1d.item() % width) + column_offset)
        indices_2d.append((row, column))

    return torch.Tensor(indices_2d).int()
