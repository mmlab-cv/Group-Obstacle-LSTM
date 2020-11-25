import numpy as np
import os
import torch
from torch.autograd import Variable
import math


def get_obst_grid_mask(frame, dataset, dimensions, neighborhood_size, grid_size):
    # define width and height
    width, height = 2, 2
    # extract mnp
    mnp = frame.shape[0]
    # define boundaries
    width_bound, height_bound = ((width * 1.0) / neighborhood_size) * 4, ((height * 1.0) / neighborhood_size) * 4
    # load obstacle map
    if dataset == 0:
        obst_data = np.load("semi/normalized_sampled/eth_univ_semi.npy")
    elif dataset == 1:
        obst_data = np.load("semi/normalized_sampled/hotel_semi.npy")
    elif dataset == 2:
        obst_data = np.load("semi/normalized_sampled/zara01_semi.npy")
    elif dataset == 3:
        obst_data = np.load("semi/normalized_sampled/zara02_semi.npy")
    else:
        obst_data = np.load("semi/normalized_sampled/ucy_univ_semi.npy")
    # find mno = max number obstacles = number of obstacles in the map
    mno = len(obst_data)
    frame_mask = np.zeros((mnp, mno, grid_size ** 2))
    # calculate mask
    for pedindex in range(mnp):
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]
        width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
        height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
        for obst_index in range(mno):
            obst_x, obst_y = obst_data[obst_index, 0], obst_data[obst_index, 1]

            # Get x and y of the obstacles
            if obst_x >= width_high or obst_x < width_low or obst_y >= height_high or obst_y < height_low:
                # Obst not in surrounding, so binary mask should be zero
                continue
            cell_x = int(np.floor(((obst_x - width_low) / width_bound) * grid_size))
            cell_y = int(np.floor(((obst_y - height_low) / height_bound) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

            frame_mask[pedindex, obst_index, cell_x + cell_y * grid_size] = 1
    return frame_mask


def distance(ped_x, ped_y, obst_x, obst_y):

    d = math.sqrt(abs(ped_x - obst_x) ** 2 + abs(ped_y - obst_y) ** 2)

    return d


def get_obst_grid_mask_inf(frame, dataset, dimensions, neighborhood_size, grid_size):
    # define width and height
    width, height = 2, 2
    # extract mnp
    mnp = frame.shape[0]
    # define boundaries
    width_bound, height_bound = ((width * 1.0) / neighborhood_size) * 4, ((height * 1.0) / neighborhood_size) * 4
    # load obstacle map
    if dataset == 0:
        obst_data = np.load("semi/normalized_sampled/eth_univ_semi.npy")
    elif dataset == 1:
        obst_data = np.load("semi/normalized_sampled/hotel_semi.npy")
    elif dataset == 2:
        obst_data = np.load("semi/normalized_sampled/zara01_semi.npy")
    elif dataset == 3:
        obst_data = np.load("semi/normalized_sampled/zara02_semi.npy")
    else:
        obst_data = np.load("semi/normalized_sampled/ucy_univ_semi.npy")
    # find mno = max number obstacles = number of obstacles in the map
    mno = len(obst_data)
    frame_mask = np.zeros((mnp, mno, grid_size ** 2))
    # calculate mask
    for pedindex in range(mnp):
        current_x, current_y = frame[pedindex, 0], frame[pedindex, 1]
        width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
        height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
        for obst_index in range(mno):
            obst_x, obst_y = obst_data[obst_index, 0], obst_data[obst_index, 1]

            # Get x and y of the obstacles
            if obst_x >= width_high or obst_x < width_low or obst_y >= height_high or obst_y < height_low:
                # Obst not in surrounding, so binary mask should be zero
                continue
            cell_x = int(np.floor(((obst_x - width_low) / width_bound) * grid_size))
            cell_y = int(np.floor(((obst_y - height_low) / height_bound) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

            frame_mask[pedindex, obst_index, cell_x + cell_y * grid_size] = 1
    return frame_mask


def get_seq_mask(sequence, dataset, dimensions, neighborhood_size, grid_size):
    '''
    Get the obstacle grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        sequence_mask.append(Variable(torch.from_numpy(get_obst_grid_mask(sequence[i], dataset, dimensions,
                                                                     neighborhood_size, grid_size)).float()).cuda())
        # SOLVED: out of range on i was given by dataset --> seq dataset is the same for the entire
        # sequence so it takes only an int
        # sequence_mask.append(Variable(torch.from_numpy(get_obst_grid_mask(sequence[i], dataset, dimensions,neighborhood_size, grid_size)).float()))
    return sequence_mask
