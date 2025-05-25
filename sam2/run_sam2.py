import torch
from sam2.build_sam import build_sam2_video_predictor
import os
import shutil
import argparse
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from glob import glob
import json
import imageio  # For reading/writing image data, including videos
import pickle
from imageio import get_writer  # Specifically for writing video files
from sklearn.neighbors import NearestNeighbors  # For finding nearest neighbors, used in point density calculation
from collections import defaultdict  # For dictionaries with default values

# DAVIS_PALETTE: A predefined color palette used for visualizing segmentation masks.
# This specific palette is often used with the DAVIS dataset. It's a byte string
# representing a sequence of RGB color values for different object indices.
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def process_invisible_traj(traj, visible_mask, confidences, state, predictor, dilation_size=8, max_iterations=10,
                           obj_id=0, timestep=100, downscale_factor=None):
    """
    Iteratively processes point trajectories to identify and segment objects using a SAM2 predictor.
    It aims to find objects that might be initially "invisible" or poorly tracked by selecting
    optimal frames for prompting the model and then refining the set of unsegmented trajectories.

    Args:
        traj (np.ndarray): Point trajectories. Expected shape [C, N, T] (e.g., [2_coords, Num_points, Num_frames]) upon input from main.
                           Internally, it's transposed for `process_points_with_memory`.
        visible_mask (np.ndarray): Boolean mask indicating visibility of each point in each frame. Shape [N, T].
        confidences (np.ndarray): Confidence scores for each point track. Shape [N, T].
        state: The current inference state of the SAM2 predictor.
        predictor: The SAM2 video predictor instance.
        dilation_size (int, optional): Size of the dilation kernel for masks. Defaults to 8.
        max_iterations (int, optional): Maximum number of iterations to attempt segmentation. Defaults to 10.
        obj_id (int, optional): Initial object ID to start assigning. Defaults to 0.
        timestep (int, optional): Total number of frames (T) in the video. Defaults to 100. Used for selecting keyframes.
        downscale_factor (tuple, optional): Tuple (scale_x, scale_y) for downscaling masks if needed. Defaults to None.

    Returns:
        dict: A dictionary `memory_dict` where keys are segmented object IDs and values are dictionaries
              containing information about the object's tracks ('pts_trajs', 'confi_trajs', 'vis_trajs'),
              the canonical time ('time') of segmentation, and number of points ('num').
    """
    iteration = 0  # Counter for the number of segmentation iterations
    memory_dict = {}  # Dictionary to store information about segmented objects
    take_all = False  # Flag, if True, forces segmentation acceptance in process_points_with_memory

    # If the number of points (traj.shape[1] which is N) is very small, set take_all to True.
    # This implies that if only a few tracks remain, they are likely a single object.
    if traj.shape[1] <= 5:  # N (number of points) is the second dimension of input traj [C, N, T]
        take_all = True

    # Main loop for iterative segmentation
    while iteration < max_iterations:
        # 1. Find the "best" frame (t) to prompt SAM2.
        # This is typically a frame with a high number of visible points among the remaining tracks.
        visible_points_per_frame = visible_mask.sum(
            axis=0)  # Sum visible points for each frame (column sum) -> shape [T]
        top_k = int(timestep * 0.2)  # Consider the top 20% of frames by visible point count
        top_k_indices = np.argsort(visible_points_per_frame)[-top_k:][
                        ::-1]  # Get indices of top_k frames, sorted descending by count
        top_k_indices = sorted(top_k_indices)  # Sort these frame indices chronologically

        # Select the middle frame from this top_k list
        middle_index = len(top_k_indices) // 2
        t = top_k_indices[middle_index]  # This is the chosen canonical frame index for prompting

        # If the chosen frame 't' has no visible points (e.g., if all top_k frames have 0),
        # fall back to the frame with the absolute maximum number of visible points.
        if visible_points_per_frame[t] == 0:
            t = visible_points_per_frame.argmax()  # Index of the frame with the most visible points

        # 2. Use process_points_with_memory to attempt segmentation in frame 't'.
        # traj is [C, N, T]. process_points_with_memory expects [N, C, T] for its trajectory input.
        # So, traj.transpose(1,0,2) changes [C,N,T] to [N,C,T].
        mem_pts, mem_confi, mem_vis, new_memory_dict, obj_id = process_points_with_memory(
            state, predictor, t, obj_id,
            traj.transpose(1, 0, 2),  # Transposed trajectories
            confidences, visible_mask, dilation_size, take_all,
            downscale_factor
        )
        # mem_pts is returned as [N_remaining, C, T]

        # 3. Update the main memory_dict with any newly segmented objects.
        memory_dict.update(new_memory_dict)

        # 4. Update trajectories, visibility, and confidences by removing points that were segmented.
        # mem_pts is [N_remaining, C, T]. Transpose back to [C, N_remaining, T] for consistency with input `traj`.
        traj = mem_pts.transpose(1, 0, 2)
        visible_mask = mem_vis  # Updated visibility mask
        confidences = mem_confi  # Updated confidences

        iteration += 1  # Increment iteration count

        # If the number of remaining points (traj.shape[1]) is very small, stop iterating.
        if traj.shape[1] < 6:
            break

    return memory_dict  # Return the dictionary of all segmented objects


def find_dense_pts(points):
    """
    Finds the densest point in a given set of 2D points.
    Density is approximated by finding the point with the minimum sum of distances to all other points.

    Args:
        points (np.ndarray): A 2D array of points, shape [N, 2], where N is the number of points.

    Returns:
        np.ndarray: The densest point, shape [1, 2].
    """
    # Initialize NearestNeighbors to find distances to all N-1 other points (n_neighbors=points.shape[0])
    nbrs = NearestNeighbors(n_neighbors=points.shape[0]).fit(points)
    # distances: [N, N] array where distances[i,j] is dist from point i to its j-th neighbor
    # indices: [N, N] array of neighbor indices
    distances, indices = nbrs.kneighbors(points)

    # Calculate "density" for each point as the sum of distances to all other points in the set.
    # A smaller sum means the point is, on average, closer to all other points.
    density = np.sum(distances, axis=1)  # Sum distances for each row (point)

    # Find the index of the point with the minimum sum of distances (i.e., the "densest" point).
    max_density_index = np.argmin(density)
    # Retrieve the coordinates of this densest point.
    max_density_point = points[max_density_index]
    # Expand dimensions to return it as a [1, 2] array, consistent for point prompts.
    max_density_point = np.expand_dims(max_density_point, axis=0)

    return max_density_point


# ---------------------- SAM2 Prompt Execution ---------------------- #
def process_points_with_memory(state, predictor, t_cano, initial_obj_id, traj, confi_t, mask_t, dilation_size=3,
                               take_all=False, downscale_factor=None):
    """
    Segments an object in a specific frame (t_cano) by prompting SAM2 with a dense point from the trajectories.
    If a valid mask is produced, the associated point tracks are stored in `memory_dict`,
    and these tracks are removed from the input `traj`, `confi_t`, and `mask_t`.

    Args:
        state: Current inference state of the SAM2 predictor.
        predictor: The SAM2 video predictor instance.
        t_cano (int): The canonical frame index in which to prompt SAM2.
        initial_obj_id (int): The current highest object ID, used to assign a new ID if an object is segmented.
        traj (np.ndarray): Point trajectories, expected shape [N_points, C_coords, T_frames] (e.g., [N, 2, T]).
        confi_t (np.ndarray): Confidence scores for tracks, shape [N_points, T_frames].
        mask_t (np.ndarray): Visibility masks for tracks, shape [N_points, T_frames].
        dilation_size (int, optional): Kernel size for dilating the SAM2 output mask. Defaults to 3.
        take_all (bool, optional): If True, forces acceptance of the segmentation. Defaults to False.
        downscale_factor (tuple, optional): (scale_x, scale_y) for downscaling. Defaults to None.

    Returns:
        tuple:
            - np.ndarray: `mem_pts` - Filtered trajectories (points belonging to segmented object removed). Shape [N_new, C, T].
            - np.ndarray: `mem_confi` - Filtered confidences. Shape [N_new, T].
            - np.ndarray: `mem_vis` - Filtered visibility masks. Shape [N_new, T].
            - dict: `memory_dict` - Dictionary containing info about the newly segmented object (if any).
            - int: `obj_id_valid` - Updated object ID after potential segmentation.
    """
    obj_id_valid = initial_obj_id  # Initialize the object ID to be assigned if segmentation is successful

    # Create copies of input arrays to modify them safely
    mem_pts = traj.copy()  # Trajectories, e.g., [N, 2, T]
    mem_vis = mask_t.copy()  # Visibility, e.g., [N, T]
    mem_confi = confi_t.copy()  # Confidences, e.g., [N, T]
    memory_dict = {}  # Dictionary to store info if an object is segmented in this call

    # Extract points and visibility for the canonical frame t_cano
    points_cano = mem_pts[:, :, t_cano]  # Points at t_cano, shape [N, 2] (assuming C=2)
    vis_cano = mem_vis[:, t_cano]  # Visibility at t_cano, shape [N]
    remain_pts = points_cano[vis_cano]  # Filter to get only visible points at t_cano, shape [N_visible, 2]

    # If no points are visible in the canonical frame, return without processing
    if remain_pts.shape[0] == 0:
        return mem_pts, mem_confi, mem_vis, memory_dict, obj_id_valid

    # Find the densest point among the visible points in t_cano to use as a prompt for SAM2
    nearest_point = find_dense_pts(remain_pts)  # Shape [1, 2]

    # Reset the predictor's internal state (e.g., clear previous masks/prompts for this object ID)
    predictor.reset_state(state)
    # Labels for the prompt point(s); 1 typically indicates a foreground point.
    labels = np.array([1], np.int32)

    # Prompt SAM2 with the selected point(s) in frame t_cano for a generic object_id=1 (local to this call)
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=t_cano,
        obj_id=1,  # Using a temporary object ID for this specific SAM2 prompt
        points=nearest_point,
        labels=labels,
    )

    # Binarize the output mask logits from SAM2 (logits > 0.0 means foreground)
    mask_sam = (out_mask_logits[0] > 0.0).cpu().numpy()  # Shape [H, W]
    # Dilate the binary mask to potentially cover more of the object
    dilated_mask = dilate_mask(mask_sam, dilation_size)  # Shape [1, H, W]

    # Determine which of the original points (at t_cano) fall within the dilated mask
    in_mask = np.zeros(mem_pts.shape[0], dtype=bool)  # Boolean array for all N points, initially all False
    in_mask_all = find_pts_in_mask(dilated_mask, points_cano, downscale_factor)  # Check all points at t_cano
    # Update 'in_mask': a point is selected if it was visible at t_cano AND falls into the dilated_mask
    in_mask[vis_cano] = in_mask_all[vis_cano]

    # Determine which points fall into the original (non-dilated) SAM mask
    prompt_mask = np.zeros(mem_pts.shape[0], dtype=bool)  # Boolean array for all N points
    prompt_mask_all = find_pts_in_mask(mask_sam, points_cano, downscale_factor)  # Check against non-dilated mask
    # Update 'prompt_mask': a point is selected if visible at t_cano AND falls into the original mask_sam
    prompt_mask[vis_cano] = prompt_mask_all[vis_cano]

    # Optional visualization (if args.vis is True)
    if args.vis:
        save_mask_with_points(mask_sam, nearest_point, "prompt.png")  # Visualize prompt point and SAM mask
        save_mask_with_points(dilated_mask, points_cano,
                              "dilated_mask.png")  # Visualize all points at t_cano and dilated mask
        # Visualize points that are considered 'in_mask' (i.e. selected by the dilated mask)
        save_mask_with_points(dilated_mask, points_cano[in_mask[vis_cano]],
                              "in_mask.png")  # Needs careful indexing if points_cano is already filtered

    # Judge if the segmentation is valid based on the number of points covered
    include = judge(in_mask, mem_pts, mem_vis)  # `in_mask` refers to points within the dilated mask

    # If the segmentation is considered valid OR if 'take_all' is True:
    if (np.sum(prompt_mask) > 0 and include) or take_all:
        obj_id_valid += 1  # Increment to get a new unique object ID
        # Store information about this newly segmented object
        memory_dict[obj_id_valid] = {
            'time': t_cano,  # Canonical frame of segmentation
            'pts_trajs': mem_pts[prompt_mask],  # Trajectories of points within the non-dilated `prompt_mask`
            'confi_trajs': mem_confi[prompt_mask],  # Corresponding confidences
            'vis_trajs': mem_vis[prompt_mask],  # Corresponding visibilities
            'num': int(in_mask.sum())  # Total number of points (from original N) falling in the `dilated_mask`
        }

    # Remove the points that were segmented (those in `in_mask`, i.e., covered by dilated_mask)
    # from the working set of trajectories, confidences, and visibilities.
    mem_pts = mem_pts[~in_mask]  # Keep points NOT in `in_mask`
    mem_confi = mem_confi[~in_mask]  # Keep corresponding confidences
    mem_vis = mem_vis[~in_mask]  # Keep corresponding visibilities

    return mem_pts, mem_confi, mem_vis, memory_dict, obj_id_valid


def judge(in_mask, mem_pts, mem_vis, thre=6):
    """
    Determines if a segmented mask (represented by `in_mask` over all points) is valid.
    A mask is valid if it contains at least `thre` points, either in total (across all frames,
    based on `in_mask` itself) or `thre` points visible in any single frame.

    Args:
        in_mask (np.ndarray): Boolean array of shape [N_total_points], indicating which points
                              (from the original set passed to `process_points_with_memory`)
                              are considered part of the current segmentation attempt (e.g., fall in dilated mask).
        mem_pts (np.ndarray): Point trajectories, shape [N_total_points, C, T_frames].
        mem_vis (np.ndarray): Visibility masks for points, shape [N_total_points, T_frames].
        thre (int, optional): Threshold for the minimum number of points. Defaults to 6.

    Returns:
        bool: True if the mask is considered valid, False otherwise.
    """
    include = False  # Flag indicating if the criteria are met

    # First check: if the total number of points selected by `in_mask` is above threshold.
    if np.sum(in_mask) >= thre:
        include = True
        return include  # If so, the mask is valid.

    # Second check (if first failed): iterate through each frame.
    T = mem_pts.shape[-1]  # Number of frames
    for t in range(T):
        points = mem_pts[:, :, t]  # Points in current frame t, shape [N_total_points, C]
        v_mask_frame_t = mem_vis[:, t]  # Visibility in current frame t, shape [N_total_points]

        # Combine `in_mask` (points belonging to the object candidate)
        # with `v_mask_frame_t` (points visible in this specific frame t).
        # `mask_valid_in_frame_t` indicates points that are part of the object candidate AND visible in frame t.
        mask_valid_in_frame_t = v_mask_frame_t & in_mask

        points_valid_in_frame_t = points[
            mask_valid_in_frame_t]  # Get actual coordinates of these points (not strictly needed for count)
        num_valid_in_frame_t = points_valid_in_frame_t.shape[0]  # Count of such points in frame t

        # If the count in this frame meets the threshold, the mask is considered valid.
        if num_valid_in_frame_t > thre:
            include = True
            return include  # Valid by frame-specific criterion.

    return include  # If neither criterion met after checking all frames, return current value of include (False).


def dilate_mask(mask, dilation_size=5):
    """
    Dilates a binary mask using a square kernel.

    Args:
        mask (np.ndarray): Input binary mask. Can be shape [1, H, W] or [H, W].
        dilation_size (int, optional): Size of the square dilation kernel. Defaults to 5.

    Returns:
        np.ndarray: The dilated mask, shape [1, H, W].
    """
    # If mask has shape [1, H, W], remove the leading dimension to get [H, W] for cv2.dilate.
    if mask.shape[0] == 1 and mask.ndim == 3:  # Ensure it's not just a single row mask
        mask_2d = mask[0]
    else:
        mask_2d = mask  # Assume it's already [H,W]

    # Create a square kernel of ones for dilation.
    kernel = np.ones((dilation_size, dilation_size), np.uint8)

    # Perform dilation. Mask must be uint8. Iterations=3 means applying dilation 3 times.
    dilated_mask_2d = cv2.dilate(mask_2d.astype(np.uint8), kernel, iterations=3)
    # Add back the leading dimension to ensure output is [1, H, W].
    dilated_mask_expanded = np.expand_dims(dilated_mask_2d, axis=0)

    return dilated_mask_expanded


def cls_iou(pred, label, thres=0.7):
    """
    Calculates Intersection over Union (IoU) for classification/segmentation masks.
    The prediction `pred` is first binarized using `thres`.

    Args:
        pred (torch.Tensor): Predicted mask logits or probabilities. Shape e.g., [B, C, N_elements] or [B, N_elements].
        label (torch.Tensor): Ground truth binary mask. Same shape as pred after binarization.
        thres (float, optional): Threshold to binarize the prediction. Defaults to 0.7.

    Returns:
        torch.Tensor: The mean IoU score over the relevant dimensions (e.g., batch/class).
    """
    # Binarize the prediction tensor based on the threshold.
    mask = (pred > thres).float()  # Convert boolean to float (0.0 or 1.0)
    b, c, n = label.shape  # Get dimensions (batch, channels, num_elements per channel)

    # Calculate Intersection: (TP) sum of element-wise product of predicted mask and true label.
    intersect = (mask * label).sum(-1)  # Sum over the last dimension (N_elements)
    # Calculate Union: (TP + FP + FN) sum(pred) + sum(label) - sum(intersection).
    union = mask.sum(-1) + label.sum(-1) - intersect
    # Calculate IoU. Add a small epsilon (1e-12) to prevent division by zero if union is zero.
    iou = intersect / (union + 1e-12)

    return iou.mean()  # Return the mean IoU (e.g., averaged over batch and/or channels).


def save_mask_with_points(mask, points, save_path, point_color='red', point_size=20):
    """
    Saves an image of a mask overlaid with specified points using Matplotlib.

    Args:
        mask (np.ndarray): The mask to display. Expected shape [H, W] or [1, H, W].
        points (np.ndarray): Points to overlay, shape [N, 2] (x, y coordinates).
        save_path (str): Path to save the output image.
        point_color (str, optional): Color of the points. Defaults to 'red'.
        point_size (int, optional): Size of the points. Defaults to 20.
    """
    # Generate a random color with some transparency (alpha=0.6) for the mask overlay.
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # [R, G, B, Alpha]
    h, w = mask.shape[-2:]  # Get height and width from the last two dimensions of the mask.
    # Reshape mask to [H, W, 1] and multiply by color [1, 1, 4] to get an RGBA image [H, W, 4].
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    fig, ax = plt.subplots()  # Create a Matplotlib figure and axes.

    ax.imshow(mask_image)  # Display the (semi-transparent) colored mask.

    # Extract x and y coordinates from the points array.
    x_coords, y_coords = points[:, 0], points[:, 1]
    # Scatter plot the points on the image.
    ax.scatter(x_coords, y_coords, color=point_color, s=point_size)

    ax.axis('off')  # Turn off axis numbers and ticks for a cleaner image.

    # Save the figure to the specified path.
    # bbox_inches='tight' removes whitespace borders. pad_inches=0 further tightens it.
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory.


def put_per_obj_mask(per_obj_mask, height, width):
    """
    Combines individual per-object binary masks into a single instance segmentation mask.
    In the output mask, pixel values correspond to object IDs.
    Objects with higher IDs will overlay objects with lower IDs if they overlap, due to sorted drawing order.

    Args:
        per_obj_mask (dict): Dictionary where keys are object IDs (int) and values are
                             binary masks (np.ndarray, HxW, boolean or 0/1) for each object.
        height (int): Height of the output combined mask.
        width (int): Width of the output combined mask.

    Returns:
        np.ndarray: Combined instance segmentation mask of shape [H, W], dtype=np.uint8.
                    Pixel values are object IDs (or 0 for background).
    """
    # Initialize an empty mask (all zeros, representing background).
    mask = np.zeros((height, width), dtype=np.uint8)
    # Get object IDs and sort them in descending order. This ensures that when objects overlap,
    # the object with the numerically higher ID is drawn "on top" (its ID will be in the final mask).
    object_ids = sorted(per_obj_mask.keys())[::-1]  # Sort keys (IDs) descending.

    # Iterate through each object ID (from highest to lowest).
    for object_id in object_ids:
        object_mask_binary = per_obj_mask[object_id]  # Get the binary mask for the current object.
        object_mask_binary = object_mask_binary.reshape(height, width)  # Ensure correct shape.
        # Where the current object's binary mask is True (or >0),
        # set the corresponding pixels in the combined `mask` to the `object_id`.
        mask[object_mask_binary] = object_id
    return mask


def save_multi_masks_to_dir(
        output_mask_dir,
        video_name,
        frame_name,
        per_obj_output_mask,
        height,
        width,
        per_obj_png_file,  # Flag to decide saving mode
        output_palette,
):
    """
    Saves segmentation masks for a frame to a directory as PNG files.
    Can save either a single combined mask for the frame (where pixel values are object IDs)
    or individual masks for each object in separate subdirectories.

    Args:
        output_mask_dir (str): Base directory to save the masks.
        video_name (str): Name of the video sequence (used as a subdirectory name).
        frame_name (str): Name of the current frame (e.g., "00001", used for filename).
        per_obj_output_mask (dict): Dictionary of masks for the current frame.
                                   Keys are object IDs, values are binary masks (HxW).
        height (int): Height of the masks.
        width (int): Width of the masks.
        per_obj_png_file (bool): If True, save each object's mask as a separate PNG file
                                 in a subfolder named after the object ID.
                                 If False, save a single combined mask for the frame.
        output_palette (bytes): Color palette to use for saving the PNGs (e.g., DAVIS_PALETTE).
    """
    # Create the directory for the video sequence if it doesn't already exist.
    # e.g., output_mask_dir/video_name/
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)

    if not per_obj_png_file:
        # Save as a single combined mask for the frame.
        # Combine individual object masks into one, where pixel values are object IDs.
        output_mask_combined = put_per_obj_mask(per_obj_output_mask, height, width)
        # Define the full path for the output PNG file.
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        # Save the combined mask as a PNG with the specified palette.
        save_ann_png(output_mask_path, output_mask_combined, output_palette)
    else:
        # Save each object's mask as a separate PNG file.
        for object_id, object_mask_binary in per_obj_output_mask.items():
            object_name_str = f"{object_id:03d}"  # Format object ID (e.g., 1 -> "001")
            # Create a subdirectory for this specific object if it doesn't exist.
            # e.g., output_mask_dir/video_name/001/
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name_str),
                exist_ok=True,
            )
            # Reshape and convert the object's binary mask to uint8.
            # For a single object mask, this will typically be 0s and 1s (or some other value for the object).
            # It should be 0 and object_id, or just 0 and 1 if palette handles coloring based on filename implicitly.
            # Given save_ann_png, it expects values that map to palette indices. If it's a binary 0/1 mask for the object,
            # it will use the first two colors of the palette.
            # To make it distinct, usually, the mask itself would contain the object_id, or 1 if it's a binary yes/no.
            output_mask_single_obj = object_mask_binary.reshape(height, width).astype(np.uint8)
            # Define the full path for this object's mask PNG file.
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name_str, f"{frame_name}.png"
            )
            # Save the object's mask as a PNG.
            save_ann_png(output_mask_path, output_mask_single_obj, output_palette)


def save_masks_to_dir(
        # This function appears to be a specific case of save_multi_masks_to_dir (when per_obj_png_file=True)
        output_mask_dir,
        video_name,
        frame_name,
        per_obj_output_mask,
        height,
        width,
        output_palette,
):
    """
    Saves each object's mask as a separate PNG file in its own subdirectory.
    This is equivalent to `save_multi_masks_to_dir` with `per_obj_png_file=True`.

    Args:
        output_mask_dir (str): Base directory to save masks.
        video_name (str): Name of the video sequence.
        frame_name (str): Name of the current frame.
        per_obj_output_mask (dict): Dictionary of masks for the current frame (obj_id -> binary_mask).
        height (int): Height of the masks.
        width (int): Width of the masks.
        output_palette (bytes): Color palette for saving PNGs.
    """
    # Create the main video directory if it doesn't exist.
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)

    # Iterate through each object and its mask.
    for object_id, object_mask_binary in per_obj_output_mask.items():
        object_name_str = f"{object_id:03d}"  # Format object ID (e.g., 1 -> "001").
        # Create a subdirectory for this specific object.
        os.makedirs(
            os.path.join(output_mask_dir, video_name, object_name_str),
            exist_ok=True,
        )
        # Reshape and convert the object's binary mask to uint8.
        output_mask_single_obj = object_mask_binary.reshape(height, width).astype(np.uint8)
        # Define the full path for this object's mask PNG file.
        output_mask_path = os.path.join(
            output_mask_dir, video_name, object_name_str, f"{frame_name}.png"
        )
        # Save the object's mask as a PNG with the specified palette.
        save_ann_png(output_mask_path, output_mask_single_obj, output_palette)


def save_ann_png(path, mask, palette):
    """
    Saves a 2D numpy array mask as a PNG file with a specified palette using PIL.

    Args:
        path (str): Path to save the PNG file.
        mask (np.ndarray): Mask data (2D array of np.uint8). Pixel values are indices into the palette.
        palette (bytes): Color palette data (e.g., DAVIS_PALETTE).
    """
    # Ensure mask is a 2D uint8 numpy array, as expected by PIL for palettized images.
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    # Create a PIL Image object from the numpy mask array.
    output_mask_image = Image.fromarray(mask)
    # Apply the specified palette to the image.
    output_mask_image.putpalette(palette)
    # Save the image to the specified path.
    output_mask_image.save(path)


def load_ann_png(path):
    """
    Loads a palettized PNG file as a numpy mask and extracts its palette using PIL.

    Args:
        path (str): Path to the PNG file.

    Returns:
        tuple:
            - np.ndarray: Loaded mask as a 2D numpy array of np.uint8.
            - list or None: The palette data from the PNG file (flattened list of R,G,B values).
    """
    # Open the image file using PIL.
    mask_image_pil = Image.open(path)
    # Get the palette from the image (if it exists).
    palette_data = mask_image_pil.getpalette()
    # Convert the PIL image to a numpy array of uint8.
    mask_numpy = np.array(mask_image_pil).astype(np.uint8)
    return mask_numpy, palette_data


def get_per_obj_mask(mask):
    """
    Splits a combined instance segmentation mask (where pixel values are object IDs)
    into a dictionary of per-object binary masks.

    Args:
        mask (np.ndarray): Combined instance segmentation mask (2D array, HxW).
                           Pixel values > 0 represent object IDs. 0 is background.

    Returns:
        dict: Dictionary where keys are object IDs (int) and values are binary boolean
              masks (np.ndarray, HxW) for each object.
    """
    # Find all unique non-zero pixel values in the mask; these are the object IDs.
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()  # Filter out background (0) and convert to list.
    # Create a dictionary: for each object_id, create a binary mask where pixels matching that ID are True.
    per_obj_mask_dict = {obj_id: (mask == obj_id) for obj_id in object_ids}
    return per_obj_mask_dict


def load_data(dynamic_dir):
    """
    Loads trajectory, visibility, and confidence data from .npy files stored in `dynamic_dir`.

    Args:
        dynamic_dir (str): Directory containing "dynamic_traj.npy", "dynamic_visibility.npy",
                           and "dynamic_confidences.npy" files.

    Returns:
        tuple:
            - np.ndarray: `traj` - Trajectories, shape [C, N_points, T_frames] (e.g., [2, N, T]).
            - np.ndarray: `visible_mask` - Visibility mask, shape [N_points, T_frames], boolean.
            - np.ndarray: `confi` - Confidence scores, shape [N_points, T_frames].
    """
    # Define paths to the .npy files based on the input directory.
    track_path = os.path.join(dynamic_dir, "dynamic_traj.npy")
    visible_path = os.path.join(dynamic_dir, "dynamic_visibility.npy")
    confi_path = os.path.join(dynamic_dir, "dynamic_confidences.npy")

    # Load data from the .npy files.
    traj = np.load(track_path)  # Tracks, e.g. [2, N, T]
    visible_mask = np.load(visible_path).astype(bool)  # Visibility, convert to boolean
    confi = np.load(confi_path)  # Confidences

    return traj, visible_mask, confi


def apply_mask_to_rgb(rgb_image, mask_image):
    """
    Applies a binary mask to an RGB image.
    Regions where the mask is active (>0) retain their original RGB values.
    Regions where the mask is inactive (0) become black.

    Args:
        rgb_image (np.ndarray): The input RGB image, shape [H, W, 3].
        mask_image (np.ndarray): The binary mask, shape [H, W] (values are 0 or >0).

    Returns:
        np.ndarray: The masked RGB image, shape [H, W, 3].
    """
    # Create an empty (black) image with the same dimensions as the input RGB image.
    masked_rgb = np.zeros_like(rgb_image)

    # Where the `mask_image` is greater than 0 (i.e., part of the foreground/object),
    # copy the corresponding pixels from the original `rgb_image` to `masked_rgb`.
    masked_rgb[mask_image > 0] = rgb_image[mask_image > 0]

    return masked_rgb


def save_video_from_images3(rgb_images, mask_images, video_dir, fps=30):
    """
    Saves multiple video variations from lists of RGB images and corresponding mask images.
    Creates:
    1. "original_rgb.mp4": The original RGB video.
    2. "mask.mp4": Video of masked regions (from RGB) on a white background.
    3. "mask_rgb.mp4": Same as "mask.mp4" (seems redundant by current implementation).
    4. "mask_rgb_color.mp4": Video of RGB content with a colored (green) transparent overlay on masked regions.

    Args:
        rgb_images (list of np.ndarray): List of RGB frames ([H, W, 3], dtype=np.uint8, RGB order).
        mask_images (list of np.ndarray): List of mask frames ([H, W], binary 0 or 1, or object IDs).
        video_dir (str): Directory to save the output video files.
        fps (int, optional): Frames per second for the output videos. Defaults to 30.
    """
    # Ensure image lists are not empty.
    assert len(rgb_images) > 0 and len(mask_images) > 0, "Image lists cannot be empty"

    # Get height and width from the first RGB image.
    height, width, _ = rgb_images[0].shape
    # Create the output video directory if it doesn't exist.
    os.makedirs(video_dir, exist_ok=True)

    # Define paths for the output video files.
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")  # Masked content on white bg
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")  # Also masked content on white bg
    mask_rgb_color_video_path = os.path.join(video_dir, "mask_rgb_color.mp4")  # RGB with green overlay on mask

    # Initialize imageio video writers for each output video.
    rgb_writer = get_writer(rgb_video_path, fps=fps)
    mask_writer = get_writer(mask_video_path, fps=fps)
    mask_rgb_writer = get_writer(mask_rgb_video_path, fps=fps)
    mask_rgb_color_writer = get_writer(mask_rgb_color_video_path, fps=fps)

    # Iterate through each pair of RGB image and mask image.
    for rgb_img, mask_img in zip(rgb_images, mask_images):
        # --- Prepare `mask_img_white_bg` for "mask.mp4" and "mask_rgb.mp4" ---
        # Create a white image of the same size as rgb_img.
        mask_img_white_bg = np.ones_like(rgb_img) * 255  # White background (255, 255, 255)
        # Where `mask_img` is active (>0), copy the corresponding pixels from `rgb_img`.
        # This effectively shows the RGB content of the masked region on a white background.
        mask_img_white_bg[mask_img > 0] = rgb_img[mask_img > 0]

        # --- Prepare `colored_mask` for "mask_rgb_color.mp4" ---
        colored_mask = rgb_img.copy()  # Start with a copy of the original RGB image.
        overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Green color for the mask overlay (RGB format).
        alpha = 0.5  # Transparency level for the overlay (0.0 fully transparent, 1.0 fully opaque).

        # Apply the colored overlay where the mask is active.
        # This blends the `overlay_color` with the original `rgb_img` pixels in masked regions.
        colored_mask[mask_img > 0] = (
                alpha * overlay_color + (1 - alpha) * rgb_img[mask_img > 0]
        ).astype(np.uint8)  # Ensure result is uint8 for image display/saving.

        # Write the processed frames to their respective video files.
        rgb_writer.append_data(rgb_img)  # Original RGB frame.
        mask_writer.append_data(mask_img_white_bg)  # Frame for "mask.mp4".
        mask_rgb_writer.append_data(mask_img_white_bg)  # Frame for "mask_rgb.mp4".
        mask_rgb_color_writer.append_data(colored_mask)  # Frame for "mask_rgb_color.mp4".

    print(f'Videos saved to {video_dir}!')
    # Close all video writers to finalize the files.
    rgb_writer.close()
    mask_writer.close()
    mask_rgb_writer.close()
    mask_rgb_color_writer.close()


def save_video_from_images2(rgb_images, mask_images, video_dir, fps=30):
    """
    Saves RGB, mask, and masked RGB videos from lists of images using imageio.
    - "original_rgb.mp4": Original RGB video.
    - "mask.mp4": Grayscale mask video (mask is white, background is black).
    - "mask_rgb.mp4": Video of RGB content where mask is active, black otherwise.

    Args:
        rgb_images (list of np.ndarray): List of RGB frames ([H, W, 3]).
        mask_images (list of np.ndarray): List of binary mask frames ([H, W], values 0 or 1).
        video_dir (str): Directory to save videos.
        fps (int, optional): Frames per second. Defaults to 30.
    """
    assert len(rgb_images) > 0 and len(mask_images) > 0, "Image lists cannot be empty"

    height, width, _ = rgb_images[0].shape  # Get frame dimensions.
    os.makedirs(video_dir, exist_ok=True)  # Ensure output directory exists.

    # Define video file paths.
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")

    # Initialize imageio video writers.
    rgb_writer = get_writer(rgb_video_path, fps=fps)
    mask_writer = get_writer(mask_video_path, fps=fps)
    mask_rgb_writer = get_writer(mask_rgb_video_path, fps=fps)

    for rgb_img, mask_img_binary in zip(rgb_images, mask_images):  # mask_img_binary is 0 or 1
        # Prepare mask for "mask.mp4":
        # Convert binary mask (0 or 1) to grayscale image (0 for black, 255 for white).
        mask_img_grayscale = (mask_img_binary.astype(np.uint8) * 255)
        # Convert grayscale (single channel) to a 3-channel image (e.g., [H,W] -> [H,W,3]) for video writer.
        mask_img_rgb_compatible = np.stack([mask_img_grayscale] * 3, axis=-1)

        # Prepare masked RGB image for "mask_rgb.mp4":
        # `apply_mask_to_rgb` expects a mask where >0 is foreground. `mask_img_binary` (0/1) works.
        masked_rgb_content = apply_mask_to_rgb(rgb_img, mask_img_binary)

        # Append frames to the respective video writers.
        rgb_writer.append_data(rgb_img)  # Original RGB.
        mask_writer.append_data(mask_img_rgb_compatible)  # Grayscale mask (white on black background).
        mask_rgb_writer.append_data(masked_rgb_content)  # RGB content within mask, black elsewhere.

    print(f'video saved to {mask_rgb_video_path} (and others in {video_dir})!')
    # Close writers.
    rgb_writer.close()
    mask_writer.close()
    mask_rgb_writer.close()


def save_video_from_images(rgb_images, mask_images, video_dir, fps=30):
    """
    Saves RGB, mask, and masked RGB videos using OpenCV's VideoWriter.
    - "original_rgb.mp4": Original RGB video (assumes input `rgb_images` are BGR or handles conversion).
    - "mask.mp4": Grayscale mask video (mask is white, background is black), saved as BGR.
    - "mask_rgb.mp4": Video of BGR content where mask is active, black otherwise.

    Args:
        rgb_images (list of np.ndarray): List of frames, expected in BGR format by cv2.VideoWriter
                                        unless cv2.imread was used and color conversion was done prior.
                                        Shape [H, W, 3].
        mask_images (list of np.ndarray): List of binary mask frames ([H, W], values 0 or 1).
        video_dir (str): Directory to save videos.
        fps (int, optional): Frames per second. Defaults to 30.
    """
    assert len(rgb_images) > 0 and len(mask_images) > 0, "image list cannot be empty"

    height, width, _ = rgb_images[0].shape  # Get frame dimensions.
    os.makedirs(video_dir, exist_ok=True)  # Ensure output directory exists.

    # Define video file paths.
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")

    # Initialize OpenCV VideoWriters. 'mp4v' is a common codec for .mp4 files.
    rgb_out = cv2.VideoWriter(rgb_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    mask_out = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    mask_rgb_out = cv2.VideoWriter(mask_rgb_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for rgb_img_bgr, mask_img_binary in zip(rgb_images, mask_images):  # rgb_img_bgr assumed BGR
        # Prepare mask for "mask.mp4":
        # Convert binary mask (0/1) to grayscale (0/255).
        mask_img_grayscale = mask_img_binary.astype(np.uint8) * 255
        # Convert grayscale to BGR for VideoWriter.
        mask_img_bgr_compatible = cv2.cvtColor(mask_img_grayscale, cv2.COLOR_GRAY2BGR)

        # Prepare masked BGR image for "mask_rgb.mp4":
        # `apply_mask_to_rgb` will work fine with BGR input if it treats channels generically.
        masked_bgr_content = apply_mask_to_rgb(rgb_img_bgr, mask_img_binary)  # mask_img_binary is 0/1

        # Write frames to video files.
        rgb_out.write(rgb_img_bgr)  # Original BGR frame.
        mask_out.write(mask_img_bgr_compatible)  # BGR mask (white on black).
        mask_rgb_out.write(masked_bgr_content)  # BGR content within mask, black elsewhere.

    # Release VideoWriters to finalize files.
    rgb_out.release()
    mask_out.release()
    mask_rgb_out.release()
    # Note: A print statement for completion was in save_video_from_images2, but not here.


def save_obj_video(args, obj_id, video_name, frame_names):
    """
    Creates and saves a video specifically for a single object ID.
    It loads the mask for this `obj_id` for each frame, applies it to the corresponding RGB frame,
    and then saves the resulting sequence as a video.

    Args:
        args: Command-line arguments object, expected to have `output_mask_dir` (where per-object masks
              are stored, e.g., .../video_name/obj_id_str/) and `video_dir` (for RGB frames).
        obj_id (int): The ID of the object for which to create the video.
        video_name (str): Name of the video sequence (used for path construction).
        frame_names (list of str): List of frame names (without extension, e.g., "00001").
    """
    # Directory where masks for the specific `obj_id` are stored.
    # e.g., args.output_mask_dir/video_name/001/ (if obj_id is 1)
    obj_mask_dir = os.path.join(args.output_mask_dir, video_name, f'{obj_id:03d}')

    # Determine the image file extension (e.g., ".jpg", ".png") from the args.video_dir.
    rgb_p_example = os.listdir(args.video_dir)[0]  # Get the first file in the video directory.
    if os.path.splitext(rgb_p_example)[-1].lower() in [".jpg", ".jpeg", ".png"]:  # Check common image extensions.
        suffix = os.path.splitext(rgb_p_example)[-1]
    else:  # Fallback if the first file is not a common image type (e.g., could be a subdirectory or hidden file).
        suffix = os.path.splitext(os.listdir(args.video_dir)[1])[-1]  # Try the second file.

    obj_masks_binary = []  # List to store binary masks (0/1) for the current object across frames.
    obj_rgbs_bgr = []  # List to store corresponding RGB (actually BGR from cv2.imread) frames.

    # Iterate through each frame name provided.
    for frame_name in frame_names:
        # Path to the mask PNG for the current object and frame.
        # Assumes this PNG is specific to `obj_id` or contains it.
        mask_path = os.path.join(obj_mask_dir, f"{frame_name}.png")

        # Load the annotation PNG (which might be a combined mask or an object-specific one).
        input_mask_from_png, _ = load_ann_png(mask_path)  # input_mask_from_png is HxW, uint8

        # Extract individual object masks from the loaded PNG.
        # If the PNG in obj_mask_dir/frame_name.png is supposed to be purely for this obj_id,
        # it might be stored with ID '1' within that file, or directly as the object_id.
        per_obj_input_mask_dict = get_per_obj_mask(input_mask_from_png)  # obj_id -> boolean mask

        current_obj_binary_mask = None  # Initialize
        if len(per_obj_input_mask_dict) == 0:  # If no objects found in the mask file for this frame.
            # Create an all-zero (empty) mask for this frame.
            obj_mask_binary = np.zeros((input_mask_from_png.shape[0], input_mask_from_png.shape[1]), dtype=bool)
        else:
            # Attempt to get the mask for the specified `obj_id`.
            # The original code checks for ID '1'. This implies masks saved in per-object folders
            # might themselves be indexed with '1' if they represent a single object.
            # Let's keep the logic: try '1' first, as it seems to be an assumption.
            if 1 in per_obj_input_mask_dict:  # Often, a single object mask is saved with ID 1.
                obj_mask_binary = per_obj_input_mask_dict[1]  # Boolean mask [H,W]
            elif obj_id in per_obj_input_mask_dict:  # Fallback: check if the actual obj_id is a key.
                obj_mask_binary = per_obj_input_mask_dict[obj_id]
            else:  # If neither '1' nor `obj_id` is present, but other objects are.
                # This might indicate a mismatch or an empty mask for this specific object in this frame.
                print(
                    f"[Warning] Object ID {obj_id} (or default ID 1) not found in {mask_path}. Using empty mask for this frame.")
                obj_mask_binary = np.zeros((input_mask_from_png.shape[0], input_mask_from_png.shape[1]), dtype=bool)

        obj_masks_binary.append(obj_mask_binary.astype(np.uint8))  # Add the 0/1 mask to the list.

        # Load the corresponding RGB image using OpenCV (reads as BGR by default).
        rgb_path = os.path.join(args.video_dir, f'{frame_name}{suffix}')
        rgb_image_bgr = cv2.imread(rgb_path)  # Shape [H, W, 3], BGR order.
        obj_rgbs_bgr.append(rgb_image_bgr)  # Add BGR image to list.

    # Define the directory to save the output video for this object.
    # It replaces "sam2" in the path with "sam2_video" to create a parallel directory structure for videos.
    video_output_dir = obj_mask_dir.replace("sam2", "sam2_video")

    # Save the video using `save_video_from_images` (which expects BGR images and 0/1 masks).
    save_video_from_images(obj_rgbs_bgr, obj_masks_binary, video_output_dir)


def downsample_mask(mask_2d, scale_x, scale_y):
    """
    Downsamples a 2D mask using specified scaling factors for x and y dimensions.
    Uses nearest neighbor interpolation to preserve sharp boundaries in the mask.

    Args:
        mask_2d (np.ndarray): The input 2D mask [H, W].
        scale_x (float): Scaling factor for the width (e.g., 0.5 for half width).
        scale_y (float): Scaling factor for the height (e.g., 0.5 for half height).

    Returns:
        np.ndarray: The downsampled 2D mask.
    """
    # Calculate new width and height based on original dimensions and scaling factors.
    new_width = int(mask_2d.shape[1] * scale_x)
    new_height = int(mask_2d.shape[0] * scale_y)
    # Resize the mask using OpenCV's resize function.
    # cv2.INTER_NEAREST is crucial for masks to avoid introducing new intermediate values.
    downsampled_mask_result = cv2.resize(mask_2d, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return downsampled_mask_result


def find_pts_in_mask(mask, pts, downscale_factor=None):
    """
    Determines which points from a given set `pts` fall inside a given `mask`.
    Optionally, the mask and point coordinates can be downscaled before checking.

    Args:
        mask (np.ndarray): The mask, expected shape [1, H, W] (binary) or [H, W] (binary).
        pts (np.ndarray): Points to check, shape [N, 2] where columns are (x, y) coordinates.
        downscale_factor (tuple, optional): (scale_x, scale_y) to downscale mask and points.
                                           If None, no downscaling is performed. Defaults to None.

    Returns:
        np.ndarray: Boolean array of shape [N], where True indicates the corresponding point
                    from `pts` is inside the (potentially downscaled) mask.
    """
    # Ensure mask_2d is a 2D array [H, W]. If mask is [1,H,W], take the first slice.
    mask_2d = mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask

    if downscale_factor is not None:
        # If downscaling is specified:
        scale_x, scale_y = downscale_factor
        # Downsample the 2D mask.
        internal_downsampled_mask = downsample_mask(mask_2d, scale_x, scale_y)

        # Scale the point coordinates according to the downscale factor.
        scaled_x_coords = (pts[:, 0] * scale_x).astype(int)
        scaled_y_coords = (pts[:, 1] * scale_y).astype(int)

        # Clip scaled coordinates to be within the bounds of the downsampled mask to prevent out-of-bounds errors.
        scaled_x_coords = np.clip(scaled_x_coords, 0, internal_downsampled_mask.shape[1] - 1)
        scaled_y_coords = np.clip(scaled_y_coords, 0, internal_downsampled_mask.shape[0] - 1)

        # Check if scaled points fall within the downsampled mask by indexing.
        # `internal_downsampled_mask` contains 0 or 1 (or other values if not binary).
        # The result `in_mask_values` will contain these values at the point locations.
        in_mask_values = internal_downsampled_mask[scaled_y_coords, scaled_x_coords]
    else:
        # If no downscaling, use original mask and point coordinates.
        # Convert point coordinates to integers for indexing.
        x_coords, y_coords = pts[:, 0].astype(int), pts[:, 1].astype(int)

        # Clip coordinates to be within the bounds of the original mask.
        x_coords = np.clip(x_coords, 0, mask_2d.shape[1] - 1)
        y_coords = np.clip(y_coords, 0, mask_2d.shape[0] - 1)

        # Check if points fall within the original mask.
        in_mask_values = mask_2d[y_coords, x_coords]

    # Convert the result (which could be 0s and 1s, or other mask values) to a boolean array.
    # Any non-zero value in the mask at the point's location means the point is "in" the mask.
    in_mask_bool_array = in_mask_values.astype(bool)

    return in_mask_bool_array


def find_centroid_and_nearest_farthest(points):
    """
    Calculates the centroid of a 2D point set. Then, it finds:
    1. The point in the set closest to this centroid.
    2. The one or two points in the set farthest from this centroid.

    Args:
        points (np.ndarray): A 2D point set of shape (N, 2), where N is the number of points.

    Returns:
        tuple:
            - nearest_point (np.ndarray): Point of shape (1, 2) closest to the centroid.
            - nearest_index (int): Index (in the input `points` array) of the point closest to the centroid.
            - farthest_points (np.ndarray): Point(s) of shape (k, 2) (k=1 if N<3, else k=2) farthest from the centroid.
            - farthest_indices (list): Indices of the farthest point(s).
            - centroid (np.ndarray): Coordinates of the centroid, shape (2,).
    """
    # Calculate the centroid (mean of x and y coordinates across all points).
    centroid = np.mean(points, axis=0)  # Shape [2]

    # Calculate Euclidean distances from each point in the set to the calculated centroid.
    distances = np.linalg.norm(points - centroid, axis=1)  # Shape [N]

    # Find the point closest to the centroid.
    nearest_index = np.argmin(distances)  # Index of the point with the minimum distance.
    nearest_point = np.expand_dims(points[nearest_index], axis=0)  # Shape [1, 2] for consistency.

    num_points = len(points)  # Total number of points in the input set.

    # Determine the farthest point(s) based on the number of points.
    if num_points == 1:
        # If only one point, it's both the nearest and the farthest.
        farthest_points = nearest_point  # Shape [1, 2]
        farthest_indices = [nearest_index]  # List containing its index.
    elif num_points == 2:
        # If two points, the one not nearest is the farthest.
        farthest_indices = [i for i in range(num_points) if i != nearest_index]  # Index of the other point.
        farthest_points = np.expand_dims(points[farthest_indices[0]], axis=0)  # Shape [1, 2].
    else:  # num_points >= 3
        # If three or more points, find the two points that are farthest from the centroid.
        # `np.argsort` sorts distances and returns indices. `[-2:]` gets indices of the two largest distances.
        farthest_indices = np.argsort(distances)[-2:]
        farthest_points = points[farthest_indices]  # Shape [2, 2], containing the two farthest points.

    return nearest_point, nearest_index, farthest_points, farthest_indices, centroid


def is_subset(mask1, mask2, coverage_threshold=0.9):
    """
    Checks if `mask1` is substantially a subset of `mask2`.
    This is determined if the intersection area of `mask1` and `mask2` covers at least
    `coverage_threshold` of `mask1`'s total area.

    Args:
        mask1 (np.ndarray): First binary mask (boolean or 0/1 array).
        mask2 (np.ndarray): Second binary mask (boolean or 0/1 array).
        coverage_threshold (float, optional): Minimum ratio of (intersection_area / mask1_area)
                                             for `mask1` to be considered a subset. Defaults to 0.9.

    Returns:
        bool: True if `mask1` is considered a subset of `mask2` by the threshold, False otherwise.
    """
    # Calculate the area of mask1 (number of True or non-zero pixels).
    mask1_area = (mask1 > 0).sum()
    # Calculate the area of intersection between mask1 and mask2.
    intersection_area = np.logical_and(mask1 > 0, mask2 > 0).sum()

    # If mask1 has no area, it cannot be a subset in a meaningful way for this ratio.
    if mask1_area == 0:
        return False

    # Calculate the coverage ratio: (area of (mask1 AND mask2)) / (area of mask1).
    coverage_ratio = intersection_area / mask1_area
    # Return True if this ratio meets or exceeds the specified threshold.
    return coverage_ratio >= coverage_threshold


def record_potential_merge(potential_merges, obj_id1, obj_id2):
    """
    Records or increments the count of a potential merge event between two object IDs.
    The `potential_merges` dictionary stores these counts. E.g. `potential_merges[obj_id1][obj_id2]`
    would be the number of times `obj_id2` was found to potentially merge with `obj_id1`.

    Args:
        potential_merges (dict): A dictionary (often a `defaultdict(lambda: defaultdict(int))`)
                                 to store merge counts between pairs of object IDs.
        obj_id1 (int): ID of the first object in the pair.
        obj_id2 (int): ID of the second object in the pair.
    """
    # If obj_id1 is not yet a key in potential_merges, initialize its value as a defaultdict(int).
    # This allows easy incrementing of counts for obj_id2 associated with obj_id1.
    if obj_id1 not in potential_merges:
        potential_merges[obj_id1] = defaultdict(int)
    # Increment the count for the pair (obj_id1, obj_id2).
    potential_merges[obj_id1][obj_id2] += 1


def compute_iou(mask1, mask2):
    """
    Computes the Intersection over Union (IoU) between two binary masks.

    Args:
        mask1 (np.ndarray): First binary mask (boolean or 0/1 array).
        mask2 (np.ndarray): Second binary mask (boolean or 0/1 array).

    Returns:
        float: The IoU score. Returns 0 if the union is 0 (to avoid division by zero).
    """
    # Calculate intersection: number of pixels where both masks are True (or >0).
    intersection = np.logical_and(mask1, mask2).sum()
    # Calculate union: number of pixels where at least one mask is True (or >0).
    union = np.logical_or(mask1, mask2).sum()
    # Compute IoU. If union is 0 (both masks are empty), IoU is 0.
    return intersection / union if union > 0 else 0


def analyze_frame_merges(video_segments, iou_threshold=0.5):
    """
    Analyzes per-frame segmentation masks across a video to identify groups of original object IDs
    that likely correspond to the same actual object and should be merged.
    Merging decisions are based on frequent high IoU or subset relationships between masks of different IDs.

    Args:
        video_segments (dict): A dictionary where keys are frame indices (int) and values are
                             `per_obj_output_mask` (a dict: original_obj_id -> binary_mask for that frame).
        iou_threshold (float, optional): IoU threshold. If IoU > this, or subset condition met,
                                       objects are considered for merging in that frame. Defaults to 0.5.

    Returns:
        dict: `result` - A dictionary mapping new, unique merged object IDs (starting from 1) to
                       groups (sorted lists) of original object IDs that belong to that merged object.
                       Example: {1: [orig_id1, orig_id5], 2: [orig_id2], 3: [orig_id3, orig_id4, orig_id7]}
    """
    potential_merges = {}  # Stores counts: {obj_id1: {obj_id2: count_of_frames_they_should_merge}}
    frame_count = len(video_segments)  # Total number of frames that have segmentation data.

    # Iterate through each frame's segmentation data.
    for per_obj_output_mask_in_frame in video_segments.values():  # per_obj_output_mask_in_frame is {obj_id: mask}
        visited_pairs_in_this_frame = set()  # To avoid redundant (id1,id2) and (id2,id1) checks within the same frame.

        # Compare all unique pairs of object masks within the current frame.
        list_of_obj_ids_in_frame = list(per_obj_output_mask_in_frame.keys())
        for i in range(len(list_of_obj_ids_in_frame)):
            obj_id1 = list_of_obj_ids_in_frame[i]
            mask1 = per_obj_output_mask_in_frame[obj_id1]
            for j in range(i + 1, len(list_of_obj_ids_in_frame)):  # Avoid self-comparison and duplicate pairs
                obj_id2 = list_of_obj_ids_in_frame[j]
                mask2 = per_obj_output_mask_in_frame[obj_id2]

                # Pair (obj_id1, obj_id2) is now unique for this frame.
                # (Original code had a more complex visited check, this nested loop structure is simpler for pairs)

                # Calculate IoU between the two masks.
                iou = compute_iou(mask1, mask2)

                # If IoU is high, or one mask is a subset of the other, record a potential merge for this frame.
                if iou > iou_threshold or is_subset(mask1, mask2) or is_subset(mask2, mask1):
                    # Record potential merge symmetrically for easier lookup later.
                    record_potential_merge(potential_merges, obj_id1, obj_id2)
                    record_potential_merge(potential_merges, obj_id2, obj_id1)
                # No need to add to visited_pairs_in_this_frame due to loop structure for unique pairs (i, j)

    # Determine final merges based on the frequency of potential merges across all frames.
    final_merges_graph = defaultdict(
        list)  # Adjacency list for graph: {obj_id1: [list of obj_ids to merge with obj_id1]}

    # Collect all unique object IDs that appeared in any merge consideration.
    all_considered_obj_ids = set(potential_merges.keys())
    for obj_id1 in potential_merges:
        all_considered_obj_ids.update(potential_merges[obj_id1].keys())

    for obj_id1 in all_considered_obj_ids:
        if obj_id1 in potential_merges:  # Check if obj_id1 was a primary key in potential_merges
            for obj_id2, count in potential_merges[obj_id1].items():
                # If objects overlapped/subset significantly in at least 30% of frames (where frame_count is total frames with any seg data)
                # This condition decides if a strong merge link exists.
                if frame_count > 0 and (count / frame_count >= 0.3):
                    if obj_id2 not in final_merges_graph[obj_id1]:
                        final_merges_graph[obj_id1].append(obj_id2)

    # Group object IDs into unique sets (connected components in the `final_merges_graph`).
    # Each set represents a single merged object.
    groups = []  # List of sets, where each set contains original IDs to be merged.
    visited_obj_ids_for_grouping = set()  # Keep track of IDs already assigned to a group.

    # Get all unique object IDs present in the original video_segments.
    all_obj_ids_in_video = set()
    for frame_data in video_segments.values():
        all_obj_ids_in_video.update(frame_data.keys())
    # Also include IDs from final_merges_graph keys if they somehow missed above (e.g. if an object was always merged into).
    all_obj_ids_in_video.update(final_merges_graph.keys())

    for obj_id_start_node in sorted(
            list(all_obj_ids_in_video)):  # Iterate sorted for deterministic group numbering later.
        if obj_id_start_node not in visited_obj_ids_for_grouping:
            current_group = set()  # Start a new group for this component.
            # Use a queue for Breadth-First Search (BFS) to find all connected IDs.
            queue = [obj_id_start_node]
            visited_obj_ids_for_grouping.add(obj_id_start_node)  # Mark start node as visited.

            head = 0
            while head < len(queue):  # While queue is not empty
                current_obj_in_bfs = queue[head]
                head += 1
                current_group.add(current_obj_in_bfs)  # Add current object to the group.

                # Add all its neighbors (objects it should merge with) from the graph to the queue.
                for neighbor_obj_id in final_merges_graph.get(current_obj_in_bfs, []):
                    if neighbor_obj_id not in visited_obj_ids_for_grouping:
                        visited_obj_ids_for_grouping.add(neighbor_obj_id)
                        queue.append(neighbor_obj_id)
            groups.append(sorted(list(current_group)))  # Add the discovered group (sorted list) to the list of groups.

    # The original code includes unmerged objects as individual groups, which is naturally handled by the
    # connected components algorithm above: if an object has no merge links, it forms a group of size 1.

    # Map each group (list of original IDs) to a new, unique merged object ID (starting from 1).
    result_merged_id_to_orig_ids = {i + 1: group for i, group in enumerate(groups)}

    return result_merged_id_to_orig_ids


def merge_masks(video_segments, merge_groups):
    """
    Merges per-object masks within each frame according to the `merge_groups` mapping.
    For each new merged object ID, it combines (logical OR) the masks of all its constituent original object IDs.

    Args:
        video_segments (dict): Original per-frame segmentation data.
                             Structure: {frame_idx: {original_obj_id: binary_mask_array}}.
        merge_groups (dict): Mapping from new merged object IDs to lists/sets of original object IDs.
                           Structure: {new_merged_obj_id: [original_obj_id1, original_obj_id2, ...]}.

    Returns:
        dict: `merged_video_segments` - Per-frame segmentation with masks merged.
                                    Structure: {frame_idx: {new_merged_obj_id: combined_binary_mask_array}}.
    """
    merged_video_segments_output = {}  # Initialize dictionary to store the results with merged masks.

    # Iterate through each frame in the original `video_segments`.
    for out_frame_idx, per_obj_output_mask_in_frame in video_segments.items():
        # `per_obj_output_mask_in_frame` is {original_obj_id: binary_mask} for the current frame.
        merged_masks_for_this_frame = {}  # Store merged masks for the current frame.

        # Iterate through each new (merged) object ID and its group of original IDs.
        for new_merged_obj_id, list_of_original_obj_ids in merge_groups.items():
            combined_mask_for_new_id = None  # Initialize the combined mask for this new_merged_obj_id.

            # Iterate through each original object ID that belongs to the current `new_merged_obj_id`.
            for original_obj_id in list_of_original_obj_ids:
                # Retrieve the mask for this `original_obj_id` in the current frame, if it exists.
                mask_of_original_obj = per_obj_output_mask_in_frame.get(original_obj_id)

                if mask_of_original_obj is not None:  # If the original object has a mask in this frame.
                    # If this is the first mask being added to `combined_mask_for_new_id`, initialize it.
                    if combined_mask_for_new_id is None:
                        combined_mask_for_new_id = np.copy(mask_of_original_obj)
                    else:
                        # Otherwise, merge (logical OR) the current `mask_of_original_obj`
                        # with the `combined_mask_for_new_id` accumulated so far.
                        combined_mask_for_new_id = np.logical_or(combined_mask_for_new_id, mask_of_original_obj)
                        # Using np.maximum would also work if masks are strictly 0/1.
                        # combined_mask_for_new_id = np.maximum(combined_mask_for_new_id, mask_of_original_obj)

            # If a `combined_mask_for_new_id` was formed (i.e., at least one original mask was found),
            # assign it to the `new_merged_obj_id` in the results for this frame.
            if combined_mask_for_new_id is not None:
                merged_masks_for_this_frame[new_merged_obj_id] = combined_mask_for_new_id

        # Store the collection of all merged masks for the current `out_frame_idx`.
        merged_video_segments_output[out_frame_idx] = merged_masks_for_this_frame

    return merged_video_segments_output


def main(args):
    """
    Main function to perform motion segmentation using SAM2 on video sequences.
    It involves:
    1. Loading pre-computed point trajectories.
    2. (If not `args.cal_only`) Iteratively segmenting objects using SAM2, prompted by these trajectories.
       - This includes selecting good frames for prompting and propagating masks.
       - Segmented objects are stored in `memory_dict`.
       - Masks from propagation are collected in `video_segments`.
    3. (If not `args.cal_only`) Analyzing and merging masks that likely belong to the same object.
    4. (If not `args.cal_only`) Saving these initial/merged masks as PNG files.
    5. Loading the saved (or pre-existing if `args.cal_only`) masks.
    6. Generating and saving output videos: original RGB, final mask, and masked RGB.
    """
    # --- Setup and Configuration ---
    # `video_name` is derived from the basename of `args.dynamic_dir` (e.g., "farm01").
    video_name = os.path.basename(args.dynamic_dir)

    # Determine `output_mask_dir` for initial mask predictions.
    # This path is where raw segmentation masks (before final video generation) will be stored or read from.
    if "baseline" in args.output_mask_dir:  # Special handling if "baseline" is in the output path.
        output_mask_dir_for_initial_preds = args.output_mask_dir
        # output_mask_dir_for_initial_preds = os.path.dirname(args.output_mask_dir) # This commented line suggests an alternative.
    else:  # Standard case: create an "initial_preds" subdirectory within `args.output_mask_dir`.
        output_mask_dir_for_initial_preds = os.path.join(args.output_mask_dir, "initial_preds")

    # --- Segmentation Phase (skipped if args.cal_only is True) ---
    if not args.cal_only:
        # --- SAM2 Predictor Initialization ---
        checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"  # Path to SAM2 model weights.
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Path to SAM2 model configuration file.
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)  # Instantiate the predictor.

        # --- Frame Information ---
        # Get a sorted list of frame names (e.g., "00001", "00002") from the `args.video_dir`.
        # Assumes frames are image files (jpg, jpeg, png).
        frame_names = sorted([
            os.path.splitext(p)[0]  # Get filename without extension.
            for p in os.listdir(args.video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            # Check common extensions, case-insensitive.
        ])
        # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0])) # Alternative sort if frame names are numeric strings.

        # Get image dimensions (width, height) from the first frame in the video directory.
        img_ext_example = os.listdir(args.video_dir)[0]  # Name of the first file.
        img_path_example = os.path.join(args.video_dir, img_ext_example)
        with Image.open(img_path_example) as img:  # Use PIL to open and get dimensions.
            width, height = img.size  # PIL's size is (width, height).

        # --- Load Trajectory Data ---
        # `traj` shape: [C, N, T] (e.g., [2_coords, Num_points, Num_frames]).
        # `visible_mask` shape: [N, T].
        # `confidences` shape: [N, T].
        traj_loaded, visible_mask_loaded, confidences_loaded = load_data(args.dynamic_dir)
        _, N_points, T_frames = traj_loaded.shape  # C (coords, usually 2), N (num_points), T (num_frames).

        # --- Parameters for `process_invisible_traj` ---
        # `q_ts` was originally used to define max_iterations; this seems simplified now.
        # max_iterations for the refinement loop in `process_invisible_traj`.
        # Original code had a q_ts list here, possibly for another purpose or for this calculation.
        # This calculation seems to derive it from T_frames directly or a fixed range.
        max_iterations_for_pit = max(T_frames // 16, 5)  # Heuristic: at least 5, or related to video length.
        max_iterations_for_pit = min(max_iterations_for_pit, 10)  # Cap at 10 iterations.

        downscale_factor_for_pit = None  # No downscaling by default.

        # --- SAM2 Inference and Initial Segmentation ---
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):  # Context for inference.
            # Initialize predictor state for the current video.
            current_predictor_state = predictor.init_state(args.video_dir)
            # Ensure the output directory for initial predictions exists.
            os.makedirs(output_mask_dir_for_initial_preds, exist_ok=True)

            # `process_invisible_traj` identifies initial object segments.
            # `traj_loaded` is [C,N,T]. `process_invisible_traj` receives it as is.
            # Internally, `process_invisible_traj` transposes it to [N,C,T] for `process_points_with_memory`.
            memory_dict_initial_segments = process_invisible_traj(
                traj_loaded, visible_mask_loaded, confidences_loaded, current_predictor_state, predictor,
                dilation_size=6, max_iterations=max_iterations_for_pit, timestep=T_frames,
                downscale_factor=downscale_factor_for_pit
            )

        # `video_segments` will store per-frame, per-object masks after propagation.
        # Structure: {frame_idx: {obj_id: binary_mask_array}}
        video_segments_after_propagation = {}
        basename_video_dir = os.path.basename(args.video_dir)  # Basename of the video directory.

        if len(memory_dict_initial_segments) == 0:  # If no objects were initially segmented.
            print(f"[WARN]: {video_name} (sequence name) don't have dynamic objects based on process_invisible_traj!")
            # The original script exits here. This means if no initial dynamic objects are found,
            # no masks or videos will be produced by the 'if not args.cal_only' block.
            exit()

        # --- Mask Propagation for Each Segmented Object ---
        # For each object found by `process_invisible_traj` (stored in `memory_dict_initial_segments`):
        for obj_id_from_mem, object_package in memory_dict_initial_segments.items():
            # Define query frames for prompting during propagation. Start with the object's canonical time.
            q_ts_propagation = list(range(0, T_frames, 16))  # Sample every 16th frame.
            predictor.reset_state(current_predictor_state)  # Reset predictor state for each new object propagation.

            canonical_time = object_package['time']  # The frame where this object was initially defined.
            if canonical_time in q_ts_propagation:
                q_ts_propagation.remove(canonical_time)
            q_ts_propagation.insert(0, canonical_time)  # Ensure canonical_time is the first prompt frame.

            # Trajectories, confidences, visibility for points belonging to this specific object.
            pts_trajs_for_obj = object_package['pts_trajs']  # Shape [N_obj_pts, C_coords, T_frames]
            confi_trajs_for_obj = object_package['confi_trajs']  # Shape [N_obj_pts, T_frames]
            vis_trajs_for_obj = object_package['vis_trajs']  # Shape [N_obj_pts, T_frames]

            require_reverse_propagation = True  # Flag to decide if backward propagation is needed.

            # Iterate through query frames (q_ts_propagation) to add/refine prompts for SAM2.
            for t_query in q_ts_propagation:
                # Get points of the current object visible at this query frame `t_query`.
                pts_at_t_query = pts_trajs_for_obj[:, :, t_query]  # Points [N_obj_pts, C_coords]
                vis_at_t_query = vis_trajs_for_obj[:, t_query]  # Visibility [N_obj_pts]

                visible_points_for_prompt = pts_at_t_query[vis_at_t_query]  # Filtered [N_visible_obj_pts, C_coords]

                if visible_points_for_prompt.shape[0] == 0:  # If no points visible for this object at t_query.
                    print(f'Propagate in time {t_query} for object {obj_id_from_mem}: No visible points for prompt.')
                    continue  # Skip to next query frame.

                if t_query == 0:  # If prompting at the very first frame.
                    require_reverse_propagation = False  # Backward propagation might not be necessary.

                # Select prompt points for SAM2 (e.g., densest point, or centroid + farthest).
                # Original code uses find_dense_pts first, then falls back. Here it's find_dense_pts.
                # The inner logic has: nearest_point, _, farthest_points, _, _ = find_centroid_and_nearest_farthest(visible_points)
                # prompt_points = np.concatenate((nearest_point, farthest_points), axis=0)
                # THEN: print('visible_points', visible_points); prompt_points = find_dense_pts(visible_points)
                # This means `find_dense_pts` is the one actually used for the first attempt.
                # print('visible_points', visible_points_for_prompt) # Original debug print
                prompt_points_sam = find_dense_pts(visible_points_for_prompt)
                # print('prompt_points', prompt_points_sam) # Original debug print

                num_prompt_pts = prompt_points_sam.shape[0]
                labels_sam = np.ones(num_prompt_pts, dtype=np.int32)  # All foreground prompts.

                # Add new points/prompts to SAM2 for the current object (`obj_id_from_mem`) at `t_query`.
                _, _, out_mask_logits_prompted = predictor.add_new_points_or_box(
                    inference_state=current_predictor_state,
                    frame_idx=t_query,
                    obj_id=obj_id_from_mem,  # Use the object's actual ID.
                    points=prompt_points_sam,
                    labels=labels_sam,
                )

                # --- Fallback prompt strategy if initial prompt is insufficient ---
                # Binarize the mask generated by the prompt.
                prompt_generated_mask_binary = (out_mask_logits_prompted[0] > 0.0).cpu().numpy()
                # save_mask_with_points((prompt_mask).cpu().numpy(), visible_points, "test.png") # Original debug save

                # Check how many of the object's known visible points are covered by this generated mask.
                points_in_generated_mask_check = find_pts_in_mask(prompt_generated_mask_binary,
                                                                  visible_points_for_prompt)
                # If coverage is less than 70% of the object's known points, try a different prompt strategy.
                if points_in_generated_mask_check.sum() < (visible_points_for_prompt.shape[0] * 0.7):
                    # Fallback strategy: use centroid and farthest points as prompts.
                    nearest_pt_fallback, _, farthest_pts_fallback, _, _ = find_centroid_and_nearest_farthest(
                        visible_points_for_prompt)
                    prompt_points_fallback = np.concatenate((nearest_pt_fallback, farthest_pts_fallback), axis=0)
                    # prompt_points = find_dense_pts(visible_points) # This line was commented out in original, but the logic above was active

                    num_prompt_pts_fallback = prompt_points_fallback.shape[0]
                    labels_fallback = np.ones(num_prompt_pts_fallback, dtype=np.int32)
                    # Re-prompt SAM2 with the fallback points.
                    _, _, out_mask_logits_prompted = predictor.add_new_points_or_box(  # Overwrites previous logits
                        inference_state=current_predictor_state,
                        frame_idx=t_query,
                        obj_id=obj_id_from_mem,
                        points=prompt_points_fallback,
                        labels=labels_fallback,
                    )
                    # save_mask_with_points((out_mask_logits[0] > 0.0).cpu().numpy(), prompt_points, "test2.png") # Original debug save

            # --- Propagate masks forward in the video for the current object ---
            for out_frame_idx_fwd, out_obj_ids_fwd, out_mask_logits_fwd in predictor.propagate_in_video(
                    current_predictor_state):
                if out_frame_idx_fwd not in video_segments_after_propagation:
                    video_segments_after_propagation[out_frame_idx_fwd] = {}  # Initialize dict for this frame if new.
                for i, current_propagated_obj_id_fwd in enumerate(out_obj_ids_fwd):
                    # Binarize the propagated mask for this object in this frame.
                    current_propagated_mask_fwd = (out_mask_logits_fwd[i] > 0.0).cpu().numpy()
                    if current_propagated_obj_id_fwd not in video_segments_after_propagation[out_frame_idx_fwd]:
                        # If this object ID isn't yet in this frame's dict, add it.
                        video_segments_after_propagation[out_frame_idx_fwd][
                            current_propagated_obj_id_fwd] = current_propagated_mask_fwd
                    else:
                        # If already present, combine (logical OR) the new mask with the existing one.
                        # This handles cases where multiple propagation steps might contribute to the same object's mask.
                        video_segments_after_propagation[out_frame_idx_fwd][current_propagated_obj_id_fwd] = \
                            np.logical_or(
                                video_segments_after_propagation[out_frame_idx_fwd][current_propagated_obj_id_fwd],
                                current_propagated_mask_fwd)

            # --- Propagate masks backward if needed ---
            if require_reverse_propagation:
                for out_frame_idx_rev, out_obj_ids_rev, out_mask_logits_rev in predictor.propagate_in_video(
                        current_predictor_state, reverse=True):
                    if out_frame_idx_rev not in video_segments_after_propagation:
                        video_segments_after_propagation[out_frame_idx_rev] = {}
                    for i, current_propagated_obj_id_rev in enumerate(out_obj_ids_rev):
                        current_propagated_mask_rev = (out_mask_logits_rev[i] > 0.0).cpu().numpy()
                        if current_propagated_obj_id_rev not in video_segments_after_propagation[out_frame_idx_rev]:
                            video_segments_after_propagation[out_frame_idx_rev][
                                current_propagated_obj_id_rev] = current_propagated_mask_rev
                        else:
                            video_segments_after_propagation[out_frame_idx_rev][current_propagated_obj_id_rev] = \
                                np.logical_or(
                                    video_segments_after_propagation[out_frame_idx_rev][current_propagated_obj_id_rev],
                                    current_propagated_mask_rev)

        # --- Mask Merging and Saving ---
        # Sort `video_segments_after_propagation` by frame index for ordered processing.
        video_segments_sorted = dict(sorted(video_segments_after_propagation.items()))

        # Directory to save the initial (pre-merge or merged by obj_id from propagation) masks.
        # e.g., output_mask_dir_for_initial_preds/video_basename/
        save_dirname_for_masks = os.path.join(output_mask_dir_for_initial_preds, basename_video_dir)
        if os.path.exists(save_dirname_for_masks):
            shutil.rmtree(save_dirname_for_masks)  # Remove old directory if it exists, to ensure clean output.

        # Analyze frames to find groups of object IDs that should be merged into single objects.
        # `final_merge_groups` will be {new_merged_id: [list_of_original_ids_to_merge]}.
        final_merge_groups = analyze_frame_merges(video_segments_sorted, iou_threshold=0.9)
        # Merge the masks in `video_segments_sorted` based on these `final_merge_groups`.
        merged_video_segments_data = merge_masks(video_segments_sorted, final_merge_groups)

        # Save the (now merged by `analyze_frame_merges`) masks as PNG files.
        # Each frame will have one PNG containing all merged objects, distinguished by pixel values (new merged IDs).
        for out_frame_idx_save, per_obj_mask_to_save in merged_video_segments_data.items():
            # `per_obj_mask_to_save` is {new_merged_id: combined_binary_mask}.
            save_multi_masks_to_dir(
                output_mask_dir=output_mask_dir_for_initial_preds,  # Save to the determined output directory.
                video_name=video_name,  # Sequence name (e.g., "farm01").
                frame_name=frame_names[out_frame_idx_save],  # Actual frame name string (e.g., "00001").
                per_obj_output_mask=per_obj_mask_to_save,  # Dict of {new_id: mask} for this frame.
                height=height,
                width=width,
                per_obj_png_file=False,  # False: save a single combined mask per frame.
                output_palette=DAVIS_PALETTE,  # Use the predefined palette.
            )

    # --- Post-processing and Video Saving Phase ---
    # This phase can run even if `args.cal_only` is True, using masks already generated and saved.

    # Get sorted list of frame names (again, in case `args.cal_only` was True and they weren't loaded).
    frame_names_for_video_gen = sorted([
        os.path.splitext(p)[0]
        for p in os.listdir(args.video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ])
    # frame_names_for_video_gen.sort(key=lambda p: int(os.path.splitext(p)[0])) # Alt sort.

    # `sam_dir` points to the directory where combined masks (one PNG per frame) are stored/expected.
    # This is the output of the `save_multi_masks_to_dir` call from the segmentation phase.
    # It should be `output_mask_dir_for_initial_preds` / `video_name`.
    sam_dir_containing_masks = os.path.join(output_mask_dir_for_initial_preds, video_name)
    # obj_ids =  sorted(os.listdir(sam_dir)) # This line seems unused, `sam_dir` contains frame PNGs not obj_id folders here.

    # Determine path for final results (e.g., output videos).
    # This creates a "final_res" directory, possibly parallel to "initial_preds".
    if "baseline" in args.output_mask_dir:
        dynamic_path_for_final_results = os.path.join(os.path.dirname(args.output_mask_dir), "final_res", video_name)
    else:
        dynamic_path_for_final_results = os.path.join(args.output_mask_dir, "final_res", video_name)
    os.makedirs(dynamic_path_for_final_results, exist_ok=True)  # Ensure this directory exists.

    # `seq_mask_dir` should point to where the per-frame combined PNG masks are.
    # This logic matches `sam_dir_containing_masks` definition.
    if "baseline" in args.output_mask_dir:
        seq_mask_dir_for_loading = os.path.join(args.output_mask_dir, video_name)
    else:
        seq_mask_dir_for_loading = os.path.join(args.output_mask_dir, "initial_preds", video_name)

    # Get list of all mask PNG files from `seq_mask_dir_for_loading`.
    mask_png_paths_for_video = sorted(glob(os.path.join(seq_mask_dir_for_loading, "*.png")))
    # These commented lines suggest jpg/jpeg masks were also considered, but typically masks are PNG.
    # mask_png_paths_for_video += sorted(glob(os.path.join(seq_mask_dir_for_loading, "*.jpg")))
    # mask_png_paths_for_video += sorted(glob(os.path.join(seq_mask_dir_for_loading, "*.jpeg")))

    all_mask_tensors_for_video = []  # List to hold loaded masks as tensors.
    # If image dimensions (height, width) were not obtained (e.g., `cal_only`=True), get them now.
    # This check `if 'height' not in locals()` might be fragile if `height` was defined in a different scope.
    # A more robust way would be to ensure they are always passed or re-fetched if needed.
    # For now, assume it works as intended by the original author.
    if 'height' not in locals() or 'width' not in locals():  # Check if height/width are defined
        first_img_path_for_dims = os.path.join(args.video_dir, os.listdir(args.video_dir)[0])
        with Image.open(first_img_path_for_dims) as img_for_dims:
            width, height = img_for_dims.size

    for d_path_mask_png in mask_png_paths_for_video:  # Iterate through found mask PNG paths.
        # d_path = os.path.join(sam_dir, f'{frame_name}.png') # This was an example, loop uses globbed paths.
        if not os.path.exists(d_path_mask_png):
            # This case should ideally not be hit if glob returns existing files.
            # If a frame's mask is missing, create an empty mask.
            mask_img_loaded = np.zeros((height, width), dtype=np.uint8)
            print(f"[Warning] Mask file {d_path_mask_png} expected but not found. Using empty mask.")
        else:
            mask_img_loaded, _ = load_ann_png(d_path_mask_png)  # Load combined mask PNG for the frame.

        # Convert the loaded mask (where pixel values are object IDs) to a binary mask (0 for background, 1 for any foreground).
        mask_img_binary = (mask_img_loaded > 0).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_img_binary)  # Convert to PyTorch tensor.
        all_mask_tensors_for_video.append(mask_tensor)

    # `predict_mask` is a stacked tensor of all binary masks [T, H, W]. (This variable is not used further.)
    # predict_mask = torch.stack(all_mask_tensors_for_video, dim=0) if all_mask_tensors_for_video else torch.empty(0)

    # --- Prepare RGB frames for video generation ---
    # Determine image suffix (e.g., ".jpg") for RGB frames in `args.video_dir`.
    rgb_p_example_for_suffix = os.listdir(args.video_dir)[0]
    if os.path.splitext(rgb_p_example_for_suffix)[-1].lower() in [".jpg", ".jpeg", ".png"]:
        suffix_rgb = os.path.splitext(rgb_p_example_for_suffix)[-1]
    else:  # Fallback if first file is not a common image type.
        suffix_rgb = os.path.splitext(os.listdir(args.video_dir)[1])[-1]

    rgbs_for_video = []  # List to store RGB frames (as numpy arrays, RGB order).
    for frame_name_rgb in frame_names_for_video_gen:  # Use the same sorted list of frame names.
        rgb_path = os.path.join(args.video_dir, f'{frame_name_rgb}{suffix_rgb}')
        rgb_image_bgr = cv2.imread(rgb_path)  # OpenCV reads images in BGR order.
        if rgb_image_bgr is None:
            print(f"[Error] Could not read RGB image {rgb_path}. Using a black frame.")
            # Create a black frame if image loading fails, to maintain sequence length.
            rgb_image_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # Black RGB frame
        else:
            rgb_image_rgb = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB.
        rgbs_for_video.append(rgb_image_rgb)

    # --- Save final output videos ---
    # `save_dir_for_videos` is where "original_rgb.mp4", "mask.mp4", etc., will be saved.
    # e.g., .../final_res/video_name/video/
    save_dir_for_videos = os.path.join(dynamic_path_for_final_results, "video")
    # Convert list of mask tensors back to list of numpy arrays for the video saving function.
    np_masks_for_video = [mask_t.numpy() for mask_t in all_mask_tensors_for_video]

    # Ensure number of RGB frames matches number of mask frames before saving videos.
    if len(rgbs_for_video) == len(np_masks_for_video) and len(rgbs_for_video) > 0:
        save_video_from_images3(rgbs_for_video, np_masks_for_video, save_dir_for_videos)
    else:
        print(
            f"[Warning] Mismatch in number of RGB frames ({len(rgbs_for_video)}) and mask frames ({len(np_masks_for_video)}). Cannot save videos.")
        if not rgbs_for_video: print("No RGB frames loaded.")
        if not np_masks_for_video: print("No mask frames loaded/generated.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Train trajectory-based motion segmentation network OR Run SAM2-based video motion segmentation.',
        # Description seems to cover two possibilities.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows default values in help message.
    )
    # Directory for outputting segmentation masks.
    parser.add_argument('--output_mask_dir', type=str, default="exp_res/sam_res/main/FBMS-moving",
                        help="Base directory to save segmentation masks and final video results.")
    # Directory containing input video frames (as individual image files).
    parser.add_argument('--video_dir', type=str, default="current-data-dir/baseline/FBMS-moving/Testset/farm01",
                        help="Directory containing the video frames (e.g., JPG, PNG sequences).")
    # Directory containing pre-calculated dynamic trajectories and related data (.npy files).
    parser.add_argument('--dynamic_dir', type=str, default="exp_res/tracks_seg/main/FBMS-moving/farm01",
                        help="Directory with dynamic_traj.npy, dynamic_visibility.npy from motion segmentation/tracking.")
    # Flag to enable saving of intermediate visualization images (e.g., masks with points).
    parser.add_argument('--vis', action="store_true",
                        help="Enable saving of visualization images during processing (e.g., prompt.png).")
    # Directory for ground truth masks (typically used for evaluation, not directly in this script's main processing flow).
    parser.add_argument('--gt_dir', type=str, default="current-data-dir/davis/DAVIS/Annotations/480p/a",
                        help="Directory of ground truth annotations (for evaluation purposes, not used in core script logic).")
    # Flag to skip the SAM2 segmentation step and only run post-processing/video generation
    # using masks assumed to be already present in `output_mask_dir`.
    parser.add_argument('--cal_only', action="store_true",
                        help="If set, skips SAM2 segmentation and only performs calculations/video saving based on existing masks in output_mask_dir.")

    args = parser.parse_args()  # Parse command-line arguments into the `args` object.

    main(args)  # Call the main processing function with the parsed arguments.