import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from rtree import index
from tqdm import tqdm
import os

import torch.nn as nn
import torch

from eval.model import SegNet, DEFAULT_CKPT_PATH

class Seger(nn.Module):
    def __init__(self, ckpt_path=None, bg_thres=200, cuda_device_id:int=0):
        super(Seger, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
        print(f"=== CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')} ===")

        if ckpt_path is None:
            ckpt_path = DEFAULT_CKPT_PATH
        self.seg_net = SegNet(ckpt_path, bg_thres)
        
        # border width
        self.border_width = 4
        self.patch_size = 128
        self.overlap = 16
        self.bg_thres = bg_thres
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, img_patch:np.ndarray):
        pass

    def get_ROI(self, shape):
        overlap = self.overlap
        patch_size = self.patch_size
        D, H, W = shape
        def cal_index(idx, BORDER):
            if idx + patch_size >= BORDER:
                start = BORDER - patch_size
                end = BORDER
                cut_left = idx + overlap//2 - start
                cut_right = 0
            else:
                start = idx
                end = start + patch_size
                cut_left = 0 if idx==0 else overlap//2
                cut_right = overlap//2
            return start, end, cut_left, cut_right
        ROI = []
        for d_idx in range(0, D, patch_size-overlap):
            d_s, d_e, d_cL, d_cR = cal_index(d_idx, D)
            for h_idx in range(0, H, patch_size-overlap):
                h_s, h_e, h_cL, h_cR = cal_index(h_idx, H)
                for w_idx in range(0, W, patch_size-overlap):
                    w_s, w_e, w_cL, w_cR = cal_index(w_idx, W)        
                    ROI.append([[d_s,d_e,d_cL,d_cR], [h_s,h_e,h_cL,h_cR], [w_s,w_e,w_cL,w_cR]])
                    if w_e==W: break
                if h_e==H: break
            if d_e==D: break
        return ROI

    def preprocess(self,img:torch.Tensor, percentiles=[0.01, 1.0]):
        # input img: tensor [0,65535]
        # output img: tensor [0,1]
        img = torch.clip(img, min=self.bg_thres, max=None) - self.bg_thres
        flattened_arr = torch.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))-1
        if flattened_arr[clip_high] < self.bg_thres:
            return None
        clipped_arr = torch.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
        min_value = torch.min(clipped_arr)
        max_value = torch.max(clip_high)
        filtered = clipped_arr
        img = (filtered - min_value) / (max_value - min_value)
        return img
    
    def postprocess(self,mask,min_size=50):
        labeled_mask, _ = label(mask, return_num=True)
        region_sizes = np.bincount(labeled_mask.ravel())
        small_regions = np.where(region_sizes < min_size)[0]
        for region in small_regions:
            mask[labeled_mask == region] = 0
        return mask
    
    def batch(self, image):
        ROI = self.get_ROI(image.shape)
        batch_in = []
        for roi in ROI:
            [d_s,d_e,d_cL,d_cR], [h_s,h_e,h_cL,h_cR], [w_s,w_e,w_cL,w_cR] = roi
            piece_in = image[d_s:d_e, h_s:h_e, w_s:w_e].astype(np.float32)
            # preprocess
            piece_in = torch.from_numpy(piece_in)
            piece_in = self.preprocess(piece_in)[None]
            batch_in.append(piece_in)
        batch_in = torch.stack(batch_in).to(self.device)
        return batch_in
    
    def process_oneImage(self, image:np.ndarray, re_batch:bool):
        ROI = self.get_ROI(image.shape)
        if re_batch:
            batch_in = self.batch(image)
        else:
            batch_in = image[None]
        batch_out = self.seg_net.model(batch_in)
        if re_batch:
            mask = np.zeros(image.shape, dtype=np.uint8)
            for i, roi in enumerate(ROI):
                [d_s,d_e,d_cL,d_cR], [h_s,h_e,h_cL,h_cR], [w_s,w_e,w_cL,w_cR] = roi
                piece_out = batch_out[i, 0, d_cL:self.patch_size-d_cR, h_cL:self.patch_size-h_cR, w_cL:self.patch_size-w_cR]
                piece_out[piece_out >= 0.5] = 1
                piece_out[piece_out < 0.5] = 0
                piece_out = piece_out.detach().cpu().numpy()
                mask[d_s+d_cL:d_e-d_cR, h_s+h_cL:h_e-h_cR, w_s+w_cL:w_e-w_cR] = piece_out
            mask = self.postprocess(mask)
        else:
            mask = batch_out[None, None]
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            mask = mask.detach().cpu().numpy()
        return mask

    def mask_to_segs(self, mask, offset=[0,0,0]):
        '''
        segment:
        {
            sid: int,
            points: [head,...,tail],
            sampled_points: points[::interval]
        }
        '''

        interval = 3

        x_border = 1
        y_border = 1
        z_border = 1

        skel = skeletonize(mask)
        skel[:x_border, :, :] = 0
        skel[-x_border:, :, :] = 0
        skel[:, :y_border, :] = 0
        skel[:, -y_border:, :] = 0
        skel[:, :, :z_border] = 0
        skel[:, :, -z_border:] = 0

        labels = label(skel, connectivity=3)
        regions = regionprops(labels)

        segments = []
        for region in regions:
            points = region.coords
            distances = cdist(points, points)
            adjacency_matrix = distances <= 1.8 # sqrt(3)
            np.fill_diagonal(adjacency_matrix, 0)
            graph = nx.from_numpy_array(adjacency_matrix.astype(np.uint8))
            spanning_tree = nx.minimum_spanning_tree(graph, algorithm='kruskal', weight=None)
            # remove circles by keeping only DFS tree
            graph.remove_edges_from(set(graph.edges) - set(spanning_tree.edges))

            branch_nodes = [node for node, degree in graph.degree() if degree >= 3]
            branch_nbrs = []
            for node in branch_nodes:
                branch_nbrs += list(graph.neighbors(node))

            for bn in branch_nodes:
                if len(list(graph.neighbors(node)))==3:
                    segments.append(
                        {
                            'sid' : None,
                            'points' : [[i+j for i,j in zip(points[bn],offset)]],
                            'sampled_points' : [[i+j for i,j in zip(points[bn],offset)]]
                        }
                    )

            graph.remove_nodes_from(branch_nbrs)
            graph.remove_nodes_from(branch_nodes)

            connected_components = list(nx.connected_components(graph))
            for nodes in connected_components:
                if len(nodes)<=interval*2:
                    continue
                subgraph = graph.subgraph(nodes).copy()
                end_nodes = [node for node, degree in subgraph.degree() if degree == 1]
                if (len(end_nodes)!=2):
                    continue
                path = nx.shortest_path(subgraph, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
                # path to segment
                seg_points = np.array([points[i].tolist() for i in path])
                seg_points = seg_points + np.array(offset)
                seg_points = seg_points.tolist()
                sampled_points = seg_points[:-(interval-1):interval]
                sampled_points.append(seg_points[-1])
                segments.append(
                    {
                        'sid' : None,
                        'points' : seg_points,
                        'sampled_points' : sampled_points
                    }
                )
        return skel, segments

