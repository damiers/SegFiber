import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from tqdm import tqdm
import os

from eval.utils.patch import patchify_without_splices, get_patch_rois
from eval.model import SegNet, DEFAULT_CKPT_PATH
from eval.utils.imageReader import ImageReader
from eval.utils.sqliteDBIO import sqliteDBIO

class Seger():
    def __init__(self, ckpt_path=None, bg_thres=200, cuda_device_id:int=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
        print(f"=== CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')} ===")

        if ckpt_path is None:
            ckpt_path = DEFAULT_CKPT_PATH
        self.seg_net = SegNet(ckpt_path, bg_thres)        
        # border width
        self.bw = 4

    def postprocess(self,mask,min_size=50):
        labeled_mask, _ = label(mask,return_num=True)
        region_sizes = np.bincount(labeled_mask.ravel())
        small_regions = np.where(region_sizes < min_size)[0]
        for region in small_regions:
            mask[labeled_mask == region] = 0
        return mask
    

    def get_large_mask(self, img):
        '''
        process one large volume(D,W,H>100) with border (default 4), return mask
        '''
        block_size = 128
        border_size = self.bw
        bordered_size = img.shape
        actual_size = [i-border_size*2 for i in bordered_size]
        block_rois = get_patch_rois([border_size,border_size,border_size]+actual_size,block_size)
        large_mask = np.zeros(img.shape,dtype=np.uint8)
        print(f'blocksize: {len(block_rois)}')
        for roi in block_rois:
            tg_size = self.bw
            # add border if possible
            x1,x2,y1,y2,z1,z2 = roi[0],roi[0]+roi[3],roi[1],roi[1]+roi[4],roi[2],roi[2]+roi[5]
            x1 = max(0,x1-tg_size)
            y1 = max(0,y1-tg_size)
            z1 = max(0,z1-tg_size)
            x2 = min(img.shape[0],x2+tg_size)
            y2 = min(img.shape[1],y2+tg_size)
            z2 = min(img.shape[2],z2+tg_size)

            block = img[x1:x2,y1:y2,z1:z2]

            x1_pad = roi[0]-x1
            y1_pad = roi[1]-y1
            z1_pad = roi[2]-z1
            x2_pad = x2-roi[0]-roi[3]
            y2_pad = y2-roi[1]-roi[4]
            z2_pad = z2-roi[2]-roi[5]

            pad_widths = [
                (tg_size-x1_pad, tg_size-x2_pad),
                (tg_size-y1_pad, tg_size-y2_pad),
                (tg_size-z1_pad, tg_size-z2_pad)
            ]
            
            # if img.shape%block_size != 0, pad to target size
            ap = [] # additional padding
            for i, (p1,p2) in enumerate(pad_widths):
                res = block_size+tg_size*2 - (block.shape[i]+p1+p2)
                ap.append(res)
                if res!=0:
                    pad_widths[i] = (p1,p2+res)

            padded_block = np.pad(block, pad_widths, mode='reflect')
            print(f'padded_block shape: {padded_block.shape}')

            mask = self.seg_net.get_mask(padded_block,thres=0.5)
            mask = mask.astype(np.uint8)
            mask = mask[tg_size:-tg_size-ap[0],tg_size:-tg_size-ap[1],tg_size:-tg_size-ap[2]]
            large_mask[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[4],roi[2]:roi[2]+roi[5]] = mask
        processed_mask = self.postprocess(large_mask)
        return processed_mask[border_size:-border_size,border_size:-border_size,border_size:-border_size]


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
            graph:nx.Graph = nx.from_numpy_array(adjacency_matrix.astype(np.uint8))
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
                            'points' : [[i+j for i,j in zip(points[bn].tolist(),offset)]],
                            'sampled_points' : [[i+j for i,j in zip(points[bn].tolist(),offset)]]
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
                        'points' : seg_points,
                        'sampled_points' : sampled_points
                    }
                )
        return skel, segments

    def process_whole(self, image_path, db_path, level=0, channel=0, chunk_size=300, splice=300, roi=None):
        '''
        cut whole brain image to [300,300,300] cubes without splices (z coordinates % 300 == 0)
        '''
        image = ImageReader(image_path)
        db = sqliteDBIO(db_path)

        if roi==None:
            image_roi = image.rois[level]
        else:
            image_roi = roi
        rois = patchify_without_splices(image_roi,chunk_size,splices=splice)
        
        # pad rois
        _, seg_version = db.get_max_sid_version()
        for roi in tqdm(rois):
            if (np.array(roi[3:])<=np.array([128,128,128])).all():
                img = image.from_roi(roi, padding='reflect', level=level, channel=channel) 
                mask = self.seg_net.get_mask(img)
                offset = roi[:3]
            else:
                roi[:3] = [i-self.bw for i in roi[:3]]
                roi[3:] = [i+self.bw*2 for i in roi[3:]]
                padded_block = image.from_roi(roi, padding='reflect', level=level, channel=channel) 
                mask = self.get_large_mask(padded_block)
                offset=[i+self.bw for i in roi[:3]]
            _, segs_in_block = self.mask_to_segs(mask, offset=offset)

            db.segs2db(segs_in_block, version=seg_version)