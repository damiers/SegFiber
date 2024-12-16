import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from rtree import index
from tqdm import tqdm
import os

from .utils.patch import patchify_without_splices, get_patch_rois
from .utils.dbio import read_nodes, read_edges, add_edges, delete_edges, uncheck_nodes
from .utils.image_reader import wrap_image

from .model import SegNet, DEFAULT_CKPT_PATH

class Seger():
    def __init__(self,ckpt_path=None,bg_thres=200,cuda_device_id:int=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
        print(f"=== CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')} ===")

        if ckpt_path is None:
            ckpt_path = DEFAULT_CKPT_PATH
        self.seg_net = SegNet(ckpt_path,bg_thres)        
        # border width
        self.bw = 4

    def postprocess(self,mask,min_size=50):
        labeled_mask, _ = label(mask,return_num=True)
        region_sizes = np.bincount(labeled_mask.ravel())
        small_regions = np.where(region_sizes < min_size)[0]
        for region in small_regions:
            mask[labeled_mask == region] = 0
        return mask
    

    def get_large_mask(self,img):
        '''
        process one large volume(D,W,H>100) with border (default 4), return mask
        '''
        block_size = 100
        border_size = self.bw
        bordered_size = img.shape
        actual_size = [i-border_size*2 for i in bordered_size]
        block_rois = get_patch_rois([border_size,border_size,border_size]+actual_size,block_size)
        large_mask = np.zeros(img.shape,dtype=np.uint8)
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


    def connect_segs(self, db_path):
        def _cal_angle(v1:np.ndarray, v2:np.ndarray):
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            v1 = v1 / norm_v1
            v2 = v2 / norm_v2
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
            return angle

        angle_disconnt = 100
        angle_connt_valid = 45
        angle_connt_invalid = 120
        dist_threshold = 12

        G = nx.Graph()
        p = index.Property(dimension=3)
        rtree = index.Index(properties=p)

        nodes = read_nodes(db_path)
        edges = read_edges(db_path)
        for node in nodes:
            G.add_node(node['nid'], nid=node['nid'], coord=node['coord'], creator=node['creator'], status=node['status'], type=node['type'], date=node['date'], checked=node['checked'])
            rtree.insert(node['nid'], tuple(list(node['coord']) + list(node['coord'])), obj=[node['nid'], node['coord']])
        for edge in edges:
            G.add_edge(edge['src'], edge['des'], date=node['date'], creator=edge['creator'])

        # check all path nodes
        path_nodes = [node for node, degree in G.degree() if degree==2]
        nodes_invalid = []
        pbar = tqdm(path_nodes, desc='proofreading')
        for nid in pbar:
            if G.degree[nid]!=2:
                continue
            nbr_nids = list(G.neighbors(nid))
            v1 = np.asarray(G.nodes[nbr_nids[0]]['coord']) - np.asarray(G.nodes[nid]['coord'])
            v2 = np.asarray(G.nodes[nbr_nids[1]]['coord']) - np.asarray(G.nodes[nid]['coord'])
            angle = _cal_angle(v1, v2)
            # if invalid
            if angle < angle_disconnt:
                G.remove_edge(nid, nbr_nids[0])
                G.remove_edge(nid, nbr_nids[1])
                delete_edges(db_path, [[nid, nbr_nids[0]],])
                delete_edges(db_path, [[nid, nbr_nids[1]],])
                nodes_invalid.append([nid, angle])
            pbar.set_description(f'proofreading, num of invalid edges: {len(nodes_invalid)*2}')
        
        # try to connect end nodes
        edges_autoConnt = []
        end_nodes = [node for node, degree in G.degree() if degree==1]
        pbar = tqdm(end_nodes)
        for end_nid in pbar:
            if G.degree[end_nid]>1:
                continue
            curr_path = [end_nid] + [des for src, des in list(nx.dfs_edges(G, end_nid, depth_limit=5))]
            curr_coords = np.asarray([G.nodes[nid]['coord'] for nid in curr_path])
            offset = [i-dist_threshold//2 for i in curr_coords[0]]
            roi = offset + [i+dist_threshold for i in offset]
            curr_direction = np.sum(curr_coords[:-1:3] - curr_coords[1::3], axis=0)

            nbr_nid_list = set(rtree.intersection(tuple(roi), objects=False)) - set(curr_path)
            nbr_nid_list = [nid for nid in nbr_nid_list if G.degree[nid]==1]
            matched_nbr_nid = None
            min_angle = angle_connt_valid
            for nbr_nid in nbr_nid_list:
                nbr_path = [nbr_nid] + [des for src, des in list(nx.dfs_edges(G, nbr_nid, depth_limit=5))]
                nbr_coords = np.asarray([G.nodes[nid]['coord'] for nid in nbr_path])
                nbr_direction = np.sum(nbr_coords[1::3] - nbr_coords[:-1:3], axis=0)

                direction_angle = _cal_angle(curr_direction, nbr_direction)
                connection_angle = min(
                    _cal_angle(v1=curr_coords[1]-curr_coords[0], v2=nbr_coords[0]-curr_coords[0]),
                    _cal_angle(v1=nbr_coords[1]-nbr_coords[0], v2=curr_coords[0]-nbr_coords[0]),
                )
                # if valid
                if direction_angle<=min_angle and connection_angle>=angle_connt_invalid:
                    min_angle = direction_angle
                    matched_nbr_nid = nbr_nid

            if matched_nbr_nid is not None:
                G.add_edge(end_nid, matched_nbr_nid)
                add_edges(db_path, edges=[[end_nid, matched_nbr_nid],], user_name='connector')
                uncheck_nodes(db_path, [end_nid, matched_nbr_nid])
                edges_autoConnt.append([end_nid, matched_nbr_nid])
            pbar.set_description(f'auto connecting, num of auto connected segs: {len(edges_autoConnt)}')

        print(f'Remove {len(nodes_invalid)*2} invalid edges.\nAuto Connect {len(edges_autoConnt)} segments.')
        return nodes_invalid, edges_autoConnt


    def process_whole(self,image_path,level=0,channel=0,chunk_size=300,splice=300,roi=None):
        '''
        cut whole brain image to [300,300,300] cubes without splices (z coordinates % 300 == 0)
        '''
        image = wrap_image(image_path)
        if roi==None:
            image_roi = image.rois[level]
        else:
            image_roi = roi
        rois = patchify_without_splices(image_roi,chunk_size,splices=splice)
        # pad rois
        segs = []
        for roi in tqdm(rois):
            if (np.array(roi[3:])<=np.array([128,128,128])).all():
                if 'tif' in image_path:
                    img = image.from_roi(roi,padding='reflect')
                else:
                    img = image.from_roi(roi,level,channel,padding='reflect') 
                mask = self.seg_net.get_mask(img)
                offset = roi[:3]
            else:
                roi[:3] = [i-self.bw for i in roi[:3]]
                roi[3:] = [i+self.bw*2 for i in roi[3:]]
                if 'tif' in image_path:
                    padded_block = image.from_roi(roi,padding='reflect')
                else:
                    padded_block = image.from_roi(roi,level,channel,padding='reflect') 
                mask = self.get_large_mask(padded_block)
                offset=[i+self.bw for i in roi[:3]]
            _, segs_in_block = self.mask_to_segs(mask,offset=offset)
            segs+=segs_in_block

        for i, seg in enumerate(segs):
            seg['sid'] = i

        return segs

