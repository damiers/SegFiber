import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from ..utils.patch import get_patch_rois
from .torch_segmenter import TorchSegmenter


NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def build_skeleton_graph(points):
    index_by_coord = {tuple(point): index for index, point in enumerate(points)}
    edges = []
    for src, (x, y, z) in enumerate(points):
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            dst = index_by_coord.get((x + dx, y + dy, z + dz))
            if dst is not None and src < dst:
                edges.append((src, dst))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    graph.add_edges_from(sorted(edges), weight=1)
    return graph


class Seger:
    def __init__(
        self,
        config,
        checkpoint,
        background_threshold,
        tile_batch_size,
        device,
    ):
        self.seg_net = TorchSegmenter(
            config,
            checkpoint,
            background_threshold,
            device,
        )
        self.border_width = 4
        self.tile_batch_size = tile_batch_size

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
        border_size = self.border_width
        bordered_size = img.shape
        actual_size = [i-border_size*2 for i in bordered_size]
        block_rois = get_patch_rois(
            [border_size, border_size, border_size] + actual_size,
            block_size,
        )
        large_mask = np.zeros(img.shape,dtype=np.uint8)
        for start in range(0, len(block_rois), self.tile_batch_size):
            blocks = []
            placements = []
            for roi in block_rois[start : start + self.tile_batch_size]:
                x1, y1, z1 = roi[:3]
                x2 = x1 + roi[3]
                y2 = y1 + roi[4]
                z2 = z1 + roi[5]
                x1 = max(0, x1 - border_size)
                y1 = max(0, y1 - border_size)
                z1 = max(0, z1 - border_size)
                x2 = min(img.shape[0], x2 + border_size)
                y2 = min(img.shape[1], y2 + border_size)
                z2 = min(img.shape[2], z2 + border_size)

                block = img[x1:x2, y1:y2, z1:z2]
                padding = [
                    (
                        border_size - (roi[0] - x1),
                        border_size - (x2 - roi[0] - roi[3]),
                    ),
                    (
                        border_size - (roi[1] - y1),
                        border_size - (y2 - roi[1] - roi[4]),
                    ),
                    (
                        border_size - (roi[2] - z1),
                        border_size - (z2 - roi[2] - roi[5]),
                    ),
                ]
                extra_padding = []
                for axis, (before, after) in enumerate(padding):
                    extra = block_size + 2 * border_size - (
                        block.shape[axis] + before + after
                    )
                    extra_padding.append(extra)
                    padding[axis] = (before, after + extra)

                blocks.append(np.pad(block, padding, mode="reflect"))
                placements.append((roi, extra_padding))

            masks = self.seg_net.get_masks(blocks, thres=0.5)
            for mask, (roi, extra_padding) in zip(masks, placements):
                mask = mask.astype(np.uint8)
                mask = mask[
                    border_size : -border_size - extra_padding[0],
                    border_size : -border_size - extra_padding[1],
                    border_size : -border_size - extra_padding[2],
                ]
                x1, y1, z1 = roi[:3]
                x2 = x1 + roi[3]
                y2 = y1 + roi[4]
                z2 = z1 + roi[5]
                large_mask[x1:x2, y1:y2, z1:z2] = mask
        processed_mask = self.postprocess(large_mask)
        return processed_mask[border_size:-border_size, border_size:-border_size, border_size:-border_size]
    
    def mask_to_segs(self, mask, keep_branch, offset=(0, 0, 0)):
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
            if len(points) == 1:
                pt = (points + offset).tolist()
                segments.append({
                    "points": pt,
                    "nodes": pt,
                    "edges": [],
                    "checked": [-1]
                })
                continue

            G:nx.Graph = build_skeleton_graph(points)
            T:nx.Graph = nx.minimum_spanning_tree(G, algorithm='kruskal', weight=None)
            G.remove_edges_from(set(G.edges) - set(T.edges))

            sampled_edges = []
            sampled_coords =[]
            checked = []
            idx = 0
            nid2idx_map = {}
            sampled_path_nids_list = []
            if keep_branch:
                deg = dict(T.degree())
                key_nodes  = {n for n,d in deg.items() if d != 2}
                deg2_nodes = {n for n,d in deg.items() if d == 2}

                subG_deg2 = T.subgraph(deg2_nodes)
                for comp in nx.connected_components(subG_deg2):
                    end_points = []
                    for src in comp:
                        for dst in T.neighbors(src):
                            if dst not in comp:
                                end_points.append(dst)
                    # if there are not exactly two end points, skip this component
                    if len(end_points) != 2:
                        continue

                    # find the shortest path between the two end points
                    src, dst = end_points
                    path_nids = nx.shortest_path(T, src, dst)
                    sampled_path_nids = path_nids[0:-(interval-1):interval] + [path_nids[-1]]
                    sampled_path_nids_list.append(sampled_path_nids)
                    for i, (src_nid, dst_nid) in enumerate(zip(sampled_path_nids[:-1], sampled_path_nids[1:])):
                        def __check_node(nid):
                            return -1 if nid in key_nodes else 0
                        if src_nid not in nid2idx_map:
                            nid2idx_map[src_nid] = idx
                            sampled_coords.append(points[src_nid] + offset)
                            checked.append(__check_node(src_nid))
                            idx += 1
                        if dst_nid not in nid2idx_map:
                            nid2idx_map[dst_nid] = idx
                            sampled_coords.append(points[dst_nid] + offset)
                            checked.append(__check_node(dst_nid))
                            idx += 1
                        sampled_edges.append([nid2idx_map[src_nid], nid2idx_map[dst_nid]])
            else:
                branch_nodes = [node for node, degree in G.degree() if degree >= 3]
                branch_nbrs = []
                for node in branch_nodes:
                    branch_nbrs += list(G.neighbors(node))
                for bnid in branch_nodes:
                    if len(list(T.neighbors(bnid))) == 3:
                        bnode_coord = (points[bnid] + offset).tolist()
                        segments.append({
                            "points": [bnode_coord],
                            "nodes": [bnode_coord],
                            "edges": [],
                            "checked": [-1]
                        })
                T.remove_nodes_from(branch_nbrs)
                T.remove_nodes_from(branch_nodes)

                connected_components = list(nx.connected_components(T))
                for CC in connected_components:
                    if len(CC)<=interval*2:
                        continue
                    subgraph:nx.Graph = T.subgraph(CC).copy()
                    end_nodes = [node for node, degree in subgraph.degree() if degree == 1]
                    if (len(end_nodes)!=2):
                        continue
                    path_nids = nx.shortest_path(subgraph, source=end_nodes[0], target=end_nodes[1], weight=None, method='dijkstra') 
                    sampled_path_nids = path_nids[0:-(interval-1):interval] + [path_nids[-1]]
                    for nid in sampled_path_nids:
                        nid2idx_map[nid] = idx
                        sampled_coords.append((points[nid] + offset).tolist())
                        idx += 1
                    checked += [-1] + [0] * (len(sampled_path_nids)-2) + [-1]
                    for i, (src_nid, dst_nid) in enumerate(zip(sampled_path_nids[:-1], sampled_path_nids[1:])):
                        sampled_edges.append([nid2idx_map[src_nid], nid2idx_map[dst_nid]])
            segments.append({
                "points": (points+offset).tolist(),
                "nodes": sampled_coords,
                "edges": sampled_edges,
                "checked": checked
            })
        return segments
    
    def process(self, img_patch, offset, re_batch, keep_branch):
        if not re_batch:
            mask = self.seg_net.get_mask(img_patch)
        else:
            mask = self.get_large_mask(img_patch)
        # mask = self.seg_net.get_mask(img_patch)
        segs = self.mask_to_segs(mask, keep_branch=keep_branch, offset=offset)
        return segs
