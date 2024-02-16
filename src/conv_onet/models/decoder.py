from turtle import window_height
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from src.common import get_rotation_from_tensor


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=False, concat=True):
        super().__init__()
        self.concat = concat
        self.mapping_size = mapping_size
        self.scale = scale
        self.learnable = learnable
        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        # x should be tensor
        x = x.squeeze(0)  # eval points use input of (1,batch_num,3)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = (2*math.pi*x) @ self._B.to(x.device)
        #x = torch.matmul(2*math.pi*x, self._B).to(x.device)
        if self.concat:
            return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
        else:
            return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.
    Note: encourage to use Fourier Feature as embedding instead
    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling           # get linspace for power
        # include x after embeded, e.g. [x,x1,x2,...]
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))  # list append, get 2 dims
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self, mapping_size=3):
        super().__init__()
        self.mapping_size = mapping_size

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP_geometry(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, cfg, name='', dim=3, c_dim=32,
                 hidden_size=128, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], pos_embedding_method='fourier',
                 concat_feature=False, use_view_direction=False):
        super().__init__()
        self.name = name
        self.feat_name = name[:3]+'_feat'
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        self.weighting = cfg['pointcloud']['nn_weighting']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.min_nn_num = cfg['pointcloud']['min_nn_num']
        self.N_surface = cfg['rendering']['N_surface']
        self.use_view_direction = use_view_direction

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                # for fine decoder, c_dim is doubled than middle, e.g. 64
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25, concat=False, learnable=True)
            if self.use_view_direction:
                self.embedder_view_direction = GaussianFourierFeatureTransform(
                    3, mapping_size=embedding_size, scale=25)
            self.embedder_rel_pos = GaussianFourierFeatureTransform(
                3, mapping_size=10, scale=32, learnable=True)
        self.mlp_col_neighbor = MLP_col_neighbor(self.c_dim, 2*self.embedder_rel_pos.mapping_size, hidden_size)

        # xyz coord. -> embedding size
        # change later
        embedding_input = embedding_size
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_input, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_input, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
            hidden_size, 1, activation="relu")

        if not leaky:
            #self.actvn = F.relu
            self.actvn = torch.nn.Softplus(beta=100)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def get_feature_at_pos(self, npc, p, npc_feats, is_tracker=False, cloud_pos=None, is_mesher=False,
                           dynamic_r_query=None):
        assert torch.is_tensor(
            p), 'point locations for get_feature_at_pos should be tensor.'
        device = p.device
        p = p.reshape(-1, 3)
        D, I, neighbor_num = npc.find_neighbors_faiss(p.clone().detach(),
                                                      step='query' if not is_mesher else 'mesh',
                                                      dynamic_radius=dynamic_r_query)
        radius_query_bound = npc.get_radius_query(
        )**2 if not self.use_dynamic_radius else dynamic_r_query.reshape(-1, 1)**2
        if is_tracker:
            # re-calculate D
            nn_num = D.shape[1]
            D = torch.sum(torch.square(
                cloud_pos[I]-p.reshape(-1, 1, 3)), dim=-1)
            D = D.reshape(-1, nn_num)
            if self.weighting == 'distance':
                D[D > radius_query_bound] = 1e4  # mark out of query radius
            else:
                D[D > radius_query_bound] = 50  # for 'expo' weighting

        c = torch.zeros([p.shape[0], self.c_dim],
                        device=device).normal_(mean=0, std=0.01)
        has_neighbors = neighbor_num > self.min_nn_num-1

        if self.weighting == 'distance':
            #neighbor_pos = cloud_pos[I]  # (N,nn_num,3)
            #neighbor_rel_pos = neighbor_pos - p[:, None, :]
            #neighbor_l1_dis = torch.abs(neighbor_rel_pos).sum(dim=-1)
            weights = 1.0/(D+1e-10)
            #weights = 1.0/(neighbor_l1_dis+1e-10)
        else:
            # try avoid over-smooth by e^(-x)
            weights = torch.exp(-20*torch.sqrt(D))
        # if not is_tracker:
        with torch.no_grad():
            weights[D > radius_query_bound] = 0.
            # can't do this when weights requires grad
        # (n_points, nn_num=8, 1)
        weights = F.normalize(weights, p=1, dim=1).unsqueeze(-1)

        # use fixed num of nearst nn
        # select neighbors within range, then interpolate feature by inverse distance weighting
        neighbor_feats = npc_feats[I]             # (n_points, nn_num=8, c_dim)

        c = weights * neighbor_feats
        c = c.sum(axis=1).reshape(-1, self.c_dim)
        c[~has_neighbors] = torch.zeros(
            [self.c_dim], device=device).normal_(mean=0, std=0.01)

        return c, None, has_neighbors  # (N_point,c_dim), mask for pts

    def forward(self, p, npc, npc_geo_feats, npc_col_feats, pts_num=16, is_tracker=False, cloud_pos=None,
                point_mask=False, pts_views_d=None, is_mesher=False, dynamic_r_query=None):
        """forwad method of NICER decoder.

        Args:
            p (tensor): sampling locations, N*3
            npc (NerualPointCloud): shared npc object
            npc_geo_feats (tensor): cloned from npc
            npc_col_feats (tensor): cloned from npc
            pts_num (int, optional): sampled pts num along each ray. Defaults to N_surface.
            is_tracker (bool, optional): whether called by tracker. Defaults to False.
            cloud_pos (tensor, optional): point cloud position. 
            point_mask (bool): if to return mask for locations that have neighbor, default is to return valid ray mask
            pts_views_d (tensor): viweing directions
            is_mesher (bool, optional): whether called by mesher.
            dynamic_r_query (tensor, optional): if enabled dynamic radius, query radius for every pixel will be different.

        Returns:
            _type_: _description_
        """
        #device = p.device
        c, pts_normals, has_neighbors = self.get_feature_at_pos(
            npc, p, npc_geo_feats, is_tracker, cloud_pos, is_mesher, dynamic_r_query=dynamic_r_query)  # get (N,c_dim), e.g. (N,32)
        # change, to return has_neighbors as default
        if not is_mesher:
            # ray is not close to the current npc, choose bar here
            valid_ray_mask = ~(
                torch.sum(has_neighbors.view(-1, pts_num), 1) < int(self.N_surface/2+1))
        else:
            valid_ray_mask = None
        p = p.float().reshape(1, -1, 3)

        embedded_pts = self.embedder(p)
        embedded_input = embedded_pts
        if self.name == 'color':
            
            if self.use_view_direction:
                pts_views_d = F.normalize(pts_views_d, p=2, dim=1)
                embedded_views_d = self.embedder_view_direction(pts_views_d)
                embedded_input = torch.cat(
                    [embedded_pts, embedded_views_d], -1)
        h = embedded_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                # hidden dim + (feature dim->hidden dim) -> hidden dim
                h = h + self.fc_c[i](c)
                # so for hidden layers in the decoder, its input comes from both its feature and embedded location.
            if i in self.skips:
                h = torch.cat([embedded_input, h], -1)
        out = self.output_linear(h)
        if not self.color:
            # (N,1)->(N,) for occupancy; if with color, still (N,4) dim.
            out = out.squeeze(-1)
        return out, valid_ray_mask, has_neighbors


class QuantizedReLU(nn.Module):
    def __init__(self, num_bits):
        super(QuantizedReLU, self).__init__()
        self.num_bits = num_bits
        self.quantization_step = 2.0 / (2**num_bits - 1)

    def forward(self, x):
        # quantize values to desired range
        x_q = torch.round(x / self.quantization_step) * self.quantization_step
        #x_q = torch.clamp(x_q, 0.0, 1.0)
        # use straight-through estimator for gradient
        x_q = x_q.detach() + (x - x_q)
        return x_q
        # cause slight larger loss than sigmoid


class MLP_col_neighbor(nn.Module):
    def __init__(self, c_dim, embedding_size_rel, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(c_dim + embedding_size_rel, hidden_size)
        self.linear2 = nn.Linear(hidden_size, c_dim)
        self.act_fn = nn.Softplus(beta=100)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        # init.normal_(self.linear1.weight, mean=0,std=0.01)
        # init.normal_(self.linear2.weight, mean=0,std=0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class MLP_exposure(nn.Module):
    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 12)
        self.act_fn = nn.Softplus(beta=100)

        # init.xavier_uniform_(self.linear1.weight)
        # init.xavier_uniform_(self.linear2.weight)
        init.normal_(self.linear1.weight, mean=0,std=0.01)
        init.normal_(self.linear2.weight, mean=0,std=0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class MLP_color(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, cfg, name='', dim=3, c_dim=32,
                 hidden_size=128, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=True, skips=[2], pos_embedding_method='fourier',
                 concat_feature=False, use_view_direction=False):
        super().__init__()
        self.name = name
        self.feat_name = name[:3]+'_feat'
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        self.weighting = cfg['pointcloud']['nn_weighting']
        self.min_nn_num = cfg['pointcloud']['min_nn_num']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.N_surface = cfg['rendering']['N_surface']
        self.use_view_direction = use_view_direction
        self.encode_rel_pos_in_col = cfg['model']['encode_rel_pos_in_col']
        self.encode_exposure = cfg['model']['encode_exposure']
        self.encode_viewd = cfg['model']['encode_viewd']

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                # for fine decoder, c_dim is doubled than middle, e.g. 64
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 20
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=32)
            if self.use_view_direction:
                if self.encode_viewd:
                    self.embedder_view_direction = GaussianFourierFeatureTransform(
                        3, mapping_size=embedding_size, scale=32)
                else:
                    self.embedder_view_direction = Same(mapping_size=3)
            self.embedder_rel_pos = GaussianFourierFeatureTransform(
                3, mapping_size=10, scale=32, learnable=True)
        self.mlp_col_neighbor = MLP_col_neighbor(
            self.c_dim, 2*self.embedder_rel_pos.mapping_size, hidden_size)
        if self.encode_exposure:
            self.mlp_exposure = MLP_exposure(
                cfg['model']['exposure_dim'], hidden_size)

        # xyz coord. -> embedding size
        embedding_input = 2*embedding_size
        if self.use_view_direction:
            embedding_input += (2 if self.encode_viewd else 1)*self.embedder_view_direction.mapping_size
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_input, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_input, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
            hidden_size, 3, activation="linear")

        if not leaky:
            #self.actvn = F.relu
            self.actvn = torch.nn.Softplus(beta=100)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        self.last_act = QuantizedReLU(8)

        self.sample_mode = sample_mode

    def get_feature_at_pos(self, npc, p, npc_feats, is_tracker=False, cloud_pos=None, is_mesher=False,
                           dynamic_r_query=None):
        assert torch.is_tensor(
            p), 'point locations for get_feature_at_pos should be tensor.'
        device = p.device
        p = p.reshape(-1, 3)
        D, I, neighbor_num = npc.find_neighbors_faiss(p.clone().detach(),
                                                      step='query' if not is_mesher else 'mesh',
                                                      dynamic_radius=dynamic_r_query)
        radius_query_bound = npc.get_radius_query(
        )**2 if not self.use_dynamic_radius else dynamic_r_query.reshape(-1, 1)**2
        if is_tracker:
            # re-calculate D
            nn_num = D.shape[1]
            D = torch.sum(torch.square(
                cloud_pos[I]-p.reshape(-1, 1, 3)), dim=-1)
            D = D.reshape(-1, nn_num)
            if self.weighting == 'distance':
                D[D > radius_query_bound] = 1e4  # mark out of query radius
            else:
                D[D > radius_query_bound] = 50  # for 'expo' weighting

        c = torch.zeros([p.shape[0], self.c_dim],
                        device=device).normal_(mean=0, std=0.01)
        has_neighbors = neighbor_num > self.min_nn_num-1

        if self.weighting == 'distance':
            #neighbor_pos = cloud_pos[I]  # (N,nn_num,3)
            #neighbor_rel_pos = neighbor_pos - p[:, None, :]
            #neighbor_l1_dis = torch.abs(neighbor_rel_pos).sum(dim=-1)
            weights = 1.0/(D+1e-10)
            #weights = 1.0/(neighbor_l1_dis+1e-10)
        else:
            # try avoid over-smooth by e^(-x)
            weights = torch.exp(-20*torch.sqrt(D))
        # if not is_tracker:
        with torch.no_grad():
            weights[D > radius_query_bound] = 0.
            # can't do this when weights requires grad
            # error_msg: one of the variables needed for gradient computation has been modified by an inplace operation
        # (n_points, nn_num=8, 1)
        weights = F.normalize(weights, p=1, dim=1).unsqueeze(-1)

        # use fixed num of nearst nn
        # select neighbors within range, then interpolate feature by inverse distance weighting
        neighbor_feats = npc_feats[I]             # (n_points, nn_num=8, c_dim)
        if self.encode_rel_pos_in_col:
            neighbor_pos = cloud_pos[I]  # (N,nn_num,3)
            neighbor_rel_pos = neighbor_pos - p[:, None, :]
            embedding_rel_pos = self.embedder_rel_pos(
                neighbor_rel_pos.reshape(-1, 3))             # (N, nn_num, 40)
            neighbor_feats = torch.cat([embedding_rel_pos.reshape(neighbor_pos.shape[0], -1, self.embedder_rel_pos.mapping_size*2),
                                        neighbor_feats], dim=-1)  # (N, nn_num, 40+c_dim)
            neighbor_feats = self.mlp_col_neighbor(
                neighbor_feats)                  # (N, nn_num, c_dim)

        c = weights * neighbor_feats
        c = c.sum(axis=1).reshape(-1, self.c_dim)
        c[~has_neighbors] = torch.zeros(
            [self.c_dim], device=device).normal_(mean=0, std=0.01)

        return c, None, has_neighbors  # (N_point,c_dim), mask for pts

    def forward(self, p, npc, npc_geo_feats, npc_col_feats, is_tracker=False, cloud_pos=None, 
                point_mask=False, pts_views_d=None, is_mesher=False, dynamic_r_query=None, exposure_feat=None):
        """forwad method of NICER decoder.

        Args:
            p (tensor): sampling locations, N*3
            npc (NerualPointCloud): shared npc object
            npc_geo_feats (tensor): cloned from npc
            npc_col_feats (tensor): cloned from npc
            pts_num (int, optional): sampled pts num along each ray. Defaults to N_surface.
            is_tracker (bool, optional): whether is called by tracker.
            cloud_pos (tensor, optional): point cloud position, used when called by tracker to re-calculate D. 
            point_mask (bool): if to return mask for locations that have neighbor, not used.

        Returns:
            _type_: _description_
        """
        c, pts_normals, _ = self.get_feature_at_pos(npc, p, npc_col_feats, is_tracker, cloud_pos, 
                                                    is_mesher, dynamic_r_query=dynamic_r_query)
        p = p.float().reshape(1, -1, 3)

        embedded_pts = self.embedder(p)
        embedded_input = embedded_pts
        if self.name == 'color':
            if self.use_view_direction:
                pts_views_d = F.normalize(pts_views_d, p=2, dim=1)
                embedded_views_d = self.embedder_view_direction(pts_views_d)
                embedded_input = torch.cat(
                    [embedded_pts, embedded_views_d], -1)
        h = embedded_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.actvn(h)
            if self.c_dim != 0:
                # hidden dim + (feature dim->hidden dim) -> hidden dim
                h = h + self.fc_c[i](c)
                # so for hidden layers in the decoder, its input comes from both its feature and embedded location.
            if i in self.skips:
                h = torch.cat([embedded_input, h], -1)
        out = self.output_linear(h)
        if self.encode_exposure:
            if exposure_feat is not None:
                affine_tensor = self.mlp_exposure(exposure_feat)
                #rot, trans = get_rotation_from_tensor(affine_tensor[:4]), affine_tensor[4:]
                rot, trans = affine_tensor[:9].reshape(3,3), affine_tensor[-3:]
                out = torch.matmul(out, rot) + trans
                out = torch.sigmoid(out)
            else:
                return out # apply exposure separately
        else:
            out = torch.sigmoid(out)
            #out = self.last_act(out)
        return out


class NICER(nn.Module):
    """    
    Decoder for point represented features.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of decoder network
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, cfg, dim=3, c_dim=32,
                 hidden_size=128,
                 pos_embedding_method='fourier', use_view_direction=False):
        super().__init__()

        self.geo_decoder = MLP_geometry(name='geometry', cfg=cfg, dim=dim, c_dim=c_dim, color=False,
                                        skips=[2], n_blocks=5, hidden_size=32,
                                        pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP_color(name='color', cfg=cfg, dim=dim, c_dim=c_dim, color=True,
                                       skips=[2], n_blocks=5, hidden_size=hidden_size,
                                       pos_embedding_method=pos_embedding_method,
                                       use_view_direction=cfg['use_view_direction'])

    def forward(self, p, npc, stage, npc_geo_feats, npc_col_feats, pts_num=16, is_tracker=False, cloud_pos=None,
                pts_views_d=None, dynamic_r_query=None, exposure_feat=None):
        """
            Output occupancy/color in different stage, output is always (N,4)

        Args:
            p (torch.Tensor): point locations
            npc (torch.Tensor): NeuralPointCloud object.
            stage (str): listed below.
            npc_geo_feats (torch.Tensor): (N,c_dim)
            npc_col_feats (torch.Tensor): (N,c_dim)
            pts_num (int): number of points in sampled in each ray, used only by geo_decoder.
            is_tracker (bool): whether called by tracker.
            cloud_pos (torch.Tensor): (N,3)
            pts_views_d(torch.Tensor): used if color decoder encodes viewing directions.
            dynamic_r_query (torch.Tensor): (N,), used if dynamic radius enabled.

        """
        device = f'cuda:{p.get_device()}'
        match stage:
            case 'geometry':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats, npc_col_feats,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 dynamic_r_query=dynamic_r_query)
                raw = torch.zeros(
                    geo_occ.shape[0], 4, device=device, dtype=torch.float)
                raw[..., -1] = geo_occ
                return raw, ray_mask, point_mask
            case 'color':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats, npc_col_feats,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 dynamic_r_query=dynamic_r_query)   # returned (N,)
                raw = self.color_decoder(p, npc, npc_geo_feats, npc_col_feats,                                # returned (N,4)
                                         is_tracker=is_tracker, cloud_pos=cloud_pos,
                                         pts_views_d=pts_views_d,
                                         dynamic_r_query=dynamic_r_query, exposure_feat=exposure_feat)
                raw = torch.cat([raw, geo_occ.unsqueeze(-1)], dim=-1)
                return raw, ray_mask, point_mask
            case 'mesh':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats, npc_col_feats,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 point_mask=True, is_mesher=True, dynamic_r_query=dynamic_r_query)
                raw = self.color_decoder(p, npc, npc_geo_feats, npc_col_feats,
                                         is_tracker=is_tracker, cloud_pos=cloud_pos,
                                         pts_views_d=pts_views_d, is_mesher=True,
                                         dynamic_r_query=dynamic_r_query, exposure_feat=exposure_feat)
                raw = torch.cat([raw, geo_occ.unsqueeze(-1)], dim=-1)
                return raw, ray_mask, point_mask
            case 'color_only':
                raw = self.color_decoder(p, npc, npc_geo_feats, npc_col_feats,                                # returned (N,4)
                                         is_tracker=is_tracker, cloud_pos=cloud_pos,
                                         pts_views_d=pts_views_d,
                                         dynamic_r_query=dynamic_r_query, exposure_feat=exposure_feat)
                return raw
