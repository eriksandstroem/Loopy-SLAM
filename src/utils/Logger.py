import os

import torch


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, mapper
                 ):
        self.verbose = mapper.verbose
        self.ckptsdir = mapper.ckptsdir
        self.gt_c2w_list = mapper.gt_c2w_list
        self.estimate_c2w_list = mapper.estimate_c2w_list
        self.decoders = mapper.decoders

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes, npc, exposure_feat=None, last_log=False):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        if last_log:
            torch.save({
                'geo_feats':npc.get_geo_feats(end=True),  
                'col_feats':npc.get_col_feats(end=True),  
                'cloud_pos':npc.get_cloud_pos(end=True),      
                'pts_num': npc.pts_num(),         
                'input_pos': npc.input_pos(),     
                'input_rgb': npc.input_rgb(),     
                'input_normal': npc.input_normal(),  
                'input_normal_cartesian': npc.input_normal_cartesian(),

                'decoder_state_dict': self.decoders.state_dict(),
                'gt_c2w_list': self.gt_c2w_list,             
                'estimate_c2w_list': self.estimate_c2w_list, 
                'keyframe_list': keyframe_list,
                'keyframe_dict': keyframe_dict,
                'selected_keyframes': selected_keyframes,
                'idx': idx,
                'fragments' : npc.get_fragments(),
                "exposure_feat_all": torch.stack(exposure_feat, dim=0)
                if exposure_feat is not None
                else None,
            }, path, _use_new_zipfile_serialization=False)
        else:
            torch.save({  
                'cloud_pos':npc.get_cloud_pos(end=True),      
                'pts_num': npc.pts_num(),         
                'input_pos': npc.input_pos(),     
                'input_rgb': npc.input_rgb(),     
                'input_normal': npc.input_normal(),  
                'input_normal_cartesian': npc.input_normal_cartesian(),

                'decoder_state_dict': self.decoders.state_dict(),
                'gt_c2w_list': self.gt_c2w_list,             
                'estimate_c2w_list': self.estimate_c2w_list, 
                'keyframe_list': keyframe_list,
                'keyframe_dict': keyframe_dict,
                'selected_keyframes': selected_keyframes,
                'idx': idx,
                'fragments' : npc.get_fragments(),
                "exposure_feat_all": torch.stack(exposure_feat, dim=0)
                if exposure_feat is not None
                else None,
            }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
