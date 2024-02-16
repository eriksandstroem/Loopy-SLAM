import torch
import cv2 as cv
import os
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import numpy as np


scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]


for scene in scenes:
    dataset_path = f"/home/notchla/point-slam-loop/Datasets/Replica/{scene}/results"
    renderings_path = f"/home/notchla/Documents/Eslam-renderings/{scene}/tracking_vis"

    rendering_files = sorted([os.path.join(renderings_path, f) for f in os.listdir(renderings_path)])

    dataset_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if "depth" not in f])
    dataset_depth_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if "frame" not in f])

    idx = 0

    counter = 0
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type='alex', normalize=True).to("cuda:0")

    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0

    for render in tqdm(rendering_files):

        render_img = cv.imread(render, cv.IMREAD_COLOR)
        dataset_img = cv.imread(dataset_files[idx], cv.IMREAD_COLOR)
        dataset_gt_img = cv.imread(dataset_depth_files[idx], cv.IMREAD_UNCHANGED)

        depth_data = dataset_gt_img.astype(np.float32)

        norm_render_img = cv.normalize(render_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        norm_dataset_img = cv.normalize(dataset_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        render_tensor = torch.from_numpy(norm_render_img).to("cuda:0")
        dataset_tensor = torch.from_numpy(norm_dataset_img).to("cuda:0")
        depth_data = torch.from_numpy(depth_data).to("cuda:0")
        
        mse_loss = torch.nn.functional.mse_loss(
            dataset_tensor[depth_data > 0], render_tensor[depth_data > 0])
        psnr_frame = -10. * torch.log10(mse_loss)

        ssim_value = ms_ssim(dataset_tensor.transpose(0, 2).unsqueeze(0).float(), render_tensor.transpose(0, 2).unsqueeze(0).float(),
                                                data_range=1.0, size_average=True)
        
        lpips_value = cal_lpips(torch.clamp(dataset_tensor.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                                torch.clamp(render_tensor.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()


        psnr_sum += psnr_frame
        ssim_sum += ssim_value
        lpips_sum += lpips_value

        idx += 5
        counter += 1

    avg_psnr = psnr_sum / counter
    avg_ssim = ssim_sum / counter
    avg_lpips = lpips_sum / counter

    print({f'{scene} avg_ms_ssim': avg_ssim})
    print({f'{scene} avg_psnr': avg_psnr})
    print({f'{scene} avg_lpips': avg_lpips})