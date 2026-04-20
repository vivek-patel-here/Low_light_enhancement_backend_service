import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class Denoiser(nn.Module):

    def denoise(self ,image):
        img = image.squeeze().permute(1,2,0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        denoised = cv2.fastNlMeansDenoisingColored(
			img,
			None,
			h=11,        
			hColor=11,
			templateWindowSize=7,
			searchWindowSize=21
		)
        denoised = torch.from_numpy(denoised / 255.0).permute(2,0,1).unsqueeze(0).float()
        return denoised.to(image.device)
    
    def laplacian_enhance(self,image):
        img = image.squeeze().permute(1,2,0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

		# Gaussian blur (noise reduction)
        blurred = cv2.GaussianBlur(img, (3,3), 0)

		# Laplacian (edge extraction)   
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))

		# Add edges back (unsharp masking style)
        enhanced = cv2.addWeighted(img, 1.0, lap, 0.3, 0)
        enhanced = torch.from_numpy(enhanced / 255.0).permute(2,0,1).unsqueeze(0).float()
        return enhanced.to(image.device)
    
    def denoisePreserveDetail(self,image):
        denoised = self.denoise(image)
        denoised = self.laplacian_enhance(denoised)
        output =  0.7 * denoised + 0.3 * image 
        return output
