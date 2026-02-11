from scipy.interpolate import interp1d
from skimage.restoration import inpaint
from simple_lama_inpainting import SimpleLama

def remove_red_scalebar(img):
    img_remove_red = img.copy()
    diff = img[:,:,0] - img[:,:,1]
    mask = diff >0
    # print(mask.shape)
    r_ch = img[:,:,0]    
    out = inpaint.inpaint_biharmonic(r_ch, mask)
    img_remove_red[:,:,0] = out*255
    img_remove_red[:,:,1] = out*255
    img_remove_red[:,:,2] = out*255

    return img_remove_red


def fill_edge_line(x,mask_missing):
    t = np.arange(len(x))
    known = ~mask_missing
    f = interp1d(t[known], x[known], kind="linear", fill_value="extrapolate")
    x_filled = x.copy()
    x_filled[mask_missing] = f(t[mask_missing])
    return x_filled

from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np

def remove_corner_blackbox(img):
    img_remove_black = img.copy()
    
    mask = (img[:,:,0] ==0) * 255
    lama = SimpleLama()
    # print(mask.shape)
    r_ch = img[:,:,0] 
    top_row =    img[0,:,0] 
    top_int = top_row[top_row>0].mean()
    bottom_row = img[-1,:,0] 
    bottom_int = bottom_row[bottom_row>0].mean()
    left_col =    img[:,0,0] 
    left_int = left_col[left_col>0].mean()
    right_col = img[:,-1,0] 
    right_int = right_col[right_col>0].mean()
    if(r_ch[0,0]==0):   r_ch[0,0] = (top_int + left_int)/2
    if(r_ch[0,-1]==0):  r_ch[0,-1] = (top_int + right_int)/2
    if(r_ch[-1,0]==0):  r_ch[-1,0] = (bottom_int + left_int)/2
    if(r_ch[-1,-1]==0): r_ch[-1,-1] = (bottom_int + right_int)/2
    mask[0,0]=0
    mask[0,-1]=0
    mask[-1,0]=0
    mask[-1,-1]=0

    mask_missing = mask[0,:]
    if(mask_missing.max()>0):
        r_ch[0,:] = fill_edge_line(r_ch[0,:],mask_missing)

    mask_missing = mask[-1,:]
    if(mask_missing.max()>0):
        r_ch[-1,:] = fill_edge_line(r_ch[-1,:],mask_missing)

    mask_missing = mask[:,0]
    if(mask_missing.max()>0):
        r_ch[:,0] = fill_edge_line(r_ch[:,0],mask_missing)

    mask_missing = mask[:,-1]
    if(mask_missing.max()>0): 
        r_ch[:,-1] = fill_edge_line(r_ch[:,-1],mask_missing)

    out = inpaint.inpaint_biharmonic(r_ch, mask)

    # out = lama(r_ch, mask)  
    img_remove_black[:,:,0] = out*255
    img_remove_black[:,:,1] = out*255
    img_remove_black[:,:,2] = out*255
    return img_remove_black