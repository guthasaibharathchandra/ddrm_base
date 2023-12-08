import torch
import numpy as np
from lib import DDPM
# The numbering of timesteps starts from 1 to T and for index 0, alpha=1, beta=0

def sample_timesteps(T_ddpm, T_short=20):

    #INPUT ASSERTIONS: BEGIN
    assert isinstance(T_ddpm, int)
    assert isinstance(T_short, int)
    assert T_short <= T_ddpm and T_short >= 1
    #INPUT ASSERTIONS: END

    #ALGORITHM: BEGIN
    T_short_list = [T_ddpm,]
    s = T_ddpm // T_short    
    while len(T_short_list) < T_short:
            t = T_short_list[0] - s
            t = t if t > 1 else 1
            T_short_list = [t,] + T_short_list
    T_short_list = np.asarray(T_short_list)
    assert len(T_short_list) == T_short
    return T_short_list
    #ALGORITHM: END  


class DDRM(DDPM):
    
    def __init__(self, schedule, model, weightedloss=True, cuda=True,):
        
        super().__init__(schedule, model, weightedloss, cuda)
        self.sigmasquare = (1.0/(self.alpha+1e-12)) - 1.0   
    
    def reverse_diffusion_ddrm(self, H, y, sigma_y, eta, eta_b, T_ddrm=20, cuda=True, onlymean=False, ddpminputshape=None): #only for generation (works with numpy inputs)

        #INPUT ASSERTIONS: BEGIN
        assert len(H.shape) == 2
        assert len(y.shape) == 2
        T = len(self.alpha) - 1
        T_ddrm_list = sample_timesteps(T, T_short=T_ddrm)
        #INPUT ASSERTIONS: END

        #ALGORITHM: BEGIN  
        #device = 'cuda' if cuda else 'cpu'
        #device = torch.device(device)
        #self.model.to(device) # model is an object of torch.nn.module class
        
        #----------------------------------------------------SVD based pre-processing BEGIN---------------------------------
        b, m = y.shape
        assert H.shape[0] == m
        m, n = H.shape
        U,S,V_T = np.linalg.svd(H)
        k = int(np.sum(S > 1e-6))
        S = S[:k]
        #--------------------------------------------
        S_inv = np.zeros_like(H).T
        S_inv[np.arange(k),np.arange(k)] = 1/S
        #--------------------------------------------
        y_bar =  y @ U @ S_inv.T
        assert y_bar.shape == (b, n)
        #--------------------------------------------
        S_diag = np.zeros((1,n))
        S_diag[:,:k] = S
        S_inv_diag = np.zeros((1,n))
        S_inv_diag[:,:k] = 1/S
        S_mask = np.zeros((1,n))
        S_mask[:,:k] = 1.0
        S_mask_bool = np.bool_(S_mask)
        #----------------------------------------------------SVD based pre-processing END---------------------------------
        
        
        #x_bar_T = np.random.normal(loc=S_mask*y_bar, scale=np.sqrt(self.sigmasquare[T]*np.ones_like(y_bar) - np.square(sigma_y*S_inv_diag)))
        if onlymean:
            x_bar_T = np.zeros_like(y_bar) #experimental
        else:
            x_bar_T = np.random.normal(loc=np.zeros_like(y_bar), scale=np.sqrt(self.sigmasquare[T]*np.ones_like(y_bar)))
        
        with torch.no_grad():
            
            #x_inp_bar = torch.from_numpy(x_bar_T)
            x_inp_bar = x_bar_T
            x_inp = x_inp_bar @ V_T
            
            T_ddrm_list = T_ddrm_list[::-1]
            T_ddrm_next_list = np.concatenate([T_ddrm_list[1:],[0]])
            for t_curr, t_prev in zip(T_ddrm_list, T_ddrm_next_list):
            
                x_inp = x_inp.reshape((b,*ddpminputshape))
                t_curr_inp = t_curr * torch.ones(x_inp.shape[0], dtype=torch.float32)
                x0_pred = x_inp - np.sqrt(self.sigmasquare[t_curr])*self.infer(np.sqrt(self.alpha[t_curr])*torch.from_numpy(np.asarray(x_inp,dtype=np.float32) ), t_curr_inp).cpu().numpy()
                x0_pred = x0_pred.reshape((b,n))
                x0_pred_bar = x0_pred @ V_T.T 

                w = np.zeros((1,n))
                mask_sigma_high = S_mask_bool * (sigma_y * S_inv_diag <= np.sqrt(self.sigmasquare[t_prev]))  # eta_b is the weight
                w[mask_sigma_high] = eta_b
                
                mask_sigma_low = S_mask_bool * (sigma_y * S_inv_diag > np.sqrt(self.sigmasquare[t_prev]))
                if sigma_y > 0:
                    w_ = np.sqrt(self.sigmasquare[t_prev]*(1.0 - eta**2)) * S_diag / sigma_y
                    w[mask_sigma_low] = w_[mask_sigma_low]
                else:
                    assert np.sum(mask_sigma_high) == k

                mask_singular = ~S_mask_bool
                w_ = np.sqrt(self.sigmasquare[t_prev]*(1.0 - eta**2)/self.sigmasquare[t_curr])
                w[mask_singular] = w_

                y_bar[:,k:] = x_inp_bar[:,k:]

                mean = (1.0 - w)*x0_pred_bar + w*y_bar
                std = np.ones_like(y_bar)*np.sqrt(eta**2 * self.sigmasquare[t_prev])
                std[:,mask_sigma_high[0]] = np.sqrt((self.sigmasquare[t_prev] - (sigma_y * eta_b * S_inv_diag)**2)[0,mask_sigma_high[0]])
                assert mean.shape == y_bar.shape and std.shape == mean.shape
                if not onlymean:
                    x_inp_bar = np.random.normal(loc=mean, scale=std)
                else:
                    x_inp_bar = mean
                x_inp = x_inp_bar @ V_T

        return x_inp            
        #ALGORITHM: END

    