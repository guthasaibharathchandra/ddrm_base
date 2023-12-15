import torch
import numpy as np
from lib import DDPM
from lib_svd import Deblurring
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


class DDRM_fast(DDPM):
    
    def __init__(self, schedule, model, weightedloss=True, cuda=True,):
        
        super().__init__(schedule, model, weightedloss, cuda)
        self.sigmasquare = (1.0/(self.alpha+1e-12)) - 1.0   
    
    def reverse_diffusion_ddrm(self, dummy_x, y, sigma_y, eta, eta_b, inp_channels, image_size, H=None, T_ddrm=20, cuda=True, onlymean=False): #only for generation

        with torch.no_grad():
                    
                #INPUT ASSERTIONS AND SETUP: BEGIN
                assert len(dummy_x.shape) == 4 # this is a random tensor of (batch, c, h, w)
                assert len(y.shape) == 2
                device = 'cuda' if cuda else 'cpu'
                device = torch.device(device)
                #assert task in ['inpaint','deblur','sr4x']
                #if task == 'deblur':
                #    H = Deblurring(torch.Tensor([1/9] * 9).to(device), inp_channels, image_size, device)
                #else:
                #    raise NotImplementedError
                assert H is not None
                #H = task_H
                T = len(self.alpha) - 1
                T_ddrm_list = sample_timesteps(T, T_short=T_ddrm)
                #INPUT ASSERTIONS AND SETUP: END


                #PREPROCESSING: BEGIN  
                #----------------------------------------------------SVD based pre-processing BEGIN---------------------------------
                y = y.to(device)
                dummy_x = dummy_x.to(device)
                b, m = y.shape
                fulldimx = dummy_x.shape[1]*dummy_x.shape[2]*dummy_x.shape[3]
                n = fulldimx
                
                #--------------------------------------------
                #singulars = H.add_zeros(H.singulars()).reshape((-1))
                singulars = torch.zeros((n),device=dummy_x.device)
                _singulars = H.singulars() 
                singulars[:_singulars.shape[0]] = _singulars 
                assert singulars.shape[0] == n       
                k = int(torch.sum(singulars > 0))
                #print(f"k == {k}, singulars.shape = {singulars.shape}")
                S_inv_diag = torch.zeros((1,n),device=dummy_x.device)
                S_inv_diag[:,singulars > 0] = (1.0 / singulars)[singulars > 0]
                
                S_diag = torch.zeros((1,n),device=dummy_x.device)
                S_diag[:,singulars > 0] = singulars[singulars > 0]
                
                S_mask = torch.zeros((1,n),device=dummy_x.device)
                S_mask[:,singulars > 0] = 1.0
                assert k == int(torch.sum(S_mask))

                S_mask_bool = S_mask > 0
                S_mask_bool = S_mask_bool.reshape((1,n))
                #--------------------------------------------
                
                
                U_t_y = H.Ut(y) #S_inv_diag * 
                #print(f"minmax__0 = {torch.min(U_t_y), torch.max(U_t_y)}")
                #print(f"singulars = {singulars[k-50:k+50]}, type = {singulars.dtype}")
                #print(f"minmax__-1 = shape = {singulars.shape} , {torch.min(singulars[:60000]), torch.max(singulars[:60000])}")
                _y_bar =  U_t_y / singulars[:U_t_y.shape[-1]]
                y_bar = torch.zeros((b,n),device=dummy_x.device)
                y_bar[:,:_y_bar.shape[1]] = _y_bar
                #print(f"y_bar.shape = {y_bar.shape}")
                #y_bar = H.add_zeros(y_bar).reshape((1,n))
                #print(torch.argmin(y_bar[:,:k]))
                #print(y_bar[:,51985])
                assert y_bar.shape == (b, n)
                #print(f"k == {k}")
                #print(f"minmax__1 = {torch.min(y_bar), torch.max(y_bar)}")
                #print(f"minmax__2 = {torch.min(y_bar[:,:k]), torch.max(y_bar[:,:k])}")

                #--------------------------------------------
                #----------------------------------------------------SVD based pre-processing END---------------------------------
                #PREPROCESSING: END
                


                #ALGORITHM: BEGIN
                #x_bar_T = np.random.normal(loc=S_mask*y_bar, scale=np.sqrt(self.sigmasquare[T]*np.ones_like(y_bar) - np.square(sigma_y*S_inv_diag)))
                
                if onlymean:
                    x_bar_T = torch.zeros_like(y_bar, dtype=torch.float32) #experimental
                else:
                    x_bar_T = np.random.normal(loc=torch.zeros_like(y_bar).cpu().numpy(), scale=torch.sqrt(self.sigmasquare[T]*torch.ones_like(y_bar)).cpu().numpy())
                    x_bar_T = torch.tensor(x_bar_T, dtype=torch.float32, device=y_bar.device)
                    
                x_inp_bar = x_bar_T
                #print(f" t = {T}, loc = {-1}, isanynan = {torch.any(torch.isnan(x_inp_bar))}, minmax = {torch.min(x_inp_bar),torch.max(x_inp_bar)}")

                x_inp = H.V(x_inp_bar)
                
                #print(f" t = {T}, loc = {0}, isanynan = {torch.any(torch.isnan(x_inp))}, minmax = {torch.min(x_inp),torch.max(x_inp)}")

                T_ddrm_list = T_ddrm_list[::-1]
                T_ddrm_next_list = np.concatenate([T_ddrm_list[1:],[0]])
                
                for t_curr, t_prev in zip(T_ddrm_list, T_ddrm_next_list):
                    
                        x_inp = x_inp.reshape(dummy_x.shape)
                        t_curr_inp = t_curr * torch.ones(x_inp.shape[0], dtype=torch.float32, device=x_inp.device)
                        eps_t_pred = self.infer(torch.tensor(np.sqrt(self.alpha[t_curr])*x_inp,dtype=torch.float32), t_curr_inp)
                        eps_t_pred = eps_t_pred[:,:3] # model also predicts sigma values so 6 channels in total, for mean we only want first 3 channels
                        x0_pred = x_inp - np.sqrt(self.sigmasquare[t_curr])*eps_t_pred
                        #print(f" t = {t_prev}, loc = {1}, isanynan = {torch.any(torch.isnan(x0_pred))}, minmax = {torch.min(x0_pred),torch.max(x0_pred)}")
                        x0_pred_bar = H.Vt(x0_pred)
                        #print(f" t = {t_prev}, loc = {2}, isanynan = {torch.any(torch.isnan(x0_pred_bar))}, minmax = {torch.min(x0_pred_bar),torch.max(x0_pred_bar)}")
                        assert x0_pred_bar.shape == (b,n)

                        # Setting up weights vector w for each case
                        w = torch.zeros((1,n),device=x0_pred_bar.device,dtype=torch.float32)
                        
                        # CASE3: when sigma_t >= sigma_y / s_i 
                        mask_sigma_high = S_mask_bool * (sigma_y * S_inv_diag <= np.sqrt(self.sigmasquare[t_prev]))  # eta_b is the weight
                        w[mask_sigma_high] = eta_b

                        # CASE2: when sigma_t < sigma_y / s_i
                        mask_sigma_low = S_mask_bool * (sigma_y * S_inv_diag > np.sqrt(self.sigmasquare[t_prev]))
                        if sigma_y > 0:
                            w_ = np.sqrt(self.sigmasquare[t_prev]*(1.0 - eta**2)) * S_diag / sigma_y
                            w[mask_sigma_low] = w_[mask_sigma_low]
                        else:
                            assert torch.sum(mask_sigma_high) == k
                        
                        # CASE1: when s_i = 0
                        mask_singular = ~S_mask_bool
                        w_ = np.sqrt(self.sigmasquare[t_prev]*(1.0 - eta**2)/self.sigmasquare[t_curr])
                        w[mask_singular] = w_

                        # Weighted average with vector w according to the case
                        #print(f" t = {t_prev}, loc = {200}, isanynan = {torch.any(torch.isnan(y_bar))}, minmax = {torch.min(y_bar[:,:k]),torch.max(y_bar[:,:k])}")
                        
                        #y_bar[:,k:] = x_inp_bar[:,k:]
                        y_bar[:,mask_singular[0]] = x_inp_bar[:,mask_singular[0]]

                        #print(f" t = {t_prev}, loc = {300}, isanynan = {torch.any(torch.isnan(y_bar))}, minmax = {torch.min(y_bar),torch.max(y_bar)}")
                        #print(f" t = {t_prev}, loc = {400}, isanynan = {torch.any(torch.isnan(w))}, minmax = {torch.min(w),torch.max(w)}")
                        mean = (1.0 - w)*x0_pred_bar + w*y_bar
                        #print(f" t = {t_prev}, loc = {3}, isanynan = {torch.any(torch.isnan(mean))}, minmax = {torch.min(mean),torch.max(mean)}")
                        std = torch.ones_like(y_bar)*np.sqrt(eta**2 * self.sigmasquare[t_prev])
                        std[:,mask_sigma_high[0]] = torch.sqrt((self.sigmasquare[t_prev] - (sigma_y * eta_b * S_inv_diag)**2)[0,mask_sigma_high[0]])
                        assert mean.shape == y_bar.shape and std.shape == mean.shape
                        if not onlymean:
                            x_inp_bar = np.random.normal(loc=mean.cpu().numpy(), scale=std.cpu().numpy())
                            x_inp_bar = torch.tensor(x_inp_bar, device=y_bar.device, dtype=torch.float32)
                        else:
                            x_inp_bar = mean
                        #print(f" t = {t_prev}, loc = {4}, isanynan = {torch.any(torch.isnan(x_inp_bar))}, minmax = {torch.min(x_inp_bar),torch.max(x_inp_bar)}")
                        x_inp = H.V(x_inp_bar)
                        #print(f" t = {t_prev}, loc = {5}, isanynan = {torch.any(torch.isnan(x_inp))}")

                return x_inp            
                #ALGORITHM: END