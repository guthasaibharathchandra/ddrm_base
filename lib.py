import numpy as np
import torch
#torch.set_default_dtype(torch.float32)
# The numbering of timesteps starts from 1 to T and for index 0, alpha=1, beta=0
def forward_diffusion(x, t_curr, t_next, schedule): #only for visualization (works with numpy inputs)
    
    #INPUT ASSERTIONS: BEGIN
    assert isinstance(x, np.ndarray)
    assert isinstance(t_curr, int)
    assert isinstance(t_next, int)
    assert isinstance(schedule, dict)
    assert 'alpha' in schedule and 'beta' in schedule
    alpha = schedule.get('alpha')
    beta  = schedule.get('beta')
    assert isinstance(alpha, (list, np.ndarray)) and isinstance(beta, (list, np.ndarray))
    assert (len(alpha) == len(beta))
    assert (alpha[0] == 1.0) and (beta[0] == 0)
    T = len(alpha) - 1
    #assert T > 10, f"Horizon T={T} is too low!"
    assert (t_curr >=0) and (t_curr <= T)
    assert t_next >= t_curr
    assert t_next <= T
    #INPUT ASSERTIONS: END

    #ALGORITHM: BEGIN
    alpha_effective = alpha[t_next]/alpha[t_curr]
    variance_factor = 1.0 - alpha_effective
    std_factor = np.sqrt(variance_factor)
    mean = np.sqrt(alpha_effective) * x
    return mean + (std_factor*np.random.normal(loc=0.0, scale=1.0, size=x.shape))
    #ALGORITHM: END

def reverse_diffusion(ddpm_model, x, t_curr, schedule, cuda=True, onlymean=False): #only for visualization (works with numpy inputs)

    #INPUT ASSERTIONS: BEGIN
    assert isinstance(x, np.ndarray)
    assert isinstance(t_curr, int)
    assert isinstance(schedule, dict)
    assert 'alpha' in schedule and 'beta' in schedule and 'rvar' in schedule
    alpha = schedule.get('alpha')
    beta  = schedule.get('beta')
    rvar = schedule.get('rvar')
    assert isinstance(alpha, (list, np.ndarray)) and isinstance(beta, (list, np.ndarray)) and isinstance(rvar, (list, np.ndarray))
    assert (len(alpha) == len(beta)) and (len(alpha) == len(rvar))
    assert (alpha[0] == 1.0) and (beta[0] == 0) and (rvar[0] == 0)
    T = len(alpha) - 1
    assert (t_curr > 0) and (t_curr <= T)
    #INPUT ASSERTIONS: END

    #ALGORITHM: BEGIN  
    x_inp = torch.from_numpy(x)
    t_curr_inp = t_curr * torch.ones(x_inp.shape[0], dtype=torch.float32)
    device = 'cuda' if cuda else 'cpu'
    device = torch.device(device)
    ddpm_model.to_device(device) # model is an object of DDPM class below
    #x_inp = x_inp.to(device)
    #t_curr_inp = t_curr_inp.to(device)
    with torch.no_grad():
        noise_pred = ddpm_model.infer(x_inp, t_curr_inp)
        noise_pred = noise_pred.cpu().numpy()
    assert noise_pred.shape == x.shape and noise_pred.dtype == x.dtype
    mean = (1.0/np.sqrt(1.0-beta[t_curr])) * ( x - ( (beta[t_curr]/np.sqrt(1-alpha[t_curr])) * noise_pred ) )
    assert mean.dtype == x.dtype
    rstd = np.sqrt(rvar[t_curr])
    if onlymean:
        return mean
    
    return mean + np.asarray((rstd*np.random.normal(loc=0.0, scale=1.0, size=tuple(x.shape))),dtype=np.float32)
    #ALGORITHM: END

def make_schedule(scheme, rvar, start_beta=0, end_beta=0.99, T=500):
    
    #INPUT ASSERTIONS: BEGIN
    assert isinstance(T, int)
    assert T > 0
    #assert T > 10, f"Horizon T={T} is too low!"
    assert scheme=='linear' or scheme == 'decay'
    assert rvar=='beta' or rvar=='fvar'
    assert end_beta >= start_beta
    #INPUT ASSERTIONS: END

    if scheme == 'linear':
        ans = 1.0
        alpha = [1.0,]
        eps = 1e-7
        beta = [0,] + np.linspace(start=start_beta, stop=end_beta, endpoint=True, num=T).tolist()
        beta = np.array(beta, dtype=np.float32)
        assert (len(beta) == T+1) and (beta[0] == 0) and (beta[T] - end_beta < eps) and (beta[1] - start_beta < eps)
        for i in range(1,len(beta)):
                ans *= 1.0-beta[i]
                alpha.append(ans)
        alpha = np.array(alpha, dtype=np.float32)

    else:
        raise NotImplementedError
    
    if rvar == 'beta':

        rvar = np.copy(beta)

    elif rvar == 'fvar':

        rvar = [0.0]
        for i in range(1,len(beta)):
            rvar.append( ((1-alpha[i-1])*beta[i])/(1-alpha[i]) )
        rvar = np.array(rvar, dtype=np.float32)     

    else:
        raise NotImplementedError
    
    assert (len(alpha) == len(beta)) and (len(beta) == len(rvar))

    return {'alpha':alpha, 'beta':beta, 'rvar':rvar, 'T':T}


class DDPM():

    def __init__(self, schedule, model, weightedloss=True, cuda=True):
        
        #INPUT ASSERTIONS: BEGIN
        assert isinstance(schedule, dict)
        assert 'alpha' in schedule and 'beta' in schedule and 'rvar' in schedule
        alpha = schedule.get('alpha')
        beta  = schedule.get('beta')
        rvar = schedule.get('rvar')
        assert isinstance(alpha, (list, np.ndarray)) and isinstance(beta, (list, np.ndarray)) and isinstance(rvar, (list, np.ndarray))
        assert (len(alpha) == len(beta)) and (len(alpha) == len(rvar))
        assert (alpha[0] == 1.0) and (beta[0] == 0) and (rvar[0] == 0)
        #INPUT ASSERTIONS: END

        #super(DDPM).__init__()
        self.T = len(alpha) - 1
        self.alpha = alpha
        self.beta = beta
        self.rvar = rvar
        self.weightedloss = weightedloss
        #assert weightedloss==False, "weightedloss not supported at the moment!"
        device = 'cuda' if cuda else 'cpu'
        self.device = torch.device(device)    
        self.model = model.to(self.device)
        self.loss = self.__build_loss().to(self.device)
        self.loss_weights = self.__get_loss_weights()
    
    
    def __build_loss(self,):
        return torch.nn.MSELoss(reduction='none')

    def __get_loss_weights(self,):
            
            t_index = np.arange(start=2, stop=1+self.T, step=1)
            numer = .5 * np.square(self.beta[t_index])
            denom = self.rvar[t_index]*(1.0 - self.beta[t_index])*(1.0-self.alpha[t_index])
            loss_weights = numer/(denom+1e-10)
            loss_weights = np.asarray([0,0,] + loss_weights.tolist(), dtype=np.float32)
            return loss_weights


    def __sample_std_gaussian_noise(self, X): # X is already supposed to be on self.device

        return torch.randn(*X.shape, device=self.device, dtype=torch.float32)
    
    def __generate_temporal_noise(self, X0): # X0 here is supposed to be already on self.device

        num_samples = X0.shape[0]
        
        #SAMPLE T
        start_time_step = 1
        if self.weightedloss:
            start_time_step = 2 # This is to ensure non-inf weight for t=1 case
        T_curr = np.random.choice(np.arange(start=start_time_step, stop=1+self.T, step=1), size=num_samples, replace=True)    
        Alpha_T_curr = self.alpha[T_curr]
        
        #GENERATE WEIGHTS FOR LOSS
        if self.weightedloss:
            loss_weights = self.loss_weights[T_curr]
            loss_weights = torch.from_numpy(loss_weights).to(self.device)
        else:
            loss_weights = 1.0

        #GENERATE X_T
        p = torch.tensor(np.sqrt(Alpha_T_curr).reshape(-1, *([1]*(len(X0.shape)-1)) ), dtype=torch.float32).to(self.device)
        q = torch.tensor(np.sqrt(1.0-Alpha_T_curr).reshape(-1, *([1]*(len(X0.shape)-1)) ), dtype=torch.float32).to(self.device)
        Noise_std_gaussian = self.__sample_std_gaussian_noise(X0)
        X_T_curr = (p*X0) + (q*Noise_std_gaussian)
        T_curr = torch.from_numpy(np.asarray(T_curr, dtype=np.float32)).to(self.device)

        return X_T_curr, T_curr, Noise_std_gaussian, loss_weights

    def to_device(self, device):
        self.device = device
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

    def infer(self, X_T_curr, T_curr): #only for inference
        self.model.eval()
        X_T_curr, T_curr = X_T_curr.to(self.device), T_curr.to(self.device)
        Noise_std_gaussian_pred = self.model(X_T_curr, T_curr)
        
        return Noise_std_gaussian_pred

    
    def run_step(self, X0): #only for training/validation run
        
        self.model.train()
        X0 = X0.to(self.device)
        X_T_curr, T_curr, Noise_std_gaussian, loss_weights = self.__generate_temporal_noise(X0)
        Noise_std_gaussian_pred = self.model(X_T_curr, T_curr)
        loss = torch.mean(self.loss(Noise_std_gaussian_pred, Noise_std_gaussian).reshape(X0.shape[0],-1).mean(dim=-1) * loss_weights)
        return loss




