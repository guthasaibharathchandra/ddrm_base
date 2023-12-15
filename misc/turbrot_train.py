import torch
from lib import forward_diffusion, reverse_diffusion, DDPM, make_schedule
from torch.utils.data import Dataset
import numpy as np
from unet import UNetModel
from torchvision.transforms import ToTensor, Compose, PILToTensor, Normalize, Resize, Lambda
import os
from PIL import Image

#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
turbrot_dataset_root = '/mimer/NOBACKUP/groups/azizpour-group-alvis/bharath/datasets/turbrot/'

class turbRotDataset(Dataset): #adopted from https://github.com/klaswijk/ddrm-fluids/blob/DDPM_Abhijeet/utils/turbRotDataloader.py
    """
    Custom dataset class for turbrot dataset. it will help dataloader to find the length of dataset (__len__)
    and access each image one by one (__getitem__)
    """
    def __init__(self, data_root, train=True):
        data_file = "turbrot_train.npy" if train else "turbrot_test.npy"
        self.data_file = os.path.join(data_root,data_file)
        self.data = np.asarray(np.load(self.data_file),dtype=np.float32)
        self.transform = Compose([          #Compose class is typically used to define pipelines to perfrom data processing (mostly for images)
                            ToTensor(),     #ToTensor() convert image to pytorch tensor in (Channel, Height, width)
                            Lambda(lambda x: x / 1.25)  #Custom lambda function: To normalize the data between [-1,1]
                        ])

    # Initialize your data or provide a list of file paths.    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load and preprocess your data here
        image = self.transform(self.data[idx])         #Note that we are applying transform here

        return image


#Dataset
turbrot_train_dataset_orig = turbRotDataset(data_root=turbrot_dataset_root, train=True)
N = len(turbrot_train_dataset_orig)
print(f"Train data len = {N}")

#schedule
T = 1000
schedule = make_schedule(scheme='linear', rvar='beta', T=T, start_beta=1e-4, end_beta=0.02)

#Model
class unet_turbrot_classic(torch.nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.model = UNetModel(image_size=64,
                            in_channels=1,
                            model_channels=32,
                            out_channels=1,
                            num_res_blocks=2,
                            attention_resolutions=[2,8],
                            dropout=0,
                            channel_mult=(1, 2, 2, 2),
                            conv_resample=True,
                            dims=2,
                            num_classes=None,
                            use_checkpoint=False,
                            use_fp16=False,
                            num_heads=1,
                            num_head_channels=-1,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=False,
                            resblock_updown=False,
                            use_new_attention_order=False)

    def forward(self, x, t):
        return self.model(x, t)

def train(ddpm_model, num_epochs, batch_size, savefreq, prefix):
    optimizer = torch.optim.Adam(ddpm_model.model.parameters(),lr=1e-4)
    train_dataset = turbrot_train_dataset_orig
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(1,num_epochs+1):
        LOSS = 0
        ITER = 0
        for data in dataloader:
            #data_x, data_y = data
            optimizer.zero_grad()
            loss = ddpm_model.run_step(data)
            loss.backward()
            optimizer.step()
            LOSS += loss.item()
            ITER += 1
        print(f"epoch = {epoch}, loss = {LOSS/ITER}")  
        if epoch % savefreq == 0:
            if not os.path.exists('./ckpts'):
                os.mkdir('ckpts')
            D = {'model_state_dict': ddpm_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch }
            torch.save(D,'ckpts/'+prefix+'-'+str(epoch)+'.ckpt')  
"""
def infer(ddpm_model, schedule, num_samples):
    
    z = np.random.normal(loc=0, scale=1.0, size=(num_samples,3,256,256))
    z = np.asarray(z, dtype=np.float32)
    x = z
    for t in range(0,T):
        x = reverse_diffusion(ddpm_model=ddpm_model, x=x, t_curr=T-t, cuda=True, schedule=schedule, onlymean=False)
    pred_samples = x
    assert pred_samples.shape == z.shape
    h, w = int(np.sqrt(num_samples)), int(np.sqrt(num_samples))
    assert h*w == num_samples
    f, ax = plt.subplots(ncols=w, nrows=h, figsize=(1.2*w,1.2*h))
    for id in range(num_samples):
        image  = pred_samples[id]
        image = np.transpose(image,(1,2,0))
        image = (1.0 + image)*0.5*255 
        image = np.clip(image, a_min=0, a_max=255)
        image = np.asarray(image, dtype=np.uint8)
        r = id//w
        c = id - (r*w)
        ax[r,c].imshow(image)
        ax[r,c].axis('off')
"""

def resume(ddpm_model, num_epochs, batch_size, savefreq, prefix, ckpt):
    ckpt = torch.load(ckpt)
    optimizer = torch.optim.Adam(ddpm_model.model.parameters(),lr=1e-4)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    ddpm_model.model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt['epoch']+1
    
    #flowers102_dataset_root = '/mimer/NOBACKUP/groups/azizpour-group-alvis/bharath/datasets/flowers102'
    #flowers102_train_dataset = Flowers102(root=flowers102_dataset_root, download=False, split='train', transform=Compose([Resize(size=(256,256)), ToTensor(), Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])  ]))
    
    dataloader = torch.utils.data.DataLoader(turbrot_train_dataset_orig, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(start_epoch,num_epochs+1):
        LOSS = 0
        ITER = 0
        for data in dataloader:
            #data_x, data_y = data
            optimizer.zero_grad()
            loss = ddpm_model.run_step(data)
            loss.backward()
            optimizer.step()
            LOSS += loss.item()
            ITER += 1
        print(f"epoch = {epoch}, loss = {LOSS/ITER}")  
        if epoch % savefreq == 0:
            if not os.path.exists('./ckpts'):
                os.mkdir('ckpts')
            D = {'model_state_dict': ddpm_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch }
            torch.save(D,'ckpts/'+prefix+'-'+str(epoch)+'.ckpt')  





model = unet_turbrot_classic()
#model.load_state_dict(torch.load('./ckpts/flowers102-uwl-100.ckpt')['model_state_dict'])
ddpm_model = DDPM(schedule=schedule,model=model,weightedloss=False,cuda=True)
#resume(ddpm_model, num_epochs=200, batch_size=32, savefreq=20, prefix='flowers102-uwl', ckpt='./ckpts/flowers102-uwl-100.ckpt')

train(ddpm_model, num_epochs=100, batch_size=32, savefreq=10, prefix='turbrot-classic')
#infer(ddpm_model, schedule, num_samples=25)


