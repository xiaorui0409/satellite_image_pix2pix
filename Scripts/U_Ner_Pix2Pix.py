import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.manual_seed(0)
import torchvision
from skimage import color
import numpy as np
from torchvision.datasets import VOCSegmentation
import os
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms import functional as TF




#### BUilding the contracting block
class Contractingblock(nn.Module):
    def __init__(self, input_channels,use_dropout=False, use_bn=True):
        super(Contractingblock, self).__init__()
        ### Downsampling process
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels*2)
        self.use_bn = use_bn

        if use_dropout:
            self.dropout = nn.Dropout2d()
        self.use_dropout = use_dropout
    
    def forward(self, x):
        # image tensor [batch_size, channels, height, width]
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        x = self.maxpool(x)

        return x
    
def crop(image, new_shape):
    # image tensor [batch_size, channels, height, width]
    ## skip_con_x: "image" from contracting path; "x.shape"
    middle_height = image.shape[2]//2
    middle_width = image.shape[3]//2

    starting_height = middle_height - new_shape[2]//2
    final_height = starting_height + new_shape[2]

    starting_width = middle_width - new_shape[3]//2
    final_width = starting_width + new_shape[3]

    cropped_images = image[:, :,starting_height:final_height, starting_width: final_width]

    return cropped_images

#### BUilding the expanding block
class Expandingblock(nn.Module):
    def __init__(self, input_channels,use_dropout=False, use_bn=True):
        super(Expandingblock, self).__init__()
        ### upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        ## Halves the number of feature channels
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        ## concatenation with the corresponding cropped feature map from the contracting path
        ## two 3*3 convolutions
        self.conv2 = nn.Conv2d(input_channels, input_channels//2,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2,kernel_size=2,padding=1)
        self.activation = nn.ReLU()

        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels//2)
        self.use_bn = use_bn

        if use_dropout:
            self.dropout = nn.Dropout2d()

        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        ## skip_con_x: the image tensor from the "contracting path" used for skip connection
        x = self.upsample(x)
        x = self.conv1(x)
        ## skip connections:
        ### feature map from contracting path (skip_con_x/"image") need to match thedimension in expanding path(x)
        x_skip = crop(skip_con_x, x.shape)
        x = torch.cat([x, x_skip], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        return x
    
### Define the featuremap Block
class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels,output_channels):
        super(FeatureMapBlock, self).__init__()
        ####1x1 convolution
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self,x):
        x = self.conv(x)

        return x
    

##### Define the U-Net network
class UNet(nn.Module):
    def __init__(self, input_channels,output_channels,hidden_channels=32):  #64
        super(UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = Contractingblock(hidden_channels*1,use_dropout=True)
        self.contract2 = Contractingblock(hidden_channels*2,use_dropout=True)
        self.contract3 = Contractingblock(hidden_channels*4,use_dropout=True)
        self.contract4 = Contractingblock(hidden_channels*8)
        self.contract5 = Contractingblock(hidden_channels*16)
        self.contract6 = Contractingblock(hidden_channels*32)

        self.expand1 = Expandingblock(hidden_channels*64)
        self.expand2 = Expandingblock(hidden_channels*32)
        self.expand3 = Expandingblock(hidden_channels*16)
        self.expand4 = Expandingblock(hidden_channels*8)
        self.expand5 = Expandingblock(hidden_channels*4)
        self.expand6 = Expandingblock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels*1, output_channels)
        self.sigmoid = torch.nn.Sigmoid()  ###output range[0, 1]

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)

        x7 = self.expand1(x6,x5)
        x8 = self.expand2(x7,x4)
        x9 = self.expand3(x8,x3)
        x10 = self.expand4(x9,x2)
        x11 = self.expand5(x10,x1)
        x12 = self.expand6(x11,x0)

        x_13 = self.downfeature(x12)
        x_final = self.sigmoid(x_13)

        return  x_final   

#### Define the Discriminator based on the contracting path of U_net
#######Output: one-channel matrix of classifications instead of a single value
class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=8): 
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = Contractingblock(hidden_channels*1, use_bn=False)
        self.contract2 = Contractingblock(hidden_channels*2)
        self.contract3 = Contractingblock(hidden_channels*4)
        self.contract4 = Contractingblock(hidden_channels*8)
        
        ### 1x1 kernel size
        self.output = nn.Conv2d(hidden_channels*16, 1, kernel_size=1)

    def forward(self,x,y):
        ### concatnate the fake image with segementation mask
        x = torch.cat([x, y], axis=1)

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
  
        x_final = self.output(x4)

        return x_final



#### define the Dataset
######pairs each "real image" with its corresponding "condition image(segmentation mask)" 
class VOCPairedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super(VOCPairedDataset, self).__init__()
        self.image_dir = image_dir ### store path for "real_image"
        self.mask_dir = mask_dir   ### store path for "condition_image"--> "mask"
        self.transform = transform

        ###Build the dictionary of "Mask": like  {"example001": "example001.png"}
        #####  "key": filename without extension;"value" filename with extension.
        self.mask = {os.path.splitext(file)[0]: file for file in os.listdir(mask_dir)}

        ## filtern "real_image" to ensure each image has its corresponding "segementation_mask"
        self.images =[img for img in os.listdir(image_dir) if os.path.splitext(img)[0] in self.mask]

        ## calcaute the filtered image
        all_images = os.listdir(image_dir)
        total_images = len(all_images)
        filtered_images = len(self.images)
        filtered_percentage = (filtered_images / total_images) * 100
        ### Filtered 2913 out of 17125 images (17.01%)
        print(f"Filtered {filtered_images} out of {total_images} images ({filtered_percentage:.2f}%)\n")

    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):
        ### retrive the image_file; filename  obtained from  "self.images[idx]" like "exmaple001.png"
        image_name = self.images[idx]
        ## construct full path of "real_image" (store path of directory + filename)
        image_path = os.path.join(self.image_dir, image_name)

        mask_name = self.mask[os.path.splitext(image_name)[0]]
        ## construct full path of "condtion_image"
        mask_path = os.path.join(self.mask_dir, mask_name)

        #####Debug check match path between image_path and mask_path
        # print('\n')
        # print(f"Debug print statements_____\n")
        # print(f"Loading image: {image_path}")
        # print(f"Loading mask: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Combine image and mask side-by-side
        try:
            combined_image = Image.new('RGB', (image.width + mask.width, image.height))
            combined_image.paste(image, (0, 0))
            combined_image.paste(mask, (image.width, 0))

            # Check if dimensions are as expected, raise an exception if they are not
            if not(combined_image.width == image.width + mask.width and combined_image.height == image.height):
                raise ValueError("Combined image dimensions do not match expected dimensions.\n")
            
        except Exception as e:
            print(f"Error combining images: {e}")
            raise # Rethrow the exception to handle it outside or stop the program


        # Convert PIL image to tensor
        if self.transform: 
            combined_image = self.transform(combined_image)
        

        return combined_image


def calculate_mean_std(initial_dataset, batch_size_mean_var): ## num_workers=4
    dataloader = DataLoader(initial_dataset, batch_size=batch_size_mean_var, shuffle=False)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        # Rearrange batch to be the shape of [B, C, W * H]
        data = data.view(data.size(0), data.size(1), -1)
        # Update total sum and squared sum
        channels_sum += torch.mean(data, dim=[0, 2])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std



#### Verifty the customized dataset

def verify_dataset_pairs_new(dataset, num_samples=10):
    for i in range(num_samples):
        combined_image = dataset[i]  # # Single combined image
        # Assuming the image and mask are concatenated along the width
        mid_point = combined_image.shape[2] // 2  # Assuming combined along width and channels are last
        image = combined_image[:, :, :mid_point]
        mask = combined_image[:, :, mid_point:]

        #print(f"Sample {i}: Combined image shape: {combined_image.shape}")  # torch.Size([3, 256, 512])
        #print("Visualizing Image and Mask...\n")
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))  # Adjust dimension if necessary
        plt.title('Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask.permute(1, 2, 0))  # Adjust dimension if necessary
        plt.title('Mask')
        plt.show()


## visuaizing the batch of image
def save_image_tensor(image_tensor, num_images=8, size =(3, 256, 256), filename="output.png"):
    image_shifted = image_tensor.cpu()
    #### detach from GPU and reshape accornding to "size"
    image_flatten = image_shifted.detach().view(-1, *size)
    ### make_grid() to arrange images in a grid of 5 rows
    image_grid = make_grid(image_flatten[: num_images], nrow=4)
    #### adjust dimension serve for matplotlib from [c, h w] to [h, w ,c], remove singleton dimenisons in case for single image or grayscale image
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()

    #####save image
    save_image(image_grid, filename)



def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)



### compute the adversial loss for generator
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    ### real: the real image to be used to evaluate the reconstruction
    ### condition: segmentation mask/source image(e.g satellite image) like label class in condtinoal GAN
    ### recon_criterio: pixel distance between fake image and real image
    ### adv_criterion: adversial loss function. feed the fake image and condition  to the disc to cimpute the loss
    ### lambda_recon: the degree to which the reconstruction loss should be weighted in the sum

    fake_image = gen(condition)
    dis_fake_prediction = disc(fake_image, condition)
    #### Generator try to fool the disc
    adv_loss = adv_criterion(dis_fake_prediction, torch.ones_like(dis_fake_prediction))
    ### pixel difference
    recon_loss = recon_criterion(real, fake_image)

    gen_loss = adv_loss + recon_loss*lambda_recon

    return gen_loss


#### define the trainign process
def train(save_model,dataloader, gen, disc, gen_opt, disc_opt, adv_criterion, recon_criterion, 
          lambda_recon, device,n_epochs,display_step,target_shape,input_dim,real_dim,):
    average_disc_loss = 0
    average_gen_loss = 0
    ## prepare dataloader
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        #  "combined_image": single batch from dataloader
        for i, combined_image in enumerate(tqdm(dataloader)):
            ### both condition and target images are drived from the same input image
            image_width = combined_image.shape[3]
            real = combined_image[:,:,:, :image_width//2] 
            condition = combined_image[:,:,:, image_width//2:] 

            # Resize images to the specific target shape
            condition = nn.functional.interpolate(condition, size=(target_shape, target_shape))
            real = nn.functional.interpolate(real, size=(target_shape, target_shape))

            condition = condition.to(device)
            real = real.to(device)
            cur_batch_size = len(condition)

            # Debugging visualization after resizing
            # if i % display_step == 0:  # Visual inspection after resizing
            #     print(f"Debugging: visualization Full Image Before Split")
            #     plt.figure(figsize=(6, 6))
            #     plt.imshow(combined_image[0].permute(1, 2, 0).cpu().numpy())
            #     plt.title("Full Image Before Split")
                
            #     # Condition Image After Resize
            #     plt.figure(figsize=(6, 6))
            #     plt.imshow(condition[0].permute(1, 2, 0).cpu().numpy())
            #     plt.title(f"Split Condition Image After Resize\n")
            #     plt.show()
                
            #     # Real Image After Resize
            #     plt.figure(figsize=(6, 6))
            #     plt.imshow(real[0].permute(1, 2, 0).cpu().numpy())
            #     plt.title(f"Split Real Image After Resize\n")
            #     plt.show()


            #### Update discriminator
            with torch.no_grad():
                fake = gen(condition)
            
            disc_fake_pred = disc(fake.detach(), condition)
            disc_fake_loss = adv_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_real_pred = disc(real,  condition)
            ##### One side Label Smoothing
            soft_real_labels = 0.95*torch.ones_like(disc_real_pred)
            disc_real_loss = adv_criterion(disc_real_pred, soft_real_labels)

            disc_loss = (disc_fake_loss + disc_real_loss)/2

            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()


            ### Update generator ###
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            ### keep record of the average discriminator loss
            average_disc_loss += disc_loss.item()/display_step

            ### keep record of the average generator loss
            average_gen_loss += gen_loss.item()/display_step


            ### Visualization code ###
            if cur_step % display_step ==0 and cur_step > 0:
                print(f"EPOCH {epoch}, STEP {cur_step}, AVER_Generator_loss {average_gen_loss}, AVER_Discri_loss {average_disc_loss}")

                save_image_tensor(condition, size=(input_dim, target_shape, target_shape),
                                  filename=f"Final_0.0002_Condition_output_{cur_step}.png")
                save_image_tensor(real, size=(real_dim, target_shape, target_shape),
                                  filename=f"Final_0.0002_Real_output_{cur_step}.png")
                save_image_tensor(fake, size=(real_dim, target_shape, target_shape),
                                  filename=f"Final_0.0002_Fake_output_{cur_step}.png")
                
                ### reset avaerge loss to zero
                average_disc_loss = 0
                average_gen_loss = 0

                ### Saving model at the regualr intervals protects against data loss 
                if save_model:
                    ### saved filename: "Satellite_Pix2pix_{cur_step}"
                    torch.save({"gen":gen.state_dict(), "gen_opt":gen_opt.state_dict(),
                                "disc":disc.state_dict(), "disc_opt":disc_opt.state_dict()}, 
                                f"Final_0.0002_Satellite_Pix2Pix_NEW{cur_step}.pth")

            cur_step +=1





def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #### Set the paramters
    adv_criterion = nn.BCEWithLogitsLoss() 
    recon_criterion = nn.L1Loss() 
    lambda_recon = 200##
    num_samples=5
    n_epochs = 201 #20

    input_dim = 3
    real_dim = 3

    display_step = 2000#1500#1000#200 #500
    batch_size = 8##8
    batch_size_mean_var = 256
        
    lr_gen = 0.002 #      0.001  0.0007  0.008
    lr_disc = 0.0002
    lr_uniform = 0.0002
    target_shape = 256 
    save_model=True
    # image_path =r'Emply_U_network'   # DFKI_MASS_New\disc_update_15step_lamda_0.7
    # os.makedirs(image_path, exist_ok=True)

    pretrained_U_network = True

    # Setup initial transformations without normalization for mean/std calculation
    initial_transforms = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])

    #  # Reinitialize dataset with the proper transformations
    dataset = VOCPairedDataset(
        image_dir='path_to_voc_train_dataset/VOCdevkit/VOC2012/JPEGImages',
        mask_dir='path_to_voc_train_dataset/VOCdevkit/VOC2012/SegmentationClass',
        transform=initial_transforms  #transformations
    )

    ###### Verify "real image" and "condtion image" correctly 
    print(f"verify the customized dataset before construct dataloader\n")
   # verify_dataset_pairs(dataset, num_samples)
    verify_dataset_pairs_new(dataset, num_samples)

    # DataLoader setup
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ### initialize your generator (U-Net) and discriminator, as well as their optimizers.
    gen = UNet(input_dim, real_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr = lr_uniform) #lr = lr_gen

    disc = Discriminator(input_dim + real_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr = lr_uniform) # lr = lr_disc

    gen = gen.apply(weight_init)
    disc = disc.apply(weight_init)


    print(f"start load the pretrained model based on the previose pretrained U_network____Final_0.0002__-_\n")

    try:
        if pretrained_U_network:
            load_state = torch.load("Final_Satellite_Pix2Pix_NEW18000.pth")

            gen.load_state_dict(load_state["gen"])
            gen_opt.load_state_dict(load_state["gen_opt"])

            disc.load_state_dict(load_state["disc"])
            disc_opt.load_state_dict(load_state["disc_opt"])
            print(f"Pretrained model loaded successfully____LAST_LR_0.002____\n")

    except FileNotFoundError:
        print("Error: No file found for the pretrained model\n")
    except Exception as e:
        print(f"An error occurred while loading the pretrained model: {e}\n")


    # Train
    train(save_model,dataloader, gen, disc, gen_opt, disc_opt, adv_criterion, recon_criterion, 
          lambda_recon, device,n_epochs,display_step,target_shape,input_dim,real_dim)

if __name__ == "__main__":
    main()
