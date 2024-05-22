from extras.BLIP.models.med import BertConfig  # Importing BertConfig class from the med module
from extras.BLIP.models.nlvr_encoder import BertModel  # Importing BertModel class from the nlvr_encoder module
from extras.BLIP.models.vit import interpolate_pos_embed  # Importing interpolate_pos_embed function from the vit module
from extras.BLIP.models.blip import create_vit, init_tokenizer, is_url  # Importing create_vit, init_tokenizer, and is_url functions from the blip module

# Importing required classes and functions from the timm.models.hub module
import timm.models.hub
from timm.models.hub import download_cached_file

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import os

class BLIP_NLVR(nn.Module):
    def __init__(self,  # Initializing the class with the following arguments
                 med_config='configs/med_config.json',  
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()  
        
        # Creating the visual encoder and getting the vision width
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        
        # Initializing the tokenizer
        self.tokenizer = init_tokenizer()   
        
        # Loading the med_config as a BertConfig object
        med_config = BertConfig.from_json_file(med_config)
        
        # Setting the encoder width of the med_config to the vision width
        med_config.encoder_width = vision_width
        
        # Creating the text encoder with the updated med_config
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 
                    
        # Initializing the classification head
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

    def forward(self, image, text, targets, train=True):
        
        # Getting the image embeddings from the visual encoder
        image_embeds = self.visual_encoder(image) 
        
        # Creating the image attributes tensor
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        
        # Splitting the image embeddings based on the targets tensor size
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))     

        # Tokenizing the text and converting it to tensors
        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device) 
        
        # Setting the first token of the input_ids tensor to the encoder token id
        text.input_ids[:,0] = self.tokenizer.enc_token_id        

        # Passing the input_ids and attention_mask tensors through the text encoder
        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        
        # Extracting the hidden state from the output tensor
        hidden_state = output.last_hidden_state[:,0,:]        

        # Passing the hidden state through the classification head
        prediction = self.cls_head(hidden_state)

        if train:            
            # Calculating the cross entropy loss
            loss = F.cross_entropy(prediction, targets)   
            return loss
        else:
            return prediction
    
def blip_nlvr(pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        # Loading the checkpoint and printing the missing keys
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

        
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        # Downloading the cached file from the given url
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        # Loading the checkpoint from the given file path
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    # Interpolating the pos_embed tensor from the checkpoint
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    # Renaming some keys in the state_dict
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  
                
    # Loading the state_dict onto the model
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
