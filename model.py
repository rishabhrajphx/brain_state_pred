# Model adapted from the following link
# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# for the following paper
# https://arxiv.org/abs/2001.08317
# This code creates a special type of AI model called a "Transformer" that can predict future values in a time series (like stock prices or weather data)


import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

# First, we create a helper class that adds position information to our data
class PositionalEncoder(nn.Module):
    def __init__(
        self,
        dropout: float=0.1,        # Randomly turns off 10% of connections to prevent over-learning
        max_seq_len: int=5000,     # Maximum length of input sequence we can handle
        d_model: int=512,          # Size of our model's internal representation
        batch_first: bool=True     # Technical detail about how data is organized
        ):
        super().__init__()         # Initialize the parent class
        
        # Store these values for later use
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # Create a special pattern that helps the model understand the order of data
        # Think of this like adding timestamps to each piece of data
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)    # Use sine waves for even positions
        pe[:, 0, 1::2] = torch.cos(position * div_term)    # Use cosine waves for odd positions
        self.register_buffer('pe', pe)                     # Save this pattern for later

    # This function adds the position information to our input data
    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
             x = x + self.pe[:x.size(self.x_dim)].squeeze().unsqueeze(0)
        else:
            x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

# Main model class that will make predictions
class TimeSeriesTransformer(nn.Module):
    def __init__(self,
        input_size: int,                    # Size of each input data point
        dec_seq_len: int,                   # Length of sequence to predict
        batch_first: bool=True,             # Technical detail about data organization
        out_seq_len: int=58,               # Length of output sequence
        max_seq_len: int=5000,             # Maximum sequence length
        dim_val: int=512,                  # Size of internal representation
        n_encoder_layers: int=4,           # Number of processing layers for input
        n_decoder_layers: int=4,           # Number of processing layers for output
        n_heads: int=8,                    # Number of attention mechanisms
        dropout_encoder: float=0.2,        # Dropout rate for encoder
        dropout_decoder: float=0.2,        # Dropout rate for decoder
        dropout_pos_enc: float=0.1,        # Dropout rate for position encoding
        dim_feedforward_encoder: int=2048, # Size of encoder's internal processing
        dim_feedforward_decoder: int=2048, # Size of decoder's internal processing
        num_predicted_features: int=1      # Number of features to predict
        ):
        super().__init__()
        
        # Store decoder sequence length
        self.dec_seq_len = dec_seq_len

        #------ ENCODER SECTION ------#
        # This part processes the input data
        
        # Convert input data to model's internal format
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
            )

        # Add position information
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            batch_first=batch_first
            )

        # Create the main processing layers for input
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )
        
        # Stack multiple encoder layers together
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
            )

        #------ DECODER SECTION ------#
        # This part generates the predictions
        
        # Convert prediction target to model's internal format
        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
            )

        # Create the main processing layers for output
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        # Stack multiple decoder layers together
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
            )

        # Convert from internal format back to prediction format
        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
            )

    # This function runs the actual prediction process
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
                tgt_mask: Tensor=None) -> Tensor:
        # Process input through encoder
        src = self.encoder_input_layer(src)                # Convert input format
        src = self.positional_encoding_layer(src)          # Add position information
        src = self.encoder(src=src)                        # Process through encoder
        
        # Generate predictions through decoder
        decoder_output = self.decoder_input_layer(tgt)     # Convert target format
        decoder_output = self.decoder(                     # Process through decoder
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )
        decoder_output = self.linear_mapping(decoder_output)  # Convert to final prediction
        return decoder_output
