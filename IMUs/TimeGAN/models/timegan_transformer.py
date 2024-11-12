# -*- coding: UTF-8 -*-
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """
    Generates a sinusoidal positional encoding.
    Args:
        seq_len (int): The length of the sequence.
        d_model (int): The dimensionality of the embeddings.
        device (torch.device): The device to run the model on.
    Returns:
        torch.Tensor: The positional encoding.
    """
    position = torch.arange(seq_len).unsqueeze(1).float().to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)).to(device)
    
    pe = torch.zeros(seq_len, d_model).to(device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

class EmbeddingNetwork(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN
    """

    def __init__(self, args, input_dim=100, latent_dim=128, seq_len=100, num_classes=4):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Embedder Architecture
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=num_classes + args.feature_dim,
            nhead=1,  # number of heads (hidden_dim must be divisible by nhead)
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        self.emb_linear = torch.nn.Linear(num_classes + args.feature_dim, self.hidden_dim) # self.hidden_dim + 4
        self.emb_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614

        # cGANs with Projection Discriminator
        self.label_embedding = torch.nn.Embedding(num_classes, num_classes)
        
    def generate_square_subsequent_mask(self, sz, T):
        # Creates a mask for sequence lengths defined in T
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # Adjusting the mask to accommodate sequences of varying lengths
        mask_expanded = mask.unsqueeze(0).expand(T.shape[0], sz, sz)
        for i, length in enumerate(T):
            mask_expanded[i, length:] = float('-inf')

        return mask_expanded

    def forward(self, X, T, labels=None, prev_step=None):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F) --> Batch x Seq Length x input_size
            - T: input temporal information (B)
        Returns:
            - H: latent space embeddings (B x S x H)
        """
        
         # Assuming X is batch first and shape is [batch, seq, feature]
        X = X.permute(1, 0, 2)  # Transformer expects [seq, batch, feature]
        
        if labels is not None:
            # Embed the labels and repeat them to match sequence length
            labels = labels.to(X.device)
            label_emb = self.label_embedding(labels)  # Shape: [batch_size, feature_dim]
            label_emb = label_emb.unsqueeze(0).repeat(X.size(0), 1, 1)  # Shape: [seq, batch, feature_dim]
            
        X = torch.cat((X, label_emb), dim=-1) if labels is not None else X
                            
        # Generate attention mask
        # Assuming X is [seq_length, batch_size, feature_dim]
        seq_length, batch_size, feature_dim = X.shape

        # Calculate positional encodings
        positional_encodings = sinusoidal_positional_encoding(seq_length, feature_dim, X.device)

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(1).expand(-1, batch_size, -1)

        # Add positional encodings to X
        X = X + positional_encodings

        # Transformer forward pass
        H = self.transformer_encoder(X)#, mask=attn_mask)

        # Convert back to [batch, seq, feature]
        H = H.permute(1, 0, 2)
        logits = self.emb_linear(H)
        H = self.emb_sigmoid(logits)
        return H


class RecoveryNetwork(torch.nn.Module):
    """The recovery network (decoder) for TimeGAN
    """

    def __init__(self, args, input_dim=6, latent_dim=128, seq_len=100, num_classes=4):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Transformer Decoder Layer
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.hidden_dim + input_dim,
            nhead=1,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.rec_linear = torch.nn.Linear(self.hidden_dim + input_dim, self.feature_dim)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614

        # cGANs with Projection Discriminator
        self.label_embedding = torch.nn.Embedding(num_classes, self.feature_dim)
        
    def generate_subsequent_mask(self, sz, T):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask_expanded = mask.unsqueeze(0).expand(T.shape[0], sz, sz)
        for i, length in enumerate(T):
            mask_expanded[i, length:] = float('-inf')
        
        return mask_expanded
        

    def forward(self, H, T, labels=None, prev_step=None):
        H = H.permute(1, 0, 2)  # Transformer expects [seq, batch, feature]

        if labels is not None:
            labels = labels.to(H.device)
            label_emb = self.label_embedding(labels)  # Shape: [batch_size, feature_dim]
            label_emb = label_emb.unsqueeze(0).repeat(H.size(0), 1, 1)  # Shape: [seq, batch, feature_dim]
            
        H = torch.cat((H, label_emb), dim=-1) if labels is not None else X

        # Generate attention mask based on sequence lengths
        # Assuming X is [seq_length, batch_size, feature_dim]
        seq_length, batch_size, feature_dim = H.shape

        # Calculate positional encodings
        positional_encodings = sinusoidal_positional_encoding(seq_length, feature_dim, H.device)

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(1).expand(-1, batch_size, -1)

        # Add positional encodings to X
        H = H + positional_encodings

        # Decoder forward pass
        H_decoded = self.transformer_decoder(H, H)#memory_mask=attn_mask)

        H_decoded = H_decoded.permute(1, 0, 2)  # Back to [batch, seq, feature]

        X_tilde = self.rec_linear(H_decoded)
        label_probabilities = torch.nn.functional.softmax(X_tilde, dim=-1)
        predicted_labels = torch.argmax(label_probabilities, dim=-1)


        return X_tilde, predicted_labels
        


class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN
    """

    def __init__(self, args, input_dim=6, latent_dim=128, seq_len=100, num_classes=4):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

         # Label embedding
        self.label_embedding = torch.nn.Embedding(num_classes, self.hidden_dim)

        # Transformer Encoder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim + self.hidden_dim,#self.hidden_dim + num_classes + 2,
            nhead=1,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.sup_linear = torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614

                    
    def generate_square_subsequent_mask(self, sz, T):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask_expanded = mask.unsqueeze(0).expand(T.shape[0], sz, sz)
        for i, length in enumerate(T):
            mask_expanded[i, length:] = float('-inf')
        
        return mask_expanded

       
    def forward(self, H, T, labels=None, prev_step=None):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        H = H.permute(1, 0, 2)  # Transformer expects [seq, batch, feature]

        if labels is not None:
            labels = labels.to(H.device)
            label_emb = self.label_embedding(labels)  # Shape: [batch_size, feature_dim]
            label_emb = label_emb.unsqueeze(0).repeat(H.size(0), 1, 1)  # Shape: [seq, batch, feature_dim]
            
        H = torch.cat((H, label_emb), dim=-1) if labels is not None else X

        # Generate attention mask
        # Assuming X is [seq_length, batch_size, feature_dim]
        seq_length, batch_size, feature_dim = H.shape

        # Calculate positional encodings
        positional_encodings = sinusoidal_positional_encoding(seq_length, feature_dim, H.device)

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(1).expand(-1, batch_size, -1)

        # Add positional encodings to X
        H = H + positional_encodings

        # Transformer forward pass
        H_transformed = self.transformer_encoder(H)#, mask=attn_mask)

        H_transformed = H_transformed.permute(1, 0, 2)  # Back to [batch, seq, feature]

        logits = self.sup_linear(H_transformed)
        H_hat = self.sup_sigmoid(logits)
        return H_hat


class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """

    def __init__(self, args, input_dim=6, latent_dim=128, seq_len=100, num_classes=4):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Label embedding
        self.label_embedding = torch.nn.Embedding(num_classes, self.Z_dim)

        # Transformer Encoder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim + input_dim,
            nhead=1,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.gen_linear = torch.nn.Linear(input_dim + input_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        self.classification = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim + input_dim),
            torch.nn.Linear(input_dim + input_dim, num_classes))
        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
                    
    def generate_square_subsequent_mask(self, sz, T):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask_expanded = mask.unsqueeze(0).expand(T.shape[0], sz, sz)
        for i, length in enumerate(T):
            mask_expanded[i, length:] = float('-inf')
        
        return mask_expanded


    def forward(self, Z, T, labels=None, prev_step=None):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns: 
            - H: embeddings (B x S x E)
        """
        
        #print(f'Inside generator Z: {Z.shape}')
        #print(f'Inside generator T: {T.shape}')
        #print(f'Inside generator labels: {labels.shape}')
        #print(f'Inside generator prev_step: {prev_step.shape}')

        Z = Z.permute(1, 0, 2)  # Transformer expects [seq, batch, feature]

        if labels is not None:
            labels = labels.to(Z.device)
            label_emb = self.label_embedding(labels)  # Shape: [batch_size, Z_dim]
            label_emb = label_emb.unsqueeze(0).repeat(Z.size(0), 1, 1)  # Shape: [seq, batch, Z_dim]
            
        Z = torch.cat((Z, label_emb), dim=-1) if labels is not None else X

        # Generate attention mask
        # Assuming X is [seq_length, batch_size, feature_dim]
        seq_length, batch_size, feature_dim = Z.shape

        # Calculate positional encodings
        positional_encodings = sinusoidal_positional_encoding(seq_length, feature_dim, Z.device)

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(1).expand(-1, batch_size, -1)

        # Add positional encodings to X
        Z = Z + positional_encodings

        # Transformer forward pass
        H_o = self.transformer_encoder(Z)#, mask=attn_mask)

        H_o = H_o.permute(1, 0, 2)  # Back to [batch, seq, feature]

        logits = self.gen_linear(H_o)
        H = self.gen_sigmoid(logits)

        class_out = self.classification(H_o)
        probabilities = torch.nn.functional.softmax(class_out, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)

        return H, predicted_labels


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, args, input_dim=6, output_dim=100, num_classes=4, batch_size=16, seq_len=0):
        super(DiscriminatorNetwork, self).__init__()
        self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

       
        # Transformer Encoder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim + num_classes,
            nhead=1,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.dis_linear = torch.nn.Linear(self.hidden_dim + num_classes, 1)
        self.classification = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim + num_classes),
            torch.nn.Linear(self.hidden_dim + num_classes, num_classes))

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(batch_size, output_dim),
            torch.nn.Sigmoid())


        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
 
    def generate_square_subsequent_mask(self, sz, T):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask_expanded = mask.unsqueeze(0).expand(T.shape[0], sz, sz)
        for i, length in enumerate(T):
            mask_expanded[i, length:] = float('-inf')
        
        return mask_expanded


    def forward(self, H, T, labels=None, prev_step=None):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        H = H.permute(1, 0, 2)  # Transformer expects [seq, batch, feature]

        if labels is not None:
            labels = labels.to(H.device)
            label_emb = self.label_embedding(labels)  # Shape: [batch_size, hidden_dim]
            label_emb = label_emb.unsqueeze(0).repeat(H.size(0), 1, 1)  # Shape: [seq, batch, hidden_dim]
            
        H = torch.cat((H, label_emb), dim=-1) if labels is not None else X

        # Generate attention mask
        # Assuming X is [seq_length, batch_size, feature_dim]
        seq_length, batch_size, feature_dim = H.shape

        # Calculate positional encodings
        positional_encodings = sinusoidal_positional_encoding(seq_length, feature_dim, H.device)

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(1).expand(-1, batch_size, -1)

        # Add positional encodings to X
        H = H + positional_encodings        

        # Transformer forward pass
        H_o = self.transformer_encoder(H)#, mask=attn_mask)

        H_o = H_o.permute(1, 0, 2)  # Back to [batch, seq, feature]

        logits = self.dis_linear(H_o).squeeze(-1)
        class_out = self.classification(H_o)

        probabilities = torch.nn.functional.softmax(class_out, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)

        return logits, class_out, predicted_labels 


class TimeGAN(torch.nn.Module):
    """Implementation of TimeGAN (Yoon et al., 2019) using PyTorch
    Reference:
    - https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
    - https://github.com/jsyoon0823/TimeGAN
    """

    def __init__(self, args):
        super(TimeGAN, self).__init__() 
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)

        self.crit_cls = torch.nn.CrossEntropyLoss()
        
    def _recovery_forward(self, X, T, labels, prev_step=None): # The embedding and recovery 
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        H = self.embedder(X, T, labels, prev_step=prev_step)
        X_tilde, _ = self.recovery(H, T, labels, prev_step=prev_step)


        # SUPERVISED LOSS (G_loss_S)
        # For Joint training
        # Predict the next step in the sequence
        H_hat_supervise = self.supervisor(H, T, labels, prev_step=prev_step)

        #  Supervised prediction for the next step (H_hat_supervise[:, :-1, :])
        #  and the actual next step in the embedded features (H[:, 1:, :]).
        #  This is known as teacher forcing, where the model is trained to predict
        #  the next step in the sequence.
        
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :],
            H[:, 1:, :]
        )  # Teacher forcing next output

        # RECONSTRUCTION LOSS (E_loss_TO)
        # How well the network can reconstruct the original data from the embedded features.
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        # This scaling could be to ensure that the reconstruction loss has a significant
        # impact on the total loss. The higher weight could be emphasizing the importance
        # of this aspect in the training process.
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        # This indicates that while the supervised loss is important, it should not dominate
        # the total loss. Too much emphasis on this component might lead the model to overfit
        # to the training data, reducing its generalization capability.
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss, E_loss0, E_loss_T0

    def _supervisor_forward(self, X, T, labels, prev_step=None):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        H = self.embedder(X, T, labels, prev_step=prev_step)
        H_hat_supervise = self.supervisor(H, T, labels, prev_step=prev_step)

        # Supervised loss (Ls)
        S_loss = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :],
            H[:, 1:, :])  # Teacher forcing next output
        return S_loss

    def _discriminator_forward(self, X, T, Z, gamma=1, labels=None, prev_step=None):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        H = self.embedder(X, T, labels, prev_step=prev_step).detach()

        # Generator
        E_hat, _ = self.generator(Z, T, labels, prev_step=prev_step)
        E_hat = E_hat.detach()
        H_hat = self.supervisor(E_hat, T, labels, prev_step=prev_step).detach()

        # Forward Pass
        Y_real, Y_class_real, _ = self.discriminator(H, T, labels, prev_step=prev_step)  # Encoded original data
        Y_fake, Y_class_fake, _ = self.discriminator(H_hat, T, labels, prev_step=prev_step)  # Output of generator + supervisor
        Y_fake_e, Y_class_fake_e, _ = self.discriminator(E_hat, T, labels, prev_step=prev_step) 
        

        Y_class_true_avg = torch.mean(Y_class_real, dim=1)  # Shape: [128, 4]
        labels = labels.to(Y_class_real.device)
        D_loss_cls_true = self.crit_cls(Y_class_true_avg, labels)

        Y_class_fake_avg = torch.mean(Y_class_fake, dim=1)  # Shape: [128, 4]
        D_loss_cls_fake = self.crit_cls(Y_class_fake_avg, labels)

        D_loss_cls = D_loss_cls_true + D_loss_cls_fake

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        
        D_loss = -torch.mean(Y_real) + torch.mean(Y_fake)
        D_loss += D_loss_cls

        return D_loss

    def _generator_forward(self, X, T, Z, gamma=1, labels=None, prev_step=None):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        # Supervisor Forward Pass
        H = self.embedder(X, T, labels, prev_step=prev_step)
        H_hat_supervise = self.supervisor(H, T, labels, prev_step=prev_step)
        # Generator Forward Pass
        E_hat, _ = self.generator(Z, T, labels, prev_step=prev_step)
        H_hat = self.supervisor(E_hat, T, labels, prev_step=prev_step)

        # Synthetic data generated
        X_hat, _ = self.recovery(H_hat, T, labels, prev_step=prev_step)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake, Y_class_fake, _ = self.discriminator(H_hat, T, labels, prev_step=prev_step)  # Output of supervisor
        Y_fake_e, Y_class_fake_e, _ = self.discriminator(E_hat, T, labels, prev_step=prev_step)  # Output of generator

        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(
            torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

        return G_loss

    def _inference(self, Z, T, labels=None, prev_step=None):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        E_hat, fake_labels_gen = self.generator(Z, T, labels, prev_step=prev_step)
        H_hat = self.supervisor(E_hat, T, labels, prev_step=prev_step) # Crucial for maintaining the temporal coherence in the generated sequences

        # Synthetic data generated
        # Recovery: Transforms the output of the supervisor (H_hat) back into the original data space,
        # resulting in the synthetic data
        X_hat, _, fake_labels_dis = self.discriminator(E_hat, T, labels,prev_step=prev_step)
        X_hat, fake_labels_recovered = self.recovery(H_hat, T, labels, prev_step=prev_step)
        return X_hat, fake_labels_dis

    def forward(self, X, T, Z, obj, gamma=1, labels=None, prev_step=None):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given") 

            X = torch.FloatTensor(X)
            X = X.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T, labels, prev_step=prev_step)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T, labels, prev_step=prev_step)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z, labels=labels, prev_step=prev_step)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Discriminator
            loss = self._discriminator_forward(X, T, Z, labels=labels, prev_step=prev_step)

            return loss

        elif obj == "inference":

            X_hat, fake_labels = self._inference(Z, T, labels, prev_step=prev_step)
            X_hat = X_hat.cpu().detach()

            return X_hat, fake_labels

        else:
            raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")

        return loss
