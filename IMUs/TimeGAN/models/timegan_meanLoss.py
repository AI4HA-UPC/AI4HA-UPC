# -*- coding: UTF-8 -*-
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.autograd as autograd
from torch.autograd import Variable

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
        """self.emb_rnn = torch.nn.GRU(
            input_size=(self.feature_dim+num_classes+self.feature_dim),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )"""
        
        self.emb_rnn = torch.nn.GRU(
            input_size=(self.feature_dim+num_classes),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.emb_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

        # cGANs with Projection Discriminator
        self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

    def forward(self, X, T, labels=None, prev_step=None):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F) --> Batch x Seq Length x input_size
            - T: input temporal information (B)
        Returns:
            - H: latent space embeddings (B x S x H)
        """
        if labels is not None:
            labels = labels.to(X.device)
            x = self.label_embedding(labels)  # Shape: [batch_size, num_classes, latent_dim]
            # Repeat or expand the label embeddings to match sequence length of X
            # Assuming the same label embedding for each time step
            sequence_length = X.size(1)
            x = x.unsqueeze(1).repeat(1, sequence_length, 1)
        if prev_step is not None:
            prev_step = prev_step.to(X.device)
        
        #X = torch.cat((X, x, prev_step), dim=-1) if labels is not None and prev_step is not None else X 
        X = torch.cat((X, x), dim=-1) if labels is not None else X 
        # Concatenate along the feature dimension
        #X = torch.cat((X, x), dim=-1)

        # Dynamic RNN input for ignoring paddings
        #########################################
        # N = batch size, L = sequence length, D = 2 if bidirectional=True, otherwise 1, Hin = input_size
        # Hout = hidden_size
        # input: Tensor of shape (L,Hin) for unbatched input, (L,N,Hin) when batch_first=False or
        # (N,L,Hin) when batch_first=True containing the features of the input sequence.
        # The input can also be a packed variable length sequence.
        # h_0: tensor of shape (D * num_layers, Hout) or (D * num_layers, N, Hout) containing the
        # initial hidden state for the input sequence. Defaults to zeros if not provided.
        #########################################
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        #########################################
        # Output: tensor (N, L, D * Hout) if bach_first=True containing the output features (h_t) from
        # the last layer of the GRU, for each t.
        # h_n: tensor of shape (D*num_layers,Hout) or (D*num_layers,N,Hout) containing the
        # final hidden state for the input sequence.
        #########################################
        H_o, H_t = self.emb_rnn(X_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        #########################################
        # transform input features into a space where they can be more easily separable or to
        # learn high-level features.
        #########################################
        logits = self.emb_linear(H_o)

        #########################################
        # Introduce non-linearities.
        # This is crucial because it allows the network to learn complex patterns in the data.
        #########################################
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

        # Recovery Architecture
        """self.rec_rnn = torch.nn.GRU(
            input_size=(self.hidden_dim+num_classes+input_dim),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )"""
        self.rec_rnn = torch.nn.GRU(
            input_size=(self.hidden_dim+num_classes),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.rec_linear = torch.nn.Linear(self.hidden_dim, self.feature_dim)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.rec_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.rec_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

        # cGANs with Projection Discriminator
        self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

    def forward(self, H, T, labels=None, prev_step=None):
        """Forward pass for the recovering features from latent space to original space
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - X_tilde: recovered data (B x S x F)
        """
        if labels is not None:
            labels = labels.to(H.device)
            x = self.label_embedding(labels)  # Shape: [batch_size, num_classes, latent_dim]
            # Repeat or expand the label embeddings to match sequence length of X
            # Assuming the same label embedding for each time step
            sequence_length = H.size(1)
            if len(x.shape) < 3:
                x = x.unsqueeze(1).repeat(1, sequence_length, 1)
                
        if prev_step is not None:
            prev_step = prev_step.to(H.device)  
            
        H = torch.cat((H, x), dim=-1) if labels is not None else H 
        
        # Concatenate along the feature dimension
        # H = torch.cat((H, x), dim=-1)


        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, H_t = self.rec_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        X_tilde = self.rec_linear(H_o)
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

        # Supervisor Architecture
        """self.sup_rnn = torch.nn.GRU(
            input_size=(self.hidden_dim+num_classes+input_dim),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers - 1,
            batch_first=True
        )"""
        self.sup_rnn = torch.nn.GRU(
            input_size=(self.hidden_dim+num_classes),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers - 1,
            batch_first=True
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

        # cGANs with Projection Discriminator
        self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

    def forward(self, H, T, labels=None, prev_step=None):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        if labels is not None:
            labels = labels.to(H.device)
            x = self.label_embedding(labels)  # Shape: [batch_size, num_classes, latent_dim]
            # Repeat or expand the label embeddings to match sequence length of X
            # Assuming the same label embedding for each time step
            sequence_length = H.size(1)
            x = x.unsqueeze(1).repeat(1, sequence_length, 1)
            
        if prev_step is not None:
            prev_step = prev_step.to(H.device)
            
        H = torch.cat((H, x), dim=-1) if labels is not None else H
        # Concatenate along the feature dimension
        #H = torch.cat((H, x), dim=-1)

        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, H_t = self.sup_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        logits = self.sup_linear(H_o)
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

        # cGANs with Projection Discriminator
        self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

        # Generator Architecture
        """self.gen_rnn = torch.nn.GRU(
            input_size=(self.Z_dim+num_classes+input_dim),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )"""
        self.gen_rnn = torch.nn.GRU(
            input_size=(self.Z_dim+num_classes),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

        self.classification = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, num_classes))

    def forward(self, Z, T, labels=None, prev_step=None):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns: 
            - H: embeddings (B x S x E)
        """

        if labels is not None:
            labels = labels.to(Z.device) 
            x = self.label_embedding(labels)  # Shape: [batch_size, num_classes, latent_dim]
            # Repeat or expand the label embeddings to match sequence length of X
            # Assuming the same label embedding for each time step
            sequence_length = Z.size(1)
            x = x.unsqueeze(1).repeat(1, sequence_length, 1)
            
        if prev_step is not None:
            prev_step = prev_step.to(Z.device)
            
        Z = torch.cat((Z, x), dim=-1) if labels is not None else Z 
        
        # Concatenate along the feature dimension
        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, H_t = self.gen_rnn(Z_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        class_out = self.classification(H_o)
        # Apply softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(class_out, dim=-1)
        # Use argmax to get the most likely class label
        predicted_labels = torch.argmax(probabilities, dim=-1)

        logits = self.gen_linear(H_o)
        H = self.gen_sigmoid(logits)

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

        # Discriminator Architecture
        """self.dis_rnn = torch.nn.GRU(
            input_size=(self.hidden_dim+num_classes+input_dim), 
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )"""
        self.dis_rnn = torch.nn.GRU(
            input_size=(self.hidden_dim+num_classes), 
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

        self.classification = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, num_classes))

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(batch_size, output_dim),
            torch.nn.Sigmoid())

    def forward(self, H, T, labels=None, prev_step=None):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        if labels is not None:
            labels = labels.to(H.device)
            x = self.label_embedding(labels)  # Shape: [batch_size, num_classes, latent_dim]
            # Repeat or expand the label embeddings to match sequence length of X
            # Assuming the same label embedding for each time step
            sequence_length = H.size(1)
            x = x.unsqueeze(1).repeat(1, sequence_length, 1)
            
        if prev_step is not None:
            prev_step = prev_step.to(H.device)
            
           
        H = torch.cat((H, x), dim=-1) if labels is not None else H
        
        # Concatenate along the feature dimension
        # H = torch.cat((H, x), dim=-1)

        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, H_t = self.dis_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        #cond_output = torch.inner(H_o, x)
        class_out = self.classification(H_o)

        logits = self.dis_linear(H_o).squeeze(-1)

        # Apply softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(class_out, dim=-1)
        # Use argmax to get the most likely class label
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
        # Loss function
        self.criterion = torch.nn.BCELoss()
        
    def compute_gradient_penalty(self, D, real_samples, fake_samples, T, labels, prev_step):
      """Calculates the gradient penalty loss for WGAN GP"""
      cuda = True if torch.cuda.is_available() else False
      Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
      # Random weight term for interpolation between real and fake samples
      alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
      # Get random interpolation between real and fake samples
      interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
      print(f'Alpha: {alpha.shape}')
      print(f'Real_samples: {real_samples.shape}')
      print(f'Fake_samples: {fake_samples.shape}')
      print(f'T: {T.shape}')
      print(f'Labels: {labels.shape}')
      print(f'Prev_shape: {prev_step.shape}')
      print(f'Interpolates: {interpolates.shape}')
      d_interpolates = D(interpolates, T, labels=labels, prev_step=prev_step)
      fake = Variable(Tensor(real_samples.shape[0], 100).fill_(1.0), requires_grad=False)
      print(f'd_interpolates: {d_interpolates[0].shape}')
      print(f'fake: {fake.shape}')
      # Get gradient w.r.t. interpolates
      gradients = autograd.grad(
          outputs=d_interpolates[0],
          inputs=interpolates,
          grad_outputs=fake,
          create_graph=True,
          retain_graph=True,
          only_inputs=True,
      )[0]
      gradients = gradients.reshape(gradients.size(0), -1)
      gradient_penalty = ((gradients.norm(2, dim=1) -1) ** 2).mean()
      return gradient_penalty



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
        # Generating fake images
        E_hat, _ = self.generator(Z, T, labels, prev_step=prev_step)
        E_hat = E_hat.detach()
        H_hat = self.supervisor(E_hat, T, labels, prev_step=prev_step).detach()

        # Forward Pass
        # Disciminating real data
        Y_real, Y_class_real, _ = self.discriminator(H, T, labels, prev_step=prev_step)  # Encoded original data
        # Disciminating fake images
        Y_fake, Y_class_fake, _ = self.discriminator(H_hat, T, labels, prev_step=prev_step)  # Output of generator + supervisor
        Y_fake_e, Y_class_fake_e, _ = self.discriminator(E_hat, T, labels, prev_step=prev_step) # Output generator
        
        # Calculating discrimination loss (real images)
        Y_real = torch.sigmoid(Y_real)
        real_loss = self.criterion(Y_real, Variable(torch.ones_like(Y_real)))
        # Calculating discrimination loss (fake images)
        Y_fake = torch.sigmoid(Y_fake)
        fake_loss = self.criterion(Y_fake, Variable(torch.zeros_like(Y_fake)))
        
        # Sum two losses
        D_loss = real_loss + fake_loss
      
        Y_class_true_avg = torch.mean(Y_class_real, dim=1)  # Shape: [128, 4]
        labels = labels.to(Y_class_real.device)
        D_loss_cls_true = self.crit_cls(Y_class_true_avg, labels)

        Y_class_fake_avg = torch.mean(Y_class_fake, dim=1)  # Shape: [128, 4]
        D_loss_cls_fake = self.crit_cls(Y_class_fake_avg, labels)

        D_loss_cls = D_loss_cls_true + D_loss_cls_fake

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
        #E_hat = self.generator(Z, T, labels, prev_step=prev_step)
        H_hat = self.supervisor(E_hat, T, labels, prev_step=prev_step)

        # Synthetic data generated
        X_hat, _ = self.recovery(H_hat, T, labels, prev_step=prev_step)
        #X_hat = self.recovery(H_hat, T, labels, prev_step=prev_step)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake, Y_class_fake, _ = self.discriminator(H_hat, T, labels, prev_step=prev_step)  # Output of supervisor
        Y_fake_e, Y_class_fake_e, _ = self.discriminator(E_hat, T, labels, prev_step=prev_step)  # Output of generator
        
        Y_fake = torch.sigmoid(Y_fake)
        G_loss_U = self.criterion(Y_fake, Variable(torch.ones_like(Y_fake)))
        Y_fake_e = torch.sigmoid(Y_fake_e)
        G_loss_U_e = self.criterion(Y_fake_e, Variable(torch.ones_like(Y_fake_e)))

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

    def forward(self, X, T, Z, obj, gamma=1, labels=None, prev_step=None, criterion=None):
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

