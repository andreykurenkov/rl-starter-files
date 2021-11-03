import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from dnc import DNC

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 use_rnn_memory=False,
                 use_text=False,
                 use_discrete_memory=False,
                 use_dnc_memory=False,
                 stack_size=1,
                 repeat_env=False,
                 rnn_mem_size = None,
                 concat_input_to_mem=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_rnn_memory = use_rnn_memory
        self.use_discrete_memory = use_discrete_memory
        self.use_dnc_memory = use_dnc_memory
        self.repeat_env = repeat_env
        self.stack_size = stack_size
        self.use_stacking = stack_size > 1
        self.rnn_mem_size = rnn_mem_size
        self.concat_input_to_mem = concat_input_to_mem

        if not self.repeat_env:
            # Define image embedding
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
            n = obs_space["image"][0]
            m = obs_space["image"][1]
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
            self.embedded_input_size = self.image_embedding_size
        else:
            self.embedded_input_size = 1

        if rnn_mem_size is None:
            self.rnn_hidden_size = self.embedded_input_size
        else:
            self.rnn_hidden_size = rnn_mem_size

        # Define memory
        if self.use_rnn_memory:
            self.memory_rnn = nn.LSTMCell(self.embedded_input_size, self.rnn_hidden_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        if self.repeat_env:
            if self.concat_input_to_mem and self.use_rnn_memory:
                self.embedding_size+= self.rnn_hidden_size
            elif self.use_rnn_memory:
                self.embedding_size= self.rnn_hidden_size
            elif self.use_stacking:
                self.embedding_size = self.stack_size
            else:
                self.embedding_size = self.embedded_input_size
        else:
            if self.use_text:
                self.embedding_size += self.text_embedding_size
            if use_discrete_memory:
                self.embedding_size += 8

        if self.use_dnc_memory:
            self.dnc = DNC(
              input_size=self.embedded_input_size,
              output_size=64,
              hidden_size=32,
              rnn_type='lstm',
              num_layers=4,
              nr_cells=20,
              cell_size=10,
              read_heads=4,
              batch_first=True,
              gpu_id=-1
            )
            self.embedding_size = 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        if self.use_dnc_memory:
            return self.dnc.hidden_size
        else:
            return 2*self.rnn_hidden_size

    def forward(self, obs, memory):
        if self.repeat_env:
            x = obs
        else:
            x = obs.image.transpose(1, 3).transpose(2, 3)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)

        if self.use_rnn_memory:
            hidden = (memory[:, :self.rnn_hidden_size], memory[:, self.rnn_hidden_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            if self.concat_input_to_mem:
                memory = torch.cat((x,hidden[0]),dim=0)
            else:
                embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        elif self.use_discrete_memory:
            embedding = torch.cat([x, torch.Tensor(obs.memory)], dim=1)
        elif self.use_dnc_memory:
            (controller_hidden, memory, read_vectors) = memory
            x_expanded = x[:,:,None]
            embedding, (controller_hidden, memory, read_vectors) = \
              self.dnc(x_expanded, (controller_hidden, memory, read_vectors))
            embedding = embedding[:,0,:]

            memory = (controller_hidden, memory, read_vectors)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
