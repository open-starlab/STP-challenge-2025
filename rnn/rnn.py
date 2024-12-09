import torch
import torch.nn as nn

from rnn.utils import get_params_str, cudafy_list
from rnn.utils import nll_gauss
from rnn.utils import batch_error, roll_out, roll_out_test
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ['x_dim', 'y_dim', 'z_dim', 'h_dim', 'rnn_dim', 'n_layers', 'n_agents']
        self.params = params
        self.params_str = get_params_str(self.model_args, params)

        x_dim = params['x_dim'] # action
        y_dim = params['y_dim'] # state 
        z_dim = params['z_dim']
        h_dim = params['h_dim']
        rnn_dim = params['rnn_dim']
        n_layers = params['n_layers']
        self.len_seq = params['len_seq']

        # embedding
        embed_size = params['embed_size']
        self.embed_size = embed_size
        embed_ball_size = params['embed_ball_size'] 
        self.embed_ball_size = embed_ball_size

        # parameters 
        n_all_agents = params['n_all_agents'] # all players        
        n_agents = params['n_agents']
        n_feat = params['n_feat']  # dim
        ball_dim = params['ball_dim']
        
        dropout = 0.5 # 
        # dropout2 = 0
        self.xavier = True # initial value
        self.att_in = False # customized attention input

        self.batchnorm = True # if self.attention >= 2 else False
        print('batchnorm = '+str(self.batchnorm))

        # network parameters 
        if n_agents <= 11:
            self.n_network = 1
        elif n_agents <= 22:
            self.n_network = 2
        elif n_agents == 23:
            self.n_network = 3

        # RNN
        if self.batchnorm:
            self.bn_dec = nn.ModuleList([nn.BatchNorm1d(h_dim) for i in range(self.n_network)]) 

        in_enc = x_dim+y_dim+rnn_dim           

        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) for i in range(self.n_network)])
        self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(self.n_network)])

        self.rnn = nn.ModuleList([nn.GRU(in_enc, rnn_dim, n_layers) for i in range(self.n_network)])
        # self.rnn = nn.ModuleList([nn.GRU(y_dim, rnn_dim, n_layers) for i in range(n_agents)])

    def forward(self, states, rollout, train, hp=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        out = {}
        out2 = {}
        out['L_rec'] = torch.zeros(1).to(device)
        out2['e_pos'] = torch.zeros(1).to(device)
        out2['e_vel'] = torch.zeros(1).to(device)
        
        n_agents = self.params['n_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        fs = self.params['fs'] # added
        x_dim = self.params['x_dim']  
        burn_in = self.params['burn_in']

        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)

        h = [torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_dim']) for i in range(n_agents)]
        if self.params['cuda']:
            h = cudafy_list(h)

        for i in range(n_agents):  
            if i <= 11:
                i_network = 0
            elif i <= 22:
                i_network = 1
            elif i == 23:
                i_network = 2
            for t in range(len_time):        
                y_t = states[t][0].clone() # state
                x_t0 = states[t+1][0][:,n_feat*i:n_feat*i+n_feat].clone() 

                # action
                x_t = x_t0[:,2:4] # vel 

                current_pos = y_t[:,n_feat*i:n_feat*i+2]
                current_vel = y_t[:,n_feat*i+2:n_feat*i+4]    
                v0_t1 = x_t0[:,2:4]                

                # RNN
                state_in = y_t
                enc_in = torch.cat([current_vel, state_in, h[i][-1]], 1)

                dec_t = self.dec[i_network](h[i][-1])
                dec_mean_t = self.dec_mean[i_network](dec_t) # + current_vel
                _, h[i] = self.rnn[i_network](enc_in.unsqueeze(0), h[i])

                # objective function
                out['L_rec'] += batch_error(dec_mean_t, x_t)

                # body constraint            
                v_t1 = dec_mean_t[:,:2]   
                next_pos = current_pos + current_vel*fs

                if t >= burn_in or burn_in==len_time:
                    # error (not used when backward)
                    out2['e_pos'] += batch_error(next_pos, x_t0[:,:2])
                    out2['e_vel'] += batch_error(v_t1, v0_t1)

                    del v_t1, current_pos, next_pos

        out2['e_pos'] /= (len_time-burn_in)*n_agents
        out2['e_vel'] /= (len_time-burn_in)*n_agents

        out['L_rec'] /= (len_time)*n_agents
        return out, out2

    def sample(self, states, rollout, burn_in=0, n_sample=1,TEST=False,Challenge=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        out = {}
        out2 = {}
        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)


        if not Challenge:
            out['L_rec'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
            out2['e_pos'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
            out2['e_vel'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
            out2['e_e_p'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
            out2['e_e_v'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        
        Sum = True if not TEST else False

        n_agents = self.params['n_agents']
        n_all_agents = self.params['n_all_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        fs = self.params['fs'] # added
        x_dim = self.params['x_dim']
        burn_in = self.params['burn_in']

        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)

        h = [[torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_dim']) for _ in range(n_sample)] for i in range(n_agents)]
        
        if self.params['cuda']:
            states = cudafy_list(states)
            for i in range(n_agents):
                h[i] = cudafy_list(h[i])    
            for i in range(self.n_network):                
                self.rnn[i] = self.rnn[i].to(device)
                self.dec[i] = self.dec[i].to(device)
                self.dec_mean[i] = self.dec_mean[i].to(device)

        # states = states.repeat(1,n_agents,1,1).clone()
        if Challenge:
            zero_frames = torch.zeros(self.len_seq-burn_in+1, states.size(1), states.size(2), states.size(3)).to(device)
            states = torch.cat((states,zero_frames), dim=0)
            states_n = [states.clone() for _ in range(n_sample)]
        else:
            states_n = [states.clone() for _ in range(n_sample)]

        for t in range(len_time):
            for n in range(n_sample):
                prediction_all = torch.zeros(batchSize, n_agents, x_dim)

                for i in range(n_agents):
                    if i <= 11:
                        i_network = 0
                    elif i <= 22:
                        i_network = 1
                    elif i == 23:
                        i_network = 2

                    y_t = states_n[n][t][0].clone() # states[t][i].clone() # state

                    if not Challenge:
                        x_t0 = states[t+1][0][:,n_feat*i:n_feat*i+n_feat].clone() 

                        # action
                        x_t = x_t0[:,2:4] # vel 

                    # for evaluation
                    current_pos = y_t[:,n_feat*i:n_feat*i+2]
                    current_vel = y_t[:,n_feat*i+2:n_feat*i+4]    
                    if not Challenge:
                        v0_t1 = x_t0[:,2:4]

                    state_in = y_t
                    enc_in = torch.cat([current_vel, state_in, h[i][n][-1]], 1)

                    dec_t = self.dec[i_network](h[i][n][-1])
                    dec_mean_t = self.dec_mean[i_network](dec_t) # + current_vel
                    # objective function
                    if not Challenge:
                        out['L_rec'][n] += batch_error(dec_mean_t, x_t, Sum)

                    v_t1 = dec_mean_t[:,:2]   
                    next_pos = current_pos + current_vel*fs

                    if t >= burn_in: 
                        # prediction
                        prediction_all[:,i,:] = dec_mean_t[:,:x_dim]
                        if not Challenge:
                            # error (not used when backward)
                            out2['e_pos'][n] += batch_error(next_pos, x_t0[:,:2], Sum)
                            out2['e_vel'][n] += batch_error(v_t1, v0_t1, Sum)
                            if t == len_time-1:
                                out2['e_e_p'][n] += batch_error(next_pos, x_t0[:,:2], Sum)
                                out2['e_e_v'][n] += batch_error(v_t1, v0_t1, Sum)

                        del current_pos, current_vel

                    _, h[i][n] = self.rnn[i_network](enc_in.unsqueeze(0), h[i][n])

                # role out
                if t >= burn_in-1 and rollout:
                    y_t = states_n[n][t].clone() 
                    if n_agents < 23:
                        y_t1_ = states_n[n][t+1][0].clone()  
                    else:
                        y_t1_ = None
                    states_new = roll_out_test(y_t,y_t1_,prediction_all,n_agents,n_feat,ball_dim,fs,batchSize)
                    states_n[n][t+1][0] = states_new.clone()
        
        if not Challenge:
            for n in range(n_sample):
                out['L_rec'][n] /= (len_time-burn_in)*n_agents
                out2['e_pos'][n] /= (len_time-burn_in)*n_agents
                out2['e_vel'][n] /= (len_time-burn_in)*n_agents
                out2['e_e_p'][n] /= n_agents
                out2['e_e_v'][n] /= n_agents

        states = states_n[0]
        return states, out, out2
