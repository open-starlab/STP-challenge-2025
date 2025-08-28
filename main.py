import glob, os, time, math, warnings, copy, re
from datetime import datetime
import argparse
import random
import pickle

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

# customized ftns 
from rnn import load_model
from rnn.utils import num_trainable_params
from tqdm import tqdm

#from scipy import signal

# Keisuke Fujii, 2024

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='robocup2D')
parser.add_argument('--data_dir', type=str, default='robocup2d_data')
parser.add_argument('--n_roles', type=int, default=23)
parser.add_argument('--burn_in', type=int, default=30)
parser.add_argument('-t_step', '--totalTimeSteps', type=int, default=60)
parser.add_argument('--overlap', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--n_epoch', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('-ev_th','--event_threshold', type=int, default=50, help='event with frames less than the threshold will be removed')
parser.add_argument('--fs', type=int, default=1)
parser.add_argument('--cont', action='store_true')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--numProcess', type=int, default=8)
parser.add_argument('--TEST', action='store_true') 
parser.add_argument('--challenge_data', type=str, default=None)
parser.add_argument('--Sanity', action='store_true')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--drop_ind', action='store_true')
args, _ = parser.parse_known_args()

# directories
path_init = './weights/' 
if args.challenge_data is not None:
    args.Challenge = True
else:
    args.Challenge = False
    
if args.Challenge:
    args.TEST = True

def run_epoch(train,rollout,hp):
    loader = train_loader if train == 1 else val_loader if train == 0 else test_loader
 
    losses = {} 
    losses2 = {}
    sample_nan = 0
    for batch_idx, (data) in enumerate(tqdm(loader, desc="Processing batches")):

        if args.cuda:
            data = data.cuda() #, data_y.cuda()
        # (batch, agents, time, feat) => (time, agents, batch, feat)
        data = data.permute(2, 1, 0, 3) #, data.transpose(0, 1)

        if torch.isnan(data).any():
            sample_nan += torch.sum(torch.isnan(data).any(dim=1).any(dim=2).any(dim=0))
            data = data[:,:,~torch.isnan(data).any(dim=1).any(dim=2).any(dim=0)]

        if train == 1:
            batch_losses, batch_losses2 = model(data, rollout, train, hp=hp)
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            if hp['model'] != 'RNN_ATTENTION': 
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            _, batch_losses, batch_losses2 = model.sample(data, rollout=True, burn_in=hp['burn_in'])# , TEST=TEST)
        
        for key in batch_losses:
            if batch_idx == 0:
                losses[key] = batch_losses[key].item()
            else:
                losses[key] += batch_losses[key].item()
        
        for key in batch_losses2:
            if batch_idx == 0:
                losses2[key] = batch_losses2[key].item()
            else:
                losses2[key] += batch_losses2[key].item()

    for key in losses:
        losses[key] /= (len(loader.dataset)-sample_nan)
    for key in losses2:
        losses2[key] /= (len(loader.dataset)-sample_nan)
    return losses, losses2

def loss_str(losses):
    ret = ''
    for key in losses:
        if False: # 'L' in key and not 'mac' in key and not 'vel' in key:
            ret += ' {}: {:.0f} |'.format(key, losses[key])
        elif 'vel' in key:
            ret += ' {}: {:.3f} |'.format(key, losses[key])
        else: 
            ret += ' {}: {:.3f} |'.format(key, losses[key])
    return ret[:-2]

def run_sanity(args,test_loader):
    data = []
    for batch_idx, batch in enumerate(test_loader):
        data.append(batch.numpy())
    data = np.concatenate(data, axis=0)
    data = data[:,0] 

    n_agents = args.n_agents
    batchSize,_,_ = data.shape
    n_feat = args.n_feat
    burn_in = args.burn_in
    fs = args.fs
    GT = data.copy()
    losses = {}
    losses['e_pos'] = np.zeros(batchSize)
    losses['e_vel'] = np.zeros(batchSize)
    losses['e_e_p'] = np.zeros(batchSize)
    losses['e_e_v'] = np.zeros(batchSize)

    for t in range(args.horizon):
        for i in range(n_agents):
            
            current_pos = data[:,t,n_feat*i+0:n_feat*i+2]
            current_vel = data[:,burn_in,n_feat*i+2:n_feat*i+4]
            next_pos0 = GT[:,t+1,n_feat*i+0:n_feat*i+2]
            next_vel0 = GT[:,t+1,n_feat*i+2:n_feat*i+4]

            if t >= burn_in: 
                next_pos = current_pos + current_vel*fs      
                next_vel = current_vel 
                losses['e_pos'] += batch_error(next_pos, next_pos0)
                losses['e_vel'] += batch_error(next_vel, next_vel0)

                data[:,t+1,n_feat*i+0:n_feat*i+2] = next_pos
            if t == args.horizon-1:
                losses['e_e_p'] += batch_error(next_pos, next_pos0)
                losses['e_e_v'] += batch_error(next_vel, next_vel0)
                

    # del data
    losses['e_pos'] /= (args.horizon-burn_in)*n_agents 
    losses['e_vel'] /= (args.horizon-burn_in)*n_agents
    losses['e_e_p'] /= n_agents 
    losses['e_e_v'] /= n_agents

    avgL2_m = {}
    avgL2_sd = {}
    for key in losses:
        avgL2_m[key] = np.mean(losses[key])
        avgL2_sd[key] = np.std(losses[key])

    print('Velocity (Sanity Check)')
    print('Mean:')
    print('  Position Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_pos'], avgL2_sd['e_pos']))
    print('  Velocity Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_vel'], avgL2_sd['e_vel']))
    print('Endpoint:')
    print('  Position Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_e_p'], avgL2_sd['e_e_p']))
    print('  Velocity Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_e_v'], avgL2_sd['e_e_v']))
    
            
    losses['e_pos'] =  np.mean(losses['e_pos'])
    losses['e_e_p'] = np.mean(losses['e_e_p'])
    return losses

def batch_error(predict, true):
    error = np.sqrt(np.sum((predict[:,:2] - true[:,:2])**2,1))
    return error

def bound(val, lower, upper):
    """Clamps val between lower and upper."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val

if __name__ == '__main__':
    numProcess = args.numProcess  
    os.environ["OMP_NUM_THREADS"]=str(numProcess) 
    TEST = args.TEST

    if not torch.cuda.is_available():
        args.cuda = False
        print('cuda is not used')
    else:
        args.cuda = True
        print('cuda is used')

    # Set manual seed
    args.seed = 42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # pre-process----------------------------------------------
    args.filter = True

    global fs
    fs = 1/args.fs
    if args.data == 'soccer':
        n_pl = 11
        
    event_threshold = args.event_threshold
    n_roles = args.n_roles
    batchSize = args.batchsize # 
    overlapWindow = args.overlap # 
    totalTimeSteps =  args.totalTimeSteps # 

    file_paths = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir)]
    os.makedirs("./metadata", exist_ok=True)

    def create_metadata(file_paths, total_time_steps, overlap, output_path):
        """
        Create metadata for play_on intervals and save it as a pickle file.
        
        Args:
            file_paths (list): List of file paths to process.
            output_path (str): Path to save the metadata.
        """
        total_time_steps += 1
        metadata = []
        for file_path in file_paths:
            # Read only the playmode column to identify play_on intervals
            playmode_col = pd.read_csv(file_path, usecols=['playmode']).playmode
            play_on_indices = playmode_col[playmode_col == 'play_on'].index.tolist()

            # Find continuous play_on ranges
            start_idx = play_on_indices[0]
            for i in range(1, len(play_on_indices)):
                if play_on_indices[i] != play_on_indices[i - 1] + 1:
                    # End of a continuous range
                    end_idx = play_on_indices[i - 1]
                    if end_idx - start_idx + 1 >= total_time_steps:  # Check total length condition
                        for j in range(start_idx, end_idx - total_time_steps + 1, total_time_steps - overlap):
                            metadata.append({
                                'file_path': file_path,
                                'start_idx': j,
                                'end_idx': j + total_time_steps
                            })
                    start_idx = play_on_indices[i]

            # Handle the last range
            end_idx = play_on_indices[-1]
            if end_idx - start_idx + 1 >= total_time_steps:
                for j in range(start_idx, end_idx - total_time_steps + 1, total_time_steps - overlap):
                    metadata.append({
                        'file_path': file_path,
                        'start_idx': j,
                        'end_idx': j + total_time_steps
                    })
            
        # Save metadata as a pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {output_path}")
        return metadata

    # update if input csv files are changed
    if not os.path.exists("./metadata/metadata.pkl"):
        print("Creating metadata...")
        metadata = create_metadata(file_paths, args.totalTimeSteps, args.overlap, output_path="./metadata/metadata.pkl")
    else:
        print("Loading metadata...")
        with open("./metadata/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        unique_file_paths = set(item['file_path'] for item in metadata)
        if len(file_paths) != len(unique_file_paths):
            print("Updating metadata...")
            metadata = create_metadata(file_paths, args.totalTimeSteps, args.overlap, output_path="./metadata/metadata.pkl")

    # Split metadata into train, validation, and test sets

    def split_metadata_by_date(metadata, val_ratio=0.1, test_ratio=0.1, val_games=None, test_games=None):
        """
        Split metadata into train, val, and test based on file dates.
        """
        
        # Extract file dates and group metadata by file
        file_date_map = {}
        pattern = re.compile(r"(\d{2})(\d{2})-(\d{4}).*?(202\d)")
        
        def extract_datetime(file_name):
            """
            Extracts a datetime object from the file name.
            """
            match = pattern.search(file_name)
            if match:
                month, day, time, year = match.groups()
                hour, minute = divmod(int(time), 100)
                return datetime(int(year), int(month), int(day), hour, minute)
            else:
                raise ValueError(f"Invalid file name format: {file_name}")

        file_times = {}
        for item in metadata:
            file_name = item['file_path'].split('/')[-1]
            file_times[item['file_path']] = extract_datetime(file_name)

        # Sort files by date
        sorted_files = sorted(file_times.items(), key=lambda x: x[1])

        # Split train, val, test
        files = [file for file, _ in sorted_files]
        if val_ratio is not None and test_ratio is not None:
            train_files, temp_files = train_test_split(files, test_size=val_ratio + test_ratio, shuffle=False)
            val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio), shuffle=False)
        elif val_games is not None and test_games is not None:
            train_files, temp_files = train_test_split(files, test_size=(val_games + test_games)/len(files), shuffle=False)
            val_files, test_files = train_test_split(temp_files, test_size=test_games / (val_games + test_games), shuffle=False)
        # Split metadata based on file assignments
        train_metadata = [item for item in metadata if item['file_path'] in train_files]
        val_metadata = [item for item in metadata if item['file_path'] in val_files]
        test_metadata = [item for item in metadata if item['file_path'] in test_files]
        
        # Shuffle train metadata

        return train_metadata, val_metadata, test_metadata

    # Example usage
    if len(metadata) < 5000:
        train_metadata, val_metadata, test_metadata = split_metadata_by_date(
            metadata, val_ratio=0.1, test_ratio=0.1)
    else:
        train_metadata, val_metadata, test_metadata = split_metadata_by_date(
            metadata, val_ratio=None, test_ratio=None, val_games=1, test_games=1)
        
    print('train: '+str(len(train_metadata))+' val:'+str(len(val_metadata))+' test: '+str(len(test_metadata)))

    class Dataset(Dataset):
        def __init__(self, args, metadata, challenge_data=None):
            self.metadata = metadata
            self.args = args
            self.challenge_data = challenge_data

        def __len__(self):
            if self.challenge_data is None:
                return len(self.metadata)
            else:
                return len(self.challenge_data) 

        def __getitem__(self, idx):
            """
            Dynamically loads the required chunk of data based on metadata.
            """
            if self.challenge_data is None:
                item = self.metadata[idx]
                file_path, start_idx, end_idx = item['file_path'], item['start_idx'], item['end_idx']
                chunk = pd.read_csv(file_path, skiprows=range(1, start_idx), nrows=end_idx - start_idx)
                # Extract agent positions and velocities (x, y, vx, vy)
                data = []
            else:
                chunk = self.challenge_data[idx]
                data = []

            for agent in self.args.agents:
                agent_data = chunk[[f'{agent}_x', f'{agent}_y', f'{agent}_vx', f'{agent}_vy']].values
                if self.challenge_data is not None:
                    agent_data = agent_data[-self.args.burn_in:] # should be modified later
                data.append(agent_data)
            
            # Stack agents and convert to tensor
            tensor = torch.tensor(data, dtype=torch.float32)  # Shape: (agents, length, dim)

            if self.args.Modify_Velocity: 
                vel = (tensor[:,1:,0:2] - tensor[:,:-1,0:2]) * self.args.fs
                tensor[:,:-1,2:4] = vel

            tensor = tensor.permute(1, 0, 2)  # Shape: (length, agents, dim)
            tensor = tensor.reshape(tensor.size(0), -1)  # Flatten the last two dimensions
            tensor = tensor.unsqueeze(0) # agents, time, dim
            return tensor

    # Challenge data
    if args.Challenge:
        urls_challenge = os.listdir(args.challenge_data)
        challenge_data,challenge_cycle = [],[]
        for url in urls_challenge:
            if url == '@eaDir':
                continue
            challenge_data.append(pd.read_csv(args.challenge_data + os.sep + url))
            challenge_cycle.append(challenge_data[-1].shape[0])
        test_metadata = None
        len_seqs_test = len(challenge_data)
        batchSize_test = len_seqs_test
    else:
        challenge_data = None
        len_seqs_test = len(test_metadata)
        batchSize_test = batchSize

    # Create dataset and dataloader using metadata
    args.agents = [f'l{i}' for i in range(1, 12)] + [f'r{i}' for i in range(1, 12)] + ['b']
    num_workers = args.num_workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    print('num_workers:'+str(num_workers))
    args.Modify_Velocity = True
    print('Modify_Velocity:'+str(args.Modify_Velocity))

    if not TEST:    
        train_loader = DataLoader(Dataset(args, train_metadata),
                batch_size=args.batchsize, shuffle=True, **kwargs)
        val_loader = DataLoader(Dataset(args, val_metadata),
                batch_size=args.batchsize, shuffle=False, **kwargs)
    
    test_loader = DataLoader(Dataset(args, test_metadata, challenge_data=challenge_data),
                batch_size=args.batchsize, shuffle=False, **kwargs)

    activeRoleInd = range(n_roles)
    activeRole = []; 
    activeRole.extend([str(n) for n in range(n_roles)]) # need to be reconsidered
    args.n_agents = len(activeRole)

    outputlen0 = 2
    n_feat = 4
    featurelen = 4*23  
    args.n_feat = n_feat
    args.fs = fs
    args.horizon = totalTimeSteps

    ####### Sanity check ##################
    if args.Sanity:
        losses = run_sanity(args,test_loader)

    # parameters for RNN -----------------------------------
    init_filename0 = path_init

    init_filename0 = init_filename0 + args.model + '_' + args.data + '/'
    init_filename0 = init_filename0 + str(batchSize) + '_' + str(totalTimeSteps)      
    if args.drop_ind:
        init_filename0 = init_filename0 + '_drop_ind' 

    if not os.path.isdir(init_filename0):
        os.makedirs(init_filename0)
    init_pthname = '{}_state_dict'.format(init_filename0)
    print('model: '+init_filename0)

    if not os.path.isdir(init_pthname):
        os.makedirs(init_pthname)

    args.dataset = args.data
    args.start_lr = 1e-3 
    args.min_lr = 1e-3 
    clip = True # gradient clipping
    save_every = 1
    args.batch_size = batchSize
    # args.cont = False # continue training previous best model
    args.x_dim = outputlen0 # output
    args.y_dim = featurelen # input
    args.z_dim = 64 
    args.h_dim = 64 #128 
    args.rnn_dim = 100 # 100
    args.n_layers = 2
    args.rnn_micro_dim = args.rnn_dim
    args.n_all_agents = 22 if args.data == 'soccer' else 10 
    ball_dim = 4 
    # Parameters to save
    temperature = 1 if args.data == 'soccer' else 1 
        
    params = {
        'model' : args.model,
        'dataset' : args.dataset,
        'x_dim' : args.x_dim,
        'y_dim' : args.y_dim,
        'z_dim' : args.z_dim,
        'h_dim' : args.h_dim,
        'rnn_dim' : args.rnn_dim,
        'n_layers' : args.n_layers, 
        'len_seq' : totalTimeSteps,  
        'n_agents' : args.n_agents,    
        'min_lr' : args.min_lr,
        'start_lr' : args.start_lr,
        'seed' : args.seed,
        'cuda' : args.cuda,
        'n_feat' : n_feat,
        'fs' : fs,
        'embed_size' : 32, # 8
        'embed_ball_size' : 32, # 8
        'burn_in' : args.burn_in,
        'horizon' : args.horizon,
        'rnn_micro_dim' : args.rnn_micro_dim,
        'ball_dim' : ball_dim,
        'n_all_agents' : args.n_all_agents,
        'temperature' : temperature,
        'drop_ind' : args.drop_ind,
    }

    # Load model
    
    model = load_model(args.model, params, parser)

    if args.cuda:
        model.cuda()
    # Update params with model parameters
    params = model.params
    params['total_params'] = num_trainable_params(model)

    # Create save path and saving parameters
    pickle.dump(params, open(init_filename0+'/params.p', 'wb'), protocol=2)

    # Continue a previous experiment, or start a new one
    if args.cont:
        if os.path.exists('{}_best.pth'.format(init_pthname)): 
            # state_dict = torch.load('{}_12.pth'.format(init_pthname))
            state_dict = torch.load('{}_best.pth'.format(init_pthname))
            model.load_state_dict(state_dict)
            print('best model was loaded')
        else:
            print('args.cont = True but file did not exist')


    print('############################################################')



    ###### TRAIN LOOP ##############
    best_val_loss = 0
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr) # if not args.finetune else 1e-4
    epoch_first_best = -1
    #print('epoch_first_best: '+str(epoch_first_best))

    pretrain_time =  0
    
    # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.burn_in}
    hyperparams = {'model': args.model,'burn_in': args.horizon, 'pretrain':(0 < pretrain_time)}
    
    if not TEST:
        for e in range(args.n_epoch):
            epoch = e+1
            print('epoch '+str(epoch))
            pretrain = (epoch <= pretrain_time)
            hyperparams['pretrain'] = pretrain

            # Set a custom learning rate schedule
            if epochs_since_best == 3: # and lr > args.min_lr:
                # Load previous best model
                filename = '{}_best.pth'.format(init_pthname)

                state_dict = torch.load(filename)

                # Decrease learning rate
                # lr = max(lr/3, args.min_lr)
                # print('########## lr {} ##########'.format(lr))
                epochs_since_best = 0
                print('##### Best model is loaded #####')
            else:
                if not hyperparams['pretrain'] and not args.finetune:
                    # lr = lr*0.99 # 9
                    print('########## lr {:.4e} ##########'.format(lr)) 
                    epochs_since_best += 1
                

            optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr)
            
            start_time = time.time()
            
            print('pretrain:'+str(hyperparams['pretrain']))
            hyperparams['burn_in'] = args.horizon
            train_loss,train_loss2 = run_epoch(train=1, rollout=False, hp=hyperparams)
            print('Train:\t'+loss_str(train_loss)+'|'+loss_str(train_loss2))

            torch.cuda.empty_cache()
            
            hyperparams['burn_in'] = args.burn_in
            val_loss,val_loss2 = run_epoch(train=0, rollout=True, hp=hyperparams)
            print('RO Val:\t'+loss_str(val_loss)+'|'+loss_str(val_loss2))

            total_val_loss = sum(val_loss.values())

            epoch_time = time.time() - start_time
            print('Time:\t {:.3f}'.format(epoch_time))

            torch.cuda.empty_cache()

            # Best model on test set
            if e > epoch_first_best and (best_val_loss == 0 or total_val_loss < best_val_loss): 
                best_val_loss_prev = best_val_loss
                best_val_loss = total_val_loss
                epochs_since_best = 0

                filename = '{}_best.pth'.format(init_pthname)

                torch.save(model.state_dict(), filename)
                print('##### Best model #####')
                if epoch > pretrain_time and (best_val_loss_prev-best_val_loss)/best_val_loss < 0.0001 and best_val_loss_prev != 0:
                    print('best loss - current loss: ' +str(best_val_loss_prev)+' - '+str(best_val_loss))
                    break 


            # Periodically save model
            if epoch % save_every == 0:
                filename = '{}_{}.pth'.format(init_pthname, epoch)
                torch.save(model.state_dict(), filename)
                print('########## Saved model ##########')

                           
        print('Best Val Loss: {:.4f}'.format(best_val_loss))
    
    # Load params
    params = pickle.load(open(init_filename0+'/params.p', 'rb'))
    
    # Load model
    state_dict = torch.load('{}_best.pth'.format(init_pthname, params['model']), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # Load ground-truth states from test set
    loader = test_loader 

    if True:
        print('test sample')
        # Sample trajectory
        samples = np.zeros((args.horizon,args.n_agents,len_seqs_test,featurelen)) 
        losses = {}
        losses2 = {}

        start_time = time.time()
        i = 0
        for batch_idx, (data) in enumerate(loader):
            if args.cuda:
                data = data.cuda() #, data_y.cuda()
                # (batch, agents, time, feat) => (time, agents, batch, feat) 
            data = data.permute(2, 1, 0, 3)
            
            sample, output, output2 = model.sample(data, rollout=True, burn_in=args.burn_in, n_sample=1, TEST = True, Challenge = args.Challenge)

            samples[:,:,batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = sample.detach().cpu().numpy()[:-1]

            del sample 
            if not args.Challenge:
                for key in output:
                    if batch_idx == 0:
                        losses[key] = np.zeros(1)
                        losses2[key] = np.zeros((len_seqs_test))
                    losses[key] += np.sum(output[key].detach().cpu().numpy(),axis=1)
                    losses2[key][batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = output[key].detach().cpu().numpy()
                    
                for key in output2:
                    if batch_idx == 0:
                        losses[key] = np.zeros(1)
                        losses2[key] = np.zeros((len_seqs_test))
                    losses[key] += np.sum(output2[key].detach().cpu().numpy(),axis=1)
                    losses2[key][batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = output2[key].detach().cpu().numpy()


        epoch_time = time.time() - start_time
        print('Time:\t {:.3f}'.format(epoch_time)) # Sample {} r*n_smp_b,
            
        if not args.Challenge: # create Mean + SD Tex Table for positions------------------------------------------------
            avgL2_m = {}
            avgL2_sd = {}
            for key in losses2:
                avgL2_m[key] =  np.mean(losses2[key])
                avgL2_sd[key] = np.std(losses2[key])

            print(args.model)
            print('(mean):'
                +' $' + '{:.2f}'.format(avgL2_m['e_pos'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_pos'])+'$ &'
                +' $' + '{:.2f}'.format(avgL2_m['e_vel'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_vel'])+'$ &'
                ) 
            print('(endpoint):'
                +' $' + '{:.2f}'.format(avgL2_m['e_e_p'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_e_p'])+'$ &'
                +' $' + '{:.2f}'.format(avgL2_m['e_e_v'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_e_v'])+'$ &'
                ) 
        else: # challenge   
            # Save samples
            experiment_path = './results/test/submission'
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)

            # Save samples to CSV        
            i = 0
            for seq in range(samples.shape[2]):
                sample_ = samples[args.burn_in:, 0, seq].reshape((-1,23,n_feat))[:,:,:2] #
                # check: samples[args.burn_in-1, 0, seq].reshape((23,n_feat))[:,:2]
                sample_path = os.path.join(experiment_path, f'{seq+1:02}.csv')
                df = pd.DataFrame(sample_.reshape(sample_.shape[0], -1), columns=[f'agent_{agent}_{coord}' for agent in range(sample_.shape[1]) for coord in ['x', 'y']])
                
                # rename columns and cycles
                df.columns = [col.replace('agent_0', 'l1').replace('agent_3', 'l4').replace('agent_4', 'l5')
                              .replace('agent_5', 'l6').replace('agent_6', 'l7').replace('agent_7', 'l8')
                              .replace('agent_8', 'l9').replace('agent_9', 'l10').replace('agent_10', 'l11')
                              .replace('agent_11', 'r1').replace('agent_12', 'r2').replace('agent_13', 'r3')
                              .replace('agent_14', 'r4').replace('agent_15', 'r5').replace('agent_16', 'r6')
                              .replace('agent_17', 'r7').replace('agent_18', 'r8').replace('agent_19', 'r9')
                              .replace('agent_20', 'r10').replace('agent_21', 'r11') for col in df.columns]
                df.columns = [col.replace('agent_22', 'b').replace('agent_1', 'l2').replace('agent_2', 'l3') for col in df.columns]

                cycle = range(challenge_cycle[i]+1, challenge_cycle[i] + len(df) + 1)
                df.insert(0, '#', cycle)

                df.to_csv(sample_path, index=False)
                i += 1
            print('Samples saved to {}'.format(experiment_path))

