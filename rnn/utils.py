import torch
import math


######################################################################
############################ MODEL UTILS #############################
######################################################################


def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        parser.add_argument('--'+arg, type=int, required=True)
    args, _ = parser.parse_known_args()

    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params


def get_params_str(model_args, params):
    ret = ''
    for arg in model_args:
        ret += ' {} {} |'.format(arg, params[arg])
    return ret[1:-2]


def cudafy_list(states):
    for i in range(len(states)):
        states[i] = states[i].cuda()
    return states

######################################################################
############################## GAUSSIAN ##############################
######################################################################

def nll_gauss(mean, std, x, pow=False,Sum=True):
    pi = torch.FloatTensor([math.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    if not pow:
        nll_element = (x - mean).pow(2) / std.pow(2) + 2*torch.log(std) + torch.log(2*pi)
    else:
        nll_element = (x - mean).pow(2) / std + torch.log(std) + torch.log(2*pi)

    nll = 0.5 * torch.sum(nll_element) if Sum else 0.5 * torch.sum(nll_element,1)
    return nll
    
    
def batch_error(predict, true, Sum=True, sqrt=True, diff=True):
    # error = torch.sum(torch.sum((predict[:,:2] - true[:,:2]),1))
    if diff:
        error = torch.sum((predict[:,:2] - true[:,:2]).pow(2),1)
    else:
        error = torch.sum(predict[:,:2].pow(2),1)
    if sqrt:
        error = torch.sqrt(error)
    if Sum:
        error = torch.sum(error)
    return error


######################################################################
############################## ROLE OUT ##############################
######################################################################

def calc_dist_cos_sin(rolePos,refPos,batchSize):
    if torch.cuda.is_available(): # rolePos.is_cuda:
        rolePos = rolePos.cuda()
        refPos = refPos.cuda()
        rolefeat = torch.zeros((batchSize,3)).cuda()
    else: 
        rolefeat = torch.zeros((batchSize,3))

    rolefeat[:,0] = torch.sqrt( (rolePos[:,0]-refPos[:,0]).pow(2)+(rolePos[:,1]-refPos[:,1]).pow(2) ) # dist
    loc0 = (rolefeat[:,0]==0).nonzero()
    loc1 = (rolefeat[:,0]!=0).nonzero()

    rolefeat[loc1,1] = (rolePos[loc1,0]-refPos[loc1,0]) / rolefeat[loc1,0] # cos
    rolefeat[loc1,2] = (rolePos[loc1,1]-refPos[loc1,1]) / rolefeat[loc1,0] # sin

    rolefeat[loc0,1] = 0.0
    rolefeat[loc0,2] = 0.0

    return rolefeat

def roll_out(y_t,y_t_1,prediction_all,n_roles,n_feat,ball_dim,fs,batchSize,i): # update feature vector using next_prediction

    prev_feature = y_t
    next_feature = y_t_1
    next_vel = prediction_all[:,:,:2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if n_roles >= 10: # soccer
        n_all_agents = 23 if n_roles == 23 else 22
        n_all_agents_ = 22
        n_half_agents = 11
        goalPosition = torch.tensor([52.5,0])
    n_feat_player = n_feat*n_all_agents
    goalPosition = goalPosition.repeat(batchSize, 1)
    
    next_current = next_feature[:,0:n_feat_player] 
    

    legacy_next = next_current.reshape(batchSize,n_all_agents,n_feat) 
    new_matrix = torch.zeros((batchSize,n_all_agents_,n_feat)).to(device) 
    if i < 11 or i == 22:
        teammateList = list(range(n_half_agents)) 
        opponentList = list(range(n_half_agents,n_all_agents_))  
    elif i < 22:
        teammateList = list(range(n_half_agents,n_all_agents_)) 
        opponentList = list(range(n_half_agents))

    roleOrderList = [role for role in range(n_roles)]
    role_long = torch.zeros((batchSize,n_feat)).to(device)
    if i < 22:
        teammateList.remove(i)

    # fix role vector
    role_long[:,2:4] = next_vel[:,i,:]
    role_long[:,0:2] = prev_feature[:,i*n_feat:(i*n_feat+2)] + prev_feature[:,i*n_feat+2:(i*n_feat+4)]*fs 

    if n_roles < 23:
        new_matrix[:,i,:] = role_long

    # fix all teammates vector
    for teammate in teammateList:
        player = legacy_next[:,teammate,:]
        if teammate in roleOrderList: # if the teammate is one of the active players: e.g. eliminate goalkeepers
            teamRoleInd = roleOrderList.index(teammate)
            
            player[:,2:4] = next_vel[:,teamRoleInd,:]
            player[:,0:2] = prev_feature[:,teamRoleInd*n_feat:(teamRoleInd*n_feat+2)] + prev_feature[:,teamRoleInd*n_feat+2:(teamRoleInd*n_feat+4)]*fs           

        new_matrix[:,teammate,:] = player

    for opponent in opponentList: 
        player = legacy_next[:,opponent,:]
        try: new_matrix[:,opponent,:] = player
        except: import pdb; pdb.set_trace()

    
    if n_roles == 23:
        ball = next_feature[:,88:88+ball_dim]  
        if i == 22:
            ball[:,2:4] = next_vel[:,i,:]
            ball[:,0:2] = prev_feature[:,i*n_feat:(i*n_feat+2)] + prev_feature[:,i*n_feat+2:(i*n_feat+4)]*fs 
        new_feature_vector = torch.cat([torch.reshape(new_matrix,(batchSize,n_all_agents_*n_feat)), ball ],dim=1) 

    else:
        ball = next_feature[:,n_feat_player:n_feat_player+ball_dim]  
        new_feature_vector = torch.cat([torch.reshape(new_matrix,(batchSize,n_all_agents*n_feat)), ball ],dim=1) 

    return new_feature_vector

def roll_out_test(y_t,y_t_1,prediction_all,n_roles,n_feat,ball_dim,fs,batchSize): # update feature vector using next_prediction

    prev_feature = y_t
    next_feature = y_t_1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    next_vel = prediction_all[:,:,:2].to(device) 

    if n_roles >= 10: # soccer
        n_all_agents = 23 
    
    prev_feature_ = prev_feature[0].reshape(batchSize,23,n_feat)
    next_position = prev_feature_[:,:,:2] + prev_feature_[:,:,2:]*fs
    if n_roles == 23:
        next_velocity = next_vel.reshape(batchSize,n_all_agents,2)
    else:
        next_vel_remain = next_feature.reshape(batchSize,n_all_agents,n_feat)
        next_velocity = torch.cat([next_vel,next_vel_remain[:,n_roles:,2:]],1)
    new_feature_vector = torch.cat([next_position,next_velocity],dim=2).reshape(batchSize,n_all_agents*n_feat)
    
    return new_feature_vector
