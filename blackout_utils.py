import sys
# sys.path.append('../scorenet') # not very portab
sys.path.append('../score_sde') # not very portab

import numpy as np
import json
from datetime import datetime
from scipy.optimize import bisect
from scipy.stats import binom

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Split, BertPreTokenizer, Digits, Sequence, WhitespaceSplit
from tokenizers import Regex

import torch 
from torch.utils.data import DataLoader
# from models import ncsnpp
from simple_model import Net, ConvNet
# from configs.vp import cifar10_ncsnpp_continuous as configLoader
from dataloader import SMILESDataset
from torch.utils.tensorboard import SummaryWriter

def get_config():
    config = configLoader.get_config()
    config.training.batch_size=256
    config.training.snapshot_freq_for_preemption=1000
    config.training.snapshot_freq=50000
    config.training.log_freq=100
    return config

def load_data(path='/gpfs/alpine/world-shared/med106/foxzr/ga/aisd-molecule-gan/data/gdb9_smiles.tsv' ):
    smiles = []
    with open(path) as f:
        for line in f:
            smiles.append(line)
    smiles = smiles[1:]
    return smiles

def get_dataloader(data, batch_size):
    return torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
def get_tokenizer(tokenizer_path='/gpfs/alpine/world-shared/med106/blnchrd/models/bert_docking_brackets/tokenizer', tokenizer_type='bert'):
    # initialize tokenizer
    pretokenizer_dict = {
        'regex': Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')]),
        'bert': BertPreTokenizer(),
        'bert_digits': Sequence([BertPreTokenizer(), Digits(individual_digits=True)]),
        'digits': Sequence([WhitespaceSplit(), Digits(individual_digits=True)])
    }
    with open(tokenizer_path + '/config.json', 'r') as f:
        tokenizer_config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
    tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[tokenizer_type]
    return tokenizer

def get_binomial_corrupter(cached_schedule=None, t_end=15, num_times=1000, max_state=64):
    if cached_schedule is not None:
        schedule = dict(np.load(cached_schedule, allow_pickle=True))
        schedule['noise_cdf'] = torch.Tensor(schedule['noise_cdf'])
        schedule['reverse_rates'] = torch.Tensor(schedule['reverse_rates'])
        schedule['observation_times'] = torch.Tensor(schedule['observation_times'])
    else:
        noise_cdf, reverse_rates, observation_times = generate_binomial_corrupter(t_end, num_times, max_state)
        schedule = {}
        schedule['noise_cdf'] = torch.Tensor(noise_cdf)
        schedule['reverse_rates'] = torch.Tensor(reverse_rates)
        schedule['observation_times'] = torch.Tensor(observation_times)
        schedule['num_times'] = num_times
        schedule['max_state'] = max_state
    
    return schedule 
    
def generate_binomial_corrupter(t_end, num_times, max_state):
        f = lambda x: np.log(x/(1-x))
        x_end = np.exp(-t_end)
        f_grid = np.linspace(-f(x_end), f(x_end), num_times)
        x_grid = np.array([bisect(lambda x: f(x)-f_grid[i], x_end/2, 1-x_end/2) for i in range(num_times)])
        observation_times = -np.log(x_grid)    
        
        # Analytically derived reverse-time transition rate/ forward pmf
        reverse_rates = np.zeros((max_state, max_state, num_times))
        support = np.arange(max_state)
        pmfs = np.zeros((num_times+1, max_state, max_state))
        pmfs[0,:,:] = np.eye(max_state)
        for ti in range(num_times):
            p = np.exp(-observation_times[ti])
            for n in range(max_state):
                pmfs[ti+1,:,n] =  binom(n, p).pmf(support)    
                for m in range(n):
                    reverse_rates[n,m,ti] = n-m 
                reverse_rates[n,n,ti] = 0
                
        ### Analytical forward solution, CDF
        cdfs = np.zeros_like(pmfs)

        for i in range(num_times+1):
            for j in range(max_state):
                cdfs[i,:,j] = np.cumsum(pmfs[i,:,j] )    
                
        np.savez('schedules/binomial_corrupter_files.npz', noise_cdf=cdfs, reverse_rates=reverse_rates,
                  observation_times=observation_times, num_times=num_times, max_state=max_state)
        return cdfs, reverse_rates, observation_times

def noisify_data(batch, noise_cdf, reverse_rates, observation_times, num_times, max_state):
    batch = batch.squeeze(1)
    with torch.no_grad():
        batch_size, seq_length = batch.shape
        batch = batch
        offset = 0.01
        t = np.copy(observation_times)
        sampling_prob = np.clip( np.exp(-t)*(1.-np.exp(-t)), offset, np.inf)
        sampling_prob = sampling_prob/np.sum(sampling_prob)
        sample_times = torch.from_numpy(np.random.choice(num_times,
                                                         size=(batch_size,1),
                                                          p=sampling_prob))
        cdf_batch = noise_cdf[(sample_times+1).long(),:,batch.long()]
        u = torch.FloatTensor(batch_size, seq_length, 1).uniform_()
        corrupted_batch =  torch.argmax((u < cdf_batch).long(), axis=-1).int()
        reverse_rates_batch = reverse_rates 
        corrupted_batch =  torch.argmax((u < cdf_batch).long(), axis=-1).int()
        # batch_indices = batch*max_state*num_times + corrupted_batch*num_times + sample_times
        batch_birth_rates = reverse_rates[batch.long(), corrupted_batch.long(), sample_times.long()]   
        return corrupted_batch.unsqueeze(1), batch_birth_rates, sample_times

def train_model(config, training_loader, validation_loader, schedule):
    score_model = ConvNet(schedule['max_state'])
    score_fun = ConvNet(schedule['max_state'])

    # setup model based on config
    # score_model = mutils.create_model(config)
    # score_fn = mutils.get_model_fn(score_model, train=True)
   
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4) 

    # run the training loop 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('logs/diffrun_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(2):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = train_epoch(score_model, epoch, writer, optimizer, training_loader, schedule)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vloss = loss_fn(score_model, vdata, schedule)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

def train_epoch(model, epoch_index, tb_writer, optimizer, training_loader, schedule):
    last_loss = 0.
    running_loss = 0.
    for i, batch in enumerate(training_loader):
        # zero gradients
        optimizer.zero_grad()

        # compute the loss 
        loss = loss_fn(model, batch, schedule)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report (within this epoch)
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def loss_fn(model, data, schedule): 
    data, data_dX, sample_times = noisify_data(data, **schedule)
    y = model(data, sample_times)
    loss = torch.mean(schedule['weights'][sample_times.long()]*(y - data_dX*torch.log(y)))
    return loss

def retokenize(unique, data):
    for new_token,old_token in enumerate(unique):
        data[data==old_token]=new_token
    return data    

if __name__=='__main__':
    # test loading scorenet
   #  config = get_config()

    # test loading data
    data = load_data('../etc/data/gdb9_smiles.tsv')
    # load tokenizer 
    tokenizer = get_tokenizer('../etc/tokenizers/tokenizer')

    # tokenize a sample 
    tokenized_data_train = tokenizer(data[:1000], padding='max_length', max_length=20)
    tokenized_data_val = tokenizer(data[1000:1200], padding='max_length', max_length=20)
    all_data_array = np.array(tokenizer(data[:1200], padding=True)['input_ids'])
    train_array = np.array(tokenized_data_train['input_ids'])
    val_array = np.array(tokenized_data_val['input_ids'])

    # re-tokenize by max value (for testing) 
    unique = np.unique(all_data_array)
    train_data = retokenize(unique, train_array)
    val_data = retokenize(unique, val_array)
            
    print('max value (train): {0}'.format(np.max(tokenized_data_train['input_ids'])))
    print('max value (val): {0}'.format(np.max(tokenized_data_val['input_ids'])))
    
    print('support size (train): {0}'.format(len(np.unique(tokenized_data_train['input_ids']))))
    print('support size (val): {0}'.format(len(np.unique(tokenized_data_val['input_ids']))))

    print('max value retoke (train): {0}'.format(np.max(train_data)))
    print('max value retoke (val): {0}'.format(np.max(val_data)))
    # get a data_loader 
    train_dataloader = DataLoader( SMILESDataset(torch.Tensor(train_data).unsqueeze(1)), batch_size=32, shuffle=True) 
    val_dataloader = DataLoader( SMILESDataset(torch.Tensor(val_data).unsqueeze(1)), batch_size=32, shuffle=True) 

    # get the schedule 
    schedule = get_binomial_corrupter('schedules/binomial_corrupter_files.npz')
    # schedule = get_binomial_corrupter(max_state=420)

    # try the simple model     
    seq_len = len(tokenized_data_train['input_ids'][0])
    model = ConvNet(seq_len, schedule['max_state'])
    model_in = torch.Tensor(tokenized_data_train['input_ids']).unsqueeze(1)
    print(model(model_in,1))

    config = []
    train_model(config, train_dataloader, val_dataloader, schedule)

