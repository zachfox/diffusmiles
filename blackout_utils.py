import numpy as np
import json
from scipy.optimize import bisect
from scipy.stats import binom

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Split, BertPreTokenizer, Digits, Sequence, WhitespaceSplit
from tokenizers import Regex

from models import ncsnpp
from configs.vp import cifar10_ncsnpp_continuous as configLoader


def load_scorenet():
    config =  configLoader.get_config()
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
    # config.model.num_scales=T
    if cached_schedule is not None:
        schedule = np.load(cached_schedule, allow_pickle=True)
        noise_cdf = schedule['cdfs']
        reverse_rates = schedule['reverse_rates']
        observation_times  = schedule['observation_times']    
    else:
        noise_cdf, reverse_rates, observation_times = generate_binomial_corrupter(t_end, num_times, max_state)
    return noise_cdf, reverse_rates, observation_times 
    
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
                
        np.savez('schedules/binomial_corrupter_files', cdfs=cdfs, reverse_rates=reverse_rates,
                  observation_times=observation_times)
        return cdfs, reverse_rates, observation_times

def noisify_batch(batch, device, noise_cdf, reverse_rates, observation_times, num_times):
    with torch.no_grad():
        batch_size, seq_length = batch.shape
        batch = batch.to(device)
        # was tIndex 
        sample_times = torch.from_numpy(np.random.choice(num_times,
                                                         size=(batch_size,1),
                                                          p=samplingProb)).to(device)
        #tIndex = (T*torch.cuda.FloatTensor(n,1,1,1).uniform_()).floor_().to(config.device)
        cdf_batch = noise_cdf[(sample_times+1).long(),:,batch.long()]
        #u = torch.from_numpy(np.random.random(size=(n,nx,ny,nc,1))).to(config.device)
        u = torch.cuda.FloatTensor(batch_size, seq_length, 1).uniform_().to(config.device)
        corrupted_batch =  torch.argmax((u < cdf_batch).long(), axis=-1).int()
        reverse_rates_batch = reverse_rates 
        corrupted_batch =  torch.argmax((u < cdf_batch).long(), axis=-1).int()
        # index = imgBatchGPU*256*T + nt*T + tIndex.long()
        batch_indices = batch*max_state*num_times + nt*num_times + sample_times
        birthRateBatch = brTableGPU[index.long()]   
        p = torch.exp(-observationTimeGPU[tIndex.long()])
        width = 1.0 #(255.0/2*p).reshape((n, 1, 1, 1))
        mean_v = (255.0/2*p).reshape((n, 1, 1, 1))
        
        #return (nt/255.*2.-1.).permute((0,3,1,2)), birthRateBatch.permute((0,3,1,2)), tIndex[:,0,0,0]
        return ((nt-mean_v)/width).permute((0,3,1,2)).to(torch.float32), birthRateBatch.permute((0,3,1,2)).to(torch.float32), tIndex[:,0,0,0]
output_image_batch, birthRate_batch, tIndexArray
def setup_training(config):
    score_model = mutils.create_model(config)
    score_fn = mutils.get_model_fn(score_model, train=True)
    optimizer = torch.optim.Adam(score_model.parameters(),lr=config.optim.lr) 
    
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    
    train_batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    train_batch = train_batch.permute(0, 3, 1, 2)
    imgBatch = scaler(train_batch)
    
    workdir = 'linearDegradation-cifar10-MLE-v2'
    
    state = dict(optimizer=optimizer, model=score_model, ema=ema, lossHistory=[], evalLossHistory=[], step=0)
#
#checkpoint_dir = os.path.join(workdir, "checkpoints")
#checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
## tf.io.gfile.makedirs(checkpoint_dir)
## tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
#state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
#initial_step = int(state['step'])
#lossHistory = state['lossHistory']
#evalLossHistory = state['evalLossHistory']
#
#
## ### Training
#for step in range(initial_step, config.training.n_iters):
#    
#    train_batch = torch.from_numpy(next(train_iter)['image']._numpy())
#    output_image_batch, birthRate_batch, tIndexArray = generateBatchDataGPU(train_batch, T)
#    
#    optimizer.zero_grad()
#
#    y = relu(score_fn(output_image_batch, tIndexArray))
#    
#    #loss = torch.mean( ((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#    #loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#    loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*(y - birthRate_batch*torch.log(y)))
#    #loss = torch.mean(torch.square(y - birthRate_batch))
#    
#    loss.backward()
#
#    state['ema'].update(state['model'].parameters())
#    
#    optimizer.step()
#    
#    lossHistory.append(loss.detach().cpu().numpy())
#
#    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
#        save_checkpoint(checkpoint_meta_dir, state)
#        
#    if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters:
#        save_step = step // config.training.snapshot_freq
#        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)    
#    
#    if np.mod(step, config.training.log_freq)==0:
#        
#        eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy())
#        output_image_batch, birthRate_batch, tIndexArray = generateBatchDataGPU(eval_batch, T)
#
#        ema.store(score_model.parameters())
#        ema.copy_to(score_model.parameters())
#        
#        y = relu(score_fn(output_image_batch, tIndexArray))
#        #loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#        #loss = torch.mean( ((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#        #loss = torch.mean(y - birthRate_batch*torch.log(y))
#        #loss = torch.mean(torch.square(y - birthRate_batch))
#        loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*(y - birthRate_batch*torch.log(y)))
#    
#        ema.restore(score_model.parameters())
#        
#        evalLossHistory.append(loss.detach().cpu().numpy())
#
#        print(f'current iter: {step}, loss: {lossHistory[-1]}, eval loss: {evalLossHistory[-1]}')
#        
#    state['step'] = step
#    state['lossHistory'] = lossHistory
#    state['evalLossHistory'] = evalLossHistory
#    
#    gc.collect()
#    torch.cuda.empty_cache()
#

if __name__=='__main__':
    # test loading scorenet
    load_scorenet()

    # test loading data
    data = load_data()

    # load tokenizer 
    tokenizer = get_tokenizer()

    # tokenize a sample 
    datai = data[50]
    print(datai)
    print(tokenizer(datai))

    # get corruptor
    out = get_binomial_corrupter()
    for thing in out:
        print(thing)

#tEnd = 15.
#T = 1000
#config.model.num_scales=T
#
#
#train_ds, eval_ds, scaler = generateData(config,'cifar10')
#
#train_iter = iter(train_ds)
#eval_iter = iter(eval_ds)
#
#try:
#
#    data = np.load('20221102-linearDegradation-cifar10-newSchedule-files.npz', allow_pickle=True)
#    cumSolArray=data['cumSolArray']
#    brTable=data['brTable']
#    observationTimes=data['observationTimes']    
#    
#except FileNotFoundError:
#
#    from scipy.optimize import bisect
#
#    def f(x):
#        return np.log(x/(1-x))
#
#    xEnd = np.exp(-tEnd)
#    fGrid = np.linspace(-f(xEnd), f(xEnd), T)
#    xGrid = np.array([bisect(lambda x: f(x)-fGrid[i], xEnd/2, 1-xEnd/2) for i in range(T)])
#    observationTimes = -np.log(xGrid)    
#    
#    ### Analytically derived reverse-time transition rate
#    brTable = np.zeros((256,256,T))
#    for tIndex in range(T):
#        p = np.exp(-observationTimes[tIndex])
#        for n in range(256):
#            for m in range(n):
#                brTable[n,m,tIndex] = n-m 
#            brTable[n,n,tIndex] = 0
#
#    ### Analytical forward solution, PDF
#    from scipy.stats import binom
#
#    support = np.arange(0,256)
#    solArray = np.zeros((T+1, 256, 256))
#    solArray[0,:,:] = np.eye(256)
#
#    for tIndex in range(T):
#        p = np.exp(-observationTimes[tIndex])
#        for IC in range(256):
#            solArray[tIndex+1,:,IC] =  binom(IC, p).pmf(support)    
#            
#    ### Analytical forward solution, CDF
#    cumSolArray = np.zeros_like(solArray)
#
#    for i in range(solArray.shape[0]):
#        for j in range(solArray.shape[1]):
#            cumSolArray[i,:,j] = np.cumsum(solArray[i,:,j] )    
#            
#    np.savez('20221102-linearDegradation-cifar10-MLE-v2-files', cumSolArray=cumSolArray, brTable=brTable, observationTimes=observationTimes)
#
#
## # Directly loading previously generated noise schedule and forward solution
#cumSolArrayGPU = torch.from_numpy(cumSolArray).to(config.device)
#brTableGPU = torch.from_numpy(np.ravel(brTable)).to(config.device)
#observationTimeGPU = torch.from_numpy(observationTimes).to(config.device)
##weightsGPU = torch.from_numpy( np.exp(-observationTimes)/(1-np.exp(-observationTimes))  ).to(config.device)
#
#eobservationTimes = np.hstack([0, observationTimes])
#
#offset = 0.01
#s = np.hstack((0,observationTimes[:-1]))
#t = np.copy(observationTimes)
#samplingProb = np.clip( np.exp(-t)*(1.-np.exp(-t)), offset, np.inf)
#samplingProb /= np.sum(samplingProb)
#weightsGPU = torch.from_numpy((np.exp(-s)-np.exp(-t))/np.clip( np.exp(-t)*(1.-np.exp(-t)), offset, np.inf)).to(config.device)
#weightsGPU /= torch.sum(weightsGPU)
#
## ### Noisifier
#torch.cuda.set_device(config.device)
#
#def generateBatchDataGPU(imgBatch,T):
#    
#    #imgBatch = torch.from_numpy(next(eval_iter)['image']._numpy())
#
#    with torch.no_grad():
#
#        n,nx,ny,nc = imgBatch.shape
#        imgBatchGPU = (255*imgBatch).to(config.device).long()
#        tIndex = torch.from_numpy(np.random.choice(T, size=(n,1,1,1), p=samplingProb)).to(config.device)
#        #tIndex = (T*torch.cuda.FloatTensor(n,1,1,1).uniform_()).floor_().to(config.device)
#        
#        cp = cumSolArrayGPU[(tIndex+1).long(),:,imgBatchGPU.long()]
#        #u = torch.from_numpy(np.random.random(size=(n,nx,ny,nc,1))).to(config.device)
#        u = torch.cuda.FloatTensor(n,nx,ny,nc,1).uniform_().to(config.device)
#        
#        nt =  torch.argmax((u < cp).long(), axis=4).int()
#        index = imgBatchGPU*256*T + nt*T + tIndex.long()
#        birthRateBatch = brTableGPU[index.long()]  
#    
#        p = torch.exp(-observationTimeGPU[tIndex.long()])
#        width = 1.0 #(255.0/2*p).reshape((n, 1, 1, 1))
#        mean_v = (255.0/2*p).reshape((n, 1, 1, 1))
#        
#        #return (nt/255.*2.-1.).permute((0,3,1,2)), birthRateBatch.permute((0,3,1,2)), tIndex[:,0,0,0]
#        return ((nt-mean_v)/width).permute((0,3,1,2)).to(torch.float32), birthRateBatch.permute((0,3,1,2)).to(torch.float32), tIndex[:,0,0,0]
#
#
## ### Visualize one batch
#train_batch = next(train_iter)['image']._numpy()
#train_batch_GPU = torch.from_numpy(train_batch)
#
#output_image_batch, brRate_batch, tIndexArray = generateBatchDataGPU(train_batch_GPU, T)
#
#output_image_batch = np.transpose(output_image_batch.detach().cpu().numpy(), (0,2,3,1))
#brRate_batch = np.transpose(brRate_batch.detach().cpu().numpy(), (0,2,3,1))
#tIndexArray = tIndexArray.detach().cpu().numpy()
#
#for i in range(20):
#    testImage = train_batch[i,:,:,:]
#    output_image = (255.0*(output_image_batch[i,:,:,:]+1.)/2.).astype('int32')
#    birthRate = brRate_batch[i,:,:,:]
#    targetTime = tIndexArray[i]
#    
##    fig, ax = plt.subplots(1,3, figsize=(4.8,1.5))
##    
##    ax[0].imshow(testImage)
##    
##    if np.amax(output_image)!=0:
##        ax[1].imshow(output_image/np.amax(output_image))
##    else:
##        ax[1].imshow(output_image)
##        
##    ax[1].set_title('$t='+str(targetTime)+'$')
##    
##    if np.amax(birthRate)-np.amin(birthRate)!=0:
##        ax[2].imshow((birthRate-np.amin(birthRate))/(np.amax(birthRate)-np.amin(birthRate)))
##    else:
##        ax[2].imshow(birthRate)
##        
##    for j in range(3):
##        
##        ax[j].set_xticklabels('')
##        ax[j].set_yticklabels('')
##    
##    fig.tight_layout()
#
#
## ### Instantiate an ML model to learn the transition rate
#score_model = mutils.create_model(config)
#score_fn = mutils.get_model_fn(score_model, train=True)
#optimizer = torch.optim.Adam(score_model.parameters(),lr=config.optim.lr) 
#
#ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
#
#train_batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
#train_batch = train_batch.permute(0, 3, 1, 2)
#imgBatch = scaler(train_batch)
#
#workdir = 'linearDegradation-cifar10-MLE-v2'
#
#state = dict(optimizer=optimizer, model=score_model, ema=ema, lossHistory=[], evalLossHistory=[], step=0)
#
#checkpoint_dir = os.path.join(workdir, "checkpoints")
#checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
## tf.io.gfile.makedirs(checkpoint_dir)
## tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
#state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
#initial_step = int(state['step'])
#lossHistory = state['lossHistory']
#evalLossHistory = state['evalLossHistory']
#
#
## ### Training
#for step in range(initial_step, config.training.n_iters):
#    
#    train_batch = torch.from_numpy(next(train_iter)['image']._numpy())
#    output_image_batch, birthRate_batch, tIndexArray = generateBatchDataGPU(train_batch, T)
#    
#    optimizer.zero_grad()
#
#    y = relu(score_fn(output_image_batch, tIndexArray))
#    
#    #loss = torch.mean( ((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#    #loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#    loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*(y - birthRate_batch*torch.log(y)))
#    #loss = torch.mean(torch.square(y - birthRate_batch))
#    
#    loss.backward()
#
#    state['ema'].update(state['model'].parameters())
#    
#    optimizer.step()
#    
#    lossHistory.append(loss.detach().cpu().numpy())
#
#    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
#        save_checkpoint(checkpoint_meta_dir, state)
#        
#    if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters:
#        save_step = step // config.training.snapshot_freq
#        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)    
#    
#    if np.mod(step, config.training.log_freq)==0:
#        
#        eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy())
#        output_image_batch, birthRate_batch, tIndexArray = generateBatchDataGPU(eval_batch, T)
#
#        ema.store(score_model.parameters())
#        ema.copy_to(score_model.parameters())
#        
#        y = relu(score_fn(output_image_batch, tIndexArray))
#        #loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#        #loss = torch.mean( ((birthRate_batch==0)*y + 0.5*(birthRate_batch!=0)*torch.square(y-birthRate_batch)/(birthRate_batch+1.0e-8)))
#        #loss = torch.mean(y - birthRate_batch*torch.log(y))
#        #loss = torch.mean(torch.square(y - birthRate_batch))
#        loss = torch.mean( weightsGPU[tIndexArray.long()].reshape([config.training.batch_size,1,1,1])*(y - birthRate_batch*torch.log(y)))
#    
#        ema.restore(score_model.parameters())
#        
#        evalLossHistory.append(loss.detach().cpu().numpy())
#
#        print(f'current iter: {step}, loss: {lossHistory[-1]}, eval loss: {evalLossHistory[-1]}')
#        
#    state['step'] = step
#    state['lossHistory'] = lossHistory
#    state['evalLossHistory'] = evalLossHistory
#    
#    gc.collect()
#    torch.cuda.empty_cache()
#
