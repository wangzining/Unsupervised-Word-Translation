import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

class TrainModel(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, src_dic,tgt_dic, optim_fn, lr, param_list):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dic = src_dic
        self.tgt_dic = tgt_dic
        self.mapping = mapping
        self.discriminator = discriminator
        self.param_list = param_list

        # optimizers
        if optim_fn == 'sgd':
            optim_fn = optim.SGD 
            lr_param = {}
            lr_param['lr'] = lr
            self.map_optimizer = optim_fn(mapping.parameters(), **lr_param)
   
            #optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **lr_param)
        else:
            print "Please use sgd optimizer in mapping and discriminator!"
            assert optim_fn == 'sgd'

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    
    
    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = 32
        mf = 50
        #mf = 75000 #only choose top mf number of most frequent words
        assert mf <= min(len(self.src_dic), len(self.tgt_dic))
        src_ids = torch.LongTensor(bs).random_(mf)
        tgt_ids = torch.LongTensor(bs).random_(mf)

        #use cuda..
        useGPU = True
        if useGPU:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        dis_smooth = 0.1 #discriminator smooth predication
        y[:bs] = 1 - dis_smooth
        y[bs:] = dis_smooth
        y = Variable(y.cuda() if useGPU else y)

        return x, y

    
    
    
    
    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        dis_clip_weights = 0
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
#         clip_parameters(self.discriminator, dis_clip_weights)

    def mapping_step(self, stats, batch_size = 32):
        
        self.discriminator.eval()  #run python code within itself
        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.param_list.feedback_coeff * loss

        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (fool discriminator)")
            #exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        #self.orthogonalize()
        # orthogonalize:
        if self.param_list.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.param_list.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

        return 2 * self.param_list.batch_size_adv

        