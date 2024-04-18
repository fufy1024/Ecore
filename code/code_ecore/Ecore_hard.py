
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from code.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb

from collections import Counter
import pickle

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

def emo_mask1(y, n_class=32):
    return 1-torch.sum(torch.eye(n_class)[y],0).cuda()


def to_one_hot(y, n_class=32):
    return 1-torch.as_tensor(np.eye(n_class)[y]).cuda()

def emotion_mask(y, n_class,max_len):
    return torch.from_numpy(1-np.eye(n_class)[y]).unsqueeze(1).repeat(1,max_len,1)



def subsequent_mask(size,max_len): #mask_emo:[max_len,32]
    if  size==1:
        all_mask=torch.ones(max_len,max_len+32).cuda()  
        all_mask[0,0]=torch.zeros(1).cuda() 
        all_mask[0,max_len:]=torch.zeros(1,32)
    else:    
        subsequent_mask1 = np.triu(m = np.ones(( size, size)), k=1).astype('uint8')
        mask=torch.from_numpy(subsequent_mask1)
        mask[0]=torch.zeros(mask.shape[1]) #[size,size]
        all_mask=torch.ones(max_len,max_len+32).cuda()
        all_mask[:size,:size]=mask
        all_mask[:size,max_len:]=torch.zeros(size,32)
    return all_mask



def subsequent_mask1(size,max_len,mask_emo): #mask_emo:[max_len,32]
    if  size==1:
        all_mask=torch.ones(max_len,max_len+32).cuda()  
        all_mask[0,0]=torch.zeros(1).cuda() 
        all_mask[0,max_len:]=mask_emo[0,:]
    else:    
        
        all_mask=torch.ones(max_len,max_len+32).cuda()  
        mask=torch.zeros(size,size).cuda() 

        all_mask=torch.ones(max_len,max_len+32).cuda()
        all_mask[:size,:size]=mask
        #emo-add
        all_mask[:size,max_len:]=mask_emo[:size,:]
        
    return all_mask




class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, args, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(self.args.act):  # Adaptive Computation Time
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, args, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(self.args, max_length)

        params =(args,
                 hidden_size,
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_loss = nn.MSELoss()

    def forward(self, inputs, encoder_output, mask=None, pred_emotion=None, emotion_contexts=None, context_vad=None):
        '''
        inputs: (bsz, tgt_len)
        encoder_output: (bsz, src_len), src_len=dialog_len+concept_len
        mask: (bsz, src_len)
        pred_emotion: (bdz, emotion_type)
        emotion_contexts: (bsz, emb_dim)
        context_vad: (bsz, src_len) emotion intensity values
        '''
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        loss_att = 0.0
        attn_dist = None
        if(self.universal):
            if(self.args.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec((x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src,dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            # Run decoder  y, encoder_outputs, pred_emotion, emotion_contexts, attention_weight, mask
            y, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec((x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src,dec_mask)))

            # Emotional attention loss
            if context_vad is not None:
                src_attn_dist = torch.mean(attn_dist, dim=1)  # (bsz, src_len)
                loss_att = self.attn_loss(src_attn_dist[:,:context_vad.shape[1]], context_vad)

            # Final layer normalization
            y = self.layer_norm(y)

        return y, attn_dist, loss_att


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, args, d_model, vocab):
        super(Generator, self).__init__()
        self.args = args
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(self.args.hidden_dim, 1)

    def forward(self, x, pred_emotion=None, emotion_context=None, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1):
        # pred_emotion (bsz, 1, embed_dim);  emotion_context: (bsz, emb_dim)
        if self.args.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        if emotion_context is not None:
            # emotion_context = emotion_context.unsqueeze(1).repeat(1, x.size(1), 1)
            pred_emotion = pred_emotion.repeat(1, x.size(1), 1)
            x = torch.cat((x, pred_emotion), dim=2)  # (bsz, tgt_len, 2 emb_dim)
            logit = self.emo_proj(x)
        else:
            logit = self.proj(x)  # x: (bsz, tgt_len, emb_dim)

        if self.args.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq

            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)
            # if beam_search:
            #     enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)
            return logit
        else:
            return F.log_softmax(logit, dim=-1)

class MLP(nn.Module):
    def __init__(self,input_dim,hid_dim,out_dim):
        super(MLP, self).__init__()
        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        return x

class Ecore(nn.Module):
    def __init__(self, args, vocab, decoder_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(Ecore, self).__init__()
        self.args = args
        self.vocab = vocab
        word2index, word2count, index2word, n_words = vocab
        self.word2index = word2index
        self.word2count = word2count
        self.index2word = index2word
        self.vocab_size = n_words

        #self.conversion=0

        self.word_freq = np.zeros(self.vocab_size)

        #self.embedding = nn.Embedding(n_words, self.args.emb_dim, padding_idx=args.PAD_idx)
        self.embedding = share_embedding(args, n_words, word2index, self.args.pretrain_emb)  # args, n_words, word2index #原
        self.encoder = Encoder(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)
        
        
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}
        
        self.h=2
        

        self.S=nn.Parameter(torch.FloatTensor(32,32))#self.args.emb_dim, self.args.emb_dim))

        self.emo_ln2 = nn.Linear(32*(1+1), 32)#bias=False
        self.emo_embedding = nn.Embedding(32, self.args.emb_dim)


       
        #self.emo_soft = MLP(args.emb_dim *4, args.emb_dim, 32)#bias=False
        self.emo_context = nn.Linear(args.emb_dim , 32)#bias=False
        self.emo_soft = nn.Linear(32*(1+1), 32)#bias=False
        
   
        

        self.GNN_L=2
        self.T_update_all=nn.ModuleList([T_update_all(args.hidden_dim,self.h) for _ in range(self.GNN_L)])



        ## GRAPH
        self.dropout = args.dropout
        self.W_q = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_k = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_v = nn.Linear(args.emb_dim, args.emb_dim)
        self.graph_out = nn.Linear(args.emb_dim, args.emb_dim)
        self.graph_layer_norm = nn.LayerNorm(args.hidden_dim)

        ## emotional signal distilling
        self.identify = nn.Linear(args.emb_dim, decoder_number, bias=False)
        
        self.activation = nn.Softmax(dim=1)

        ## multiple decoders
        self.emotion_embedding = nn.Linear(decoder_number, args.emb_dim)
        self.decoder = Decoder(args, args.emb_dim, hidden_size=args.hidden_dim,  num_layers=args.hop, num_heads=args.heads,
                               total_key_depth=args.depth,total_value_depth=args.depth, filter_size=args.filter, max_length=args.max_seq_length,)
        
        self.decoder_key = nn.Linear(args.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(args, args.hidden_dim, self.vocab_size)
        if args.projection:
            self.embedding_proj_in = nn.Linear(args.emb_dim, args.hidden_dim, bias=False)
        if args.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=args.PAD_idx)
        if args.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=args.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=args.PAD_idx)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        if args.noam:
            # We used the Adam optimizer here with β1 = 0.9, β2 = 0.98, and ϵ = 10^−9. We varied the learning rate over the course of training.
            # This corresponds to increasing the learning rate linearly for the first warmup training steps,
            # and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup step = 8000.
            self.optimizer = NoamOpt(args.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict']) 
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = args.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.tar'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def concept_graph(self, context, concept1, adjacency_mask1,mask_cem=None,con_cem_emb=None):
        '''

        :param context: (bsz, max_context_len, embed_dim)
        :param concept: (bsz, max_concept_len, embed_dim)
        :param adjacency_mask: (bsz, max_context_len, max_context_len + max_concpet_len)
        :return:
        '''
        # target = self.W_sem_emo(context)  # (bsz, max_context_len, emb_dim)
        # concept = self.W_sem_emo(concept)
        if mask_cem is not None:
            concept=torch.cat([concept1,con_cem_emb],1)
            adjacency_mask=torch.cat([adjacency_mask1.cuda(),mask_cem.cuda()],-1)
        else:
            concept=concept1
            adjacency_mask=adjacency_mask1


        target = context
        src = torch.cat((target, concept), dim=1)  # (bsz, max_context_len + max_concept_len, emb_dim)

        # QK attention
        q = self.W_q(target)  # (bsz, tgt_len, emb_dim)
        k, v = self.W_k(src), self.W_v(src)  # (bsz, src_len, emb_dim); (bsz, src_len, emb_dim)
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2))  # batch matrix multiply (bsz, tgt_len, src_len)

        adjacency_mask = adjacency_mask.bool()
        attn_weights_ori.masked_fill_(
            adjacency_mask,
            1e-24
        )  # mask PAD
        attn_weights = torch.softmax(attn_weights_ori, dim=-1)  # (bsz, tgt_len, src_len)

        if torch.isnan(attn_weights).sum() != 0:
            pdb.set_trace()

        #print('attn_weights',attn_weights)
        #print('self.training',self.training)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # weigted sum
        attn = torch.bmm(attn_weights, v)  # (bsz, tgt_len, emb_dim)
        attn = self.graph_out(attn)

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        new_context = self.graph_layer_norm(target + attn)

        new_context = torch.cat((new_context, concept), dim=1)
        return new_context

    def train_one_batch(self, batch, iter, train=True,outputs_emo=None):
        
        self.iter=iter
        self.tr=train
        enc_batch = batch["context_batch"]
        enc_batch_extend_vocab = batch["context_ext_batch"]
        enc_vad_batch = batch['context_vad']
        concept_input = batch["concept_batch"]  # (bsz, max_concept_len)
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(self.args.device)

        dec_batch = batch["target_batch"]
        dec_ext_batch = batch["target_ext_batch"]

        if self.args.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Embedding - context
        mask_src = enc_batch.data.eq(self.args.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        

        emb_mask = self.embedding(batch["mask_context"])  # dialogue state embedding
        src_emb1 = self.embedding(enc_batch)+emb_mask
        src_vad = enc_vad_batch  # (bsz, len, 1)  emotion intensity values

        if concept_input.size()[0] != 0:
                mask_con = concept_input.data.eq(self.args.PAD_idx).unsqueeze(1)  # real mask
                con_mask = self.embedding(batch["mask_concept"])  # kg embedding
                con_emb = self.embedding(concept_input)+con_mask
                
           
                ## Knowledge Update
                src_emb = self.concept_graph(src_emb1, con_emb, batch["adjacency_mask_batch"])  # (bsz, context+concept, emb_dim)
               
               
                mask_src = torch.cat((mask_src, mask_con), dim=2)  # ,cs_masks_cem
                src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)  # (bsz, len)

        ## Encode - context & concept
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)
        ## emotional signal distilling
        src_vad = torch.softmax(src_vad, dim=-1)
     
        

        emotion_context1=torch.mean(encoder_outputs,1)#原[16, 300]
       
        
        emo_emb=self.emo_embedding(torch.arange(32).cuda()) #+ self.emo_embedding(torch.ones([32]).long().cuda()*32) 
        emo_words,emo_mask,length_all, emo_weight_all,emo_words_emb,edge_all, mask_m_all = self.extract_emo(enc_batch,src_emb1, batch["emotion_score"],batch['emotion_label'],emo_emb)        
        emotion_context, emotion_logit, loss_emotion, emotion_acc,src_skep,loss_emo_edge=self.emo_graph3(emo_words, emo_words_emb, emo_mask, emo_emb, batch['emotion_label'],length_all,emotion_context1,emo_weight_all,edge_all, mask_m_all)
           
        # Decode
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        dec_emb = self.embedding(dec_batch[:, :-1])  # (bsz, tgt_len, emb_dim)
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, tgt_len, emb_dim)

        mask_trg = dec_batch.data.eq(self.args.PAD_idx).unsqueeze(1)
        
        pre_logit, attn_dist, loss_attn = self.decoder(inputs=dec_emb,
                                                           encoder_output=encoder_outputs,
                                                           mask=(mask_src,mask_trg),
                                                           pred_emotion=None,
                                                           emotion_contexts=emotion_context,
                                                           context_vad=src_vad)#src_skep)

        ## compute output dist
        if concept_input.size()[0] != 0:
                #enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab, concept_ext_input,cd_words_cem), dim=1)
                enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)
        logit = self.generator(pre_logit, None, None, attn_dist, enc_batch_extend_vocab if self.args.pointer_gen else None, extra_zeros)
        ctx_loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)),
                              dec_batch.contiguous().view(-1) if self.args.pointer_gen else dec_ext_batch.contiguous().view(-1))
        loss = ctx_loss + loss_emotion + loss_emo_edge #*0.8
        #loss = loss_emotion + loss_emo_edge *0.8

        if False:#+div_loss
            _, preds = logit.max(dim=-1)
            preds = self.clean_preds(preds)
            self.update_frequency(preds)
            self.criterion.weight = self.calc_weight()
            not_pad = dec_batch.ne(self.args.PAD_idx)
            target_tokens = not_pad.long().sum().item()
            div_loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
            div_loss /= target_tokens
            loss += 2 * div_loss

        

        loss_ppl = 0.0
        if self.args.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1) if self.args.pointer_gen else dec_ext_batch.contiguous().view(-1)).item()

        if torch.sum(torch.isnan(loss)) != 0:
            print('loss is NAN :(')
            pdb.set_trace()

        if train:
            loss.backward()
            self.optimizer.step()

        #return ctx_loss.item(), math.exp(min(ctx_loss.item(), 100)), loss_emotion.item(), emotion_acc
        
        if self.args.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_emotion.item(), emotion_acc

    def compute_act_loss(self,module):    
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t,dim=1)/p_t.size(1))/p_t.size(0)
        loss = self.args.act_loss_weight * avg_p_t.item()
        return loss
    
    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch_extend_vocab, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_vad_batch = batch['context_vad']
        enc_batch_extend_vocab = batch["context_ext_batch"]

        concept_input = batch["concept_batch"]  # (bsz, max_concept_len)
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(self.args.device)

        ## Encode - context
        mask_src = enc_batch.data.eq(self.args.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        emb_mask = self.embedding(batch["mask_context"])
        src_emb1 = self.embedding(enc_batch) + emb_mask
        src_vad = enc_vad_batch  # (bsz, len, 1)

      

        if concept_input.size()[0] != 0:
                mask_con = concept_input.data.eq(self.args.PAD_idx).unsqueeze(1)  # real mask
                con_mask = self.embedding(batch["mask_concept"])  # dialogue state
                con_emb = self.embedding(concept_input) + con_mask

                ## Knowledge Update
                src_emb = self.concept_graph(src_emb1, con_emb,
                                             batch["adjacency_mask_batch"])  # (bsz, context+concept, emb_dim)
                mask_src = torch.cat((mask_src, mask_con), dim=2)  # (bsz, 1, context+concept)

                src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)  # (bsz, len)
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)

        emotion_context1=torch.mean(encoder_outputs,1)

        ## Identify
        src_vad = torch.softmax(src_vad, dim=-1)


        emo_emb=self.emo_embedding(torch.arange(32).cuda()) #+ self.emo_embedding(torch.ones([32]).long().cuda()*32)
            

        emo_words,emo_mask,length_all, emo_weight_all,emo_words_emb,edge_all, mask_m_all = self.extract_emo(enc_batch,src_emb1, batch["emotion_score"],batch['emotion_label'],emo_emb)        
        emotion_context, emotion_logit, loss_emotion, emotion_acc,src_skep,loss_emo_edge=self.emo_graph3(emo_words, emo_words_emb, emo_mask, emo_emb, batch['emotion_label'],length_all,emotion_context1,emo_weight_all,edge_all, mask_m_all)
           


        if concept_input.size()[0] != 0 :
            enc_ext_batch = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)
        else:
            enc_ext_batch = enc_batch_extend_vocab

        ys = torch.ones(1, 1).fill_(self.args.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        sos_emb = ys_emb  
        if self.args.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(self.args.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if self.args.project:
                out, attn_dist, _ = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg))
            else:
                out, attn_dist, _ = self.decoder(inputs=ys_emb,
                                                 encoder_output=encoder_outputs,
                                                 mask=(mask_src,mask_trg),
                                                 pred_emotion=None,
                                                 emotion_contexts=emotion_context,
                                                 context_vad=src_vad)

            prob = self.generator(out, None, None, attn_dist, enc_ext_batch if self.args.pointer_gen else None, extra_zeros)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == self.args.EOS_idx else self.index2word[str(ni.item())] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            if self.args.use_cuda:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(self.args.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent


    def extract_emo(self,text,text_emb,emotion_score,emo,emo_emb):    
        #text:token list, [B,len] ,vads: token VAD, [B,len,3]
        #text=enc_batch, vads=batch['context_vads']

        Valence_=torch.abs(emotion_score-0.5)
        max_l=text.shape[1]


        edge_all,mask_all,length_all,emo_weight_all,mask_m_all=[],[],[],[],[]
        Bs=text.shape[0]
        multi_mask=[0.15,0]#[0.3, 0.15, 0]# [0.4, 0.3, 0.2, 0.1, 0]##
        for i in range(Bs): 
            l_i=max_l-torch.sum(text[i].data.eq(1))
            length_all.append(l_i)
            mask=subsequent_mask(l_i,max_l)#[L, L+32]
            mask_all.append(mask)

            V_i=Valence_[i]**2
            
            V_i[0]=1
            Valence_i=torch.cat([V_i,torch.ones([32]).cuda()],0).unsqueeze(0)#[1,L+32]
            Valence_i1=V_i.unsqueeze(1)#[L,1]

            mask_m_i=[]
            for i_m in range(self.h):
                v_i = multi_mask[i_m]
              

                mask0 = mask.masked_fill(Valence_i < v_i, 1)
                mask0 = mask0.masked_fill(Valence_i1 < v_i, 1)
                mask_m_i.append(mask0)
            
           
            mask_m_i=torch.stack(mask_m_i,0)
            mask_m_all.append(mask_m_i)
            edge_value=(V_i)**2
            if torch.sum(edge_value>0) and torch.min(edge_value)!=torch.max(edge_value):
                edge_value=edge_value/torch.max(edge_value)
                None
            edge_value=torch.cat([torch.ones([1]).cuda(),edge_value[1:],torch.ones([32]).cuda()/32],-1)
            edge_value1=edge_value.unsqueeze(0).repeat(max_l,1)
            edge_all.append(edge_value1)

            emo_weight= V_i
            emo_weight_all.append(emo_weight)


        emo_weight_all=torch.stack(emo_weight_all,0)
        edge_all=torch.stack(edge_all,0)
       
        N_emo=emo_emb.shape[0]
        emo2emo_edge=torch.mm(self.S,self.S.transpose(1,0))
        
        self.emo2emo_edge=torch.softmax(emo2emo_edge,-1)

        
        emo2word_edge=torch.zeros(N_emo,max_l).cuda()#[q,L]
        emo2word_edge[:,0]=1

        emo_node_edge=torch.cat([emo2word_edge,self.emo2emo_edge],1).expand(Bs,-1,-1) #[q,q+L]=>#[B,q,q+L]


        edge_all=torch.cat([edge_all,emo_node_edge],1)

        mask_all=torch.stack(mask_all,0)#[B,max_l,max_l+32]
        mask_emo=torch.ones([Bs,max_l+32,max_l+32]).cuda()
        mask_emo[:,:max_l,:]=mask_all.cuda()
        mask_emo[:,max_l:,:max_l]=torch.ones([Bs,32,max_l]).cuda()

        mask_m_all=torch.stack(mask_m_all,0)#[B,H,L,L+32],0为连接，1为不连接

       
        emo2word_mask=torch.ones(N_emo,max_l).cuda()#[q,L]
        emo2word_mask[:,0]=0

        mask_e_all=torch.cat([emo2word_mask, torch.eye(N_emo).cuda()],-1).expand(Bs,self.h,-1,-1) #[32,L],[32,32]=>[32,L+32]=>[B,H,32,L+32]
        
        mask_node_all=torch.cat([mask_e_all,mask_m_all],-2)

        return text,mask_emo,length_all,emo_weight_all,text_emb,edge_all,mask_node_all#emo_weight_all 
    
    
  
    def multi_new_emo_mask(self,mask,emo_logit,length,emo=None,edge=None):
       
        emo_logits_sort,emo_logit_id=torch.sort(emo_logit,1)#[B,32],[B,32]
        emo_logits_mean=torch.mean(emo_logit,1)#[B]
        K=5
        S_topK=[]
        for i  in range(K):
            k=i+1
            S=k/32*((torch.mean(emo_logit[:,-k:],1)-emo_logits_mean)**2)+ (32-k)/32*((torch.mean(emo_logit[:,:-k],1)-emo_logits_mean)**2) #[B]
            S_topK.append(S)
        S_topK=torch.stack(S_topK,-1)#[B,5]
        #print('S_topK',S_topK.shape)

        _,n_select=torch.max(S_topK,1)#[B,5]=>[B]
        n_select+=1

        max_l=mask.shape[2]
        
        for i in range(mask.shape[0]):
            mask_emo= emo_mask1(emo_logit_id[i,-n_select[i]:])#[32],除保留的节点为0外其他 为1
            mask_emo_node=mask_emo.expand(self.h,max_l,-1)#[32]=>[L+32,32]=>[H,L+32,32]
            mask[i,:,:,-32:] += mask_emo_node

            
        if self.iter%1000==0 and self.tr:
            print('len',n_select)

        return mask,mask_emo,edge
    
    def emo_graph3(self,emo_words,emo_words_emb, emo_mask,emo_emb,emo_GT,length_all,emotion_context=None,emo_weight=None,edge=None,short_mask=None):

        #emo_words:[B,len], emo_mask:[B,len,len+32],emo_emb:[32,300] , emo_GT:batch['emotion_label']
        #emo_mask=emo_mask[:,:-32,:]
        emo_words_emb = self.embedding(emo_words)# [B,len,300]
        emo_emb=emo_emb.unsqueeze(0).repeat(emo_words_emb.shape[0],1,1)# [B,32,300]
  
        edge=edge.unsqueeze(1).repeat(1,self.h,1,1)
        target = emo_words_emb
        src = torch.cat((target, emo_emb), dim=1)  # (bsz, len + 32, emb_dim)

        #short_mask[:,:,:,-32:]=torch.ones([target.shape[0],self.h,target.shape[1],32])
        
        for l in range(self.GNN_L):  
            if l==0:
                q,k,v = src, src, src  
                edge_before=None
                edge_new=edge
            q,k,v,edge_new = self.T_update_all[l](q,k,v, short_mask, edge_new,emo_emb)#两层使用不同T_update_all：

          
        
        attn_=q



        edge_pro=torch.sum(torch.sum(edge_new[:,:,:-32,-32:],1),1)#[B,32],只取L个节点到emotion节点的 edge score计算
      
        edge_pro_new=edge_pro+torch.sum(torch.mul(edge_pro.unsqueeze(-1),torch.sum(edge_new[:,:,-32:,-32:],1)),1) #[B,32,32]*[B,32,1]=>[B,32,32]=>[B,32] +[B,32] =>[B,32]
       
    

        attn_1=torch.cat([edge_pro_new,self.emo_context(emotion_context)],-1) 
        #print('attn_1',attn_1.shape)
        emotion_logit=self.emo_ln2(attn_1) #[B,32*(H+2)]=> [B,32]
       
 
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_logit, emo_GT)

        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)#[bsz]
        emotion_acc = accuracy_score(emo_GT.cpu().numpy(), pred_emotion)
    
       
        
        W1=torch.sigmoid(self.emo_soft(attn_1)) #[B,32]
        emotion_soft1=torch.mul(W1,emotion_logit)+emotion_logit

        if len(emotion_soft1.size())==1:
            emotion_soft1=emotion_soft1.unsqueeze(0)
        sort_emo=torch.sort(emotion_soft1)[1]
        
        loss_emo_edge=-torch.sum(self.emo2emo_edge[sort_emo[:,-1],sort_emo[:,-2]]+self.emo2emo_edge[sort_emo[:,-1],sort_emo[:,-3]]+self.emo2emo_edge[sort_emo[:,-2],sort_emo[:,-3]])
        
        
        edge1=None
        e_m, e_m1,edge1 = self.multi_new_emo_mask(short_mask,emotion_soft1,length_all,pred_emotion,edge1)


        for l in range(self.GNN_L):  
            if l==0:
                q,k,v = src, src, src  
                edge_before=None
                edge_new=edge
            q,k,v,edge_new = self.T_update_all[l](q,k,v, e_m, edge_new,emo_emb)#两层使用不同T_update_all：
            
        new_context = torch.sum(q[:,:-32,:], dim=1)#原
        
        
        return  new_context, emotion_logit, loss_emotion, emotion_acc, edge[:,0,0,:-32],loss_emo_edge 
    
    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if self.args.EOS_idx in pred:
                ind = pred.index(self.args.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == self.args.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != self.args.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(self.args.device)            



class GraphLayer_1(nn.Module):#LN + LeakyReLU + dropout + LN + 残差连接 + LayerNorm 
    def __init__(self,input_dim,hid_dim,out_dim,dropout=0.1):
        super(GraphLayer_1, self).__init__()

        self.lin_1 = nn.Linear(input_dim, hid_dim,bias=False)#bias=False
        self.lin_2 = nn.Linear(hid_dim, out_dim,bias=False) #bias=False
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout=nn.Dropout(dropout)
      

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.dropout(x)#原
        x = self.lin_2(x)
        return x





def mask_logic(alpha, adj):
    #adj=1，置 -1e30, adj=0,保持不变
    return alpha -  adj * 1e30

def mask_logic1(alpha, adj):
    #adj=1，置 1e-24, adj=0,保持不变
    return (1-adj)*alpha +  adj * 1e-24  #1e-24    


class T_update_all(nn.Module):
    def __init__(self,emb_dim,dropout=0.1):
        super(T_update_all, self).__init__()
        self.W_q_emo = nn.Linear(emb_dim, emb_dim)#bias=False
        self.W_k_emo = nn.Linear(emb_dim, emb_dim)#bias=False
        self.W_v_emo = nn.Linear(emb_dim, emb_dim)#bias=False
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout=dropout
        self.emo_LN = nn.Linear(emb_dim, emb_dim)

    def forward(self, query, key, value, emo_mask, edge,emo_emb):
    
        q = self.W_q_emo(query) #[B,L,300]
        k, v = self.W_k_emo(key), self.W_v_emo(value)#[B,L1+32,300]
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2))  #[B,L1,L1+32]

        attn_weights_ori = attn_weights_ori * edge

        attn_weights_ori=mask_logic1(attn_weights_ori, emo_mask)


        attn_weights_ = F.dropout(torch.softmax(attn_weights_ori, dim=-1) , p=self.dropout, training=self.training)
        attn_ = torch.bmm(attn_weights_, v) #[B,L1,300]
        #print('attn_',attn_.shape,v.shape,attn_weights_.shape)
        attn_ = F.dropout(self.emo_LN(attn_) , p=self.dropout, training=self.training) #[16, L1, 300]
        attn_ = self.layer_norm(query + attn_) #[B,,L1,300]
        #print('target[:,i,:]',target[:,i,:].shape,target.shape,attn_.shape)
        
        key_new=torch.cat((attn_, emo_emb), dim=1)  
        
        return attn_,key_new,key_new





class T_update_all(nn.Module):
    def __init__(self,emb_dim,h,dropout=0.1):
        super(T_update_all, self).__init__()
        self.W_q_emo = nn.Linear(emb_dim, emb_dim)#bias=False
        self.W_k_emo = nn.Linear(emb_dim, emb_dim)#bias=False
        self.W_v_emo = nn.Linear(emb_dim, emb_dim)#bias=False
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout=dropout
        self.emo_LN = nn.Linear(emb_dim, emb_dim)
        self.L_EG=2
        self.GraphLayers = nn.ModuleList([GraphLayer_1(emb_dim,emb_dim*2,emb_dim) for _ in range(self.L_EG)])
      
        self.h = h
        self.d_k = emb_dim // self.h  
        self.edge_LN = nn.Linear(self.d_k, 1)
        self.edge_LN1 = nn.Linear(1, 1)

        


    def forward(self,query, key, value, emo_mask, edge,emo_emb,e_m1=None,emotion_soft=None,edge_before=None,query_0=None):
        #edge:[B,L1,L1+32]
        B=query.shape[0]
        len2=key.shape[1]
        len1=query.shape[1]

        q = self.W_q_emo(query).view(B, -1, self.h, self.d_k).transpose(1, 2) #[B,L1,300] => [B,H,L1,D/H]
        k, v = self.W_k_emo(key).view(B, -1, self.h, self.d_k).transpose(1, 2), self.W_v_emo(value)# [B,H,L1+32,D/H]
        attn_weights_ori = torch.matmul(q, k.transpose(-2, -1))  #[B,H,L1,L1+32]
        
        v=v.reshape([B, len2, self.h, self.d_k]).transpose(1, 2) #[B,H,L1+32,D/H]
        attn_weights_ori1 = attn_weights_ori * (self.edge_LN(v).squeeze(-1).unsqueeze(-2)*edge)

 
        attn_weights_ori=mask_logic(attn_weights_ori1, emo_mask)

        if e_m1 is not None:    
            emotion_soft = torch.softmax(mask_logic(emotion_soft, e_m1),-1) # [B,H,L,32], mask_logic
            emotion_soft=torch.cat([torch.ones([B,self.h,len1,len1]).cuda(),emotion_soft],-1)
            attn_weights_ori = emotion_soft * attn_weights_ori#[:,:,-32:] #[B,32,300]
        elif emotion_soft is not None:    
            emotion_soft=torch.cat([torch.ones([B,self.h,len1,len1]).cuda(),emotion_soft],-1)
            attn_weights_ori = emotion_soft * attn_weights_ori#[:,:,-32:] #[B,32,300]
        
        attn_weights_1 = F.dropout(torch.softmax(attn_weights_ori, dim=-1) , p=self.dropout, training=self.training)

        attn_weights_=mask_logic1(attn_weights_1, emo_mask)
        edge_new = (self.edge_LN1((edge + attn_weights_ori1).unsqueeze(-1))).squeeze(-1)
       


        
        
        attn_ = torch.matmul(attn_weights_, v).squeeze(1) #[B,H,L1,D/H]
        attn_=attn_.transpose(1, 2).reshape([B, len1, -1]) # [B,L1,H,D/H] => [B,L1,D] 
        
        for i in range(self.L_EG):
            attn_= self.GraphLayers[i](attn_)
       
   
        
        return attn_, attn_, attn_ ,edge_new       
