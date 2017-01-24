 #!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, logging
from collections import Counter

import numpy as np
import tensorflow as tf

from configurable import Configurable
from lib.linalg import tanh_const

#-------------- Logging  ----------------#
program = os.path.basename(sys.argv[0])
L = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
L.info("Running %s" % ' '.join(sys.argv))

# TODO MetaVocab?
#***************************************************************
class Vocab(Configurable):
  """"""
  
  SPECIAL_TOKENS = ('<PAD>', '<ROOT>', '<unk>')
  START_IDX = len(SPECIAL_TOKENS)
  PAD, ROOT, UNK = range(START_IDX)
  UNK_TOP_PRET = UNK
  
  #=============================================================
  def __init__(self, vocab_file, conll_idx, *args, **kwargs):
    """"""
    
    self._vocab_file = vocab_file
    self._conll_idx = conll_idx
    load_embed_file = kwargs.pop('load_embed_file', False)
    global_step = kwargs.pop('global_step', None)
    cased = kwargs.pop('cased', None)
    super(Vocab, self).__init__(*args, **kwargs)
    if cased is None:
      self._cased = super(Vocab, self).cased
    else:
      self._cased = cased
    if self.name == 'Tags':
      self.SPECIAL_TOKENS = ('PAD', 'ROOT', 'UNK')
    elif self.name == 'Rels':
      self.SPECIAL_TOKENS = ('pad', 'root', 'unk')
    self._counts = Counter()
    self._str2idx = dict(zip(self.SPECIAL_TOKENS, range(Vocab.START_IDX)))
    self._idx2str = dict(zip(range(Vocab.START_IDX), self.SPECIAL_TOKENS))
    self._str2embed = {}
    self._embed2str = {}
    self._str2embed_stack = {} # self._str2embed_extra = {}
    self._embed2str_stack = {} # self._embed2str_extra = {}
    self.trainable_embeddings = None
    self.trainable_embeddings_stack = None
    self.pretrained_embeddings = None
    self.pretrained_embeddings_stack = None # self.pretrained_embeddings_extra = None
    if os.path.isfile(self.vocab_file):
      self.load_vocab_file()
    else:
      self.add_train_file()
      self.save_vocab_file()
    if load_embed_file:
      self.load_embed_file()
      if self.stack:
        self.load_stack_embed_file()
      # if self.extra_emb and not self.stack:
      #   self.load_extra_embed_file()
    self._finalize()
    
    if global_step is not None:
      self._global_sigmoid = 1-tf.nn.sigmoid(3*(2*global_step/(self.train_iters-1)-1))
    else:
      self._global_sigmoid = 1
    return
  
  #=============================================================
  def add(self, word, count=1):
    """"""
    
    if not self.cased:
      word = word.lower()
    
    self._counts[word] += int(count)
    return
  
  #=============================================================
  def update(self, iterable):
    """"""
    
    for elt in iterable:
      if isinstance(elt, basestring):
        self.add(elt)
      elif isinstance(iterable, dict):
        self.add(elt, iterable[elt])
      elif isinstance(elt, (tuple, list)) and len(elt) == 2:
        self.add(*elt)
      else:
        raise ValueError('WTF did you just pass to Vocab.update?')
    return
  
  #=============================================================
  def index_vocab(self):
    """"""
    
    cur_idx = Vocab.START_IDX
    buff = []
    for word_and_count in self._counts.most_common():
      if (not buff) or buff[-1][1] == word_and_count[1]:
        buff.append(word_and_count)
      else:
        buff.sort()
        for word, count in buff: 
          if count >= self.min_occur_count and word not in self._str2idx:
            self._str2idx[word] = cur_idx
            self._idx2str[cur_idx] = word
            cur_idx += 1
        buff = [word_and_count]
    buff.sort()
    for word, count in buff: 
      if count >= self.min_occur_count and word not in self._str2idx:
        self._str2idx[word] = cur_idx
        self._idx2str[cur_idx] = word
        cur_idx += 1
    return
  
  #=============================================================
  def add_train_file(self):
    """"""
    
    with open(self.train_file) as f:
      buff = []
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          if len(line) == 10:
            if hasattr(self.conll_idx, '__iter__'):
              for idx in self.conll_idx:
                self.add(line[idx])
            else:
              self.add(line[self.conll_idx])
          else:
            raise ValueError('The training file is misformatted at line %d' % (line_num+1))
    self.index_vocab()
    return

  # #=============================================================
  # def load_extra_embed_file(self):
  #   """"""
    
  #   self._str2embed_extra = dict(zip(self.SPECIAL_TOKENS, range(Vocab.START_IDX)))
  #   self._embed2str_extra = dict(zip(range(Vocab.START_IDX), self.SPECIAL_TOKENS))
  #   embeds = []
  #   with open(self.embed_file_extra) as f:
  #     cur_idx = Vocab.START_IDX
  #     for line_num, line in enumerate(f):
  #       line = line.strip().split()
  #       if line:
  #         try:
  #           self._str2embed_extra[line[0]] = cur_idx
  #           self._embed2str_extra[cur_idx] = line[0]
  #           embeds.append(line[1:])
  #           cur_idx += 1
  #         except:
  #           raise ValueError('The embedding file is misformatted at line %d' % (line_num+1))
  #   self.pretrained_embeddings_extra = np.array(embeds, dtype=np.float32)
  #   del embeds
  #   return
 
  #=============================================================
  def load_stack_embed_file(self):
    """"""
    
    self._str2embed_stack = dict(zip(self.SPECIAL_TOKENS, range(Vocab.START_IDX)))
    self._embed2str_stack = dict(zip(range(Vocab.START_IDX), self.SPECIAL_TOKENS))
    L.info('Embedding file for top LSTM : %s'%self.embed_file_stack)
    embeds = []
    with open(self.embed_file_stack) as f:
      cur_idx = Vocab.START_IDX
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          try:
            self._str2embed_stack[line[0]] = cur_idx
            if line[0] == "<unk>" and self.use_unk:
              self.UNK_TOP_PRET = cur_idx # point to <unk> vector
            self._embed2str_stack[cur_idx] = line[0]
            embeds.append(line[1:])
            cur_idx += 1
          except:
            raise ValueError('The embedding file is misformatted at line %d' % (line_num+1))
    self.pretrained_embeddings_stack = np.array(embeds, dtype=np.float32)
    del embeds
    return

  #=============================================================
  def load_embed_file(self):
    """"""
    
    self._str2embed = dict(zip(self.SPECIAL_TOKENS, range(Vocab.START_IDX)))
    self._embed2str = dict(zip(range(Vocab.START_IDX), self.SPECIAL_TOKENS))
    L.info('Embedding file for bottom LSTM : %s'%self.embed_file)
    embeds = []
    with open(self.embed_file) as f:
      cur_idx = Vocab.START_IDX
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          try:
            self._str2embed[line[0]] = cur_idx
            self._embed2str[cur_idx] = line[0]
            embeds.append(line[1:])
            cur_idx += 1
          except:
            raise ValueError('The embedding file is misformatted at line %d' % (line_num+1))
    self.pretrained_embeddings = np.array(embeds, dtype=np.float32)
    del embeds
    return
  
  #=============================================================
  def load_vocab_file(self):
    """"""
    
    with open(self.vocab_file) as f:
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          if len(line) == 1:
            line.insert(0, '')
          if len(line) == 2:
            self.add(*line)
          else:
            raise ValueError('The vocab file is misformatted at line %d' % (line_num+1))
    self.index_vocab()
  
  #=============================================================
  def save_vocab_file(self):
    """"""
    
    with open(self.vocab_file, 'w') as f:
      for word_and_count in self._counts.most_common():
        f.write('%s\t%d\n' % (word_and_count))
    return
  
  #=============================================================
  @staticmethod
  def idxs2str(indices):
    """"""
    
    shape = Configurable.tupleshape(indices)
    if len(shape) == 2:
      return ' '.join(':'.join(str(subidx) for subidx in index) if index[0] == Vocab.UNK else str(index[0]) for index in indices)
    elif len(shape) == 1:
      return ' '.join(str(index) for index in indices)
    elif len(shape) == 0:
      return ''
    else:
      raise ValueError('Indices should have len(shape) 1 or 2, not %d' % len(shape))
    return
  
  #=============================================================
  def get_embed(self, key, is_stack=False):
    """"""
    if is_stack:
      return self._embed2str_stack[key]
    # elif is_extra:
    #   return self._embed2str_extra[key]
    else:
      return self._embed2str[key]
  
  #=============================================================
  def _finalize(self):
    """"""
    
    if self.pretrained_embeddings is None:
      initializer = tf.random_normal_initializer()
    else:
      initializer = tf.zeros_initializer
      self.pretrained_embeddings = np.pad(self.pretrained_embeddings, ((self.START_IDX, 0), (0, max(0, self.embed_size - self.pretrained_embeddings.shape[1]))), 'constant')
      self.pretrained_embeddings = self.pretrained_embeddings[:,:self.embed_size]

      if self.stack:
        self.pretrained_embeddings_stack = np.pad(self.pretrained_embeddings_stack, ((self.START_IDX, 0), (0, max(0, self.stack_embed_size - self.pretrained_embeddings_stack.shape[1]))), 'constant')
        self.pretrained_embeddings_stack = self.pretrained_embeddings_stack[:,:self.stack_embed_size]      

      # if self.extra_emb:
      #   self.pretrained_embeddings_extra = np.pad(self.pretrained_embeddings_extra, ((self.START_IDX, 0), (0, max(0, self.embed_size_extra - self.pretrained_embeddings_extra.shape[1]))), 'constant')
      #   self.pretrained_embeddings_extra = self.pretrained_embeddings_extra[:,:self.embed_size_extra]
    
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.name):
        self.trainable_embeddings = tanh_const * tf.get_variable('Trainable', shape=(len(self._str2idx), self.embed_size), initializer=initializer)
        if self.stack:
          self.trainable_embeddings_stack = tanh_const * tf.get_variable('Trainable_stack', shape=(len(self._str2idx), self.embed_size), initializer=initializer)

        if self.pretrained_embeddings is not None:
          self.pretrained_embeddings /= np.std(self.pretrained_embeddings)
          self.pretrained_embeddings = tf.Variable(self.pretrained_embeddings, trainable=False, name='Pretrained')
          # if self.extra_emb:
          #   self.pretrained_embeddings_extra /= np.std(self.pretrained_embeddings_extra)
          #   self.pretrained_embeddings_extra = tf.Variable(self.pretrained_embeddings_extra, trainable=False, name='Pretrained_extra')
          if self.stack:
            self.pretrained_embeddings_stack /= np.std(self.pretrained_embeddings_stack)
            self.pretrained_embeddings_stack = tf.Variable(self.pretrained_embeddings_stack, trainable=False, name='Pretrained_stack')
    return
  
  #=============================================================
  def embedding_lookup(self, inputs, pret_inputs=None, pret_inputs_stack=None, keep_prob=1, moving_params=None, top=False):
    """"""
    
    if moving_params is not None:
      trainable_embeddings = moving_params.average(self.trainable_embeddings)
      if self.stack:
        trainable_embeddings_stack = moving_params.average(self.trainable_embeddings_stack)
      keep_prob = 1
    else:
      if self.drop_gradually:
        s = self.global_sigmoid
        keep_prob = s + (1-s)*keep_prob
      trainable_embeddings = self.trainable_embeddings
      if self.stack:
        trainable_embeddings_stack = self.trainable_embeddings_stack
    
    if top:
      embed_input = tf.nn.embedding_lookup(trainable_embeddings_stack, inputs)
    else:
      embed_input = tf.nn.embedding_lookup(trainable_embeddings, inputs)

    if moving_params is None:
      tf.add_to_collection('Weights', embed_input)
    if self.pretrained_embeddings is not None and pret_inputs is not None:
      embed_input += tf.nn.embedding_lookup(self.pretrained_embeddings, pret_inputs)
    elif self.pretrained_embeddings_stack is not None and pret_inputs_stack is not None:
      embed_input += tf.nn.embedding_lookup(self.pretrained_embeddings_stack, pret_inputs_stack)
      
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.pack([tf.shape(embed_input)[0], tf.shape(embed_input)[1], 1])
      embed_input = tf.nn.dropout(embed_input, keep_prob=keep_prob, noise_shape=noise_shape)
    return embed_input
  
  #=============================================================
  def weighted_average(self, inputs, moving_params=None):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = len(self)
    
    if moving_params is not None:
      trainable_embeddings = moving_params.average(self.trainable_embeddings)
    else:
      trainable_embeddings = self.trainable_embeddings
    
    embed_input = tf.matmul(tf.reshape(inputs, [-1, input_size]),
                            trainable_embeddings)
    embed_input = tf.reshape(embed_input, tf.pack([batch_size, bucket_size, self.embed_size]))
    embed_input.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(self.embed_size)]) 
    if moving_params is None:
      tf.add_to_collection('Weights', embed_input)
    return embed_input
  
  #=============================================================
  @property
  def vocab_file(self):
    return self._vocab_file
  @property
  def cased(self):
    return self._cased
  @property
  def conll_idx(self):
    return self._conll_idx
  @property
  def global_sigmoid(self):
    return self._global_sigmoid
  
  #=============================================================
  def keys(self):
    return self._str2idx.keys()
  def values(self):
    return self._str2idx.values()
  def iteritems(self):
    return self._str2idx.iteritems()
  
  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, basestring):
      if not self.cased:
        key = key.lower()
      if self._str2embed:
        if self._str2embed_stack:
          return (self._str2idx.get(key, Vocab.UNK), self._str2embed.get(key.lower(), Vocab.UNK), self._str2embed_stack.get(key.lower(), Vocab.UNK_TOP_PRET)) # top pret_emb may have <unk>
        else:
          return (self._str2idx.get(key, Vocab.UNK), self._str2embed.get(key.lower(), Vocab.UNK))
      else:
        return (self._str2idx.get(key, Vocab.UNK),)
    elif isinstance(key, (int, long, np.int32, np.int64)):
      return self._idx2str.get(key, self.SPECIAL_TOKENS[Vocab.UNK])
    elif hasattr(key, '__iter__'):
      return tuple(self[k] for k in key)
    else:
      raise ValueError('key to Vocab.__getitem__ must be (iterable of) string or integer')
    return
  
  def __contains__(self, key):
    if isinstance(key, basestring):
      if not self.cased:
        key = key.lower()
      return key in self._str2idx
    elif isinstance(key, (int, long)):
      return key in self._idx2str
    else:
      raise ValueError('key to Vocab.__contains__ must be string or integer')
    return
  
  def __len__(self):
    return len(self._str2idx)
  
  def __iter__(self):
    return (key for key in self._str2idx)
  
  