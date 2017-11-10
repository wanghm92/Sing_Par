#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class Parser(BaseParser):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None, multi=False):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets
    
    reuse = (moving_params is not None) or multi
    # self.reuse = reuse
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    
    if self.load_emb:
      if self.stack:
        word_inputs_top = vocabs[0].embedding_lookup(inputs[:,:,1], pret_inputs_stack=inputs[:,:,3], moving_params=self.moving_params, top=True)
        word_inputs_btm = vocabs[0].embedding_lookup(inputs[:,:,0], pret_inputs=inputs[:,:,2], moving_params=self.moving_params)
        tag_inputs_top  = vocabs[1].embedding_lookup(inputs[:,:,4], moving_params=self.moving_params, top=True)
        tag_inputs_btm  = vocabs[1].embedding_lookup(inputs[:,:,4], moving_params=self.moving_params)
      else:
        word_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
        tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    else:
      word_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], moving_params=self.moving_params)
      tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,1], moving_params=self.moving_params)
    
    if self.stack:
      top_recur = self.embed_concat(word_inputs_btm, tag_inputs_btm)
    else:
      top_recur = self.embed_concat(word_inputs, tag_inputs)

    # Bottom/Original RNN layers
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)
    
    # Top/Stack RNN layers
    if self.stack:
      top_recur_stack = self.embed_concat(word_inputs_top, tag_inputs_top, top_recur)
      for i in xrange(self.stack_n_recur):
        with tf.variable_scope('RNN%d'%(i+int(self.n_recur)), reuse=reuse):
          top_recur_stack, _ = self.RNN(top_recur_stack)
      top_mlp_stack = top_recur_stack

    top_mlp = top_recur

    if self.n_mlp > 0:
      with tf.variable_scope('MLP_base0', reuse=reuse):
        dep_mlp, head_dep_mlp, rel_mlp, head_rel_mlp = self.MLP(top_mlp, n_splits=4)
      for i in xrange(1,self.n_mlp):
        with tf.variable_scope('DepMLP%d' % i, reuse=reuse):
          dep_mlp = self.MLP(dep_mlp)
        with tf.variable_scope('HeadDepMLP%d' % i, reuse=reuse):
          head_dep_mlp = self.MLP(head_dep_mlp)
        with tf.variable_scope('RelMLP%d' % i, reuse=reuse):
          rel_mlp = self.MLP(rel_mlp)
        with tf.variable_scope('HeadRelMLP%d' % i, reuse=reuse):
          head_rel_mlp = self.MLP(head_rel_mlp)
    else:
      dep_mlp = head_dep_mlp = rel_mlp = head_rel_mlp = top_mlp

    if self.multi:
      if self.multi_n_mlp > 0:
        with tf.variable_scope('MLP_multi0', reuse=reuse):
          dep_mlp_multi, head_dep_mlp_multi, rel_mlp_multi, head_rel_mlp_multi = self.MLP(top_mlp, n_splits=4)
        for i in xrange(1,self.multi_n_mlp):
          with tf.variable_scope('DepMLP_multi%d' % i, reuse=reuse):
            dep_mlp_multi = self.MLP(dep_mlp_multi)
          with tf.variable_scope('HeadDepMLP_multi%d' % i, reuse=reuse):
            head_dep_mlp_multi = self.MLP(head_dep_mlp_multi)
          with tf.variable_scope('RelMLP_multi%d' % i, reuse=reuse):
            rel_mlp_multi = self.MLP(rel_mlp_multi)
          with tf.variable_scope('HeadRelMLP_multi%d' % i, reuse=reuse):
            head_rel_mlp_multi = self.MLP(head_rel_mlp_multi)
      else:
        dep_mlp_multi = head_dep_mlp_multi = rel_mlp_multi = head_rel_mlp_multi = top_mlp

    if self.stack:
      if self.stack_n_mlp > 0:
        with tf.variable_scope('MLP_stack0', reuse=reuse):
          dep_mlp_stack, head_dep_mlp_stack, rel_mlp_stack, head_rel_mlp_stack = self.MLP(top_mlp_stack, n_splits=4)
        for i in xrange(1,self.stack_n_mlp):
          with tf.variable_scope('DepMLP_stack%d' % i, reuse=reuse):
            dep_mlp_stack = self.MLP(dep_mlp_stack)
          with tf.variable_scope('HeadDepMLP_stack%d' % i, reuse=reuse):
            head_dep_mlp_stack = self.MLP(head_dep_mlp_stack)
          with tf.variable_scope('RelMLP_stack%d' % i, reuse=reuse):
            rel_mlp_stack = self.MLP(rel_mlp_stack)
          with tf.variable_scope('HeadRelMLP_stack%d' % i, reuse=reuse):
            head_rel_mlp_stack = self.MLP(head_rel_mlp_stack)
      else:
        dep_mlp_stack = head_dep_mlp_stack = rel_mlp_stack = head_rel_mlp_stack = top_mlp_stack

      dep_mlp += dep_mlp_stack
      head_dep_mlp += head_dep_mlp_stack
      rel_mlp += rel_mlp_stack
      head_rel_mlp += head_rel_mlp_stack
    
    with tf.variable_scope('Parses_base', reuse=reuse):
      parse_logits = self.bilinear_classifier(dep_mlp, head_dep_mlp, add_bias1=True)
      parse_output = self.output(parse_logits, targets[:,:,1])
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = parse_output['predictions']
    with tf.variable_scope('Rels_base', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:,:,2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)

    if self.multi:
      with tf.variable_scope('Parses_multi', reuse=reuse):
        parse_logits_multi = self.bilinear_classifier(dep_mlp_multi, head_dep_mlp_multi, add_bias1=True)
        parse_output_multi = self.output(parse_logits_multi, targets[:,:,1])
        if moving_params is None:
          predictions = targets[:,:,1]
        else:
          predictions = parse_output_multi['predictions']
      with tf.variable_scope('Rels_multi', reuse=reuse):
        rel_logits_multi, rel_logits_cond_multi = self.conditional_bilinear_classifier(rel_mlp_multi, head_rel_mlp_multi, len(vocabs[2]), predictions)
        rel_output_multi = self.output(rel_logits_multi, targets[:,:,2])
        rel_output_multi['probabilities'] = self.conditional_probabilities(rel_logits_cond_multi)

    output = {}
    output['probabilities'] = tf.tuple([parse_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions'] = tf.pack([parse_output['predictions'],
                                     rel_output['predictions']])
    output['correct'] = parse_output['correct'] * rel_output['correct']
    output['tokens'] = parse_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']
    output['loss'] = parse_output['loss'] + rel_output['loss'] 
    
    if self.stack:
      output['embed_top'] = tf.pack([word_inputs_top, tag_inputs_top])
      output['embed'] = tf.pack([word_inputs_btm, tag_inputs_btm])
      output['recur_top'] = top_recur_stack
    else:
      output['embed'] = tf.pack([word_inputs, tag_inputs])
    output['recur'] = top_recur
    output['dep'] = dep_mlp
    output['head_dep'] = head_dep_mlp
    output['rel'] = rel_mlp
    output['head_rel'] = head_rel_mlp
    output['parse_logits'] = parse_logits
    output['rel_logits'] = rel_logits

    if self.multi:
      output['probabilities_multi'] = tf.tuple([parse_output_multi['probabilities'],
                                          rel_output_multi['probabilities']])
      output['predictions_multi'] = tf.pack([parse_output_multi['predictions'],
                                       rel_output_multi['predictions']])
      output['correct_multi'] = parse_output_multi['correct'] * rel_output_multi['correct']
      output['tokens_multi'] = parse_output_multi['tokens']
      output['n_correct_multi'] = tf.reduce_sum(output['correct_multi'])
      output['n_tokens_multi'] = self.n_tokens
      output['accuracy_multi'] = output['n_correct_multi'] / output['n_tokens_multi']
      output['loss_multi'] = parse_output_multi['loss'] + rel_output_multi['loss']
      output['dep_multi'] = dep_mlp_multi
      output['head_dep_multi'] = head_dep_mlp_multi
      output['rel_multi'] = rel_mlp_multi
      output['head_rel_multi'] = head_rel_mlp_multi
      output['parse_logits_multi'] = parse_logits_multi
      output['rel_logits_multi'] = rel_logits_cond_multi

    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
