#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simple preprocessor to convert PyTorch model dictionary into weights-only uncompressed
# zip file for use with FyuseNet. Only meant to be used with models that have been 
# quantized using GPTQ
#
# Copyright (c) 2023 Martin Wawro
# SPDX-License-Identifier: MIT

__author__ = "Martin Wawro"


import sys
import torch
import zipfile


def reformat_quantized(params, num_layers, output_file, downcast=False):
   paramlist = [('model.embed_tokens.weight', 'embedding.embed', True, False)]
   prefix = 'model.layers.'
   # NOTE (mw) the biases in the attention layers are usually all zero, we ignore them here
   for layer in range(num_layers):
       paramlist.append((prefix+'%d.self_attn.q_proj.qweight' % layer,        'dec%datt.query.weights' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.q_proj.scales' % layer,         'dec%datt.query.scales' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.q_proj.qzeros' % layer,         'dec%datt.query.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.k_proj.qweight' % layer,        'dec%datt.key.weights' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.k_proj.scales' % layer,         'dec%datt.key.scales' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.k_proj.qzeros' % layer,         'dec%datt.key.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.v_proj.qweight' % layer,        'dec%datt.value.weights' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.v_proj.scales' % layer,         'dec%datt.value.scales' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.v_proj.qzeros' % layer,         'dec%datt.value.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.o_proj.qweight' % layer,        'dec%datt.out.weights' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.o_proj.scales' % layer,         'dec%datt.out.scales' % layer, False, False))
       paramlist.append((prefix+'%d.self_attn.o_proj.qzeros' % layer,         'dec%datt.out.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.gate_proj.qweight' % layer,           'dec%dgate.weights' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.gate_proj.bias' % layer,              'dec%dgate.bias' % layer, False, True))
       paramlist.append((prefix+'%d.mlp.gate_proj.scales' % layer,            'dec%dgate.scales' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.gate_proj.qzeros' % layer,            'dec%dgate.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.up_proj.qweight' % layer,             'dec%dup.weights' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.up_proj.bias' % layer,                'dec%dup.bias' % layer, False, True))
       paramlist.append((prefix+'%d.mlp.up_proj.scales' % layer,              'dec%dup.scales' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.up_proj.qzeros' % layer,              'dec%dup.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.down_proj.qweight' % layer,           'dec%ddown.weights' % layer, False, True))
       paramlist.append((prefix+'%d.mlp.down_proj.bias' % layer,              'dec%ddown.bias' % layer, False, True))
       paramlist.append((prefix+'%d.mlp.down_proj.scales' % layer,            'dec%ddown.scales' % layer, False, False))
       paramlist.append((prefix+'%d.mlp.down_proj.qzeros' % layer,            'dec%ddown.zeros' % layer, False, False))
       paramlist.append((prefix+'%d.input_layernorm.weight' % layer,          'dec%dln0.weights' % layer , True, False))
       paramlist.append((prefix+'%d.post_attention_layernorm.weight' % layer, 'dec%dln1.weights' % layer, True, False))
   paramlist.append(('model.norm.weight','modelnorm.weights', True, False))
   paramlist.append(('lm_head.weight', 'tokenscoring.embed', True, False))
   keys = params.keys()
   with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_STORED) as outfile:
        for param, oparam, castable, optional in paramlist:
            if not param in keys:
                if optional:
                    continue
                print('Parameters for %s not found, bailing out' % param)
                return False
            else:
                tensor = params[param]
                if downcast and castable:
                    tensor = tensor.to(torch.float16)
                if tensor.dtype == torch.float32:
                    prefix = 'float32/'
                elif tensor.dtype == torch.float16:
                    prefix = 'float16/'
                elif tensor.dtype == torch.int32:
                    prefix = 'int32/'
                else:
                    print('Unknown data type')
                    return False
                filename = zipfile.ZipInfo(prefix + oparam)
                outfile.writestr(filename, tensor.numpy().tobytes())


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print('Usage: %s <input pt file> <#layers> <outputfile>' % sys.argv[0])
        sys.exit(1)

    numlayers = int(sys.argv[2])
    modeldata = torch.load(sys.argv[1])
    reformat_quantized(modeldata, numlayers, sys.argv[3], True)

