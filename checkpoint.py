import os
import torch
#from tensorflow.io import gfile
import numpy as np

def get_l16_config(config):
    """ ViT-L/16 configuration """
    config.patch_size = 16
    config.emb_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def get_b16_config(config):
    """ ViT-B/16 configuration """
    config.patch_size = 16
    config.emb_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def load_checkpoint(path, new_img=384, patch=16, emb_dim=768, layers=12):
    """ Load weights from a given checkpoint path in npz/pth """
    if path.endswith('npz'):
        keys, values = load_jax(path)
        state_dict = convert_jax_pytorch(keys, values)
    elif path.endswith('pth') or path.endswith('pt'):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
    else:
        raise ValueError("checkpoint format {} not supported yet!".format(path.split('.')[-1]))

    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    # if 'pos_embed' in state_dict.keys():
    #     # Deal with class token
    #     posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    #     model_grid_seq = new_img//patch
    #     ckpt_grid_seq = int(np.sqrt(posemb_grid.shape[0]))

    #     if model_grid_seq!=ckpt_grid_seq:
    #         # Get old and new grid sizes
    #         posemb_grid = posemb_grid.reshape(ckpt_grid_seq, ckpt_grid_seq, -1)

    #         posemb_grid = torch.unsqueeze(posemb_grid.permute(2, 0, 1), dim=0)
    #         posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(model_grid_seq, model_grid_seq), mode='bicubic', align_corners=False)
    #         posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)

    #         # Deal with class token and return
    #         posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    #         # if 'jx' in path:
    #         #     state_dict['pos_embed'] = posemb
    #         # else:
    #         state_dict['transformer.pos_embedding.pos_embedding'] = posemb
    #         print('Resized positional embedding from (%d,%d) to (%d,%d)'%(ckpt_grid_seq,ckpt_grid_seq,model_grid_seq,model_grid_seq))
    return state_dict


def load_jax(path):
    """ Loads params from a npz checkpoint previously stored with `save()` in jax implemetation """
    ckpt_dict = np.load(path, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
    # with gfile.GFile(path, 'rb') as f:
    #     ckpt_dict = np.load(f, allow_pickle=False)
    #     keys, values = zip(*list(ckpt_dict.items()))
    return keys, values


def save_jax_to_pytorch(jax_path, save_path):
    model_name = jax_path.split('/')[-1].split('.')[0]
    keys, values = load_jax(jax_path)
    state_dict = convert_jax_pytorch(keys, values)
    checkpoint = {'state_dict': state_dict}
    torch.save(checkpoint, os.path.join(save_path, model_name + '.pth'))


def replace_names(names):
    """ Replace jax model names with pytorch model names """
    new_names = []
    for name in names:
        if name == 'Transformer':
            new_names.append('transformer')
        elif name == 'encoder_norm':
            new_names.append('norm')
        elif 'encoderblock' in name:
            num = name.split('_')[-1]
            new_names.append('encoder_layers')
            new_names.append(num)
        elif 'LayerNorm' in name:
            num = name.split('_')[-1]
            if num == '0':
                new_names.append('norm{}'.format(1))
            elif num == '2':
                new_names.append('norm{}'.format(2))
        elif 'MlpBlock' in name:
            new_names.append('mlp')
        elif 'Dense' in name:
            num = name.split('_')[-1]
            new_names.append('fc{}'.format(int(num) + 1))
        elif 'MultiHeadDotProductAttention' in name:
            new_names.append('attn')
        elif name == 'kernel' or name == 'scale':
            new_names.append('weight')
        elif name == 'bias':
            new_names.append(name)
        elif name == 'posembed_input':
            new_names.append('pos_embedding')
        elif name == 'pos_embedding':
            new_names.append('pos_embedding')
        elif name == 'embedding':
            new_names.append('embedding')
        elif name == 'head':
            new_names.append('classifier')
        elif name == 'cls':
            new_names.append('cls_token')
        else:
            new_names.append(name)
    return new_names


def convert_jax_pytorch(keys, values):
    """ Convert jax model parameters with pytorch model parameters """
    state_dict = {}
    for key, value in zip(keys, values):

        # convert name to torch names
        names = key.split('/')
        torch_names = replace_names(names)
        torch_key = '.'.join(w for w in torch_names)

        # convert values to tensor and check shapes
        tensor_value = torch.tensor(value, dtype=torch.float)
        # check shape
        num_dim = len(tensor_value.shape)

        if num_dim == 1:
            tensor_value = tensor_value.squeeze()
        elif num_dim == 2 and torch_names[-1] == 'weight':
            # for normal weight, transpose it
            tensor_value = tensor_value.T
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] in ['query', 'key', 'value']:
            feat_dim, num_heads, head_dim = tensor_value.shape
            # for multi head attention q/k/v weight
            tensor_value = tensor_value
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['query', 'key', 'value']:
            # for multi head attention q/k/v bias
            tensor_value = tensor_value
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            # for multi head attention out weight
            tensor_value = tensor_value
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.permute(3, 2, 0, 1)

        # print("{}: {}".format(torch_key, tensor_value.shape))
        state_dict[torch_key] = tensor_value
    return state_dict


if __name__ == '__main__':
    save_jax_to_pytorch('/Users/leon/Downloads/jax/imagenet21k+imagenet2012_ViT-L_16-224.npz', '/Users/leon/Downloads/pytorch')


