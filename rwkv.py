import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptq import *
from modelutils import *
from quant import *

# MODEL_NAME = "sgugger/rwkv-7b-pile"
MODEL_NAME = "sgugger/rwkv-430M-pile"
DEV = torch.device('cuda:0')
# DEV = torch.device('cpu')

def get_wikitext2_rwkv(nsamples, seed, seqlen):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

@torch.no_grad()
def rwkv_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.rwkv.blocks

    model.rwkv.embeddings = model.rwkv.embeddings.to(dev)
    # Set all LayerNorm layer to device
    for block in model.rwkv.blocks:
        all_layernorm = find_layers(block, layers=[nn.LayerNorm])
        for layernorm in all_layernorm.values():
            layernorm = layernorm.to(dev)
    model.rwkv.ln_out = model.rwkv.ln_out.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0}
    
    # Should infer once to turn off self.layers_are_rescaled and not make Catcher crash
    for batch in dataloader:
        model(batch[0].to(dev))
        break

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            #TODO: add back attention_mask and inputs_ids
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for block in model.rwkv.blocks:
        all_layernorm = find_layers(block, layers=[nn.LayerNorm])
        for layernorm in all_layernorm.values():
            layernorm = layernorm.cpu()
    model.rwkv.ln_out = model.rwkv.ln_out.cpu()
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    print('Ready.')

    quantizers = {}
    # for i in range(len(layers)):
    for i in range(1): # Quantize only 1st  block
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            raise Exception('true sequential Not implemented')
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
                scale,zero,g_idx = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers['model.rwkv.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
                gptq[name].free()
                
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def benchmark(model):
    # Model
    print(model.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

    # Dataset
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    ctx_len = model.seqlen
    stride = ctx_len // 2
    seq_len = encodings.input_ids.size(1)

    from tqdm import tqdm
    nlls = []
    prev_end_loc = 0
    # for begin_loc in tqdm(range(0, seq_len, stride)):
    for begin_loc in tqdm(range(0, stride*10, stride)):
        end_loc = min(begin_loc + ctx_len, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEV)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='evaluate quantized model.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--save_safetensors', type=str, default='',
        help='Save quantized `.safetensors` checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    
    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        # model = load_quant(args.model, args.load, args.wbits, args.groupsize)
        raise Exception("load_quant not implemented")
    else:
        # device_map = {
        #     'rwkv.embeddings': 0,
        #     'rwkv.blocks.0': 0,
        #     'rwkv.blocks.1': 0,
        #     'rwkv.blocks.2': 0,
        #     'rwkv.blocks.3': 0,
        #     'rwkv.blocks.4': 0,
        #     'rwkv.blocks.5': 0,
        #     'rwkv.blocks.6': 0,
        #     'rwkv.blocks.7': 0,
        #     'rwkv.blocks.8': 0,
        #     'rwkv.blocks.9': 0,
        #     'rwkv.blocks.10': 0,
        #     'rwkv.blocks.11': 0,
        #     'rwkv.blocks.12': 0,
        #     'rwkv.blocks.13': 0,
        #     'rwkv.blocks.14': 0,
        #     'rwkv.blocks.15': 0,
        #     'rwkv.blocks.16': 0,
        #     'rwkv.blocks.17': 0,
        #     'rwkv.blocks.18': 0,
        #     'rwkv.blocks.19': 0,
        #     'rwkv.blocks.20': 0,
        #     'rwkv.blocks.21': 0,
        #     'rwkv.blocks.22': 0,
        #     'rwkv.blocks.23': 0,
        #     'rwkv.blocks.24': 0,
        #     'rwkv.blocks.25': 0,
        #     'rwkv.blocks.26': 0,
        #     'rwkv.blocks.27': 0,
        #     'rwkv.blocks.28': 'cpu',
        #     'rwkv.blocks.29': 'cpu',
        #     'rwkv.blocks.30': 'cpu',
        #     'rwkv.blocks.31': 'cpu',
        #     'rwkv.ln_out': 'cpu',
        #     'head': 'cpu'
        # }

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
        print(model.hf_device_map)
        model.seqlen = 1024 # https://huggingface.co/BlinkDL/rwkv-4-pile-430m
        model.eval()
    
    dataloader, testloader = get_wikitext2_rwkv(nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen)

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = rwkv_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    #TODO: Implment save     
    #TODO: Implement load
    
    # if args.benchmark:
    #     # model = model.to(DEV)
    #     # input_ids = next(iter(dataloader))[0][:, :args.benchmark]
    #     # benchmark(model)
            
    # if args.load:
    #     exit()
        
    # if args.eval:
    #     datasets = ['wikitext2', 'ptb', 'c4'] 
    #     if args.new_eval:
    #       datasets = ['wikitext2', 'ptb-new', 'c4-new']
    #     for dataset in datasets: 
    #         dataloader, testloader = get_loaders(
    #             dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    #         )
    #         print(dataset)
    #         llama_eval(model, testloader, DEV)

    # if args.save:
    #     llama_pack(model, quantizers, args.wbits, args.groupsize)
    #     torch.save(model.state_dict(), args.save) 