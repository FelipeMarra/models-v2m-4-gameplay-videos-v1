import torch

checkpoint_path = "/home/es119256/dados/xps/audiocraft_vivit_felipe/xps/8782370b/checkpoint.th"

state = torch.load(checkpoint_path, 'cpu')

print(state.keys())

#print(f"sate:\n{state.keys()}")
# dict_keys(['history', 'xp.cfg', 'xp.sig', 'best_state', 'fsdp_best_state', 'model', 'optimizer', 'lr_scheduler', 'scaler', 'ema'])

#print(f"best_state:\n{list(state['best_state']['model'].keys())[:10]}\n")

#print(f"model:\n{list(state['model'].keys())[:10]}\n")

#print(f"ema:\n{state['ema']}\n")

print(f"xp.cfg:\n{state['xp.cfg']}\n")
