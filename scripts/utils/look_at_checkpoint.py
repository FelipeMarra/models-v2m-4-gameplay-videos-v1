import torch

checkpoint_path = "/app/xps/audiocraft_felipe/xps/c16da64f/checkpoint.th"

state = torch.load(checkpoint_path, 'cpu')

print(state.keys())

#print(f"sate:\n{state.keys()}")
# dict_keys(['history', 'xp.cfg', 'xp.sig', 'best_state', 'fsdp_best_state', 'model', 'optimizer', 'lr_scheduler', 'scaler', 'ema'])

print(f"best_state:\n{list(state['best_state']['model'].keys())[:10]}\n")

print(f"model:\n{list(state['model'].keys())[:10]}\n")

print(f"ema:\n{state['ema']}\n")
