import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from tensorboardX import SummaryWriter
from fn import config, datacore
from fn.trainer import Trainer
from fn.checkpoints import CheckpointIO
import pickle
if __name__ == '__main__':  
# Arguments
	cfg = config.load_config('configs/fn.yaml')
	is_cuda = (torch.cuda.is_available() )
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	# Set t0
	t0 = time.time()

	# Shorthands
	out_dir = 'out/fn'
	logfile = open('out/fn/log.txt','a')
	batch_size=cfg['training']['batch_size']
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	train_dataset = config.get_dataset('train', cfg)
	val_dataset = config.get_dataset('val', cfg)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
		collate_fn=datacore.collate_remove_none,
		worker_init_fn=datacore.worker_init_fn)

	#for eachkey in temp.keys():
	#	print(eachkey)
	#a_file = open("data.pkl", "wb")
	#pickle.dump(temp, a_file)
	#a_file.close()
	#input()

	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
		collate_fn=datacore.collate_remove_none,
		worker_init_fn=datacore.worker_init_fn)

	model = config.get_model(cfg, device)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	trainer = Trainer(model, optimizer, device=device)


	checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

	try:
		load_dict = checkpoint_io.load('model.pt')
	except FileExistsError:
		load_dict = dict()
	epoch_it = load_dict.get('epoch_it', -1)
	it = load_dict.get('it', -1)
	metric_val_best = np.inf


	logger = SummaryWriter(os.path.join(out_dir, 'logs'))

	# Shorthands


	nparameters = sum(p.numel() for p in model.parameters())

	logfile.write('Total number of parameters: %d' % nparameters)

	print_every = cfg['training']['print_every']
	checkpoint_every = cfg['training']['checkpoint_every']
	validate_every = cfg['training']['validate_every']
	
	while True:
		epoch_it += 1
	#     scheduler.step()
		logfile.flush()
		if epoch_it>20000:
			logfile.close()
			break
		for batch in train_loader:
			it += 1

			if  batch['input'].shape[0]==1:
				continue
			loss = trainer.train_step(batch)
			logger.add_scalar('train/loss', loss, it)
			
			if print_every > 0 and (it % print_every) == 0 and it > 0 :
				logfile.write('[Epoch %02d] it=%03d, loss=%.6f\n'
					  % (epoch_it, it, loss))
				print('[Epoch %02d] it=%03d, loss=%.6f'
					  % (epoch_it, it, loss))



			# Save checkpoint
			if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and it > 0 :
				logfile.write('Saving checkpoint')
				checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
								   loss_val_best=metric_val_best)

			# Run validation
			if validate_every > 0 and (it % validate_every) == 0 and it > 0 :
				metric_val = trainer.evaluate(val_loader)
				metric_val=metric_val.float()
				logfile.write('Validation metric : %.6f\n'
					  % (metric_val))
				if metric_val < metric_val_best:
					metric_val_best = metric_val
					logfile.write('New best model (loss %.6f)\n' % metric_val_best)
					checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
									   loss_val_best=metric_val_best)