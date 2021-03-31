# coding: utf-8

import torch
from modules import model, data_utils, lr_scheduler
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, itertools

logger = getLogger(__name__)

def update_log_handler(file_dir):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	log_file_path = os.path.join(file_dir,'history.log')
	if os.path.isfile(log_file_path):
		retrieval = True
	else:
		retrieval = False
	handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	if retrieval:
		logger.info("LEARNING RETRIEVED.")
	else:
		logger.info("Logger set up.")
		logger.info("PyTorch ver.: {ver}".format(ver=torch.__version__))
	return retrieval,log_file_path



class Learner(object):
	def __init__(self,
			input_size,
			output_size,
			num_individuals,
			save_dir,
			attention_hidden_size = 512,
			num_attention_heads = 8,
			num_attention_layers = 1,
			bottleneck_layers=list(),
			encoder_hidden_dropout= 0.0,
			discrete_pred=False,
			sparsemax=False,
			device='cpu',
			seed=1111,
			):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))
		self.distributed = False
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
				# torch.distributed.init_process_group('nccl', rank=0, world_size=1) # Currently only support single-process with multi-gpu situation.
				self.distributed = True
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')

		if discrete_pred:
			self.loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
		else:
			self.loss_func = torch.nn.MSELoss(reduction='sum')

		self.save_dir = save_dir

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		if self.retrieval:
			self.last_iteration = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			self.seed = seed
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info('# of individuals: {}'.format(num_individuals))
			logger.info('# of attention layers: {}'.format(num_attention_layers))
			logger.info('# of attention hidden units per layer: {}'.format(attention_hidden_size))
			logger.info('# of attention heads: {}'.format(num_attention_heads))
			logger.info('Bottleneck layers where # of heads is 1 (ID# starts with 0): {}'.format(bottleneck_layers))
			logger.info('Dropout rate at the top of the sublayers: {}'.format(encoder_hidden_dropout))
			if sparsemax:
				logger.info('Sparsemax, instead of softmax, is used for attention in the bottleneck layers.')
			self.attention = model.SelfAttentionEncoder(
								input_size,
								attention_hidden_size,
								num_individuals,
								num_heads=num_attention_heads,
								num_layers=num_attention_layers,
								dropout=encoder_hidden_dropout,
								bottleneck_layers=bottleneck_layers,
								sparsemax=sparsemax
								)
			self.attention_init_args = self.attention.pack_init_args()
			if self.distributed:
				self.attention = torch.nn.DataParallel(self.attention)

			logger.info('Mask the diagonal entries of the attention weights.')
			self.last_linear = torch.nn.Linear(attention_hidden_size, output_size)
			
			self.modules = [self.attention, self.last_linear]
			[m.to(self.device) for m in self.modules]
			self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])



	def train(self, dataloader, saving_interval, start_iter=0):
		"""
		Training phase. Updates weights.
		"""
		[m.train() for m in self.modules]
		self.loss_func.train()

		num_iterations = len(dataloader)
		total_loss = 0.0
		data_size = 0.0

		for iteration,(batched_input, batched_target,_) in enumerate(dataloader, start_iter):
			iteration += 1 # Original starts with 0.
			

			batched_input = batched_input.to(self.device)
			batched_target = batched_target.to(self.device)

			self.optimizer.zero_grad()
			torch.manual_seed(iteration+self.seed)
			torch.cuda.manual_seed_all(iteration+self.seed)


			att,_ = self.attention(batched_input)
			pred = self.last_linear(att.view(-1,att.size(-1))).view(att.size()[:-1]+(-1,))
			
			if isinstance(self.loss_func, torch.nn.MSELoss):
				loss = self.loss_func(pred, batched_target)
			else:
				loss = self.loss_func(
							pred.view(-1,pred.size(-1)),
							batched_target.view(-1)
							)
			batch_x_indiv = batched_input.size(0) * batched_input.size(1)
			(loss / batch_x_indiv).backward()

			self.optimizer.step()
			self.lr_scheduler.step()

			total_loss += loss.item()
			data_size += batch_x_indiv

			if iteration % saving_interval == 0:
				logger.info('{iteration}/{num_iterations} iterations complete. mean loss (per individual): {loss}'.format(iteration=iteration, num_iterations=num_iterations, loss=total_loss / data_size))
				total_loss = 0.0
				data_size = 0.0
				self.save_model(iteration-1)
		self.save_model(iteration-1)



	def learn(self, train_dataset, num_iterations, batch_size_train, learning_rate=1e-4, betas=(0.9, 0.999), decay=0.01, warmup_iters=0, saving_interval=200, num_workers=1):
		if self.retrieval:
			start_iter = self.last_iteration + 1
			logger.info('To be restarted from the beginning of iteration #: {iteration}'.format(iteration=start_iter+1))
		else:
			self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
			self.lr_scheduler = lr_scheduler.LinearWarmUp(self.optimizer, warmup_iters)
			logger.info("START LEARNING.")
			logger.info("max # of iterations: {ep}".format(ep=num_iterations))
			logger.info("batch size for training data: {size}".format(size=batch_size_train))
			logger.info("initial learning rate: {lr}".format(lr=learning_rate))
			logger.info("weight decay: {decay}".format(decay=decay))
			logger.info("Betas: {betas}".format(betas=betas))
			logger.info("First {warmup_iters} iterations for warm-up.".format(warmup_iters=warmup_iters))
			start_iter = 0
		train_dataloader = data_utils.get_data_loader(
			train_dataset,
			batch_size=batch_size_train,
			start_iter=start_iter,
			num_iterations=num_iterations,
			shuffle=True,
			num_workers=num_workers,
			random_seed=self.seed)
		self.train(train_dataloader, saving_interval, start_iter=start_iter)
		logger.info('END OF TRAINING')


	def save_model(self, iteration):
		"""
		Save model config.
		Allow multiple tries to prevent immediate I/O errors.
		"""
		checkpoint = {
			'iteration':iteration,
			'attention':self.attention.state_dict(),
			'attention_init_args':self.attention_init_args,
			'last_linear':self.last_linear.state_dict(),
			'last_linear_init_args':{
				'in_features':self.last_linear.in_features,
				'out_features':self.last_linear.out_features
				},
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'warmup_iters':self.lr_scheduler.warmup_iters,
			'distributed':self.distributed,
			'random_seed':self.seed,
		}
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_after-{iteration}-iters.pt'.format(iteration=iteration+1)))
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu'):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.


		self.attention = model.SelfAttentionEncoder(**checkpoint['attention_init_args'])
		attention_state_dict = {('^'+key).replace('^module.', '').replace('^',''):value
							for key,value in checkpoint['attention'].items()}
		self.attention.load_state_dict(attention_state_dict)
		self.attention_init_args = self.attention.pack_init_args()
		if checkpoint['distributed']:
			self.attention = torch.nn.DataParallel(self.attention)

		self.last_linear = torch.nn.Linear(**checkpoint['last_linear_init_args'])
		self.last_linear.load_state_dict(checkpoint['last_linear'])
		self.modules = [self.attention, self.last_linear]
		[m.to(self.device) for m in self.modules]
		self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])

		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		self.optimizer.load_state_dict(checkpoint['optimizer'])

		self.lr_scheduler = lr_scheduler.LinearWarmUp(self.optimizer, checkpoint['warmup_iters'])
		self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
		
		self.seed = checkpoint['random_seed']
		return checkpoint['iteration']




def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the annotationy where results are saved.')
	parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('-i', '--iterations', type=int, default=32000, help='# of iterations to train the model.')
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size for training.')
	parser.add_argument('--attention_hidden_size', type=int, default=512, help='Dimensionality of the hidden space of the attention.')
	parser.add_argument('--num_attention_heads', type=int, default=8, help='# of attention heads.')
	parser.add_argument('--num_attention_layers', type=int, default=1, help='# of layers of attention.')
	parser.add_argument('--bottleneck_layers', type=int, default=[], nargs='+', help='ID# of layers where # of heads is 1 regardless of the value on num_attention_heads. ID# starts with 0.')
	parser.add_argument('--encoder_hidden_dropout', type=float, default=0.1, help='Dropout rate in the non-top layers of the encoder RNN.')
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--decay', type=float, default=0.01, help='Weight decay.')
	parser.add_argument('--betas', type=float, default=[0.9, 0.999], nargs=2, help='Adam coefficients used for computing running averages of gradient and its square.')
	parser.add_argument('--warmup_iters', type=int, default=0, help='# of iterations for warmup.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of workers for dataloading.')
	parser.add_argument('--saving_interval', type=int, default=200, help='# of iterations in which model parameters are saved once.')
	parser.add_argument('--xmin', type=float, default=0.0, help='Minimum on the x axis.')
	parser.add_argument('--xmax', type=float, default=5.0, help='Maximum on the x axis.')
	parser.add_argument('--xstep', type=float, default=0.5, help='Step size between grids on the x axis.')
	parser.add_argument('--ymin', type=float, default=0.0, help='Minimum on the y axis.')
	parser.add_argument('--ymax', type=float, default=4.0, help='Maximum on the y axis.')
	parser.add_argument('--ystep', type=float, default=0.5, help='Step size between grids on the y axis.')
	parser.add_argument('--zmin', type=float, default=0.0, help='Minimum on the z axis.')
	parser.add_argument('--zmax', type=float, default=2.5, help='Maximum on the z axis.')
	parser.add_argument('--zstep', type=float, default=0.5, help='Step size between grids on the z axis.')
	parser.add_argument('--discrete_pred', action='store_true', help='If selected, the prediction is discretized based on the grid space.')
	parser.add_argument('--sparsemax', action='store_true', help='If selected, use sparsemax instead of softmax to compute the attention weights in the bottleneck layers.')


	return parser.parse_args()


def get_save_dir(save_root, job_id_str):
	save_dir = os.path.join(
					save_root,
					job_id_str # + '_START-AT-' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
				)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	return save_dir

if __name__ == '__main__':
	args = parse_args()

	save_root = args.save_root
	if save_root is None:
		save_root = args.input_root
	save_dir = get_save_dir(save_root, args.job_id)

	normalizer = data_utils.build_normalizer(
							args.xmin, args.xmax,
							args.ymin, args.ymax,
							args.zmin, args.zmax
							)
	if args.discrete_pred:
		assign2block, num_blocks = data_utils.build_blocks(
										args.xmin, args.xmax, args.xstep,
										args.ymin, args.ymax, args.ystep,
										args.zmin, args.zmax, args.zstep
									)
		transform = data_utils.Transform(normalizer,assign2block)
		output_size = num_blocks
	else:
		transform = data_utils.Transform(normalizer,normalizer)
		output_size = 3

	train_dataset = data_utils.Dataset(args.input_root, args.annotation_file, transform=transform)

	num_individuals = train_dataset.get_individuals().size

	# Get a model.
	learner = Learner(
				3,
				output_size,
				num_individuals,
				save_dir,
				attention_hidden_size = args.attention_hidden_size,
				num_attention_heads = args.num_attention_heads,
				num_attention_layers = args.num_attention_layers,
				bottleneck_layers=args.bottleneck_layers,
				encoder_hidden_dropout=args.encoder_hidden_dropout,
				discrete_pred=args.discrete_pred,
				sparsemax=args.sparsemax,
				device = args.device,
				seed = args.seed,
				)


	if args.discrete_pred:
		logger.info('Output prediction is discrete based on the partitioned space.')
		logger.info('x axis ({xmin}<=x<={xmax}) is partitioned by every {xstep}.'.format(xmin=args.xmin, xmax=args.xmax, xstep=args.xstep))
		logger.info('y axis ({ymin}<=y<={ymax}) is partitioned by every {ystep}.'.format(ymin=args.ymin, ymax=args.ymax, ystep=args.ystep))
		logger.info('z axis ({zmin}<=z<={zmax}) is partitioned by every {zstep}.'.format(zmin=args.zmin, zmax=args.zmax, zstep=args.zstep))
		logger.info('{num_blocks} partitions in total.'.format(num_blocks=num_blocks))
	



	# Train the model.
	learner.learn(
			train_dataset,
			args.iterations,
			args.batch_size,
			learning_rate=args.learning_rate,
			decay = args.decay,
			betas=args.betas,
			warmup_iters=args.warmup_iters,
			saving_interval=args.saving_interval,
			num_workers=args.num_workers
			)