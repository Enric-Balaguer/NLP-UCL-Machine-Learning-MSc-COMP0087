# Train a model on english subset with no active forgetting

out_dir = 'out-en-sub-no-af' # directory to save the results in the end

check_overfitting = False # Will run periodic evaluation to find the best model (slows down the process a lot)
eval_interval = 100 # Interval for complete evaluation (checkpoints with train and val loss)

log_interval = 10 # Number of steps between each record of the training loss (used for the plot and for the printed messages)

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False # if True, always save a checkpoint after each eval

dataset = 'en_sub' # Folder of the train and val bin

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
gradient_accumulation_steps = 1 
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# Define the model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

max_iters = 5000

# Learning rate
learning_rate = 1e-3 # learning rate
decay_lr = True # whether to decay the learning rate
lr_decay_iters = 5000 # Number of iterations in the lr schedule, make equal to max_iters usually
min_lr = 1e-4 # Minimal lr at the end of the schedule, learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # Number of warmup iterations

# Active forgetting
active_forgetting = False # should we train with active forgetting or not
freeze_body = False # Freeze the body and only keep embedding layer to be trained (used for finetuning)

# Results
training_loss_results = 'results_pretrain.pkl' # File with the loss curve
complete_results = 'results_pretrain_complete.pkl' # File with complete results on the test set
save_model = True # Should we save the model
model_checkpoint_name = 'en_pretrain.pt' # Model file

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model