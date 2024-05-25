# Take the model trained with no active forgetting and finetune it with german subset

out_dir = 'out-en-sub-no-af' # directory to save the results in the end

resume_from = 'en_pretrain.pt' # Model to be finetuned

check_overfitting = False # Will run periodic evaluation to find the best model (slows down the process a lot)
eval_interval = 100 # Interval for complete evaluation (checkpoints with train and val loss)

log_interval = 10 # Number of steps between each record of the training loss (used for the plot and for the printed messages)

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False # if True, always save a checkpoint after each eval

dataset = 'german_sub' # Folder of the train and val bin
init_from = 'resume' # We resume from another model

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

max_iters = 500 # Number of iterations

# Active forgetting
active_forgetting = False # should we train with active forgetting or not
freeze_body = True # Freeze the body and only keep embedding layer to be trained

# Learning rate
learning_rate = 1e-4 #use minimum learning rate of the pre training
decay_lr = False # finetune at constant LR

# Results
training_loss_results = 'results_finetune.pkl' # File with the loss curve
complete_results = 'results_complete_finetune.pkl' # File with complete results on the test set
save_model = True # Should we save the model
model_checkpoint_name = 'german_finetune.pt'