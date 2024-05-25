import glob
import logging
import os
import pickle
import signal
import subprocess
import time

from embedding_layer_reset import read_initial_optimizer_state, reset_embedding_layer

# read environment variables
repository_location = os.environ.get("REPOSITORY_LOCATION")
assert (
    repository_location is not None
), "REPOSITORY_LOCATION environment variable not set"

training_config = os.environ.get("TRAINING_CONFIG")
assert training_config is not None, "TRAINING_CONFIG environment variable not set"

checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
assert checkpoint_dir is not None, "CHECKPOINT_DIR environment variable not set"

af_interval = os.environ.get("AF_INTERVAL")
assert af_interval is not None, "AF_INTERVAL environment variable not set"

af_first_reset = os.environ.get("AF_FIRST_RESET")
assert af_first_reset is not None, "AF_FIRST_RESET environment variable not set"

af_last_reset = os.environ.get("AF_LAST_RESET")
assert af_last_reset is not None, "AF_LAST_RESET environment variable not set"

wandb_api_key = os.environ.get("WANDB_API_KEY")
assert wandb_api_key is not None, "WANDB_API_KEY environment variable not set"

# create active forgetting schedule
checkpoints_to_reset = range(
    int(af_first_reset), int(af_last_reset) + 1, int(af_interval)
)

logger = logging.getLogger(__name__)


def wandb_login():
    os.system(f"wandb login {wandb_api_key}")
    logger.info("Logged in to wandb.")


def neox_training_run(training_output_file=None):
    return subprocess.Popen(
        [
            "gpt-neox/deepy.py",
            "gpt-neox/train.py",
            training_config,
        ],
        stdout=training_output_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

def checkpoint_already_exists(checkpoint_dir, checkpoint: int):
    potential_path1 = os.path.join(checkpoint_dir, f"global_step{int(checkpoint)}", "layer_00-model_00-model_states.pt")
    potential_path2 = os.path.join(checkpoint_dir, f"global_step{int(checkpoint)}", "mp_rank_00_model_states.pt")
    return os.path.exists(potential_path1) or os.path.exists(potential_path2)


def active_forgetting_pipeline():
    if os.environ.get("BASELINE_RUN") == "true":
        logger.info("Running baseline training run ...")
        apptainer_process = neox_training_run()
        apptainer_process.wait()
        return

    # create folder for run logs
    run_wdir = os.path.join(
        repository_location, f"active_forgetting_run_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(
        run_wdir,
        exist_ok=True,
    )

    logger.info(
        f"Starting active forgetting pipeline. Logs will be saved to: {run_wdir}"
    )

    initial_optimizer_state = None
    initial_checkpoint_path = os.path.join(
        checkpoint_dir,
        "global_step0",
        "mp_rank_00_model_states.pt",
    )

    # try to load initial optimizer state from pickle file
    if os.path.exists(os.path.join(checkpoint_dir, "initial_optimizer_state.pkl")):
        with open(
            os.path.join(checkpoint_dir, "initial_optimizer_state.pkl"),
            "rb",
        ) as f:
            initial_optimizer_state = pickle.load(f)
        logger.info("Loaded initial optimizer state from pickle file.")

    for checkpoint in checkpoints_to_reset:
        # check if checkpoint already exists, in that case skip
        if checkpoint_already_exists(checkpoint_dir, checkpoint):
            logger.info(f"Checkpoint {checkpoint} already exists, skipping ...")
            continue

        logger.info(f"Starting training run to arrive at checkpoint {checkpoint} ...")

        # start apptainer training run
        with open(
            os.path.join(run_wdir, f"training_run_cp{int(checkpoint)}.txt"),
            "w",
        ) as training_output_file:
            apptainer_process = neox_training_run(training_output_file)
            time.sleep(5)

        container_start_time = time.time()

        target_checkpoint = os.path.join(
            checkpoint_dir,
            f"global_step{int(checkpoint)}",
            "mp_rank_00_model_states.pt",
        )

        # wait for the necessary checkpoint to be created
        logger.info(f"Waiting for checkpoint to be created: {target_checkpoint}")
        while True:
            # if initial optimizer state not yet read, do so now
            if initial_optimizer_state is None and os.path.exists(
                initial_checkpoint_path
            ):
                # wait a bit to make sure the checkpoint is fully written
                time.sleep(20)

                # find all optimizer state .pt files
                optimizer_state_files = glob.glob(
                    os.path.join(
                        checkpoint_dir,
                        "global_step0",
                        "zero_pp_rank_*_mp_rank_00_optim_states.pt",
                    )
                )

                initial_optimizer_state = read_initial_optimizer_state(
                    optimizer_state_files
                )

                # store the initial optimizer state as pickle file in checkpoint directory
                with open(
                    os.path.join(checkpoint_dir, "initial_optimizer_state.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(initial_optimizer_state, f)

                logger.info("Stored initial optimizer state as pickle file.")

            if os.path.exists(target_checkpoint):
                break

            # also check if the apptainer_process is still running, if not restart
            try:
                os.killpg(os.getpgid(apptainer_process.pid), 0)
            except ProcessLookupError:
                logger.error(
                    "Training run process has stopped unexpectedly, restarting ..."
                )
                with open(
                    os.path.join(
                        run_wdir,
                        f"training_run_cp{int(checkpoint)}_restart{time.strftime('%Y%m%d-%H%M%S')}.txt",
                    ),
                    "w",
                ) as training_output_file:
                    apptainer_process = neox_training_run(training_output_file)
                time.sleep(5)

            time.sleep(5)

        # stop the training run
        logger.info(
            f"Checkpoint found (took {time.time() - container_start_time:.1f} seconds). Stopping training run ..."
        )

        time.sleep(40)
        os.killpg(os.getpgid(apptainer_process.pid), signal.SIGTERM)
        time.sleep(20)

        # reset the embedding layer
        logger.info("Resetting embedding layer ...")
        reset_embedding_layer(
            os.path.join(checkpoint_dir, f"global_step{int(checkpoint)}"),
            initial_optimizer_state,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    active_forgetting_pipeline()
