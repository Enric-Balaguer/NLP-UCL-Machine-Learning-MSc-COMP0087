import os

import torch


def read_initial_optimizer_state(optimizer_state_files: str):
    optimizer_states = []
    for file in optimizer_state_files:
        optimizer_states.append(torch.load(file).copy())
    return optimizer_states


def reset_embedding_layer(checkpoint_dir: str, initial_optimizer_state: list):
    potential_path1 = os.path.join(checkpoint_dir, "layer_00-model_00-model_states.pt")
    potential_path2 = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")

    assert os.path.exists(potential_path1) or os.path.exists(potential_path2), "Checkpoint not found"

    if os.path.exists(potential_path1):
        checkpoint = torch.load(potential_path1)
        embedding_layer_shape = checkpoint["word_embeddings.weight"].shape
    else:
        checkpoint = torch.load(potential_path2)
        embedding_layer_shape = checkpoint["module"]["sequential.0.word_embeddings.weight"].shape

    # create new weights
    distribution_bounds = torch.sqrt(torch.tensor(3.0) / embedding_layer_shape[1])
    new_weights = torch.empty(embedding_layer_shape).uniform_(
        -distribution_bounds, distribution_bounds
    )

    new_weights = new_weights.to(device="cuda:0")  # Use the appropriate device

    if os.path.exists(potential_path1):
        checkpoint["word_embeddings.weight"] = new_weights
        storage_path = potential_path1
    else:
        checkpoint["module"]["sequential.0.word_embeddings.weight"] = new_weights
        storage_path = potential_path2

    # save updated checkpoint
    with open(
        storage_path, "wb"
    ) as f:
        torch.save(checkpoint, f)

    # reset optimizer state
    for i, initial_optimizer_state_part in enumerate(initial_optimizer_state):
        optimizer_state_part = initial_optimizer_state_part
        torch.save(
            optimizer_state_part,
            os.path.join(
                checkpoint_dir, f"zero_pp_rank_{i}_mp_rank_00_optim_states.pt"
            ),
        )


if __name__ == "__main__":
    checkpoint_path = os.path.join(
        os.getcwd(), "checkpoints", "global_step0", "mp_rank_00_model_states.pt"
    )

    model_weights_key = "module"
    embedding_layer_name = "sequential.0.word_embeddings.weight"

    reset_embedding_layer(checkpoint_path)
