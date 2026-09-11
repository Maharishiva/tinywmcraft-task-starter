"""Implement your world model and training here."""
import numpy as np

# from eval import DATA_DIR, load_data; data = load_data(DATA_DIR / "train.npz")


def load_model(path=None):
    """Return (model, checkpoint metadata), with measured training seconds in 'train_s'."""
    if path is not None:
        raise NotImplementedError("Implement checkpoint loading.")
    return None, {"train_s": 0.0}


def rollout(model, context_frames, context_actions, actions, seed):
    """Return uint8 frames [B, T, ...], where T = actions.shape[1]."""
    if model is not None:
        raise NotImplementedError("Implement your model's rollout.")
    return np.repeat(context_frames[:, -1:], actions.shape[1], axis=1)


if __name__ == "__main__":
    raise NotImplementedError("Implement training here.")
