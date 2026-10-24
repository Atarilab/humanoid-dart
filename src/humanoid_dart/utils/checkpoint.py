import torch


def smart_load_weights(model, checkpoint_path, device):
  print(f"[INFO] Loading weights from {checkpoint_path} with shape adaptation...")
  checkpoint = torch.load(checkpoint_path, map_location=device)
  if "model_state_dict" in checkpoint:
    loaded_state_dict = checkpoint["model_state_dict"]
  else:
    loaded_state_dict = checkpoint  # Assume it's the state dict directly

  current_state_dict = model.state_dict()
  adapted_state_dict = {}

  for name, current_param in current_state_dict.items():
    if name in loaded_state_dict:
      loaded_param = loaded_state_dict[name]

      if current_param.shape == loaded_param.shape:
        adapted_state_dict[name] = loaded_param
      else:
        # Handle mismatch
        print(
          f"[WARN] Shape mismatch for {name}: Current {current_param.shape} vs Loaded {loaded_param.shape}"
        )

        # Case: Input dimension increased (e.g. extra observations)
        # Weight matrix is (Out, In)
        if (
          current_param.ndim == 2
          and loaded_param.ndim == 2
          and current_param.shape[0] == loaded_param.shape[0]
          and current_param.shape[1] > loaded_param.shape[1]
        ):
          print(f"[INFO] Adapting {name} by zero-padding new input dimensions.")
          new_param = current_param.clone()
          # Copy old weights to the beginning
          # Assuming new observations are appended
          n_loaded = loaded_param.shape[1]
          new_param[:, :n_loaded] = loaded_param
          # Zero out the rest
          new_param[:, n_loaded:] = 0.0
          adapted_state_dict[name] = new_param

        else:
          print(f"[WARN] Could not adapt {name}. Using random initialization.")
          adapted_state_dict[name] = current_param
    else:
      print(f"[INFO] {name} not in checkpoint. Using random initialization.")
      adapted_state_dict[name] = current_param

  model.load_state_dict(adapted_state_dict)
  print("[INFO] Model weights loaded successfully (optimizer state reset).")
