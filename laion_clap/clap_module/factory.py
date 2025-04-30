def load_state_dict(checkpoint_path: str, map_location: str = "cpu", skip_params=True):
    """Load state_dict from checkpoint path
    
    Args:
        checkpoint_path (str): path to checkpoint
        map_location (str, optional): map location. Defaults to "cpu".
        skip_params (bool, optional): skip parameters. Defaults to True.
    
    Returns:
        dict: state_dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    # --- ADDED weights_only=False to handle older checkpoints with PyTorch >= 2.6 ---
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    # -------------------------------------------------------------------------------

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint 