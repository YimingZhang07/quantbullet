def cache_variables(save_dir, subfolder=None, **kwargs):
    """
    Save variables to a directory as .pkl files. The directory will be created if it does not exist.
    
    Parameters
    ----------
    save_dir : str
        Directory where the variables will be saved. If it does not exist, it will be created.
    subfolder : str, optional
        Subfolder name to create inside the save_dir. If None, a timestamped folder will be created.
    kwargs : dict
        Variables to save. The keys will be used as the names of the .pkl files.
    """
    import os
    import pandas as pd
    import pickle
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = subfolder or timestamp
    full_path = os.path.join(save_dir, folder_name)

    # only allowed to create a new directory inside the save_dir
    if not os.path.exists(save_dir):
        raise ValueError(f"save_dir {save_dir} does not exist. Please create it first.")
        
    os.makedirs(full_path, exist_ok=True)

    for name, obj in kwargs.items():
        file_path = os.path.join(full_path, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    return full_path

def load_cache_variables(load_dir, *var_names, assign_to_globals=False):
    """
    Load cached variables from a directory. The directory should contain .pkl files with the same names as the variables.
    
    Parameters
    ----------
    load_dir : str
        Directory where the cached variables are stored.
    var_names : str
        Names of the variables to load. The function will look for .pkl files with these names in the load_dir.
    assign_to_globals : bool, optional
        If True, the loaded variables will be assigned to the global namespace. Default is False.
    """
    import os
    import pickle

    results = {}
    for name in var_names:
        file_path = os.path.join(load_dir, f"{name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ {file_path} not found")
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
            results[name] = obj
            if assign_to_globals:
                globals()[name] = obj
    return results