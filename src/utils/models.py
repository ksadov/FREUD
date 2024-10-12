def get_n_dict_components(activation_size: int, expansion_factor: int, n_dict_components: int) -> int:
    if n_dict_components == 0:
        return activation_size * expansion_factor
    return n_dict_components
