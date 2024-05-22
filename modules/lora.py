def match_lora(lora, to_load):
    # Initialize a dictionary to store the patch values
    patch_dict = {}
    # Initialize a set to store the keys that have been loaded
    loaded_keys = set()

    # Iterate over the keys in to_load
    for x in to_load:
        # Get the corresponding value in lora
        real_load_key = to_load[x]
        if real_load_key in lora:
            # If the key is found in lora, add it to the patch dictionary
            patch_dict[real_load_key] = ('fooocus', lora[real_load_key])
            # And add it to the loaded keys
            loaded_keys.add(real_load_key)
            continue

        # If the key is not found, try to find a corresponding key with a suffix
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        # Try to find a corresponding key with a suffix
        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        # If a corresponding key is found, extract the relevant values
        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name ="{}.lora_linear_layer.down.weight".format(x)
            mid_name = None

        # If a corresponding key is found, add the values to the patch dictionary
        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = ("lora", (lora[A_name], lora[B_name], alpha, mid))
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)

        # Repeat the above process for other keys with different suffixes
        # ...

        # If a key is found, add it to the patch dictionary
        # and the loaded keys set

    # Create a dictionary of keys that were not found in lora
    remaining_dict = {x: y for x, y in lora.items() if x not in loaded_keys}
    # Return the patch dictionary and the remaining dictionary
    return patch_dict, remaining_dict
