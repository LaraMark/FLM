# https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py 

import torch

class LatentRebatch:
    # Define the required input types for the LatentRebatch class
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latents": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              }}
    # Define the return types for the LatentRebatch class
    RETURN_TYPES = ("LATENT",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    FUNCTION = "rebatch"

    CATEGORY = "latent/batch"

    @staticmethod
    def get_batch(latents, list_ind, offset):
        '''prepare a batch out of the list of latents'''
        # Extract the samples and noise mask from the current latent
        samples = latents[list_ind]['samples']
        shape = samples.shape
        mask = latents[list_ind]['noise_mask'] if 'noise_mask' in latents[list_ind] else torch.ones((shape[0], 1, shape[2]*8, shape[3]*8), device='cpu')
        # If the noise mask shape doesn't match the samples shape, resize it
        if mask.shape[-1] != shape[-1] * 8 or mask.shape[-2] != shape[-2]:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[-2]*8, shape[-1]*8), mode="bilinear")
        # If the noise mask length is less than the samples length, repeat it
        if mask.shape[0] < samples.shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        # If the current latent has batch_index, assign it to batch_inds
        if 'batch_index' in latents[list_ind]:
            batch_inds = latents[list_ind]['batch_index']
        else:
            batch_inds = [x+offset for x in range(shape[0])]
        # Return the prepared batch
        return samples, mask, batch_inds

    @staticmethod
    def get_slices(indexable, num, batch_size):
        '''divides an indexable object into num slices of length batch_size, and a remainder'''
        slices = []
        for i in range(num):
            slices.append(indexable[i*batch_size:(i+1)*batch_size])
        # If the length of indexable is greater than the calculated length, return the remainder
        if num * batch_size < len(indexable):
            return slices, indexable[num * batch_size:]
        else:
            return slices, None
    
    @staticmethod
    def slice_batch(batch, num, batch_size):
        result = [LatentRebatch.get_slices(x, num, batch_size) for x in batch]
        return list(zip(*result))

    @staticmethod
    def cat_batch(batch1, batch2):
        if batch1[0] is None:
            return batch2
        result = [torch.cat((b1, b2)) if torch.is_tensor(b1) else b1 + b2 for b1, b2 in zip(batch1, batch2)]
        return result

    def rebatch(self, latents, batch_size):
        batch_size = batch_size[0]

        output_list = []
        current_batch = (None, None, None)
        processed = 0

        for i in range(len(latents)):
            # Fetch the next batch from the list of latents
            next_batch = self.get_batch(latents, i, processed)
            processed += len(next_batch[2])
            # If the current batch is None, set it to the next batch
            if current_batch[0] is None:
                current_batch = next_batch
            # If the shapes of the current and next batches don't match, add the current batch to the output list
            elif next_batch[0].shape[-1] != current_batch[0].shape[-1] or next_batch[0].shape[-2] != current_batch[0].shape[-2]:
                sliced, _ = self.slice_batch(current_batch, 1, batch_size)
                output_list.append({'samples': sliced[0][0], 'noise_mask': sliced[1][0], 'batch_index': sliced[2][0]})
                current_batch = next_batch
            # Concatenate the current and next batches if their shapes match
            else:
                current_batch = self.cat_batch(current_batch, next_batch)

            # If the length of the current batch exceeds the target batch size, slice it and add the slices to the output list
            if current_batch[0].shape[0] > batch_size:
                num = current_batch[0].shape[0] // batch_size
                sliced, remainder = self.slice_batch(current_batch, num, batch_size)

                for i in range(num):
                    output_list.append({'samples': sliced[0][i], 'noise_mask': sliced[1][i], 'batch_index': sliced[2][i]})

                current_batch = remainder

        # If the current batch is not empty, slice it and add it to the output list
        if current_batch[0] is not None:
            sliced, _ = self.slice_batch(current_batch, 1, batch_size)
            output_list.append({'samples': sliced[0][0], 'noise_mask': sliced[1][0], 'batch_index': sliced[2][0]})

        # Remove empty noise masks from the output list
        for s in output_list:
            if s['noise_mask'].mean() == 1.0:
                del s['noise_mask']

        return (output_list,)

class ImageRebatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "images": ("IMAGE",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    FUNCTION = "rebatch"

    CATEGORY = "image/batch"

    def rebatch(self, images, batch_size):
        batch_size = batch_size[0]

        output_list = []
        all_images = []
        for img in images:
            for i in range(img.shape[0]):
                all_images.append(img[i:i+1])

        for i in range(0, len(all_images), batch_size):
            output_list.append(torch.cat(all_images[i:i+batch_size], dim=0))

        return (output_list,)

NODE_CLASS_MAPPINGS = {
    "RebatchLatents": LatentRebatch,
    "RebatchImages": ImageRebatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebatchLatents": "Rebatch Latents",
    "RebatchImages": "Rebatch Images",
}
