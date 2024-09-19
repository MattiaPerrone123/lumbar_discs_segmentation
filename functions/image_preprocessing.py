import numpy as np
import SimpleITK as sitk
import scipy
import math

def load_image_and_mask(directory_segm, directory_no_segm):
    #Load the image and mask
    image = nib.load(directory_no_segm)
    mask = nib.load(directory_segm)
    return image, mask





def process_array(array_main):
    #Separates vertebrae, discs and spinal cord masks based on voxel values

    #Get the unique values in the array
    unique_values = np.unique(array_main)

    #Dictionary to hold the new arrays
    new_arrays = {'array_vertebrae': np.zeros_like(array_main),
                  'array_spinal': np.zeros_like(array_main),
                  'array_disk': np.zeros_like(array_main)}

    #Group the values and create the arrays
    for value in unique_values:
        if value != 0:  # Skip 0 as it's included in all arrays
            if 1 <= value <= 99:  # Vertebrae values
                new_arrays['array_vertebrae'] += np.where(array_main == value, array_main, 0)
            elif value == 100:  # Spinal cord value
                new_arrays['array_spinal'] = np.where(array_main == value, array_main, 0)
            elif 201 <= value:  # Disk values
                new_arrays['array_disk'] += np.where(array_main == value, array_main, 0)

    return new_arrays






def resample_array_to_target_shape(np_array, target_shape):
    #Resamples a mask to a new specified spacing

    #Calculate the scaling factors for each dimension
    factors = [float(target) / float(original) for target, original in zip(target_shape, np_array.shape)]
    
    #Resample the array using linear interpolation (order=1) for non-binary data
    rescaled_array = scipy.ndimage.zoom(np_array, factors, order=1)
    
    return rescaled_array




def resample_sitk_image(sitk_image, new_spacing):
    #Resamples a SimpleITK Image to a new specified spacing
    
    # Get the original size and spacing
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()

    # Calculate the new size
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    # Resample the image
    resampled_sitk_image = sitk.Resample(sitk_image, new_size, sitk.Transform(), sitk.sitkBSpline,
                                         sitk_image.GetOrigin(), new_spacing, sitk_image.GetDirection(), 0,
                                         sitk_image.GetPixelID())

    return resampled_sitk_image



def expand_bounding_box(data, expansion_margin=10):
    #Identifies the smallest 3D bounding box around non-zero elements 
    #in a 3d Numpy array (mask) and expands it by a specified margin
    
    #Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a 3D numpy array.")

    #Check if data is 3D
    if data.ndim != 3:
        raise ValueError("Data must be a 3D numpy array.")

    #Find min and max indices along each dimension where data is non-zero
    non_zero_indices = np.argwhere(data)
    min_idx = non_zero_indices.min(axis=0)
    max_idx = non_zero_indices.max(axis=0)

    #Expand the bounding box by the specified margin, ensuring we don't go out of the array bounds
    min_x, min_y, min_z = np.maximum(min_idx - expansion_margin, 0)
    max_x, max_y, max_z = np.minimum(max_idx + expansion_margin, np.array(data.shape) - 1)

    # Return the slices for the bounding box
    return (slice(min_x, max_x + 1), slice(min_y, max_y + 1), slice(min_z, max_z + 1))


def next_multiple_of_32(number):
    #Calculates the smallest multiple of 32 that is greater than or equal to a given number 
    return math.ceil(number / 32) * 32


def find_max_dimensions(image_list):
    #Iterates through a list of 3d images to determine the maximum size for each dimension 
    #across all images, ensuring they are compatible for operations requiring uniform dimensions

    max_dim1 = max_dim2 = max_dim3 = 0

    #Iterate through all images and update the max values
    for img in image_list:
        if img.ndim != 3:
            raise ValueError("All images must be 3D.")
        max_dim1 = max(max_dim1, img.shape[0])
        max_dim2 = max(max_dim2, img.shape[1])
        max_dim3 = max(max_dim3, img.shape[2])

    return max_dim1, max_dim2, max_dim3



def resample_image(image, target_resolution):
    #Resamples the given image to the target resolution.
    return resample_sitk_image(image, target_resolution)


def process_image(img_file, target_resolution):
    #Resamples a sitk file to a target resolution, converting it to a NumPy array

    image = sitk.ReadImage(img_file)
    img_rescaled = resample_image(image, target_resolution)
    np_image = sitk.GetArrayFromImage(img_rescaled)

    if 2 * np_image.shape[-1] > np_image.shape[1] or 2 * np_image.shape[-1] > np_image.shape[0]:
        return None

    return np_image


def process_mask(msk_file, np_image_shape):
    #Reads and processes a mask file, categorizing it into vertebrae, disk and spinal arrays 

    msk = sitk.ReadImage(msk_file)
    np_msk = sitk.GetArrayFromImage(msk)
    processed_masks = process_array(np_msk)

    np_msk_vert_rescaled = resample_array_to_target_shape(processed_masks["array_vertebrae"], np_image_shape)
    np_msk_disk_rescaled = resample_array_to_target_shape(processed_masks["array_disk"], np_image_shape)
    np_msk_spinal_rescaled = resample_array_to_target_shape(processed_masks["array_spinal"], np_image_shape)

    # Ensure mask values are within binary limits
    np_msk_vert_rescaled = np.clip(np_msk_vert_rescaled, 0, 1)
    np_msk_disk_rescaled = np.clip(np_msk_disk_rescaled, 0, 1)
    np_msk_spinal_rescaled = np.clip(np_msk_spinal_rescaled, 0, 1)

    return {
        "vert": np_msk_vert_rescaled,
        "disk": np_msk_disk_rescaled,
        "spinal": np_msk_spinal_rescaled
    }




def crop_images_masks(image_list, mask_list):
    #Crop a list of images and masks according to the masks' bounding boxes.

    cropped_images = []
    cropped_masks = []
    for img, msk in zip(image_list, mask_list):
        bbox_slices_msk = expand_bounding_box(msk)
        cropped_images.append(img[bbox_slices_msk])
        cropped_masks.append(msk[bbox_slices_msk])
    return cropped_images, cropped_masks






def sigmoid(x):
    #Sigmoid function to squash values between 0 and 1
    return 1 / (1 + np.exp(-x))


def process_image_4d(volumes, mask):
    #Function performing the following steps on each slice of each volume:
    #1) Min-Max Normalization: Scales pixel values to the range [0, 1].
    #2) Z-Score Normalization: Standardizes the pixel values to have a mean of 0 and a standard deviation of 1.
    #3) Sigmoid Function: Applies a sigmoid function to squash values between 0 and 1.


    processed_volumes = []
    epsilon = 1e-7  # small constant to prevent division by zero

    if mask == 0:
        for volume in volumes:
            processed_slices = []

            for i in range(volume.shape[2]):
                slice_img = volume[:, :, i]

                #Add a check here for the slice_img size
                if slice_img.size == 0:
                    raise ValueError(f"slice_img is empty at index {i}. Volume shape was {volume.shape}")

                #1) Min-max normalization to [0, 1] range
                min_value = np.min(slice_img)
                max_value = np.max(slice_img)
                scaled_slice = (slice_img - min_value) / (max_value - min_value + epsilon)

                #2) Z-score normalization
                mean_value = np.mean(scaled_slice)
                std_value = np.std(scaled_slice)
                standardized_slice = (scaled_slice - mean_value) / (std_value + epsilon)

                #3) Apply sigmoid to squash values between 0 and 1
                normalized_slice = sigmoid(standardized_slice)
                processed_slices.append(normalized_slice)

            stacked_slices = np.stack(processed_slices, axis=2)
            processed_volumes.append(np.expand_dims(stacked_slices, axis=0))

        return np.concatenate(processed_volumes, axis=0)




def pad_images_masks(image_list, mask_list, target_dims, pad_value_img, pad_value_msk):
    #Function checking if the target dimensions are larger than the current image dimensions 
    #and calculates symmetric padding for each dimension
    
    padded_images = []
    padded_masks = []
    for img, msk in zip(image_list, mask_list):
        # Validate that the target dimensions are larger or equal to the image dimensions
        if not all(current <= target for current, target in zip(img.shape, target_dims)):
            raise ValueError("Target dimensions must be greater than or equal to image dimensions.")

        # Calculate symmetric padding for each dimension
        padding = []
        for current, target in zip(img.shape, target_dims):
            total_padding = target - current
            # Even padding on both sides
            padding_before = total_padding // 2
            padding_after = total_padding - padding_before
            padding.append((padding_before, padding_after))

        # Apply symmetric padding
        padded_img = np.pad(img, padding, mode='constant', constant_values=pad_value_img)
        padded_msk = np.pad(msk, padding, mode='constant', constant_values=pad_value_msk)

        padded_images.append(padded_img)
        padded_masks.append(padded_msk)

    return padded_images, padded_masks



