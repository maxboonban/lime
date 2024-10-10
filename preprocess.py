import pickle
import numpy as np


#
# this unpickle function is a slight modification from Spring 2022's HW2.
#
def unpickle_CIFAR(file):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.

    :param file: the file to unpickle
    :return: inputs (NumPy array of uint8) and labels (NumPy array of string)
    """
    with open(file, "rb") as fo:
        cifar_dict = pickle.load(fo, encoding="bytes")
        
    image = cifar_dict[b"data"]
    label = cifar_dict[b"labels"]
    descriptive_names = np.array(["airplane", "automobile", "bird", "cat", "deer", 
                                  "dog", "frog", "horse", "ship", "truck"])
    
    # TODO #1: 
    #     Currently, the variable "label" is a list of integers between 0 and 9,
    #     with 0 meaning "airplane", 1 meaning "automobile" and so on.
    #     You should change the label with more descriptive names, given in the 
    #     Numpy array variable "descriptive_names". 
    #     Also, this variable "label" should be a Numpy array, not a Python list.

    label = np.array(label)
    label = descriptive_names[label]
    
    # TODO #2: 
    #     You should reshape the input image np.array to (num, width, height, channels).
    #     Currently, it is a 2D array in the shape of (images, flattened pixels)
    #     You should reshape it into (num, 3, 32, 32), because the pickled images are in
    #     three channels(RGB), and 32 pixels by 32 pixels. 
    #     However, we want the order of the axis to be in (num, width, height, channels),
    #     with the RGB channel in the last dimension. 
    
    image = np.reshape(image, (-1, 3, 32 ,32)) 
    image = np.transpose(image, axes = [0,2,3,1])
    
    # DO NOT normalize the images by dividing them with 255.0.
    # With the MNIST digits, we did normalize the images, but not with CIFAR,
    # because we will be using the pre-trained ResNet50 model, which requires
    # the pixel values to be unsigned integer values between 0 and 255.
    
    return image, label


def shuffle_data(image_full, label_full, seed):
    """
    Shuffles the full dataset with the given random seed.
    
    NOTE: DO NOT EDIT
    
    It's important that you don't edit this function, 
    so that the autograder won't be confused.
    
    :param: the dataset before shuffling
    :return: the dataset after shuffling
    """
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    
    return image_full, label_full


def get_specific_class(image_full, label_full, 
                       specific_class=0, num=None):
    """
    The MNIST dataset includes all ten digits, but they are not sorted, 
        and it does not have the same number of images for each digits.  
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a specific digit. 
    
    The same for the CIFAR dataset. We only need a small subset of CIFAR.
    
    :param image_full: the image array returned by the get_data function 
    :param label_full: the label array returned by the get_data function 
    :param specific_class: the specific class you want
    :param num: number of the images and labels to return
    :return image: Numpy array of inputs (float32) 
    :return label: Numpy array of labels 
                   (either uint8 or string, whichever type it was originally)
    """
    # TODO: 
    #     return the first "num" number of images and labels from the full dataset
    #     for example, if your dataset looks something like this:
    #         example = ["a1", "b1", "a2", "a3", "c1", b2", "c2", "a4", "c3", ...]
    #     and if you want only the first 3 samples of the class "a"
    #     you will return ["a1", "a2", "a3"]
    # Hint 1: Numpy mask operations and index slicing will be useful.
    # Hint 2: You may follow the suggested steps below, but you don't have to.
    
    # prepare the mask for that specific class
    mask = (label_full == specific_class) 

    # keep only the image and label for the specific class
    image_full = image_full[mask]
    label_full = label_full[mask]

    # keep only the first num-number entries for image and label
    # if num == None, then keep everything
    image = image_full[:num]
    label = label_full[:num]
    
    return image, label


def get_subset(image_full, label_full, 
                class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                num = 100):
    """
    The MNIST dataset includes all ten digits, but they are not sorted, 
        and it does not have the same number of images for each digits.  
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a list of specific digits. 
    
    The same for the CIFAR dataset. We only need a small subset of CIFAR.
    
    :param image: the image array returned by the get_data function 
    :param label: the label array returned by the get_data function 
    :param class_list: the list of specific classes you want
    :param num: number of the images and labels to return for each class
    :return image: Numpy array of inputs (float32) 
    :return label: Numpy array of labels 
                   (either uint8 or string, whichever type it was originally)
    """
    # TODO: 
    #     repeat collecting the first "num" number of images and labels 
    #     from the full dataset for each class in class_list, and concatenate them all
    #     for example, if your dataset looks something like this:
    #         example = ["a1", "b1", "a2", "a3", "c1", b2", "c2", "a4", "c3", ...]
    #     and if you want only the first 3 samples from each class
    #     you will return ["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3"]
    # Hint 1: Use the get_specific_class function as a helper function. 
    # Hint 2: You may follow the suggested steps below, but you don't have to.
    
    image_list = []
    label_list = []
    
    for each_class in class_list:
        # use the get_specific_class function for each class in class_list
        image_each_class, label_each_class = get_specific_class(image_full, 
                                                                label_full, 
                                                                specific_class = each_class, 
                                                                num = num)
        image_list.append(image_each_class)
        label_list.append(label_each_class)

    # concatenate the image and label arrays for all classes 
    image = np.concatenate(image_list)
    label = np.concatenate(label_list)
    
    return image, label


def one_hot_encode(label, cifar_class_list):
    assert set(label) == set(cifar_class_list)
    
    oh_label = label.copy()

    for each_class_index, each_class_name in enumerate(cifar_class_list):
        mask_each_class = (oh_label == each_class_name)
        oh_label[mask_each_class] = each_class_index

    oh_label = oh_label.astype(np.int32)
    oh_label = np.eye(len(cifar_class_list))[oh_label]
    
    return oh_label