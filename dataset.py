import shutil
import os
import numpy as np


DATASET_ROOT_DIR = 'dataset'
DATASET_SUBSETS = ['train', 'validation', 'test']
CLASSES = ['cat', 'dog']



def get_files_from_folder(path):
    files = os.listdir(path)
    abs_paths = [os.path.join(path, file) for file in np.asarray(files)]
    return abs_paths



def find_classes(files, classes):
    sets = list()
    for _class in classes:
        sets.append(list(filter(lambda img_name: _class in img_name, files)))
    return tuple(sets)



def create_dirs(ds_root, sets, classes):
    for dir in sets:
        for _class in classes:
            path = os.path.join(ds_root, dir, f'{_class}s')
            if os.path.isdir(path):
                return False
            os.makedirs(path)
    return True



def shuffle_and_split(data, val_ratio=0.15, test_ratio=0.05):
    dataset = np.array(data)
    np.random.shuffle(dataset)
    train, val, test = np.split(dataset, [int(len(dataset)* (1 - (val_ratio + test_ratio))), int(len(dataset)* (1 - test_ratio))])
    return train, val, test


def copy_to_dir(files, dir):
    
    for file in files:
        print(f'{file} -> {os.path.join(dir, os.path.basename(file))}', end='\r')
        shutil.copyfile(file, os.path.join(dir, os.path.basename(file)))
    print()


def split_dataset():

    # Create train, val and test directories
    dirs_alrdy_exist = create_dirs(DATASET_ROOT_DIR, DATASET_SUBSETS, CLASSES)
    if dirs_alrdy_exist is not True:
        print('Directories already exist')
    #    exit(1)

    # Find all files and split into cats and dogs
    files = get_files_from_folder(os.path.join(DATASET_ROOT_DIR, 'dataset'))
    cats, dogs = find_classes(files, CLASSES)
    print( f'Found {len(files)} images containing {len(cats)} cats and {len(dogs)} dogs')

    # Split each class into train, val and test sets
    train_cats, val_cats, test_cats = shuffle_and_split(cats)
    train_dogs, val_dogs, test_dogs = shuffle_and_split(dogs)
    print( f'Cats: train: {len(train_cats)}\tval: {len(val_cats)}\ttest: {len(test_cats)}')
    print( f'Dogs: train: {len(train_dogs)}\tval: {len(val_dogs)}\ttest: {len(test_dogs)}')

    # Copy files to directories
    copy_to_dir(train_cats, os.path.join(DATASET_ROOT_DIR, 'train', 'cats'))
    copy_to_dir(val_cats, os.path.join(DATASET_ROOT_DIR, 'validation', 'cats'))
    copy_to_dir(test_cats, os.path.join(DATASET_ROOT_DIR, 'test', 'cats'))

    copy_to_dir(train_dogs, os.path.join(DATASET_ROOT_DIR, 'train', 'dogs'))
    copy_to_dir(val_dogs, os.path.join(DATASET_ROOT_DIR, 'validation', 'dogs'))
    copy_to_dir(test_dogs, os.path.join(DATASET_ROOT_DIR, 'test', 'dogs'))

    print('Finished')


if __name__ == "__main__":
    split_dataset()
