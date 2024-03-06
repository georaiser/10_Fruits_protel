# make directories  (for testing code use subset of total image.)

import os
import random
import shutil

def create_data_splits(root_path,
                       classes,
                       train_dir, test_dir, val_dir,
                       train_ratio, test_ratio, val_ratio,
                       SEED):
    
    # Set the manual seeds
    random.seed(SEED)
   
    for directory in [train_dir, test_dir, val_dir]:
        
        if os.path.exists(directory):
            shutil.rmtree(directory)
        
        if not os.path.exists(directory):          
            os.makedirs(directory, exist_ok=True)
            [os.makedirs(os.path.join(directory, folder), exist_ok=True) for folder in classes]
  
    # Iterate through each class
    for class_name in classes:
        class_path = root_path / class_name
        images = os.listdir(class_path)
    
        # Shuffle the images to get a random sample
        random.shuffle(images)
    
        # Calculate the number of images for each split
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_test = int(test_ratio * num_images)
        num_val = int(val_ratio * num_images)
    
        # Create symbolic links in train_dir
        for image in images[:num_train]:
            src_path = class_path / image
            dst_path = train_dir / class_name / image
            os.symlink(src_path, dst_path)
    
        # Create symbolic links in test_dir
        for image in images[num_train:num_train + num_test]:
            src_path = class_path / image
            dst_path = test_dir / class_name / image
            os.symlink(src_path, dst_path)
    
        # Create symbolic links in val_dir
        for image in images[num_train + num_test:num_train + num_test + num_val]:
            src_path = class_path / image
            dst_path = val_dir / class_name / image
            os.symlink(src_path, dst_path)
