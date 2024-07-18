import errno
import os
import re
import torch

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
        
def update_checkpoint_link(checkpoint_dir, target_link_list, num_epochs):
    old_target_list = []
    target_list = []
    for target_name, link_name in target_link_list:
        target_path = os.path.join(checkpoint_dir, target_name)
        link_path = os.path.join(checkpoint_dir, link_name)
        if os.path.exists(link_path):
            old_target_path = os.path.join(checkpoint_dir, os.readlink(link_path))
            old_target_list.append(old_target_path)
        target_list.append(target_path)
        symlink_force(target_name, link_path)

    for old_target_path in set(old_target_list):
        old_epoch = int(re.findall(r'\d+', os.path.basename(old_target_path))[0])
        if old_target_path not in target_list and old_epoch % 10 != 0 and old_epoch != num_epochs - 1:
            os.remove(old_target_path)
