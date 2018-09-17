import torch

def SaveModel(model, path, current_epoch, save_interval):
    if (current_epoch + 1) % save_interval == 0:
        torch.save(model, str(path))
        print('Model saved in {}'.format(str(path)))