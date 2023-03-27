from DeepPurpose import PPI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
def eva(filename1,j):
    # load DB Binary Data
    X_targets, X_targets_, y = read_file_training_dataset_protein_protein_pairs(filename1)

    target_encoding = 'CNN'
    train, val, test = data_process(X_target = X_targets, X_target_ = X_targets_, y = y,
                    target_encoding = target_encoding,
                    split_method='random',
                    random_seed = 1)
    config = generate_config(target_encoding=target_encoding,
                             cls_hidden_dims=[512],
                             train_epoch=20,
                             LR = 0.001,
                             batch_size = 128,
                            )

    model = models.model_initialize(**config)
    model.train(train, val,j, test)


if __name__ == '__main__':
    for i in range(5):
       filename = "human/human"+str(i)
       eva(filename,i)
