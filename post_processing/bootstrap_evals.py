import gzip
from data.c24_data import C24_Dataset
from torch.utils.data import DataLoader
import json
from models.patch_tst import PatchTST
import yaml
import torch
import numpy as np
from torchmetrics.classification import MulticlassF1Score, ConfusionMatrix, CohenKappa, MulticlassMatthewsCorrCoef
import tqdm
import os


def run_evaluation(model, test_loader, X_test, Y_test, idx_to_label, label_to_idx):
    f1_score = MulticlassF1Score(num_classes=10, average='macro')
    f1_score_micro = MulticlassF1Score(num_classes=10, average='micro')
    f1_score_none = MulticlassF1Score(num_classes=10, average=None)
    cm = ConfusionMatrix(num_classes=10, task='multiclass')
    ck = CohenKappa(task = 'multiclass', num_classes=10)
    mcc = MulticlassMatthewsCorrCoef(num_classes=10)

    f1_scores_boot = []
    f1_scores_micro_boot = []
    f1_scores_none_boot = []
    ck_boot = []
    mcc_boot = []
    N = len(Y_test)
    for i in tqdm.tqdm(range(10)):
        # For parallel efficiency, we can pre-generate all bootstrap indices in advance using numpy,
        # then pass indices[i] to each parallel process/worker later. This alone doesn't parallelize, but enables batch generation.
        # Use joblib/Pool for actual parallel processing (outside this snippet), but here just: 
        if i == 0:
            all_bootstrap_indices = np.random.choice(N, size=(10, N), replace=True)
        indices = all_bootstrap_indices[i]
        X_test_boot = X_test[indices]
        Y_test_boot = Y_test[indices]
        test_dataset_boot = C24_Dataset(X_test_boot, Y_test_boot, idx_to_label, label_to_idx)
        test_loader_boot = DataLoader(test_dataset_boot, batch_size=128, shuffle=False, num_workers=0)
        for x, y in tqdm.tqdm(test_loader_boot):
            outputs = model(x)
            f1_score.update(outputs, y)
            f1_score_micro.update(outputs, y)
            f1_score_none.update(outputs, y)
            cm.update(outputs, y)
            ck.update(outputs, y)
            mcc.update(outputs, y)
        

        # Evaluation Metrics - append to the lists
        f1_scores_boot.append(f1_score.compute())
        f1_scores_micro_boot.append(f1_score_micro.compute())
        f1_scores_none_boot.append(f1_score_none.compute())
        ck_boot.append(ck.compute())
        mcc_boot.append(mcc.compute())
        # Reset the metrics
        f1_score.reset()
        f1_score_micro.reset()
        f1_score_none.reset()
        ck.reset()
        mcc.reset()

    return f1_scores_boot, f1_scores_micro_boot, f1_scores_none_boot, ck_boot, mcc_boot

def main():
    # load the config file
    # load the config file
    with open('config_patchtst.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = PatchTST(config)
    model.load_state_dict(torch.load('baseline_11_29/patchtst_model.pth', weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    print(model)
    print(f"Model loaded")

    # Run Single Evaluation  on the Test Set
    with gzip.open('final_data_512/X_test.npy.gz', 'rb') as f:
        X_test = np.load(f)

    with gzip.open('final_data_512/Y_test.npy.gz', 'rb') as f:
        Y_test = np.load(f)

    # Load the index to label and label to index
    with open('final_data_512/label_to_index.json', 'r') as f:
        data = json.load(f)

    idx_to_label = data['index_to_label']
    label_to_idx = data['label_to_index']

    # INsert into loader
    test_dataset = C24_Dataset(X_test, Y_test, idx_to_label, label_to_idx)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False, 
        drop_last=False, 
        persistent_workers=False, 
        prefetch_factor=None,
    )

    # Run the evaluation
    f1_scores_boot, f1_scores_micro_boot, f1_scores_none_boot, ck_boot, mcc_boot = run_evaluation(model, test_loader, X_test, Y_test, idx_to_label, label_to_idx)

    if not os.path.exists('bootstrap_evals'):
        os.makedirs('bootstrap_evals')

    # Save the results
    np.save('bootstrap_evals/f1_scores_boot.npy', f1_scores_boot)
    np.save('bootstrap_evals/f1_scores_micro_boot.npy', f1_scores_micro_boot)
    np.save('bootstrap_evals/f1_scores_none_boot.npy', f1_scores_none_boot)
    np.save('bootstrap_evals/ck_boot.npy', ck_boot)
    np.save('bootstrap_evals/mcc_boot.npy', mcc_boot)



if __name__ == "__main__":
    main()