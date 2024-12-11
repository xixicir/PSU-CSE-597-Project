# Label-Retrieval-Augmented Diffusion Models for Learning from Noisy Labels
This is my implementation and replication of the [LRA-Diffusion](https://arxiv.org/abs/2305.19518v2) model. I replicated both SimCLR and CLIP versions on CIFAR-10 and CIFAR-100 datasets with various noise types.

<!-- ![CIFAR-10_TSNE](https://user-images.githubusercontent.com/123635107/214941573-02dfafbc-6e18-400d-87e6-fa604aab2501.png) -->

## 1. Preparing python environment
Install requirements.
```
pip install -r requirements.txt
```
## 2. Dataset
1. Download CIFAR datasets:
- CIFAR-10 and CIFAR-100 can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
- Place the downloaded files in the root directory

2. Dataset Information:
- CIFAR-10: 50,000 training images, 10 classes
- CIFAR-100: 50,000 training images, 100 classes

## 3. Pre-trained model & Checkpoints
* The pre-trianed SimCLR encoder for CIFAR-10 and CIFAR-100 is provided in the [model](https://github.com/xixicir/PSU-CSE-597-Project/tree/main/model) folder.
* CLIP models are available in the python package at [here](https://github.com/openai/CLIP). Install without dependency:
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git  --no-dependencies
```
* For Clothing1M, the pre-trained ["Centrality and Consistency"](https://github.com/uitrbn/tscsi_idn) (CC) classification model is also provided.

Trained checkpoints for the diffusion models are available at [here](https://drive.google.com/drive/folders/1SXzlQoOAksw349J2jnBSh5aCprDWdTQb?usp=share_link).

## 4. Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in the experiments are provided in folder `noisy_label`. The noisy labels are generated following the original [paper](https://openreview.net/pdf?id=ZPa2SyGcbwh).

## 5. Run demo script to train the LRA-diffusion
### 5.1 SimCLR: CIFAR-10 under 35% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.2 SimCLR: CIFAR-10 under 70% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.70 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.3 SimCLR: CIFAR-10 under 35% PMD + 30% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35_U_0.3 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.4 SimCLR: CIFAR-10 under 35% PMD + 60% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35_U_0.6 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.5 SimCLR: CIFAR-10 under 35% PMD  + 30% asymmetric noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35_A_0.3 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.6 CLIP: CIFAR-10 under 35% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.7 CLIP: CIFAR-10 under 70% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.70 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.8 CLIP: CIFAR-10 under 35% PMD + 30% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35_U_0.3 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.9 CLIP: CIFAR-10 under 35% PMD + 60% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35_U_0.6 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.10 CLIP: CIFAR-10 under 35% PMD  + 30% asymmetric noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35_A_0.3 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.11 SimCLR: CIFAR-100 under 35% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.12 SimCLR: CIFAR-100 under 70% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.70 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.13 SimCLR: CIFAR-100 under 35% PMD + 30% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35_U_0.3 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.14 SimCLR: CIFAR-100 under 35% PMD + 60% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35_U_0.6 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.15 SimCLR: CIFAR-100 under 35% PMD  + 30% asymmetric noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35_A_0.3 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
### 5.16 CLIP: CIFAR-100 under 35% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.17 CLIP: CIFAR-100 under 70% PMD noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.70 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.18 CLIP: CIFAR-100 under 35% PMD + 30% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35_U_0.3 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.19 CLIP: CIFAR-100 under 35% PMD + 60% uniform noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35_U_0.6 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```
### 5.20 CLIP: CIFAR-100 under 35% PMD  + 30% asymmetric noise
```
!python train_CIFAR.py --device cuda:0 --noise_type cifar100-1-0.35_A_0.3 --fp_encoder CLIP --nepoch 31 --warmup_epochs 5
```

## 6. Experiments with Different Settings
All experiments use CIFAR-10 under 35% PMD noise as the base setting.

### 6.1 Different Batch Sizes
Test model performance with different batch sizes:
```bash
# Batch size = 128
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5 --batch_size 128

# Batch size = 256
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5 --batch_size 256
```
### 6.2 Modified Learning Rate
1. Modify train_CIFAR.py:
```bash
# Replace:
optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)

# With:
optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)


# Replace:
adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.01)

# With:
adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.01)
```
2. Run the training:
```bash
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```

### 6.3 Different KNN Settings
Test different numbers of neighbors:
```bash
# k = 8 neighbors
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5 --k 8

# k = 12 neighbors
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5 --k 12
```

### 6.4 Dynamic KNN weighting
To implement dynamic KNN weighting:
1. Modify utils/knn_utils.py:
```bash
# modify the sample_knn_labels function to:
def sample_knn_labels_dynamic(query_embd, y_query, prior_embd, labels, k=10, n_class=10):

    n_sample = query_embd.shape[0]

    # Get distances and indices of k nearest neighbors
    distances, neighbour_ind = knn(query_embd, prior_embd, k=k)
    
    # Compute dynamic weights based on distances
    mean_distances = torch.mean(distances, dim=1, keepdim=True)
    # Distance-based weights
    weights = torch.exp(-distances / mean_distances)
    # Normalize weights
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # Get labels of neighbors
    neighbour_label_distribution = labels[neighbour_ind]
    
    # Add query labels
    neighbour_label_distribution = torch.cat((neighbour_label_distribution, y_query[:, None]), 1)
    weights = torch.cat((weights, torch.ones_like(weights[:, :1])), 1)  # Add weight for query label
    # Renormalize
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # Sample labels based on weighted distribution
    sampled_indices = torch.multinomial(weights, 1).squeeze()
    sampled_labels = neighbour_label_distribution[torch.arange(n_sample), sampled_indices]
    
    # Convert labels to one-hot for weight calculation
    y_one_hot_batch = nn.functional.one_hot(neighbour_label_distribution, num_classes=n_class).float()
    
    # Calculate frequency-based weights (keeping similar to original)
    neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]
    final_weights = neighbour_freq / torch.sum(neighbour_freq)
    
    return sampled_labels, torch.squeeze(final_weights)
```

2. Modify train_CIFAR.py:
```bash
# Replace:
y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch.to(device), train_embed,
                                              torch.tensor(train_dataset.targets).to(device),
                                              k=k, n_class=n_class, weighted=True)

# With:
y_labels_batch, sample_weight = sample_knn_labels_dynamic(fp_embd, y_batch.to(device), train_embed,
                                                      torch.tensor(train_dataset.targets).to(device),
                                                      k=k, n_class=n_class)
```

3. Run the training:
```bash
!python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 31 --warmup_epochs 5
```
## 7. Results and Findings
### 7.1Replication Results:
- Successfully replicated on both CIFAR-10 and CIFAR-100 datasets
- Results within reasonable range of original paper despite reduced training epochs

### 7.2Modification Effects:
- Batch size: Larger batch size (256) showed more stable training
- KNN settings: k=10 provided best balance of performance and efficiency
- Learning rate: 0.001 worked best with my training setup
- Training duration significantly impacts performance (my 30 epochs vs paper's 200)