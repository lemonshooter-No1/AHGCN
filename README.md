<div align="center">
<p align="center">
  <img src="figs/1.png" width="90%"/>
</p>
</div>

<div align="center">
<h1>Adaptive Multi-Scale Spatial-Temporal Hypergraph Neural Network for Skeleton-Based Human Action Assessment
</h1>
</div>





<p align="center">
  <i><b>Xun Li, Yongjian Zhang, Hongyun Cai, Changan Di, Minglai Shao and Hongbo Xie*</b></i><br>
  <i>Tianjin University, Nanjing University of Science and Technology</i><br>
  <i>HeBei University, Queensland University of Technology</i>
</p>


# 📌 Introduction

- We propose a unified multi-scale hypergraph convolution framework that adaptively extracts spatial context from skeleton data while aggregating temporal dependency features to improve human action assessment.
- By representing the human skeleton as a hypergraph, we design an Adaptive Spatial
Convolution (ASC) mechanism that explicitly captures and fuses high-order joint interaction across multiple spatial scales.
- We introduce an Adaptive Temporal Convolution (ATC) on hypergraph that replaces conventional temporal convolutions and enables effective modeling of interactions across
multiple frames.


# 🖇️ Framework

<div align="center">
    <img src="figs/Framework.png" width="90%" height="auto" />
</div>
<div align="center">
Fig. 1.  Illustration of the adaptive multi-scale spatial-temporal hypergraph neural network.
</div>


# 🤗 Datesets

<table>
  <tr>
    <th width="10%">Dataset</th>
    <th width="45%">Source Paper</th>
    <th width="25%">Download link</th>
    <th width="20%">Policy</th>
  </tr>
  <tr>
    <td>KIMORE</td>
    <td>•<a href="https://ieeexplore.ieee.org/document/8736767">The KIMORE Dataset: KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical REhabilitation</a></td>
    <td><a href="https://drive.google.com/drive/folders/1b1anGzSytePiCUyoz8AlGMV2G54e_ZeC">Google Drive</a></td>
    <td>For the RGB data, interested researchers can contact: Prof. Marianna Capecci (m.capecci@staff.univpm.it) and Prof. Luca Romeo (luca.romeo@unimc.it) </td>
  </tr>
  <tr>
    <td>UI-PRMD</td>
    <td>•<a href="https://www.mdpi.com/2306-5729/3/1/2">A Data Set of Human Body Movements for Physical Rehabilitation Exercises</a></td>
    <td><a href="http://webpages.uidaho.edu/ui-prmd/">webpages.uidaho.edu</a></td>
    <td></td>
  </tr>
  <tr>
    <td>MIT-Skate</td>
    <td>•<a href="https://ieeexplore.ieee.org/abstract/document/8756030">Learning to Score Figure Skating Sport Videos</a></td>
    <td><a href="https://github.com/loadder/MS_LSTM.git">GitHub</a></td>
    <td></td>
  </tr>
</table>

# Environment and dependencies

The server experiments used Python 3.10 and PyTorch 2.5.1 with CUDA 11.8 on NVIDIA RTX 3090 GPUs. Create an environment and install the training dependencies:

```bash
conda create -n ahgcn python=3.10 -y
conda activate ahgcn
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run commands from the repository root. Dataset files are supplied separately and are not included in the repository.

# Training and evaluation

## Prepared data

Place each exercise directory beside `main.py`:

- `Kimore_ex1/` through `Kimore_ex5/`: `Train_X.csv`, `Train_Y.csv`.
- `UI_PRMD_ex1/` through `UI_PRMD_ex10/`: `Data_Correct.csv`, `Labels_Correct.csv`, `Data_Incorrect.csv`, `Labels_Incorrect.csv`.
- `MIT_Skate/`: `train_x_4500.xlsx`, `scores.xlsx`.

The existing KIMORE and UI-PRMD loaders retain their preprocessing and score transformations. UI-PRMD paths are resolved from the exercise directory rather than a machine-specific Windows path.

MIT-Skate skeleton sequences are extracted beforehand from the original videos using OpenPose. Its coordinate workbook has no header, 54 columns per frame (18 joints × 3 exported values), and 4500 consecutive frame rows per video. `scores.xlsx` has no header and contains one score per video, in matching order. Missing-point handling and sequence-length preparation must already be completed. The loader preserves the supplied values without additional normalization or resampling.

MIT-Skate joint order: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar. Each joint occupies three consecutive columns.

## Model switches

`--use_topk` enables frame scoring, importance weighting, selection, and temporal hypergraph restriction in ASTHC blocks 4 and 7. Without this flag, these operations are disabled. `--topk_ratio 0.5` keeps half of the current frames at each pooling block; a ratio of 1 keeps all frames but still applies importance weighting. The full architecture includes this branch; default invocation leaves it disabled for ablations.

`--init_variant original` preserves the original initialization. `all` initializes ASC output BatchNorm scales to one, ATC output BatchNorm scales to zero, and GRU gate matrices using Xavier input weights and orthogonal recurrent weights, with input update-gate bias one. These parameters remain trainable. The individual initialization variants are also available through `--help`.

`--gru_norm 1` uses L1 normalization before the GRU; `2` uses L2. `--model_dropout` is retained for interface compatibility but is not applied in the current forward pass; `--tcn_dropout` controls dropout in the temporal hypergraph branch.

## Example training commands

Enable the complete architecture with frame selection:

```bash
python main.py --dataset Kimore_ex1 --batch_size 20 --num_epochs 800 --learning_rate 0.0003 --weight_decay 0.001 --tcn_dropout 0 --loss_beta 0.1 --lr_decay_epochs 400 --num_center 12 --sparse_lambda 0.003 --sparse_threshold 0.001 --temporal_k 4 --gru_norm 1 --init_variant all --use_topk --topk_ratio 0.5 --random_state 2 --device cuda --save_dir saved_models/Kimore_ex1_topk --model_name ahgcn
```

An explicitly selected no-pooling configuration:

```bash
python main.py --dataset Kimore_ex1 --batch_size 20 --num_epochs 800 --learning_rate 0.0003 --weight_decay 0.001 --tcn_dropout 0 --loss_beta 0.1 --lr_decay_epochs 400 --num_center 12 --sparse_lambda 0.003 --sparse_threshold 0.001 --temporal_k 4 --gru_norm 1 --init_variant all --random_state 2 --device cuda --save_dir saved_models/Kimore_ex1_no_topk --model_name ahgcn
```

Other dataset entry points use the same interface. The following are starting configurations; hyperparameters should be selected for the dataset, and these examples do not establish reproduction of all manuscript results:

```bash
python main.py --dataset UI_PRMD_ex1 --batch_size 10 --num_epochs 800 --use_topk --topk_ratio 0.5 --device cuda --save_dir saved_models/UI_PRMD_ex1 --model_name ahgcn
python main.py --dataset MIT_Skate --batch_size 4 --num_epochs 800 --use_topk --topk_ratio 0.5 --device cuda --save_dir saved_models/MIT_Skate --model_name ahgcn
```

## Checkpoints and splits

KIMORE and UI-PRMD use sample-wise random splits: 20% test, then 20% of the remaining pool for validation, giving approximately 64%/16%/20%. `--random_state` controls both splitting and training randomness. Each run saves its exact indices in `ahgcn_split.npz`.

For rehabilitation datasets, training saves `ahgcn_best_mae.pth`, `ahgcn_best_rmse.pth`, and `ahgcn_best_mape.pth`, selected by the respective validation metrics, and evaluates each on the test set. `ahgcn_best.pth` is the validation-RMSE checkpoint alias. Each checkpoint records the selected epoch, all validation metrics at that epoch, model configuration, and weights.

MIT-Skate selects 100 training samples and uses the remaining samples for testing (71 with the full 171-video dataset). It has no separate validation split and saves `ahgcn_final.pth` after the specified number of epochs.

Standalone evaluation requires the split file from the same run. Pooling, normalization and initialization configuration are restored automatically from the checkpoint:

```bash
python TestModel.py --dataset Kimore_ex1 --model_path saved_models/Kimore_ex1_topk/ahgcn_best_rmse.pth --split_path saved_models/Kimore_ex1_topk/ahgcn_split.npz --batch_size 20 --device cuda
python TestModel.py --dataset MIT_Skate --model_path saved_models/MIT_Skate/ahgcn_final.pth --split_path saved_models/MIT_Skate/ahgcn_split.npz --batch_size 4 --device cuda
```

Use a separate output directory for every experiment. Architecture versions preceding the temporal hypergraph implementation are not checkpoint-compatible. Preserve the original initialization definitions when comparing runs: even unused layers can affect random-number consumption during model construction.

## Hypergraph implementation

- `graph/Hypergraph.py`: movement-based center selection, displacement-descriptor sparse reconstruction, temporal coordinate-space k-NN, batched sparse aggregation, graph restriction and caching.
- `model/AdaptivescaleHyperConv.py`: interjoint, part and body streams with adaptive scale fusion.
- `model/AdaptivescaleTimConv.py`: temporal hypergraph convolution and optional frame selection.
- `model/module.py`: ASTHC blocks, input normalization, pooling and GRU regression.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--num_center` | 5 | Movement-selected spatial centers |
| `--sparse_lambda` | 0.01 | Sparse reconstruction L1 penalty |
| `--sparse_threshold` | 0.001 | Membership coefficient threshold |
| `--sparse_iter` | 1000 | Maximum solver iterations |
| `--sparse_tol` | 0.000001 | Solver convergence tolerance |
| `--temporal_k` | 4 | Cross-frame neighbors per hyperedge |
| `--use_topk` | off | Enable frame weighting and selection |
| `--topk_ratio` | 0.5 | Retention ratio at each pooling block |
| `--gru_norm` | 1 | GRU input normalization norm |
| `--init_variant` | original | Module initialization scheme |

`--hypergraph_cache auto` constructs spatial and temporal graphs once per run and stores them on the training GPU if their estimated size is below 10% of currently free GPU memory, otherwise on CPU. `cuda`, `cpu`, and `none` explicitly select GPU caching, CPU caching, or reconstruction on each forward pass. Batches retrieve graphs by sample index. Pooling and restriction still run during the forward pass. Caching assumes unchanged input coordinates and should be disabled for online coordinate augmentation. It is not stored in checkpoints.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests use synthetic inputs to check construction, sparse aggregation, input layout, optional pooling, gradients and cached execution. They do not validate manuscript benchmark scores.

# 💡 Performance

### 📊 The State-of-the-art Performance

<div align="center">
    <img src="figs/3.png" width="80%" height="auto" />
</div>

<div align="center">
    <img src="figs/4.png" width="80%" height="auto" />
</div>

### 📊  Different Scale Performance

<div align="center">
    <img src="figs/5.png" width="80%" height="auto" />
</div>

<div align="center">
    <img src="figs/6.png" width="80%" height="auto" />
</div>



# 🙏 Acknowledgements

This work is implemented based on [ST-GCN](https://ojs.aaai.org/index.php/aaai/article/view/12328) and [STGCN-rehab](https://ieeexplore.ieee.org/abstract/document/9709340). We sincerely thank the authors of these projects for their valuable contributions to the open-source community.



# 📧 Contact

This repository is under continuous maintenance and updates. We will keep improving the code and documentation in future updates.<br>
If you have any questions, feel free to reach out to me via email: [lixun@tju.edu.cn]

