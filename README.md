## 	E-CORE: Emotion Correlation Enhanced Empathetic Dialogue Generation

This is the official implementation for paper [E-CORE: Emotion Correlation Enhanced Empathetic Dialogue Generation] (EMNLP 2023).



## Setup

- Check the packages needed or simply run the command:
```console
pip install -r requirements.txt
```
- Download GloVe vectors from [**here (glove.6B.300d.txt)**](http://nlp.stanford.edu/data/glove.6B.zip) and put it into `/data_ecore/`.

- Download other data sources, please visit [**Google Drive**](https://drive.google.com/file/d/1gNjD8V_dZVfCafUZIZuYaGsJc2Zm9jLA/view?usp=sharing) and place processed dataset `skep_dataset_preproc.json` into `/data_ecore/`.


## Training

```bash

CUDA_VISIBLE_DEVICES=0  python main_graph.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300  --heads 2 --pretrain_emb  --device_id 0 --concept_num 1 --total_concept_num 10 --attn_loss --pointer_gen  --emb_file [glove_path] --hop 4 --train_then_test --model [model name] --dataset [dataset path]


```
## Citation
If you find our work useful, please cite our paper as follows:

```bash

@conference{fu2023core,
	author = {Fu, Fengyi and Zhang, Lei and Wang, Quan and Mao, Zhendong},
	booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing and the 13th International Joint Conference on Natural Language Processing, EMNLP 2023},
	journal = {arXiv preprint arXiv:2311.15016},
	title = {E-core: Emotion correlation enhanced empathetic dialogue generation},
	year = {2023}}
```

