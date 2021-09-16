# Pre-training of deep bidirectional protein sequence representations with structural information (IEEE Access 2021)
Official Pytorch implementation of **PLUS** | [Paper](https://ieeexplore.ieee.org/abstract/document/9529198)
<p align="center"><img src="http://ailab.snu.ac.kr/plus/images/PLUS_Logo_ver2.jpg"></p>
<br/>

## Abstract
Bridging the exponentially growing gap between the numbers of unlabeled and labeled protein sequences, several studies adopted semi-supervised learning for protein sequence modeling. In these studies, models were pre-trained with a substantial amount of unlabeled data, and the representations were transferred to various downstream tasks. Most pre-training methods solely rely on language modeling and often exhibit limited performance. In this paper, we introduce a novel pre-training scheme called PLUS, which stands for Protein sequence representations Learned Using Structural information. PLUS consists of masked language modeling and a complementary protein-specific pre-training task, namely same-family prediction. PLUS can be used to pre-train various model architectures. In this work, we use PLUS to pre-train a bidirectional recurrent neural network and refer to the resulting model as PLUS-RNN. Our experiment results demonstrate that PLUS-RNN outperforms other models of similar size solely pre-trained with the language modeling in six out of seven widely used protein biology tasks. Furthermore, we present the results from our qualitative interpretation analyses to illustrate the strengths of PLUS-RNN. PLUS provides a novel way to exploit evolutionary relationships among unlabeled proteins and is broadly applicable across a variety of protein biology tasks. We expect that the gap between the numbers of unlabeled and labeled proteins will continue to grow exponentially, and the proposed pre-training method will play a larger role.
<br/><br/>

## Data & Models
Available at our webpage (<a href="http://ailab.snu.ac.kr/PLUS/">http://ailab.snu.ac.kr/PLUS/)
<br/><br/>

## How to Run
#### Example:
```
python plus_embedding.py --data-config config/data/embedding.json --model-config config/model/plus-rnn_large.json --run-config config/run/embedding.json --pretrained-model pretrained_models/PLUS-RNN_LARGE.pt --device 0 --output-path results/plus-rnn_large
```
<br/>

## Requirements
- Python >=3.6
- PyTorch 1.3.1
- Numpy 1.17.4
- SciPy 1.4.1
- Pandas 1.1.1
- Pillow 7.0.0
- Scikit-learn 0.22.1
<br/><br/>
