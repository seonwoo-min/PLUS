# PLUS
<p align="center"><img src="http://ailab.snu.ac.kr/plus/images/PLUS_Logo_ver2.jpg"></p>
<h4><p align="center"><strong><a href="//arxiv.org/abs/1912.05625">(https://arxiv.org/abs/1912.05625)</a></strong></p></h4>
<br/>

# Abstract
<p style="text-align:justify">
<strong>Motivation:</strong> Bridging the exponential gap between the number of unlabeled and labeled protein sequences, a couple of works have adopted semi-supervised learning for protein sequence modeling. They pre-train a model with a substantial amount of unlabeled data and transfer the learned representations to various downstream tasks. Nonetheless, the current pre-training methods mostly rely on a language modeling pre-training task and often show limited performances. Therefore, a pertinent protein-specific pre-training task is necessary to better capture the information contained within the protein sequences.
<br/>
<strong>Results:</strong> In this paper, we introduce a novel pre-training scheme called <strong>PLUS</strong>, which stands for <strong>P</strong>rotein sequence representations <strong>L</strong>earned <strong>U</strong>sing <strong>S</strong>tructural information. PLUS consists of masked language modeling and a protein-specific pre-training task, namely same family prediction. PLUS can be used to pre-train various model architectures. In this work, we mainly use PLUS to pre-train a recurrent neural network (RNN) and refer to the resulting model as PLUS-RNN. It advances the state-of-the-art pre-training methods on six out of seven tasks, <em>i.e.</em>, (1) three protein(-pair)-level classification, (2) two protein-level regression, and (3) two amino-acid-level classification tasks. Furthermore, we present results from our ablation studies and qualitative interpretation analyses to better understand the strengths of PLUS-RNN.
<br/><br/>
</p>

# Requirements
- Python 3.7
- PyTorch 1.3.1
- Numpy 1.17.4
- SciPy 1.4.1
- Pandas 0.25.3
- Pillow 7.0.0
- Scikit-learn 0.22.1
<br/>

# Data & Models
Available at our webpage <a href="http://ailab.snu.ac.kr/plus/">(http://ailab.snu.ac.kr/plus/)</a>
<br/><br/><br/>
