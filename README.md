# PLUS
<p align="center"><img src="http://ailab.snu.ac.kr/plus/images/PLUS_Logo_ver2.jpg"></p>
<h4><p align="center"><strong><a href="//arxiv.org/abs/1912.05625">(https://arxiv.org/abs/1912.05625)</a></strong></p></h4>
<br/>

# Abstract
<p style="text-align:justify">
<strong>Motivation:</strong> Several studies adopted semi-supervised learning for protein sequence modeling to bridge the exponentially growing gap between the numbers of unlabeled and labeled proteins. In these studies, models were pre-trained with a substantial amount of unlabeled data, and the learned representations were transferred to various downstream tasks. Current pre-training methods mostly rely on language modeling and often exhibit limited performance. Therefore, a complementary protein-specific task for pre-training is necessary to more accurately capture the structural information contained within unlabeled proteins. 
<br/>
<strong>Results:</strong> In this paper, we introduce a novel pre-training scheme called <strong>PLUS</strong>, which stands for <strong>P</strong>rotein sequence representations <strong>L</strong>earned <strong>U</strong>sing <strong>S</strong>tructural information. PLUS consists of masked language modeling and a complementary protein-specific pre-training task, namely same-family prediction. PLUS can be used to pre-train various model architectures. In this work, we use PLUS to pre-train a bidirectional recurrent neural network and refer to the resulting model as PLUS-RNN. PLUS-RNN advances the state-of-the-art pre-training methods in six out of seven widely used protein biology tasks including (1) three protein(-pair)-level classification, (2) two protein-level regression, and (3) two amino-acid-level classification. Furthermore, we present the results from our qualitative interpretation analyses to illustrate the strengths of PLUS-RNN.
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
Available at our webpage <a href="http://ailab.snu.ac.kr/PLUS/">(http://ailab.snu.ac.kr/PLUS/)</a>
<br/><br/><br/>
