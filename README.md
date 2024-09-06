# Predicting the Target Word of Game-playing Conversations using a Low-Rank Dialect Adapter for Decoder Models
**Authors:** Dipankar Srirag and Aditya Joshi and Jacob Eisenstein

**DOI:** [10.48550/arXiv.2409.00358](https://doi.org/10.48550/arXiv.2409.00358)

## Abstract
Dialect adapters that improve the performance of LLMs for NLU tasks on certain sociolects/dialects/national varieties ('dialects' for the sake of brevity) have been reported for encoder models. In this paper, we extend the idea of dialect adapters to decoder models in our architecture called `LoRDD`. Using [`MD-3`]((https://doi.org/10.48550/arXiv.2305.11355)), a publicly available dataset of word game-playing conversations between dialectal speakers, our task is Target Word Prediction (TWP) from a masked conversation. `LoRDD` combines task adapters and dialect adapters where the latter employ contrastive learning on pseudo-parallel conversations from MD-3. Our results for `en-IN` conversations on two models (`Mistral` and  `Gemma`) show that `LoRDD` outperforms four baselines on TWP, while bridging the performance gap with `en-US` by 12% on word similarity and 25% on accuracy. The focused contribution of `LoRDD` is in its promise for dialect adaptation of decoder models.

## Keywords
- Large Language Models
- Dialect Robustness
- Conversation Understanding
- Word-Guessing Game
- Adapters

## BibTeX Citation
<tab><tab>
```bibtex
@misc{srirag2024predictingtargetwordgameplaying,
      title={Predicting the Target Word of Game-playing Conversations using a Low-Rank Dialect Adapter for Decoder Models}, 
      author={Dipankar Srirag and Aditya Joshi and Jacob Eisenstein},
      year={2024},
      eprint={2409.00358},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.00358}, 
}
```

## Contact
- [Dipankar Srirag](mailto:d.srirag@unsw.edu.au); [University of New South Wales](https://dipankarsrirag.github.io)
- [Aditya Joshi](mailto:aditya.joshi@unsw.edu.au); [University of New South Wales](https://www.unsw.edu.au/staff/aditya-joshi)
- [Jacob Eisenstein](mailto:jeisenstein@google.com); [Google DeepMind](https://jacobeisenstein.github.io)