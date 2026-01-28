# Twitter Emo Classification

Deep learning system for classifying emotions in social media text using BiLSTM with attention mechanisms. This project achievies **90.87% F1-macro** across 6 emotion categories.


## Datasets
Datasets used:
1. https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification/input
2. https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2018_task_1
3. https://github.com/RoozbehBandpey/ELTEA17/tree/main

## Project Structure


## Reporducibility
Experiments are reproducible with fixed seeds:

```python
import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Limitations
1. **Sarcasm detection**: Model struggles with ironic statements
2. **Multi-label**: Assumes single emotion per tweet
3. **Language**: English only, may not generalize to other languages
4. **Temporal drift**: Slang evolves; requires periodic recollecting data and retraining 

## License

This project is licensed under the AGPL3.0 License - see [LICENSE](LICENSE) file.

---