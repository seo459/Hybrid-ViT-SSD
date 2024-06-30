# ViT Reimplementation
- [ ] model.py
- [ ] train.py
- [ ] loss.py
- [ ] dataset.py

출처 -- https://daebaq27.tistory.com/112



- augmentation.py: 간단한 Dataset Transform을 구현
- dataset.py: ImageNet-1k dataset 클래스 구현
- model.py: ViT 모델 구현
- scheduler.py: linear warm-up + cosine annealing 등의 스케쥴러 구현.
- train.py: single gpu 상황을 가정한 train.py
- train_multi.py: multi-gpu 하에서의 train.py
- utils.py: metrics 계산, checkpoint load 등의 여러 함수 구현