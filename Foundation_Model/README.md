## Traditional ML / Classic DL / Foundation Model 비교

|                     | Traditional ML | Classic DL Model | Foundation Model |
|---------------------|-----------------------------------|--------------------------------------------|--------------------------|
| **정의**            | 전통적인 머신 러닝은 데이터로부터 패턴을 학습하는 머신 러닝 기술의 초기 형태입니다. 주로 특징 엔지니어링과 일반적인 알고리즘을 사용합니다. | 고전적인 딥 러닝 모델은 다층 신경망과 같은 깊은 구조를 사용하여 데이터를 학습하고 패턴을 인식하는 머신 러닝 방법입니다. | 기반 모델은 대규모 텍스트 데이터를 활용하여 사전 훈련된 모델로, 다양한 자연어 처리 작업에 적용할 수 있는 머신 러닝 모델입니다. |
| **특징**            | - 주로 특징 엔지니어링이 필요하며 데이터의 성격에 따라 다양한 알고리즘을 사용합니다. | - 다층 신경망과 같은 신경망 아키텍처를 사용하여 특징 엔지니어링이 감소하였습니다. | - 대량의 텍스트 데이터를 기반으로 사전 훈련되어 다양한 자연어 처리 작업에 재사용 가능합니다. |
| **예시 모델**      | - 결정 트리, 랜덤 포레스트, 서포트 벡터 머신 등 | - 다층 퍼셉트론 (MLP), 컨볼루션 신경망 (CNN), 순환 신경망 (RNN) 등 | - BERT, GPT-3, T5 등 대표적인 기반 모델들 |
| **적용 분야**      | - 이미지 분류, 텍스트 분류, 회귀 분석 등 다양한 머신 러닝 작업 | - 컴퓨터 비전, 자연어 처리, 음성 처리 등 다양한 딥 러닝 작업 | - 자연어 처리, 기계 번역, 질의응답, 텍스트 생성 등 다양한 자연어 처리 작업 |





## Foundation Model의 태동
- 전통적 모델의 경우 Task에 맞게 데이터를 모으고, Task에 맞게 모델을 훈련하는 과정
  - 필요사항
    - 많은 양의 데이터, 태깅
    - 태깅에 대한 수정
    - ML 기법 실험 및 튜닝
  - 데이터 수집의 한계와 태깅에 대한 수정, Task에 따라 모델을 구성함에 있어 많은 비용이 소모
  - Foundation Model의 등장 배경이 됨

- Foundation Model
  - 기본적 언어 지식 가르치고 그 위에 필요한 전문 분야만 쌓아나가는 형태
  - Model 하나로 조금의 Tuning을 거쳐 여러 Task 수행
  - 의미 있기 위한 조건
    - 기존 방식 대비 데이터를 조금 쓰거나
    - 같은 양의 데이터 대비 성능이 좋아야 함
  - 현재 NLP 영역에서 LLM이 두드러짐





## Foundation Model 종류 (확인 및 수정 필요)
| 자연어 처리 (NLP)                             | 컴퓨터 비전 (CV)                              | 강화 학습 (RL)                            | 생성 모델링 <br> (GAN)                             | 텍스트 마이닝 <br> (Text Mining)                          | 동영상 분석 <br> (Video Analysis)                        |
| --------------------------------------------- | --------------------------------------------- | ----------------------------------------- | --------------------------------------------- | -------------------------------------------------- | --------------------------------------------- |
| - GPT-3 <br> - BERT <br> - RoBERTa <br> - T5           | - CNN <br> - ResNet <br> - VGGNet <br> - Inception     | - DDPG <br> - PPO <br> - A3C                   | - DCGAN <br> - CycleGAN <br> - StyleGAN             | - LDA <br> - FastText <br> - Word2Vec              | - 3D CNN <br> - I3D <br> - C3D                    |
| **추천 시스템** <br> (Recommendation System)                   | **음성 처리** <br> (Speech Processing)                          | **시계열 데이터 예측** <br> (Time Series Prediction)              | **그래프 데이터 분석** <br> (Graph Data Analysis)                 | **강화 학습** <br> (Reinforcement Learning)                     |
| - MF (Matrix Factorization) <br> - Neural Collaborative Filtering | - ASR <br> (Automatic Speech Recognition) <br> - DeepSpeech | - LSTM <br> - GRU <br> - ARIMA <br> - Prophet                   | - GCN (Graph Convolutional Network) <br> - GAT (Graph Attention Network) | - DQN (Deep Q-Network) <br> - SAC (Soft Actor-Critic) <br> - TRPO (Trust Region Policy Optimization) |


