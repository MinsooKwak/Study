## 1. Traditional ML / Classic DL / Foundation Model 비교

- 통계적 모델, 신경망 모델을 거쳐 Transformer 등장 이후
- 사전훈련된 모델을 이용해 downstream task 수행하는 패러다임으로 전환
  

|                     | Traditional ML | Classic DL Model | Foundation Model |
|---------------------|-----------------------------------|--------------------------------------------|--------------------------|
| **정의**            | - 데이터로부터 패턴을 학습하는 머신 러닝 기술의 초기 형태 <br> - Feature engineering </br> - 알고리즘 활용 | - DL 활용해 데이터를 학습하고 패턴을 인식 | - 대규모 텍스트 데이터를 활용하여 사전 훈련된 모델 |
| **특징**            | - Feature engineering 필요 <br> - 데이터의 성격에 따라 다양한 알고리즘을 사용 | - DL 아키텍처를 사용해 Feature engineering 감소 | - 대량의 텍스트 데이터를 기반으로 사전 훈련 <br> - 다양한 자연어 처리 작업에 재사용 가능 |
| **예시 모델**      | - 결정 트리, 랜덤 포레스트, 서포트 벡터 머신 등 | - 다층 퍼셉트론 (MLP), 컨볼루션 신경망 (CNN), 순환 신경망 (RNN) 등 | - BERT, GPT-3, T5 등 대표적인 기반 모델들 |
| **적용 분야**      | - 이미지 분류, 텍스트 분류, 회귀 분석 등 다양한 머신 러닝 작업 | - 컴퓨터 비전, 자연어 처리, 음성 처리 등 다양한 딥 러닝 작업 | - 자연어 처리, 기계 번역, 질의응답, 텍스트 생성 등 다양한 자연어 처리 작업 |

- 2022년 **On the Opportunities and Risks of Foundation Models**라는 논문
  - **Pre-trained model**을 Foundation Model로 정의
  - 2가지 특징
    > Emergence : 시스템의 행동은 데이터를 통해 **유추**된다 <br><br> Homongenization : 하나의 거대 모델이 다양한 문제를 풀기 위한 기반이 된다
     
    - **1. emergence (출현, 창발)**
      - 복잡하게 생각하는 능력
      - 학습된 도메인 밖에서도 적용이 이뤄지는 것
      - 구성요소(하위계층)에는 없는 특성이나 행동이 전체 구조(상위계층)에서 자발적으로 돌연히 출현하게 됨
      - AI 시스템을 구축하려는 입장에서 연구자가 직접 설계하고 제어하는 측면이 아닌데 나타난 또는 나타나게 유도한 특성
      - 확률 통계학적 모델
      - 단점 : 모델의 출력 이유 설명이 어려움 
      - 파라미터 증대 > In-context learning(2020) > Emergent Abilities(2022)
        > In-context learning : zero-shot, one-shot, few-shot 개념 정의 <br> (논문 : Language Models are Few-shot learners)
        
        > Emergent Abilities (Google Research, Stanfold Univ, UNC Chapel Hill, DeepMind)
          - [Emergent Abilities of Large Language Models 논문](https://arxiv.org/pdf/2206.07682.pdf)
          - 소규모 모델에는 없으나 대규모 모델에는 존재하는 능력
          - 모델의 크기가 특정 임계값 넘어서는 순간 모델 performance 향상
            ![image](https://github.com/MinsooKwak/Study/assets/89770691/b4fa6b48-f1bf-47ad-a641-df313e239817)

        - zero shot, one-shot, few-shot : 추론 단계에서 이뤄짐 <br> (파라미터 업데이트 x) => Prompt Learning 등장 배경
        - 적절한 instruction 또는 Prompt를 주었을 때 나은 답변을 얻어낼 수 있다
          - Instruction Tuning : FLAN(google, 2022)
            - Few-shot learning + Fine-tuning
          - Prompt Engineering > chain-of-thought prompting(CoT) : PaLM(google, 2022)
            - Multi-step reasoning에 좋은 성능
              - 산술 추론 (Arithmetic reasoning) : 2단계 이상의 추론 거쳐야 풀 수 있는 산술 문제
              - 상식 추론 (Commonsense reasoning) : 일반 지식으로 추론하는 문제
            - CoT prompting : 문제에 대한 답을 바로 주는 것이 아닌 문제 푸는데 필요한 사고과정을 함께 주는 것 (풀이과정 포함)
              - 오류 분석이 가능해짐
              - 추론에 대한 해석 가능성을 높일 수 있음
    
    - **2. homogenization(균질화)**
      - SOTA 모델은 BERT, RoBERTa, BART, T5 등 몇가지 기본 모델 중 채택되게 되는 것
      - (장점) 백본 모델의 개선이 NLP 전반의 개선에 도움이 됨 : 높은 leverage
      - (단점) 기본 모델의 문제점을 동일하게 가져옴 (ex. 데이터 편향)

<br>


## 2. Foundation Model의 태동
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


<br>


## 3. Foundation Model 종류 (확인 및 수정 필요)
| 자연어 처리 (NLP)                             | 컴퓨터 비전 (CV)                              | 강화 학습 (RL)                            | 생성 모델링 <br> (GAN)                             | 텍스트 마이닝 <br> (Text Mining)                          | 동영상 분석 <br> (Video Analysis)                        |
| --------------------------------------------- | --------------------------------------------- | ----------------------------------------- | --------------------------------------------- | -------------------------------------------------- | --------------------------------------------- |
| - GPT-3 <br> - BERT <br> - RoBERTa <br> - T5           | - CNN <br> - ResNet <br> - VGGNet <br> - Inception     | - DDPG <br> - PPO <br> - A3C                   | - DCGAN <br> - CycleGAN <br> - StyleGAN             | - LDA <br> - FastText <br> - Word2Vec              | - 3D CNN <br> - I3D <br> - C3D                    |
| **추천 시스템** <br> (Recommendation System)                   | **음성 처리** <br> (Speech Processing)                          | **시계열 데이터 예측** <br> (Time Series Prediction)              | **그래프 데이터 분석** <br> (Graph Data Analysis)                 | **강화 학습** <br> (Reinforcement Learning)                     |
| - MF (Matrix Factorization) <br> - Neural Collaborative Filtering | - ASR <br> (Automatic Speech Recognition) <br> - DeepSpeech | - LSTM <br> - GRU <br> - ARIMA <br> - Prophet                   | - GCN (Graph Convolutional Network) <br> - GAT (Graph Attention Network) | - DQN (Deep Q-Network) <br> - SAC (Soft Actor-Critic) <br> - TRPO (Trust Region Policy Optimization) |



## 4. Foundation Model이 강력해지는 이유
- Tool이 강력해짐 (출처 : https://www.madrona.com/foundation-models/)
- LLM에서 LangChai이 App store와 검색 엔진의 역할을 함
- Chain 통해 합성이 가능함
- 실시간으로 application이 모델에 반영이 됨
  - 명령 프롬프트 > Data를 DB에서 가져옴
  - 응답 확인 > 다음 프롬프트의 context
  - 그것들을 chain과 결합해 LLM을 구성
- 모들의 ensemble이 나타나기 시작 (max closed and open models)

- 툴 관련 자료
   ![image](https://github.com/MinsooKwak/Study/assets/89770691/621a3e43-1017-4895-b173-7ef4b4422284)


## 5. GitHub Star Chart 통한 추세

![Pasted image 20231002180414](https://github.com/MinsooKwak/Study/assets/89770691/2de4b630-e552-4dae-83a0-44141e8a337b)
- (좌) DBT
- (중) LangChain
- (우) AutoGPT


## 6. AGI
- (추후 관련해 내용 추가 예정)
  
- Sam Altman(OpenAI CEO, ChatGPT 아버지)와 AI 연구자 Lex Fridman의 대담에서 내용
  - ChatGPT
    - 사람들이 GPT-4를 아주 초기의 인공지능으로 생각하게 될 것
    - ChatGPT가 "사용성"을 고려했기 때문에 변곡점이 될 것 [관련 기사](https://www.tech42.co.kr/%EC%B1%97gpt%EA%B0%80-%EA%BF%88%EA%BE%B8%EB%8A%94-agi%EB%A1%9C%EC%9D%98-%EC%A7%84%ED%99%94-%EC%98%A4%ED%94%88ai-ceo-%EC%83%98-%EC%95%8C%ED%8A%B8%EB%A7%8C/)
    - chatGPT화한 GPT-3모델이 AGI의 문제에 고안된 방식인 '점진적 진화'의 첫 사례
    - pretrained model
    - 대화형 interface로 다양한 NLP 처리에 용이
    - 휴먼 피드백 강화학습(Reinforcement Learning with Human Feedback)
      - Human Feedback 통해 모델의 정확도와 관련성, 공정성 높임
      - 모델에 대한 2가지 결과물에 대해 어떤 결과물이 나은지 Human Feedback 하는 방식
      - 강화학습 통해 이전보다 적은 데이터로 모델을 훨씬 강력하게 만들 수 있다는 주장
      - 반대 입장도 있음 (ex. 앤드류 응, 23.07.20, 초거대 AI 모델 플랫폼 최적화 센터 개소식 연사 중)
          > AGI가 등장하려면 수십년이 걸리며, 슈퍼인텔리전스가 등장한다는 건 비현실적
          
          > 강화학습에 대해서는 근본적으로 데이터가 많이 필요하기 때문에 크게 성장하기 힘들 것
