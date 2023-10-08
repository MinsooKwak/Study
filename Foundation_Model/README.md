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
 
     ---
    - **1. emergence (출현, 창발)**
      > 창발은 시스템에서의 양적 변화가 질적 변화를 가져오는 때를 의미한다. <br> - Philip Anerson (1972)
      ```
      [Emergence(창발)의 정의]
      
        - 복잡하게 생각하는 능력
        - 학습된 도메인 밖에서도 적용이 이뤄지는 것
        - 구성요소(하위계층)에는 없는 특성이나 행동이 전체 구조(상위계층)에서 자발적으로 돌연히 출현

      [AI 관점에서 창발]

        - 직접 설계하고 제어하는 측면이 아닌데 나타난 또는 나타나게 유도한 특성
        - 확률 통계학적 모델
        - 단점 : 모델의 출력 이유 설명이 어려움 
      ```

      - 파라미터 증대 > In-context learning(2020) > Emergent Abilities(2022)
        > In-context learning : zero-shot, one-shot, few-shot 개념 정의 <br> (논문 : Language Models are Few-shot learners)]
        
      - Emergent Abilities of Large Language Models <br> (Google Research, Stanfold Univ, UNC Chapel Hill, DeepMind, 2022) [논문 링크](https://arxiv.org/pdf/2206.07682.pdf)
        - 소규모 모델에는 없으나 대규모 모델에는 존재하는 능력
          - 모델의 크기
          
            **[ Model scale 따른 Few-shot Prompting에서의 Emergent ]**
            ![image](https://github.com/MinsooKwak/Study/assets/89770691/b4fa6b48-f1bf-47ad-a641-df313e239817)
              - BenchMark
                - BIG-Bench : A, B, C, D (언어학, 아동발달, 수학, 상식추론, 물리학, 사회적 편견, 소프트웨어 개발 등 204개 과제)
                - TruthfulQA benchmark : E
                - Grounded conceptual mappings : F [논문 링크](https://openreview.net/pdf?id=gJcEM8sxHK)
                - Massive Multi-task Language Understanding (MMLU) : G (인문과학, 사회과학 등 57개 과목)
                - WiC : H
   
            **[ CoT Prompting에서의 Emergent ]**
          ![image](https://github.com/MinsooKwak/Study/assets/89770691/be886b4b-8fa8-4f01-9d01-1b7ec1cddece)
            - 참조
              - A : PaLM, Chain of thought
              - B : FLAN, instruction tuning
              - C : scratch pad 기법, 8숫자 연산 task [참조 논문](https://browse.arxiv.org/pdf/2112.00114.pdf)
              - D : model calibration, Emergence [참조 논문](https://browse.arxiv.org/pdf/2207.05221.pdf)
 
          <br>
          
          > 모델의 크기가 일정 임계값을 넘어야 해당 고급 기법을 사용했을 때 Emergent abilities가 나타남
          > 여전히 GPT3, PaLM 등이 힘 못쓰는 task들이 남아있음
          > - BigBench Dataset <br> anachronisms, formal fallacies syllogisms negation, mathematical induction 등


        **1) zero shot, one-shot, few-shot** :
          - 학습 단계 아닌 추론 단계에서 이뤄짐 <br> (파라미터 업데이트 x) => Prompt Learning 등장 배경
          - zero shot : 적절한 instruction 또는 Prompt를 주었을 때 나은 답변을 얻어낼 수 있다
          - one shot, few shot : 지시문 뿐 아니라 구체적 예제 던져 원하는 답변 이끌어냄
          => instruction tuning, prompt Engineering 등 방법론으로 발전
          <br>

        **2) Instruction Tuning (명령/지침 조정 방법)** :
        - Instruction Following은 emergent abilities 중 하나
        - FLAN이 GPT3 보다 작은 모델을 활용(LaMDA)했지만 역시 큰 사이즈 LLM 
          ```
          [기존]
          - 기존 일반적 fine-tuning은 특정 downstream task에 국한
          - prompting 역시 의도적으로 prompting 해야 했음
          
          [instruction tuning]
          - 서로 다른 종류의 downstream task를 instruction 형태의 데이터셋으로 한번에 fine-tuning
          - User가 별도로 prompt engineering 하지 않고 모델이 user의 instruction 따른 답변 수행 
          ```
          
          - Instruction Tuning 방법이란
            - Few-shot learning + Fine-tuning 
            > LLM 모델을 Instruction 데이터셋 (pair dataset)으로 Fine-tuning하고 zero-shot 성능을 높이는 방법 <br>
              - LLM model instruction fine-tuning <br>
                - 다양한 종류의 NLP Task <br>
                - Pair Dataset : instruction, 상응하는 label <br>
              - zero-shot : 한 번도 보지 못한 task에서 inference <br>
              
        
          - FLAN(google, 2022) 논문에서 처음 등장
            > 목적 : zero-shot learning abilities의 개선
            - LaMDA 사용 : **GPT-3 대비 작은 137B 사이즈** 디코더 기반 Transformer 모델
            - LaMDA 기반 Instruction Tuning 사용해 GPT-3보다 나은 성능의 zero-shot abilities
            - 몇 몇 dataset에서는 GPT3의 few-shot 성능까지 뛰어넘음
              
              **[ zero-shot 성능 비교 ]**
              ![image](https://github.com/MinsooKwak/Study/assets/89770691/b34a47d7-f85b-4386-926b-e6df1a0993e4)
              ![image](https://github.com/MinsooKwak/Study/assets/89770691/fc6f5311-ea1a-4aec-a803-f7cf9e5504d8)

            - Instruction Fine-Tuning
              ![image](https://github.com/MinsooKwak/Study/assets/89770691/fd6be913-1500-482e-bb7d-890ff00dea98)
              - instruction 지문에서 참고될 목표를 지시하고, 추론 가능한 답변을 target으로 주는 dataset을 fine-tuning
              
              **[ 사용 데이터셋 ]**
              ![image](https://github.com/MinsooKwak/Study/assets/89770691/ba7fab55-f421-4915-a2aa-100f37598a83)
           
              **[ Dataset Instruction 수정 형식 ]**
              ![image](https://github.com/MinsooKwak/Study/assets/89770691/a8e401e9-84af-4fcc-98c3-c81aec857fc2)


            
            - FLAN 실험 발견
              > 1.Instruction data 안의 task-cluster가 많을수록 unseen task에서 성능 향상 <br> 2.Instruction tuning의 효과는 일정 수준 이상 크기의 model에서만 발생

            <br>
              
        **3) Prompt Engineering**
          - 2021년 모델의 크기 관련해 2개의 연구 : Gopher(Deepmind), PaLM(Google)
            - Gopher : 단순히 모델 크기를 키우는 것이 논리, 수학적 추론 문제 풀기 어렵다
            - PaLM : LLM 규모를 극한으로 몰아넣었을 때 few-shot 능력이 얼마나 상승되는지 확인
              - 번역, 요약, QA 등 NLP Task에서도 SOTA
              - 다국어로도 좋은 성능
            <br>
        
          - **chain-of-thought prompting (CoT) : PaLM (google, 2022)**
            - Prompt Enineering의 하위 분야
            - Multi-step reasoning에 좋은 성능 (비교 : Gopher)
              - ex. 산술 추론, 상식 추론
                ```
                - 산술 추론 (Arithmetic reasoning)
                  : 2단계 이상의 추론 거쳐야 풀 수 있는 산술 문제
                
                - 상식 추론 (Commonsense reasoning)
                  : 세계에 대한 일반 지식으로 추론하는 문제
                ```

            - **LLM few-shot vs. CoT prompting**
              - **Few shot**
                - prompt -> 답
              - **Chain of thought(CoT)**
                - 프롬프트 줄 때 문제 푸는데 필요한 **사고과정을 함께** 줌
                - 풀이 과정을 포함
                - **오류 분석이 가능**해짐
                - **추론에 대한 해석 가능성**을 높일 수 있음
            
            <br>
 
        4) Emergent Abilities에 대한 기타 문제들 (추가 스터디 필요)
          - **pre-training objectives의 영향력**
            - [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?]( https://arxiv.org/pdf/2204.05832.pdf)
          - **pre-train model이 downstream task 잘 수행하는 이유**
            - [Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning]( https://arxiv.org/pdf/2106.09226.pdf)
          - Emergence abilities의 측정
            - [Emergent Abilities of Large Language Models 논문 부록 A](https://browse.arxiv.org/pdf/2206.07682.pdf)
          - Emergence abilities의 위험성
            - truthfulness, bias, toxicity, hallucination 등의 문제
            - 추론 능력이 역설적으로 Emergent abilities가 확장됨에 따라 강화
              - [On the Opportunities and Risks of Foundation Models, 5.society 참조](https://browse.arxiv.org/pdf/2108.07258.pdf)
            - HHH기준 (Helpful, Honest, Harmless)
              - [A General Language Assistant as a Laboratory for Alignment, 부록E 참조]( https://arxiv.org/pdf/2112.00861.pdf)
              - alignment problem : HHH 충족하지 않은 문제
    
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

## 6. 모델 파라미터 / Size
- 인간 뇌 : 860억개 뉴런, 100조개 시냅스 
- 주요 모델 파라미터 스케일의 변화
  - PaLM 대비 인간 뇌에 근접하나 인간 뇌는 언어에 국한되지 않음
    
    **[ 10B 이상의 파라미터 모델 ]**
    ![image](https://github.com/MinsooKwak/Study/assets/89770691/76a34d02-2e18-43d5-8f73-acc9cba5b73c)
    - 2020, GPT3 [Language Models are Few-Shot Learners 논문](https://browse.arxiv.org/pdf/2005.14165.pdf)
      - Model sclaing-up : 기존 가장 큰 모델 대비 10배 많은 파라미터의 모델 사용
      - Few shot에서도 Task-specific한 기존 fine-tuning 모델에 필적하는 성능 보임

- Gopher(2021, Deepmind) [Gopher 논문 링크](https://browse.arxiv.org/pdf/2112.11446.pdf)
  - 단순히 모델 size 키우는 것이 논리적, 수학적 추론 필요한 문제를 풀어내지 못한다.

## 7. AGI
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
