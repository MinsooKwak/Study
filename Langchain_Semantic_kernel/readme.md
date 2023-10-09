## 서비스

### 서비스 유형
- 선택형 서비스 : 현재 대부분 웹, 앱 서비스 (Graphical User Interface : click, typing, drag...)
- 대화형  서비스 : 챗봇 기반 대화를 통해 사용
- 임베디드형 서비스 : 특정 조건이 만족되면 등장

### 선택형 서비스
- 유저의 입력, 클릭 기반 소통
- 사용자의 자유도 제한
- 사용자의 행동 패턴을 예측, 분석하기 쉬움

### 대화형 서비스 + (Plugin)
- 유저와 서비스가 "대화"를 통해 상호작용
- 정해진 순서를 따를 필요 없어짐.
- 자유도 증가 (많은 제한 사라짐)
- 사용자의 행동 패턴을 예측하기 어려움
- Copilot : 단순 챗봇 아니라 기능 녹이고, DB 사용하고, 경험을 채팅으로 줄 수 있을 때 (Microsoft)
- https://github.com/microsoft/TaskMatrix


### 임베디드형 서비스
- 선택형, 대화형 서비스는 유저와 직접적 소통을 했음
- 임베디드형은 유저와 직접적 소통보다 트리거 형태
  - 유저의 행동을 지켜보다가 등장해서 자동 완성을 제공하거나, 추천을 제공하는 등의 서비스 제공
- 주로 에디터에서 활용됨 (vscode, powerpoint design ideas, notion ai 등)
  - extension, plugin 설치에 따라 작동
- ex) github copilot

---

### ChatGPT API 기초
- https://openai.com/ > login > API > API keys > create new secret key
  - 한 번만 알려주므로 복사해서 저장
- Billing 먼저 등록해야 함 : `pip install openai`

> API 사용하기
<details>
  <summary> 새 파일에 API 숨기기 </summary>
  <div markdown="1">
    
      새 파일 -> .env> OPENAI_API_KEY = '~~~'
      
  </div>
</details>


<details>
  <summary> openai api </summary>
  <div markdown="1">
    
      import os
      import openai
      from dotenv import load_dotenv

      load_dotenv()
      openai.api_key = os.getenv("OPENAI_API_KEY")
      
  </div>
</details>


<details>
  <summary> API 사용하기 </summary>
  <div markdown="1">
    
      response = openai.ChatCompletion.create(
        model = 'gpt=3.5-turbo',  #gpt-4 가능
        messages = [
          {"role" : "system", "content" : "You are a helpful assistant."},
          {"role" : "user", "content" : "What can you do?"}
          ],
        # 파라미터들 (chatGPT의 랜덤성 조절) : temperature, top_p (동시 비추, temperature 위주로)
        # 창작 제외하고 동일하게 맞추기 위해 temperature 0으로 설정. 창작시 0.8 권장
        temperature = 0.8,
          )
      response = response.choices[0].message.content
      print(response)
      
  </div>
</details>

</br>

> 파라미터
<details>
  <summary> 요청 관련 파라미터 : n, stream </summary>
  <div markdown="1">

    response = openai.ChatCompletion.create(
      model = 'gpt=3.5-turbo',  #gpt-4 가능
      messages = [
        {"role" : "system", "content" : "You are a helpful assistant."},
        {"role" : "user", "content" : "What can you do?"}
        ],
      temperature = 0.8,
      # 요청 관련 파라미터 : stream, n값을 조정해 여러개의 답 생성하게
      n=4,
      )

    for res in response.choices:
       print(res.message)
      
  </div>
</details>

<details>
  <summary> 생성물 길이 관여 파라미터 : stop, max_tokens </summary>
  <div markdown="1">

    response = openai.ChatCompletion.create(
      model = 'gpt=3.5-turbo',  #gpt-4 가능
      messages = [
        {"role" : "system", "content" : "You are a helpful assistant."},
        {"role" : "user", "content" : "What can you do?"}
        ],
      temperature = 0.8,
      n=4,
      # 생성물 길이 관여 파라미터 : stop, max_tokens
      stop = [",", "."],
      max_tokens = 30,
        )

    for res in response.choices:
       print(res.message)
      
  </div>
</details>

<details>
  <summary> 기타 고급 파라미터 (잘 사용은 안함) </summary>
  <div markdown="1">

    - presence_penalty, frequency_penalty : 내용 중복을 얼만큼 허용할 것인가
      - frequency_penalty : 양수시 덜 중복 / 음수시 더 중복
    - logit_bias : 특정 단어가 무조건 등장하게 조정. 토큰 값을 넣어 한국어 제한.
    - user
      
  </div>
</details>

---

## ChatGPT 서비스 세대 분류

### 1세대 : 기본적 ChatGPT API 사용 세대
> User <-> APP <-> LLM
- 단순히 ChatGPT API 중개하는 역할로 빠르게 만들기 쉬움
- 제품과 잘 융화되지는 않아서 빠르게 등장했다가 사라짐
- LLM : ChatGPT, Claude, HyperClova, ...

<details>
  <summary> 1세대 서비스 예시 코드 </summary>
  <div markdown="1">

    import os
    
    import openai
    from dotenv import load_dotenv
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    load_dotenv()
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    app = FastAPI(debug=False)
    app.add_middleware(
      CORSMiddleware,
      allow_origins=['*'],
      allow_credentials=True,
      allow_methos=['*'],
      allow_headers=['*'],
      )
    
    response = openai.ChatCompletion.create(
      model = 'gpt=3.5-turbo',  #gpt-4 가능
      messages = [
        {"role" : "system", "content" : "You are a helpful assistant."},
        {"role" : "user", "content" : "What can you do?"}
        ],
      temperature = 0.8,
      # 요청 관련 파라미터 : stream, n값을 조정해 여러개의 답 생성하게
      n=4,
        )
    
    for res in response.choices:
       print(res.message)
    
    class ChatRequest(BaseModel):
      message : str
      temperature : float = 1
    
    # user와 소통할 때 항상 갖고 있는 정보 (persona, 대화 나눌 때 기본 정보) 
    SYSTEM_MEG = "You are a helpful travel assistant, Your name is Jini, 27 years old"
    
    @app.post("/chat")
    def chat(req : ChatRequest):
      response = openai.ChatCompletion.create(
        model = 'gpt=3.5-turbo',
        messages = [
          {"role" : "system", "content" : SYSTEM_MEG}
          {"role" : "user", "content" : "What can you do?"}
          ],
          temperature = req.temperature,
          )
      return {"message": response.choices[0].message.content}
    
    if __name__ =="__main__":
      # 백앤드 실행되게
      import uvicorn
    
      uvicorn.run(app, host='0.0.0.0', port=8000)
      
  </div>
</details>


### 2세대 : ChatGPT API 오케스트레이션 (Chain 방식)

> 유저 <-> APP <-> LLM <-> (LLM, LLM, LLM, LLM) 연쇄 작용으로 결과 생성
![image](https://github.com/MinsooKwak/Study/assets/89770691/d76e88e3-1c31-4594-9fcc-1a6e16955aed)
- 기존 챗봇 빌더 방식과 유사
- Intent 기반 챗봇 빌더
  - 사용자의 발화 의도를 파악하고 미리 정의한 **시나리오**대로 실행
  - 기획 단계에서 고민 기간이 필요
- 시나리오 기반으로 여러개의 Prompt가 작성되어야 함
  - Prompt 하나에 모든 정보 넣어도 모든 작업 가능x
- Prompt Chaing 사용
  - 유저가 여행 계획 요청시 계획을 생성하고 표 형태로 정리 (두개로 쪼개는 것이 더 잘 동작)
- Prompt 관리가 어려워짐. 버전 관리도 불편
- 프레임워크 있으면 훨씬 편리해짐

- 구성
  - Prompt + 예시
  - Chain : 연결
  - User와 대화 나는 메모리 기록 / 내부 동작 logging 위해 callback 필요
  - 모델 : LLM / Chat models 구현되어 있음

<details>
  <summary> 2세대 서비스 예시 </summary>
  <div markdown="1">
    
    import os
    
    import openai
    from dotenv import load_dotenv
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    load_dotenv()
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    app = FastAPI(debug=False)
    app.add_middleware(
      CORSMiddleware,
      allow_origins=['*'],
      allow_credentials=True,
      allow_methos=['*'],
      allow_headers=['*'],
      )
    
    response = openai.ChatCompletion.create(
      model = 'gpt=3.5-turbo',  #gpt-4 가능
      messages = [
        {"role" : "system", "content" : "You are a helpful assistant."},
        {"role" : "user", "content" : "What can you do?"}
        ],
      temperature = 0.8,
      # 요청 관련 파라미터 : stream, n값을 조정해 여러개의 답 생성하게
      n=4,
        )
    
    for res in response.choices:
       print(res.message)
    
    class ChatRequest(BaseModel):
      message : str
      temperature : float = 1
    
    # user와 소통할 때 항상 갖고 있는 정보 (persona, 대화 나눌 때 기본 정보) 
    SYSTEM_MEG = "You are a helpful travel assistant, Your name is Jini, 27 years old"
    
    def classify_intent(msg):
      prompt = """ Your job is to classify intent.
    
      Choose one of the following intents.
      - travel_plan
      - customer_support
      - reservation
    
      User : {msg}
      Intent : 
      """
      response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = [
          {"role" : "user", "content" SYSTEM_MSG},
        ],
      )
      return response.choices[0].message.content.strip()
    
    @app.post("/chat")
    def chat(req : ChatRequest):
    
      # 의도 파악이 중요
      intent = classify_intent(req.message)
    
      if intent == "travel_plan":
        response = openai.ChatCompletion.create(
          model = 'gpt=3.5-turbo',
          messages = [
            {"role" : "system", "content" : SYSTEM_MEG}
            {"role" : "user", "content" : "What can you do?"}
            ],
            temperature = req.temperature,
            )
        return {"message": response.choices[0].message.content}
    
      elif inent == "customer_support":
        return{"message" : "Here is customer support number : 1234567"}
    
      elif intent == "reservation":
        return{"message" : "Here is reservation number : 12345t6"}
    
    
    if __name__ =="__main__":
      # 백앤드 실행되게
      import uvicorn
    
      uvicorn.run(app, host='0.0.0.0', port=8000)
      
  </div>
</details>



### 3세대 : 외부 데이터 연결 / 오케스트레이션(Pipe-line)
![image](https://github.com/MinsooKwak/Study/assets/89770691/223743a8-0abe-42fd-b75b-e8bb23249530)

- 발화 의도 > Pipe-line(시나리오) > 답변 생성 (Copilot ; 외부 데이터 활용)
- 오케스트레이션
  - user가 어떤 말 했을 때 intent 기반으로 이해하고 정해진 시나리오대로 실행
  - 외부 데이터 들어왔을 때 : Copilot
> User <->APP <-> CoPilot<->LLM, LLM, LLM

> User <->APP <-> CoPilot<->외부 데이터
- **프롬프트에 데이터를 포함해 풍부한 답변 생성**할 수 있게 됨
  - 다양한 API, Plugin, Database 연동해 사용
  - **외부 데이터**
    - User Data (**개인 데이터**/ 선호 정보) 담아 prompt에 같이 작성 
    - APIs : Web Search -> 날씨, 예약 등
    - Vector Database : Documnets
    - Plugins : ChatGPT Plugins

- System Message 수정해서 개인화된 답변을 생성할 수 있음

- 선호 정보들을 받아오게 하고, planning에서 반영할 내용

- 구성
  - Prmopt/ example
  - Chain
  - Memory / callback
  - LLM
  - **Retrievers** : 관련성 있는 데이터 확보 후 Prompt Templete와 연결하기 위한 로직 클래스 여러개
    - 쿼리와 유사한 것 가져오는 것 통제
  - **Document Loaders**(PDF, TXT), **VectorStore**(VectorDB), **Document Transformer**(데이터 정제)

<details>
  <summary> 3세대 서비스 예시 </summary>
  <div markdown="1">

    import os
    
    import openai
    from dotenv import load_dotenv
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    load_dotenv()
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    app = FastAPI(debug=False)
    app.add_middleware(
      CORSMiddleware,
      allow_origins=['*'],
      allow_credentials=True,
      allow_methos=['*'],
      allow_headers=['*'],
      )
    
    response = openai.ChatCompletion.create(
      model = 'gpt=3.5-turbo',  #gpt-4 가능
      messages = [
        {"role" : "system", "content" : "You are a helpful assistant."},
        {"role" : "user", "content" : "What can you do?"}
        ],
      temperature = 0.8,
      # 요청 관련 파라미터 : stream, n값을 조정해 여러개의 답 생성하게
      n=4,
        )
    
    for res in response.choices:
       print(res.message)
    
    class ChatRequest(BaseModel):
      message : str
      temperature : float = 1
    
    def request_user_info():
      # import requests
      # requests.get("https://api.xx.com/users/username/info")
      return """"
      - Like Asia food
      - Like to travel to Spain,
      - 30 years old.
      """
    
    def request_planning_manual():
      # 회사 database에 접근해 가져와야 하는 정보임
      return """
      - 30 years old man likes eating food.
      - 30 years old man likes walking.
      """
    
    # user와 소통할 때 항상 갖고 있는 정보 (persona, 대화 나눌 때 기본 정보) 
    SYSTEM_MEG = f"""You are a helpful travel assistant, Your name is Jini, 27 years old
    
    Current User :
    {request_user_info()}
    
    Planning Manual :
    {request_planning_manual()}
    """
    
    def classify_intent(msg):
      prompt = """ Your job is to classify intent.
    
      Choose one of the following intents.
      - travel_plan
      - customer_support
      - reservation
    
      User : {msg}
      Intent : 
      """
      response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = [
          {"role" : "user", "content" SYSTEM_MSG},
        ],
      )
      return response.choices[0].message.content.strip()
    
    @app.post("/chat")
    def chat(req : ChatRequest):
    
      # 의도 파악이 중요
      intent = classify_intent(req.message)
    
      if intent == "travel_plan":
        response = openai.ChatCompletion.create(
          model = 'gpt=3.5-turbo',
          messages = [
            {"role" : "system", "content" : SYSTEM_MEG}
            {"role" : "user", "content" : "What can you do?"}
            ],
            temperature = req.temperature,
            )
        return {"message": response.choices[0].message.content}
    
      elif inent == "customer_support":
        return{"message" : "Here is customer support number : 1234567"}
    
      elif intent == "reservation":
        return{"message" : "Here is reservation number : 12345t6"}
    
    
    if __name__ =="__main__":
      # 백앤드 실행되게
      import uvicorn
    
      uvicorn.run(app, host='0.0.0.0', port=8000)
      
  </div>
</details>

### 4세대 : Agent 사용
![image](https://github.com/MinsooKwak/Study/assets/89770691/87685f53-03b2-453a-812e-91fc39d4b6bf)

- Agent란 :
  주어진 목표와 환경에서 어떤 행동을 취할지 생각하고 결과값에 대해 계획하는 것 반복하면서 목표 해결해나감
- Action
  - Langchain에서는 `Tool`이라 부름
  - Semantic Kernel에서는 `Skill`이라 부름
- Agent 직접 구현시 Prompt 많이 작성 필요하고 복잡
- Agent 활용시 Langchain 권장
- Agent 종류
  - Conversational : 사용 가능한 스킬셋 기반 1~2턴 이내 빠르게 해결
  - Planning : 목표가 주어졌을 때 스킬셋 기반으로 해결 과정 설계하고 하나씩 수행
  - MultiAgent : 목표가 주어졌을 때 정체성을 가진 여러 에이전트들이 협력
    - https://github.com/101dotxyz/GPTeam
- Agent 특징
  - LLM이 사용자 메세지에 답을 주기 위해 스스로 해결 방식을 고민
  - 시나리오 방식에서 벗어날 수 있음 (최소한의 Intent 분기는 필요함)
    - Agent 권한 관리 중요
  - 현재 실험 단계, GPT-4로 인한 제약 (비용, 사용량 제한, 생성 속도 느림)
  - 디버깅이 까다로움, Wandb같은 Tracing 라이브러리 사용 권장
    - 디버깅시 prompt 바꿔가며 실험 (경험적)해야
    - LangChain 사용시 prompt 작성도 까다로움
  - 사용 가능 case
    - 시나리오 기반 제품 만들기엔 경우의 수가 너무 많은 경우
      - 문제 해결 위한 툴이 준비가 잘 되어있을 때
    - 비용 걱정이 없을 때
      - 추후 고객에 비싼 가격을 받을 수 있거나
      - 한 번 실행시 여러번 사용 가능한 경우
    - 대기 길어도 사용자가 이해할 수 있는 경우 (채팅 상황 아니면 보통 괜찮음)

- 구성
  - Chain -> Agent
  - Tool/Toolkits 가져와서 웹서치, 다양한 도구 연결되는 확장성

<details>
  <summary> 4세대 Agent 활용 예시 </summary>
  <div markdown="1">

    import os

    from dotenv impmort load_dotenv
    from langchain import LLMMathChain, SerpAPIWrapper
    from langchain.agents.tools import Tool
    from langchain.chat_models import ChatOpenAI
    from langchain.experimental.plan_and_execute import (
      PlanAndExecute,
      load_agent_executor,
      load_chat_planner,
    )
    from langchain.llms import OpenAI

    load_dotenv()

    # https://serpapi.com/manage-api-key
    # pip install google-search-results of poetry add google-search-results
    llm = OpneAI(temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    search = SerpAPIWrapper(serpapi_api_key = os.getenv["SERPAPI_API_KEY"])
    tools = [
      Tool(
        name = "Search",
        func = search.run,
        description="useful for when you need to answer questions about current events",
      ),
      Tool(
        name = "Calculater",
        func = llm_math_chain.run,
        description="useful for when you need to answer questions about math",
      ),
    ]

    model = ChatOpenAPI(temperature=0)
    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

    agent.run(
      "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
      )
      
  </div>
</details>

---
## UseCase
### 선택형 + 오케스트레이션
- ChatGPT로 문서 생성, 요약 등

### 대화형 + 오케스트레이션
- 제품을 대화형으로 확장
- 챗봇 없는 기업이 코파일럿 만들어가는 사례

### 선택형 + Agent
- 시간 오래 걸리지만 어려운 작업 대상
- 웹서치 포함한 리포트 생성 case
- Planning Agent, MultiAgent 주로 활용

### 대화형 + Agent
- 복잡하고 어려운 목표보다 간단하지만 경우의 수가 많은 경우
- 멀티모달 환경에서 활약할 수 있음
  - 이미지 올렸을 때 사용자와 어떤 대화 나눌 수 있는지 등 연구되고 있음
- https://github.com/microsoft/TaskMatrix

---
## 프레임워크
- 서비스를 만들기 위한 기능들을 제공해줌
- 개발 시간이 단축됨
  - 프롬프트 관리 및 사용 최적화
  - 다양한 Integration 제공
  - 다양한 유틸리티 (Chunk 등)
  - 참고할만한 코드 쉽게 얻을 수 
- 운영 비용 절약
  - 오픈소스 포함해 자유로운 모델 전환
  - HuggingFace 이용, Fine-Tuning
    ```
      from langchain import HuggingFacePipeline
  
      llm = HuggingFacePipeline.from_model_id(
        model_id = '~~',
        task = 'text-generation',
        model_kwargs={"temperature":0, "max_length":64},
      )
  
      # Semantic Kernel
      import semantic_kernel as sk
      import semantic_kernel.connectors.ai.hugging_face as sk_hf
  
      kernel = sk.Kernel()
      kernel.config.add_text_completion_service(
        "gpt", sk_hf.HuggingFaceTextCompletion("gpt2", task="text-generation")
        )
      kernel.config.add_text_embedding_generation_service(
        "sentence-transformers/all-MiniLM-L6-v2",
        sk_hf.HuggingFaceTextEmbedding("sentence-transformers/all-MiniLM-L6-v2"),
        )
    ```
  - Token 최소화 아이디어들이 구현되어 있음
    - VectorDB 연계해 가장 관련있는 내용만 가져와 사용
    - Few Shot prompt
    - 시간 가중치 등..
- Hallucination 억제 위한 노력
  - LLM 활용 검토 방식
    - LangChain : CriteriaEvalChain
    - SemanticKernel : Grounding Skills
  - Retrieval 최적화 방식(최대한 관련 정보만 가져오기)
    - LangChain : DocumentCompressor
    

### LangChain vs Semantic Kernel
| 구분 | Lang Chain |Semantic Kernel|
|:----:|-------------------------------------------------|:---------------:|
| 장점 |- 기능을 붙이는 속도가 빠르다                      |                |
|      |- ecosystem이 풍부하다                            |                 |
|      |- 필요한 것 대부분 구현되어 있다                    |                 |
|      |- 많은 연구 & 기업이 활용해 자료 많다               |                 |
|      |- 최신 연구 코드를 쉽게 얻을 수 있다                |                 |
|      |- Production Case Study 만들기 위해 지원해준다      |                 |
| 단점 |- 오픈소스 관리 측면에서 아쉬움 (너무 많은 Issue, PR) |                 |
|      |- 패치가 너무 많다 (지속적으로 버전 맞춰야)           |                 |
|      |- 자사의 유료 제품이 기본 디펜던시로 포함되어 있음     |                 |

- ChatGPT api 3/1
- LangChain 3/2
- ChatGPT Plugin 3/21
- LangChain-Plugin 3/22

- LLM 중 최신 연구 LangChain으로 만들어서 공유되고 있음
  - [코드 공개](https://www.google.com/search?q=langchain+tree+of+thought&sca_esv=569725060&sxsrf=AM9HkKkLvT_ycZyqtSvvBzvTxtJzl_QCaA%3A1696087527140&ei=5z0YZauQCPuu2roPpoeGiAw&oq=langchain+tree+of&gs_lp=Egxnd3Mtd2l6LXNlcnAiEWxhbmdjaGFpbiB0cmVlIG9mKgIIADIIEAAYgAQYsAMyCRAAGAgYHhiwA0jLDVAAWABwAXgAkAEAmAEAoAEAqgEAuAEDyAEA4gMEGAEgQYgGAZAGAg&sclient=gws-wiz-serp)

  
