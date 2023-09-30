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


### 2세대 : ChatGPT API 오케스트레이션
> 유저 <-> APP <-> LLM <-> (LLM, LLM, LLM, LLM) 연쇄 작용으로 결과 생성
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


### 3세대 : 외부 데이터 연결
- Copilot이라고 불릴 자격이 있음
  - 발화 의도 > Pipe-line(시나리오) > 답변 생성 (Copilot ; 외부 데이터 활용)
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

