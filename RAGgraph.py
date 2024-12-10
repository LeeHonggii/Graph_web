from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


def get_answer(
    input_text, memo_text, model_name="gpt-4", temperature=0.1, max_tokens=2048
):
    """
    질문에 대한 답변을 생성하는 함수

    Args:
        input_text (str): 사용자 질문
        memo_text (str): 참고할 내용
        model_name (str, optional): 사용할 모델명. Defaults to "gpt-4"
        temperature (float, optional): 모델의 temperature 값. Defaults to 0.1
        max_tokens (int, optional): 최대 토큰 수. Defaults to 2048

    Returns:
        str: AI 모델의 답변 텍스트
    """
    template = """
다음 참고 문서를 바탕으로 질문에 답변해주세요.

질문: {input}

참고 문서:
{memo}

답변:"""

    prompt_template = PromptTemplate.from_template(template)

    model = ChatOpenAI(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    chain = prompt_template | model

    input_dict = {"input": input_text, "memo": memo_text}

    # 응답 받기
    response = chain.invoke(input_dict)

    # 실제 답변 텍스트만 추출
    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)
