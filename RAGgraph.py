from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


def get_answer(input_text, memo_text):
    template = """
    사용자가 질문한 내용: {input}
    참고할만한 내용: {memo}
    
    위 내용을 참고하여 사용자의 질문에 답변해주세요.
    """

    prompt_template = PromptTemplate.from_template(template)

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        max_tokens=2048,
        temperature=0.1,
    )

    chain = prompt_template | model

    input_dict = {"input": input_text, "memo": memo_text}

    return chain.invoke(input_dict)


if __name__ == "__main__":
    memo = "데이터 마스킹이란 데이터의 속성은 유지한 채, 익명으로 생성"
    input_question = "데이터마스킹은 뭐야?"

    response = get_answer(input_question, memo)
    print("질문:", input_question)
    print("답변:", response)
