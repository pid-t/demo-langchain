import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        description="The language the text is written in",
        enum=["spanish", "english", "french", "german", "italian"],
    )


llm = init_chat_model(
    model=os.getenv("MODEL"),
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


def main():
    inps = [
        "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!",
        "Estoy muy enojado con vos! Te voy a dar tu merecido!",
        "Weather is ok here, I can go outside without much more than a coat",
    ]
    for inp in inps:
        prompt = tagging_prompt.invoke({"input": inp})
        structured_llm = llm.with_structured_output(Classification)

        response: Classification = structured_llm.invoke(prompt)
        print(f"Input: {inp}")
        print(response.model_dump_json(indent=4))
        print("\n")

    # inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
    # prompt = tagging_prompt.invoke({"input": inp})
    # structured_llm = llm.with_structured_output(Classification)

    # response: Classification = structured_llm.invoke(prompt)
    # print(response.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
