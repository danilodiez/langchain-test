from decouple import config
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

OPENAI_API_KEY = config('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = config('ACTIVELOOP_TOKEN')

def token_usage():
    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
    with get_openai_callback() as cb:
        result = llm("Tell me a joke")
        print(cb)


def few_shot_learning():
    # create our examples
    examples = [
        {
            "query": "What's the weather like?",
            "answer": "It's raining cats and dogs, better bring an umbrella"
        },
        {
            "query": "How old are you",
            "answer": "Age is just a number, but I'm timeless"
        }
    ]

    # create an example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """The following are excerpts from conversations with an AI
    assistant. The assistant is known for its humor and wit, providing
    entertaining and amusing responses to users' questions. Here are some
    examples:
    """

    # and a suffix
    suffix = """
    User: {query}
    AI: """


    # now we create a few-show prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    chat = ChatOpenAI(model_name="gpt-3", temperature=0.0, openai_api_key=OPENAI_API_KEY)


    chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
    chain.run("What's the meaning of life?")




# token_usage()
few_shot_learning()
