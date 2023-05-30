import utils

"""
langchain.llms 使用示例
"""

from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

# ============================ LLMs ============================ #

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    # n=2, best_of=2,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
    streaming=True, 
    verbose=True, 
)
print(llm("请给我解释 Langchain 是什么"))

llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"])
print(llm_result.llm_output)	# 返回 tokens 使用量
