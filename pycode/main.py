
# Secret API key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SequentialChain
import argparse
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()
llm = OpenAI()

## prompt template for generating code
code_prompt = PromptTemplate(
    template="Write a very short function using {language} program that that will {task}.",
    input_variables=["language", "task"],
)

## prompt template for testing code.
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template= "Write a test for the following {language} code :\n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key= "code"
)

test_chain = LLMChain(
    llm = llm,
    prompt=test_prompt,
    output_key="test"
)



chain  = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)


result  = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>>>>> Generated Code <<<<<<<<")
print(result["code"])

print(">>>>>>>> Generated Test <<<<<<<<")
print(result["test"])

