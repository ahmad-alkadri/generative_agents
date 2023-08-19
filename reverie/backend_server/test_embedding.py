import openai
from gpt4all import Embed4All
from utils import openai_api_key

def openai_get_embedding(text: str, model: str="text-embedding-ada-002") -> list:
	openai.api_key = openai_api_key
	text = text.replace("\n", " ")
	print(text)
	if not text:
		text = "this is blank"
	return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def loc_get_embedding(text: str) -> list:
	text = text.replace("\n", " ")
	embedder = Embed4All()
	output = embedder.embed(text)
	return output

if __name__=="__main__":

	text_prompt = "The quick brown fox jumps over the lazy dog"
	output_loc = loc_get_embedding(text_prompt)
	output_openai = openai_get_embedding(text_prompt)
	
	print("Output from local:", output_loc)
	print("Output from openAI:", output_openai)

	print("Length of local output:", len(output_loc))
	print("Length of OpenAI output:", len(output_openai))