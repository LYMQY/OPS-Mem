import prompts.augment_prompt as pt
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

class Augment():
    def __init__(self):
        self.gpt = genai.Client(api_key=os.getenv("Gemini_API_Key_1"))
        self.system_info = "You are a professional AI expert. "
    
    def __call__(self, ques, seed=1, ques2=""):
        switcher = {
            0: pt.aug_0(ques, ques2),
            1: pt.aug_1(ques),
            2: pt.aug_2(ques),
            3: pt.aug_3(ques),
            4: pt.aug_4(ques),
            5: pt.aug_5(ques),
            6: pt.aug_6(ques),
        }
        query = switcher.get(seed)
        prompt = self.system_info + query
        response = self.gpt.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt]
        )

        clean_response = response.text.strip()
        if clean_response.startswith('```'):
            clean_response = clean_response[3:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()

        return clean_response

