import prompts.augment_prompt as prompt
from utils.gpt4_infer import GPT4
from utils.extract import Extractor


class Augment():
    def __init__(self):
        self.gpt = GPT4(version="gpt-4o")
        self.system_info = "You are a professional AI expert. "
        self.etr = Extractor()
    
    def __call__(self, ques, seed=1, ques2=""):
        switcher = {
            0: prompt.aug_0(ques, ques2),
            1: prompt.aug_1(ques),
            2: prompt.aug_2(ques),
            3: prompt.aug_3(ques),
            4: prompt.aug_4(ques),
            5: prompt.aug_5(ques),
            6: prompt.aug_6(ques),
        }
        query = switcher.get(seed)
        rtn = self.gpt(query, self.system_info, infos=None)
        return self.etr.extract(rtn)[1:-1]

