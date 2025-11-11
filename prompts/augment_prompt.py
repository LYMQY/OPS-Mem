evo_1 = "The following is an optimization problem. Please construct a new optimization problem based on the context of this problem. "
evo_2 = "The following is an optimization problem. Please find similar problems in other fields and construct a new optimization problem with a new background. "

evo_3 = "There are two optimization problems. Please construct a new optimization problem based on the background of problem A and the optimization problem type of problem B. " 

mdf_1 = "The following is an optimization problem. Please modify the constraints of this problem and construct a new optimization problem. "
mdf_2 = "The following is an optimization problem. Please modify the constraints and object of this problem and construct a new optimization problem. "
mdf_3 = "The following is an optimization problem. Please modify the variables and parameters of this problem reasonably and construct a new optimization problem. "
mdf_4 = "The following is an optimization problem. Please modify the description of some statements and construct a new optimization problem without changing the meaning of the original problem. "

general_1 = "The original optimization problem is as follows: "
general_2 = "The original optimization problem A is as follows: "
general_3 = "The original optimization problem B is as follows: "

general_0 = """Please construct a new optimization problem according to the above requirements and the provided questions and in the following format:
```
[you should write the new problem here]
```
"""


def aug_0(ques1, ques2):
    return evo_3 + "\n" + general_2 + "\n" + ques1 + "\n" + general_3 + "\n" + ques2 + "\n" + general_0

def aug_1(ques):
    return evo_1 + "\n" + general_1 + "\n" + ques + "\n" + general_0

def aug_2(ques):
    return evo_2 + "\n" + general_1 + "\n" + ques + "\n" + general_0

def aug_3(ques):
    return mdf_1 + "\n" + general_1 + "\n" + ques + "\n" + general_0

def aug_4(ques):
    return mdf_2 + "\n" + general_1 + "\n" + ques + "\n" + general_0

def aug_5(ques):
    return mdf_3 + "\n" + general_1 + "\n" + ques + "\n" + general_0

def aug_6(ques):
    return mdf_4 + "\n" + general_1 + "\n" + ques + "\n" + general_0
