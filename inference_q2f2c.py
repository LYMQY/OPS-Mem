from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import prompts.generate_prompt as generate_prompt


# load model and tokenizer


# inference to get five elements
def infer_five_elem(question):
    messages = [
        {"role": "user", "content": generate_prompt.Q2F(question)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")

    if "```text" in response:
        return response.split("```text")[1].split("```")[0]
    elif "```plaintext" in response:
        return response.split("```plaintext")[1].split("```")[0]
    elif "```" in response:
        return response.split("```")[1].split("```")[0]
    else:
        return None


# inference to get pyomo python code
def infer_code(five_elem):
    messages = [
        {"role": "user", "content": generate_prompt.F2C(five_elem)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')


# execute the code
def test_code(code_str):
    code_path = f"./test.py"
    with open(code_path, "w") as f1:
        f1.write(code_str)

    ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return answer logs, error code
    return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))


# example usage
question = ""
five_elem = infer_five_elem(question)
code_str = infer_code(five_elem)
out_log, err_log = test_code(code_str)
print(out_log)
