import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_adv_snippet(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = message,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    answer = response.choices[0]["message"]["content"].strip()
    return answer

def generate_new_prompt(content, pre_context, post_context):

    new_prompts = []

    for idx in range(len(content)):
      prompt = content[idx]['prompt'].replace('\n', '').replace('\r', '')
      pre_con = pre_context[idx]
      post_con = post_context[idx]
      front_context = "With the partial preceding codes provided as:"
      end_context = "and with the partial following codes provided as"
      new_promp = front_context + pre_con + ". " + prompt + " " + end_context + post_con
      new_promp = new_promp.replace(r' """ ', "")
      new_prompts.append(new_promp)

    return new_prompts

def save_adv_snippet(new_prompts):
    adv_snippets = []

    for new_prom in new_prompts:
        message = [{ "role": "system", "content": "You are an experienced programmer." },
                {"role": "user", "content":new_prom},]
        answer = generate_adv_snippet(message)
        adv_snippets.append(answer)
    return adv_snippets
