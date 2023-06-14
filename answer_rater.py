#!pip install tiktoken
#Add this to your main file: from answer_rater import rank_answer
import tiktoken  # for counting tokens

#utility function for rank_answer, no need to import that
def num_tokens(text: str, model: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
    
def rank_answer(
    answer: str,
    question: str,
    openai_api,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 4096 - 500,
    ftf_answer: str = 'Ei leidnud seadustest küsimusele vastust, proovige küsimus ümber sõnastada.' #failed to find answer
) -> str:
    """Rate the answer relevants to question"""
    if answer == ftf_answer: return 'Hinnang vastusele: -'
    
    message = f'Does this "{answer}" answer "{question}" give just rating 0-10'
    if (num_tokens(message, model=model) > token_budget): 
        return 'Hinnang vastusele: Hinnangu päring liialt pikk'
        
    messages = [
        {"role": "user", "content": message},
    ]
    
    response = openai_api.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return f'Hinnang vastusele: {response["choices"][0]["message"]["content"]}'