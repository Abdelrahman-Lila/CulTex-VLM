from huggingface_hub import InferenceClient
import pandas as pd
import re
import time

data = pd.read_csv("Dataset_CultureVLM_style - Concepts Introductions.csv")
questions_phase2 = pd.DataFrame()
questions_phase2['concept'] = data['concept']
print(questions_phase2.head())


client = InferenceClient(
	provider="hf-inference",
	api_key=""
)

pattern = r"</think>([\s\S]*)"
questions = []

for concept in data['concept']:

    introduction = data.loc[data['concept'] == concept, 'intro']
    introduction = introduction.to_list()[0]


    messages = [
        {
            "role": "user",
            "content": f"Assume you are given a public image showing the {concept} of Egypt. Generate a multiple-choice\
                        question based on this image and the introduction of {concept}. Provide the correct\
                        answer immediately following the question.\
                        Ensure the question delves into deeper cultural knowledge but does not directly name\
                        the {concept}. The options should be somewhat confusing to increase the difficulty,\
                        but there must be only one correct answer. Users can only answer based on the image,\
                        so don’t mention any introduction or {concept} in the question. Use the following\
                        format for your generated question:\
                        - Question: [Your Question] Options: (A) [Option 1] (B) [Option 2] (C) [Option 3] (D)\
                        [Option 4]\
                        - Answer: (X) [Option X]\
                        Here are two examples:\
                        Image: Peking Duck\
                        Introduction of Peking Duck: Peking Duck is a famous Chinese dish that originated in\
                        Beijing during the Imperial era. The dish dates back to the Yuan Dynasty (1271-1368)\
                        and became a staple in the Ming Dynasty (1368-1644). Traditionally, Peking Duck is\
                        known for its thin, crispy skin and is served with pancakes, hoisin sauce, and\
                        scallions. The preparation involves inflating the duck to separate the skin from the\
                        fat, marinating it, and roasting it in a closed or hung oven. It is considered a\
                        national dish of China and a symbol of Chinese culinary art.\
                        - Question: During which dynasty did the dish shown in the image become a staple in\
                        the cuisine of its country? Options: (A) Tang Dynasty (B) Song Dynasty (C) Ming\
                        Dynasty (D) Qing Dynasty\
                        - Answer: (C) Ming Dynasty\
                        Image: The White House\
                        Introduction of The White House: The White House, located at 1600 Pennsylvania Avenue\
                        NW in Washington, D.C., is the official residence and workplace of the President of\
                        the United States. Construction began in 1792 and was completed in 1800. The building\
                        was designed by Irish-born architect James Hoban in the neoclassical style. It has\
                        been the residence of every U.S. president since John Adams. The White House has\
                        undergone several renovations and expansions, including the addition of the West Wing\
                        and the Oval Office. It is a symbol of the U.S. government and a site of significant\
                        historical events.\
                        - Question: Who was the architect responsible for designing the building shown in the\
                        image? Options: (A) James Hoban (B) Benjamin Latrobe (C) Thomas Jefferson (D) Charles\
                        Bulfinch\
                        - Answer: (A) James Hoban\
                        Now please generate the question for the Image: {concept} of Egypt\
                        Introduction: {introduction}",
        }
    ]

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
        messages=messages, 
        max_tokens=1500
    )

    question = completion.choices[0].message
    question = str(question)
    print(question)
    match = re.search(pattern, question)
    question = match.group(1)
    questions.append(question)
    time.sleep(60)

questions_phase2['questions'] = pd.Series(questions)
questions_phase2.to_csv("Questions_phase_2.csv")
