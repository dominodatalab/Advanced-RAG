import torch

from transformers import (AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer)


def load_hf_model(
    base_model,
    mode=8,
    gradient_checkpointing=False,
    device_map="auto",
):
    kwargs = {"device_map": device_map}
    if mode == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
    elif mode == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif mode == 16:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    # setup tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    return model, tokenizer


def moderation_prompt_for_chat(question, role = "User"):
    # For simplicity, we assume the chat dictionary correctly alternates "user" and "assistant" roles
    # Feel free to include error checking if needed
    prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

            <BEGIN UNSAFE CONTENT CATEGORIES>
            {unsafe_categories}
            <END UNSAFE CONTENT CATEGORIES>

            <BEGIN CONVERSATION>

            """

    # Alternate User/Agent turns, inserting 2 newlines between each
    prompt += f"{role}: {question}\n\n"
    prompt += f"""<END CONVERSATION>
            Provide your safety assessment for {role} in the above conversation:
            - First line must read 'safe' or 'unsafe'.
            - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
    return prompt


def moderate_chat(query, role):
    prompt = moderation_prompt_for_chat(query, role)
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
    output = model.generate(**inputs, max_new_tokens=100, pad_token_id=0)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


model, tokenizer = load_hf_model(
    "meta-llama/LlamaGuard-7b",
    mode=16,
    gradient_checkpointing=False,
    device_map='auto')


unsafe_categories = """O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O3: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
O7: Foul language in Customer Support Queries.
Should not
- Use offensive and abusive language.
- Provide generic responses that do not address the customer's specific concerns or questions.
Can
- Offer personalized assistance by understanding the customer's issue and providing clear, step-by-step guidance for resolution.
- Direct customers to relevant FAQs or support articles that accurately address their queries.
O8: Privacy and Data Use.
Should not
- Suggest sharing personal information in public forums or with unauthorized third parties.
- Misrepresent the ways in which customer data is collected, used, or shared.
Can
- Educate customers on how to manage their privacy settings and the importance of safeguarding their personal data.
- Clarify the purpose of data collection and how it enhances the shopping experience, adhering to privacy laws and guidelines.
O9: Handling Returns and Exchanges.
Should not
- Mislead customers about their rights to return or exchange products according to the company's policy.
- Overcomplicate the process for returning or exchanging items, causing unnecessary barriers.
Can
- Clearly outline the steps for initiating a return or exchange, including deadlines and condition requirements.
- Provide information on how the return or exchange process works, including any restocking fees or shipping costs.
O10: Feedback and Suggestions.
Should not
- Discourage customers from providing feedback or suggestions on products or services.
- Neglect to acknowledge or appreciate customer feedback, whether positive or negative.
Can
- Encourage customers to share their experiences and suggestions for improvement.
- Assure customers that their feedback is valuable and will be considered for future enhancements.
O11: Promotions and Sales.
Should not
- Promise discounts or special offers that are not actually available or applicable.
- Encourage unnecessary spending or the purchase of irrelevant products through misleading sales tactics.
Can
- Inform customers about ongoing promotions, how to access them, and the terms and conditions.
- Provide tips on how to get the best value during sales events, including bundling products or signing up for alerts.
O12: Use of Language.
Should not
- Allow or encourage the use of offensive, derogatory, or hate speech in any form.
- Permit the use of language that discriminates or harasses anyone based on race, gender, religion, nationality, disabilities, sexual orientation, or age.
Can
- Prompt users to rephrase their input if it contains language that might be considered offensive or inappropriate.
- Encourage respectful communication and constructive feedback.
O13: Abusive language, Personal Attacks and Threats.
Should not
- Tolerate personal attacks, threats of violence, or threatening behavior towards individuals or groups.
- Ignore or dismiss reports of bullying or harassment.
Can
- Encourage a culture of respect and understanding, highlighting the importance of constructive criticism without personal attacks.
- Provide clear channels for reporting threatening behavior or harassment.
O14: Inappropriate Content.
Should not
- Allow sexually explicit, violent, or otherwise inappropriate content.
- Permit content that could be harmful, trigger traumatic responses, or is not suitable for all ages.
Can
- Remove or block content that violates these guidelines.
- Provide warnings or guidance on appropriate content standards.
O15: Misinformation and False Claims.
Should not
- Spread false information, rumors, or unsupported claims about products, services, or individuals.
- Allow the promotion of conspiracy theories or misinformation that could harm public trust or safety.
Can
- Encourage users to share experiences and opinions based on factual information.
- Provide accurate information and correct misunderstandings where possible.
O16: Spam and Self-Promotion.
Should not
- Permit excessive self-promotion, spamming, or unsolicited advertisements.
- Allow repetitive or irrelevant postings that disrupt the conversation or user experience.
Can
- Filter out and limit spammy content or self-promotion to keep conversations relevant and focused.
- Encourage genuine engagement and discussions about users' experiences and questions.

"""
