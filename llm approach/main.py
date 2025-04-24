# My first approach is to fine-tune a GPT-2 model to generate LEGO building instructions using LoRA.


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data Preparation: Create a mock dataset of LEGO instructions
#- my real dataset was over 200gb which i built using my webscraper https://github.com/meliksahyorulmazlar/Web-Scraping/tree/main/Lego%20Sets%20Instructions%20Webscraping
#- the real dataset was too big to share over github, this is an example of a small dataset of how it worked out
training_texts = [
    # Example 1
    "Build a simple chair. Steps: " \
    "Step 1: Attach a **red 2x2 brick** to a **blue 2x4 plate**. " \
    "Step 2: Connect a **yellow 2x2 brick** on top of the red brick. " \
    "Step 3: Place a **green flat tile** on the top as the seat.",
    # Example 2
    "Construct a small tower. Steps: " \
    "Step 1: Place a **gray 2x2 brick** on the base. " \
    "Step 2: Stack a **gray 2x2 brick** on the previous brick. " \
    "Step 3: Stack another **gray 2x2 brick** on top to complete the tower.",
    # Example 3
    "Assemble a car. Steps: " \
    "Step 1: Attach **4 wheels** to the **chassis**. " \
    "Step 2: Fix a **red car body** onto the chassis. " \
    "Step 3: Snap a **windshield** to the front of the car body."
]


# i used the GPT-2 small model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
# GPT-2 does not have an official padding token
# therefore i set the pad token to EOS to avoid warnings during tokenization.
tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)

# starting the LoRA fine-tuning setup
# this will freeze all original GPT-2 weights to prevent full fine-tuning (LoRA will train small added matrices).
for param in model.parameters():
    param.requires_grad = False

# this code will onfigure LoRA parameters: e.g., rank (r), alpha, dropout.
lora_config = LoraConfig(
    r=8,             
    lora_alpha=16,   
    lora_dropout=0.05,
    bias="none",     
    task_type="CAUSAL_LM"  
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  

encodings = tokenizer(training_texts, padding=True, truncation=True, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

labels = input_ids.clone()


# AdamW optimizerto update LoRA parameters.
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)  

model.train()  
epochs = 5     
batch_size = 2 

num_samples = input_ids.size(0)  
num_batches = (num_samples + batch_size - 1) // batch_size

for epoch in range(epochs):
    total_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_samples)
        batch_input = input_ids[start:end]
        batch_mask = attention_mask[start:end]
        batch_labels = labels[start:end]
        outputs = model(input_ids=batch_input, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs.loss  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_loss:.4f}")


# this will part of the code will define an inference function that uses the fine-tuned model to generate instructions from a prompt.
from transformers import StoppingCriteria, StoppingCriteriaList

#A class that will stop if the generated sequence ends with two newline tokens 
class StopOnRepeatNewlines(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if input_ids.shape[-1] < 2:
            return False
        last_two = input_ids[0, -2:]
        text = tokenizer.decode(last_two)
        return "\n\n" in text  

stop_criteria = StoppingCriteriaList([StopOnRepeatNewlines()])

# this method when given a prompt describing a lego model will generate a sequence of building instructions
def generate_instructions(prompt: str, max_steps: int = 5) -> str:
    model.eval() 
    input_ids = tokenizer.encode(prompt + " Steps: ", return_tensors='pt').to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,      
        num_beams=5,              
        early_stopping=True,
        stopping_criteria=stop_criteria
    )
    generated_text = tokenizer.decode(output_ids[0])
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text


# This will test the inference function with a new prompt (or a variation).
test_prompt = "Build a spaceship."
generated_instructions = generate_instructions(test_prompt)
print(f"Prompt: {test_prompt}\nGenerated Instructions:\n{generated_instructions}")
