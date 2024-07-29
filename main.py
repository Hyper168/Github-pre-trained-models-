from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text(prompt, max_length=500, num_return_sequences=1):
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')


    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

    

    # Tokenize input with padding
    inputs = tokenizer(prompt,
                       return_tensors='pt',
                       padding=True,
                       truncation=True)

    # Extract input_ids and attention_mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate text
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  # Prevent repeating the same n-grams
        num_beams=5,
        early_stopping=True)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Start the loop
while True:
    # Get input from the user
    user_input = input("Enter a starting sentence (or type 'exit' to quit): ")

    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        break

    # Generate and show the text
    generated_text = generate_text(user_input)
    print("Generated text:")
    print(generated_text)
    print("\n")  # Add a newline for better readability
