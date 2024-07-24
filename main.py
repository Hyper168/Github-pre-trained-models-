from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/')
def index():
    user_id = request.headers.get('X-Replit-User-Id')
    user_name = request.headers.get('X-Replit-User-Name')
    return render_template('index.html', user_id=user_id, user_name=user_name)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text', '')

    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(
        inputs['input_ids'],
        max_length=200,  # Adjust this value based on your needs
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
  
