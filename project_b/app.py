from flask import Flask, request, render_template, send_file
from model import load_model_and_tokenizer, generate_output, build_prompt
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model, tokenizer = load_model_and_tokenizer()


@app.route('/', methods=['GET', 'POST'])
def index():
    result_text = None
    download_link = False
    author = ''
    user_text = ''
    example_text = None

    if request.method == 'POST':
        author = request.form.get('author', '').strip()

        uploaded_input = request.files.get('input_file')
        if uploaded_input and uploaded_input.filename.endswith('.txt'):
            filename_input = secure_filename(uploaded_input.filename)
            path_input = os.path.join(UPLOAD_FOLDER, filename_input)
            uploaded_input.save(path_input)
            with open(path_input, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
            user_text = content[:2000]
        else:
            user_text = request.form.get('input_text', '').strip()

        uploaded_example = request.files.get('example_file')
        if uploaded_example and uploaded_example.filename.endswith('.txt'):
            filename_example = secure_filename(uploaded_example.filename)
            path_example = os.path.join(UPLOAD_FOLDER, filename_example)
            uploaded_example.save(path_example)
            with open(path_example, 'r', encoding='utf-8') as f_ex:
                content_ex = f_ex.read()
            example_text = content_ex[:2000]

        prompt = build_prompt(author, user_text, example_text)
        result_text = generate_output(model, tokenizer, prompt)
        download_link = True

        output_path = os.path.join(OUTPUT_FOLDER, 'result.txt')
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(result_text)

    return render_template(
        'index.html',
        author=author,
        input_text=user_text,
        example_text=example_text,
        result=result_text,
        download_link=download_link
    )


@app.route('/download')
def download_file():
    path = os.path.join(OUTPUT_FOLDER, 'result.txt')
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    # app.run(debug=True, use_reloader=False)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
