import argparse
from pathlib import Path

import requests
from data_downloader import downloader

from llama_cpp import Llama
import gradio as gr


DEFAULT_MODEL_URL = 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf'

parser = argparse.ArgumentParser(description='Чат-бот с возможностью указания URL модели')
parser.add_argument('--model_url', type=str, default=DEFAULT_MODEL_URL, help='URL модели для загрузки')
args = parser.parse_args()

model_url = args.model_url
if '.gguf' not in model_url:
    raise Exception('Ссылка на модель должна быть прямой ссылкой на файл в формате GGUF')

model_path = Path('models') / model_url.rsplit('/')[-1]

model_path.parent.mkdir(exist_ok=True)
if not model_path.is_file():
    print('Загрузка модели ...')
    downloader.download_data(model_url, file_name=model_path)

print('Инициализация модели ...')
model = Llama(model_path=str(model_path), n_gpu_layers=-1)

def generate(user_message: str, history: list):

    messages = []
    messages.append({'role': 'user', 'content': user_message})

    # добавление системного промта
    system_prompt = "Ты умный ассистент. Отвечай на вопросы пользователя кратко и по существу. Отвечай всегда на русском языке."
    messages.append({'role': 'system', 'content': system_prompt})

    stream_response = model.create_chat_completion(
        messages=messages,  # входной промт на который надо сгенерировать ответ
        temperature=0.3,  # настройка разнообразия генерации текста моделью
        stream=True,  # вернуть генератор
        )

    # пустую строку будем соединять с ответом модели
    response = ''
    # итерация и последовательная генерация текста моделью в цикле
    for chunk in stream_response:
        # извлечение и текущего сгененированного моделью токена
        token = chunk['choices'][0]['delta'].get('content')
        if token is not None:
            response += token
             # отправка ответа в окошко чат бота
            yield response


chatbot_interface = gr.ChatInterface(
    fn=generate,  # главная функция которая будет вызываться при нажатии кнопки Отправить
    type='messages',  # новое в версии Gradio 5
    # настройки оформления
    title='Чат-бот Т10',  # название страницы
    description='Окно переписки с ботом',  # описание окошка переписки
    css='.gradio-container {width: 60% !important}',
    )


if __name__ == '__main__':
    chatbot_interface.launch(server_port=7860, server_name='0.0.0.0')  # server_name='0.0.0.0'