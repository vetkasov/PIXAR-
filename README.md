# PIXAR-
Проект по стилизации текста под автора из русской классической литературы

Проект можно запустить в двух вариантах: просто с моделью yandex gpt-5 lite или c LoRA.
По умолчанию используется вариант с yandex gpt-5 lite. Чтобы запустить вариант с LoRA необходимо обучить, запустив train_LoRA.py, затем обращаться в app.py к model_LoRA.py
## Структура проекта:
```text
project_b
├── data/                 # Данные с текстами авторов (необходимы для LoRA, можно использовать как примеры в промпте)
├── outputs/              # Папка для сохранения результатов генерации
│   └── result.txt        # Результат генерации
├── static/               # Материалы для сайта
│   ├── author_icon.png   # Логотип
│   ├── styles.css        # Стиль для сайта
│   ├── welcome.mp3       # Звук оповещения о генерации (включается отдельно)
│   └── writers_bg.jpg    # Фоновая картинка
├── templates/            # Папка для веб интерфейсов
│   └── index.html        # Сам веб интерфейс
├── uploads/              # Папка для хранения текстов для ввода
│   ├── chehov.txt        # Пример текста Чехова
│   ├── dostoevski.txt    # Пример текста Достоевского
│   └── to_make.txt       # Пример текста для стилизации
├── app.py                # Файл для запуском веб интерфейса и модели
├── metric.ipynb          # Ноутбук метрикой (обучение и получение вероятностей принадлежности автору)
├── model_LoRA.py         # Файл с fine-tune моделью
├── model.py              # Файл с моделью (к нему обращается app.py для выдачи)
└── train_LoRA.py         # Дообучение LoRA
