# Facial Emotion Recognition Module

Застосунок для розпізнавання емоцій обличчя в реальному часі з вебкамери. Використовує `mediapipe` (Face Landmarker) для пошуку лендмарок, `opencv-python` для роботи з камерою та відображення, і rule-based класифікатор емоцій. Є 2 сценарії роботи: `Try` (без збереження) та `Login` (із реєстрацією/входом і збереженням профілю користувача в PostgreSQL). UI застосунку — англійською.

## Необхідне ПЗ

- ОС: Windows / macOS / Linux.
- Python 3.x (версія має підтримуватись установленою версією `mediapipe`).
- Вебкамера/доступ до камери.
- PostgreSQL 13+.

## Встановлення

1) Створіть і активуйте віртуальне середовище:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\\Scripts\\activate  # Windows (PowerShell)
```

2) Встановіть залежності:

```bash
pip install -r requirements.txt
```

## PostgreSQL

За замовчуванням використовується `postgresql:///emotion_recognition`. Підключення можна задати через `DATABASE_URL`:

- змінна середовища `DATABASE_URL`, або
- файл `.env` у корені (див. приклад у `.env.example`).

Таблиця `users` створюється автоматично при першому використанні авторизації.

## Запуск

```bash
python main.py
```

## Перевірка (тести)

```bash
python -m unittest discover
```
