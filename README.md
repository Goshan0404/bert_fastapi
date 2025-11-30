# Web Application Russian Sentiment Analysis with BERT
[hosting: https://reviewanalyzer.mixdev.me/](https://reviewanalyzer.mixdev.me/)
## Structure
**[/backend]()** - бэкэнд для веб приложения, обеспечивающее хранение отзывов их анализ и сообщение с сервером модели.

**[/frontend]()** - веб приложение, демонстрирующее анализ отзывов моделью и их анализ

**[ml-fastapi]()** - сервер на fastapi с классификатором bert , для сентиметтального анализа отзывов
# Start up
Для запуска всех 3 сервисов **запустите docker-compse в /backend**:
```docker-compose up```
# Web (React js)

# Backend (C#)

# Bert Model (FastApi)
**[/fastapi]()** - сервер для инференса bert модели на fastapi, с необходимыми ручками:
- */labels/* - предсказание класса отзывов с телом json
- */labels/file* - предсказание класса отзывов с телом csv file

**[/bert]()** - каталог с весами модели

**[/tokenizer]()** - каталог с конфигами и словарем

**[/bert-notebook.ipynb]()** - ноутбук с обучением bert модели

**frameworks**: `fastapi, torch, pandas, transformers`

---
**Команда**: Мопсы

**Участнники**:
- **Чернихов Георгий Павлович**- Ml разработчик
- **БуяновБот** - Backend разработчик
- **Саша Ф О**- Frontend разработчик