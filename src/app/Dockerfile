# Используем официальный образ Python
FROM python:3.9

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8082
# Копируем файлы приложения
COPY . .

# Указываем команду запуска приложения
CMD ["python", "main.py"]