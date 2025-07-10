print("Тест запуска Python скрипта")
print("Текущая директория:", __file__)

try:
    import finrl
    print("FinRL импортирован успешно")
except Exception as e:
    print(f"Ошибка импорта FinRL: {e}")