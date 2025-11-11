import os
import shutil
import fnmatch
import stat
import time


def remove_readonly(func, path, excinfo):
    """
    Обработчик для удаления файлов с атрибутом 'readonly'
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Не удалось удалить {path}: {e}")


def clean_project_directory(directory_path):
    """
    Очищает папку Python проекта от служебных файлов
    """
    items_to_remove = []

    for root, dirs, files in os.walk(directory_path, topdown=False):
        # Удаление файлов
        for file in files:
            file_path = os.path.join(root, file)

            # Удаление .log файлов
            if fnmatch.fnmatch(file, '*.log'):
                items_to_remove.append(file_path)

            # Удаление файлов Python кэша
            if (fnmatch.fnmatch(file, '*.pyc') or
                fnmatch.fnmatch(file, '*.pyo') or
                    fnmatch.fnmatch(file, '*.pyd')):
                items_to_remove.append(file_path)

        # Удаление папок
        for dir in dirs:
            dir_path = os.path.join(root, dir)

            # Удаление папки .git
            if dir == '.git':
                items_to_remove.append(dir_path)

            # Удаление папок __pycache__
            if dir == '__pycache__':
                items_to_remove.append(dir_path)

    # Фильтруем только существующие элементы
    items_to_remove = [
        item for item in items_to_remove if os.path.exists(item)]

    # Подтверждение удаления
    if items_to_remove:
        print(f"\nНайдено {len(items_to_remove)} элементов для удаления:")
        for item in items_to_remove:
            item_type = "папка" if os.path.isdir(item) else "файл"
            print(f"  - {item} ({item_type})")

        confirm = input("\nПродолжить удаление? (y/n): ").strip().lower()
        if confirm == 'y':
            for item_path in items_to_remove:
                try:
                    if os.path.isfile(item_path):
                        # Для файлов
                        os.chmod(item_path, stat.S_IWRITE)
                        os.remove(item_path)
                        print(f"✓ Успешно удален файл: {item_path}")
                    elif os.path.isdir(item_path):
                        # Для папок, особенно .git
                        if os.path.basename(item_path) == '.git':
                            print(f"Пытаюсь удалить папку .git...")
                            # Дополнительные попытки для .git
                            shutil.rmtree(item_path, onerror=remove_readonly)
                        else:
                            shutil.rmtree(item_path, onerror=remove_readonly)
                        print(f"✓ Успешно удалена папка: {item_path}")
                except Exception as e:
                    print(f"✗ Ошибка при удалении {item_path}: {e}")
                    # Дополнительная попытка с задержкой
                    if os.path.basename(item_path) == '.git':
                        print("Повторная попытка удаления .git через 2 секунды...")
                        time.sleep(2)
                        try:
                            shutil.rmtree(item_path, onerror=remove_readonly)
                            print(
                                f"✓ Успешно удалена папка .git после повторной попытки")
                        except Exception as e2:
                            print(f"✗ Не удалось удалить .git: {e2}")
            print("\nОчистка завершена!")
        else:
            print("Удаление отменено.")
    else:
        print("Не найдено файлов для удаления.")


def force_clean_git(directory="."):
    """
    Принудительное удаление папки .git
    """
    git_path = os.path.join(directory, '.git')

    if not os.path.exists(git_path):
        print("Папка .git не найдена")
        return

    print(f"Принудительное удаление папки .git: {git_path}")

    try:
        # Множественные попытки удаления
        for attempt in range(3):
            try:
                shutil.rmtree(git_path, onerror=remove_readonly)
                print(f"✓ Папка .git успешно удалена")
                break
            except Exception as e:
                print(f"Попытка {attempt + 1}: Ошибка - {e}")
                if attempt < 2:  # Не ждать после последней попытки
                    time.sleep(2)
        else:
            print("✗ Не удалось удалить папку .git после 3 попыток")
            print("Возможно, папка используется другим процессом")
    except Exception as e:
        print(f"Критическая ошибка: {e}")


def main():
    """
    Основная функция скрипта
    """
    print("=== Очистка Python проекта от служебных файлов ===\n")

    current_dir = os.getcwd()
    print(f"Текущая директория: {current_dir}")

    use_current = input("Очистить текущую директорию? (y/n): ").strip().lower()

    if use_current == 'y':
        directory_to_clean = current_dir
    else:
        directory_to_clean = input(
            "Введите путь к директории для очистки: ").strip()
        if not os.path.exists(directory_to_clean):
            print(f"Ошибка: Директория '{directory_to_clean}' не существует!")
            return

    # Проверяем наличие .git
    git_path = os.path.join(directory_to_clean, '.git')
    if os.path.exists(git_path):
        print("\n⚠️  ВНИМАНИЕ: Обнаружена папка .git!")
        print("Её удаление приведет к полной потере истории версий!")

        git_choice = input("Удалить папку .git? (y/n): ").strip().lower()
        if git_choice == 'y':
            force_clean_git(directory_to_clean)
        else:
            print("Папка .git сохранена")

    print(f"\nНачинаем очистку директории: {directory_to_clean}")
    clean_project_directory(directory_to_clean)


if __name__ == "__main__":
    main()
