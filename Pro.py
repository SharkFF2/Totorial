import os
import shutil
import magic
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

mime_detector = magic.Magic(mime=True)

MINE_KLASSER = {0: 'Annet', 1: 'Sport', 2: 'Politikk', 3: 'Teknologi'}

EXTENSION_MAPPING = {
    '.txt': "Text Files",
    '.md': "Markdown Files",
    '.csv': "CSV Files",
    '.rtf': "Rich Text Files",
    '.log': "Log Files"
}


def sanitize_folder_name(name):
    """Sanitize folder name to be valid in the filesystem."""
    return re.sub(r'[<>:"/\\|?*]', '', name)


def get_mime_type(file_path):
    """Get the MIME type of a file safely."""
    try:
        return mime_detector.from_file(file_path)
    except Exception as e:
        print(f"[ADVARSEL] MIME-deteksjon feilet for {file_path}: {e}")
        return "unknown/unknown"


def get_readable_folder_name(mime_type):
    """Convert MIME type to a more readable folder name."""
    try:
        main, sub = mime_type.split('/')
        return f"{main.capitalize()} - {sub.replace('.', ' ').replace('-', ' ').capitalize()}"
    except ValueError:
        return "Unknown"


def move_file(file_path, folder_path):
    """Move a file to a specified folder."""
    os.makedirs(folder_path, exist_ok=True)
    try:
        shutil.move(file_path, folder_path)
    except Exception as e:
        print(f"[FEIL] Kunne ikke flytte '{file_path}' til '{folder_path}': {e}")


def predict_file_category(file_path, classifier, vectorizer):
    """Predict the category of a text file, or return a fallback based on MIME type / extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in EXTENSION_MAPPING:
        return EXTENSION_MAPPING[ext]

    mime_type = get_mime_type(file_path)

    if mime_type.startswith("text"):
        # Special cases for large or empty files
        file_size = os.path.getsize(file_path)
        if file_size > 5_000_000:  # Over 5MB
            return "Text - Large File"
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read()

            if len(content.strip()) < 10:
                return "Text - Empty or Small File"

            # Predict category using trained model
            features = vectorizer.transform([content])
            prediction = classifier.predict(features)[0]
            kategori = MINE_KLASSER.get(prediction, f"Ukjent_{prediction}")
            return f"Text - {kategori}"

        except Exception as e:
            print(f"[ADVARSEL] AI-analyse feilet for '{file_path}': {e}")
            return "Text - Failed to Read"

    return get_readable_folder_name(mime_type)


def sort_files_with_ai(directory, classifier, vectorizer):
    """Sort files in a directory based on their content and type."""
    for filename in os.listdir(directory):
        if filename.startswith('.'):
            continue

        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue

        category = predict_file_category(file_path, classifier, vectorizer)
        safe_category = sanitize_folder_name(category)

        folder_path = os.path.join(directory, safe_category)
        print(f"[SORTERING] Flytter '{file_path}' --> '{folder_path}'")
        move_file(file_path, folder_path)


if __name__ == "__main__":
    directory_to_sort = input("Skriv inn banen til mappen du vil rydde opp i: ").strip()

    classifier_path = "incremental_text_classifier.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
        print(f"[FEIL] Modellfil eller vektorizer mangler. Sørg for at '{classifier_path}' og '{vectorizer_path}' finnes.")
        exit()

    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    sort_files_with_ai(directory_to_sort, classifier, vectorizer)
