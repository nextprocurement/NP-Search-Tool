import fasttext
from langdetect import detect


class LanguageDetector:
    def __init__(self, library="fasttext", ft_model: str = None):
        """
        library: str
            fasttext|langdetect
        """
        if library == "langdetect":
            self.library = "langdetect"
            self.lang_detector = detect
        elif library == "fasttext":
            self.library = "fasttext"
            self.lang_detector = fasttext.FastText._FastText(model_path=ft_model)

    def identify_language(self, text):
        try:
            if self.library == "langdetect":
                lang = self.lang_detector(text)
            elif self.library == "fasttext":
                lang = self.lang_detector.predict(text)[0][0][9:]
        except:
            lang = "unknown"
        return str(lang)

    def filter_language(self, text, lang="es"):
        # print(self.identify_language(text))
        if self.identify_language(text) == lang:
            return text
        return None
