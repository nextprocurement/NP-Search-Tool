import regex
import spacy
from unidecode import unidecode

from .utils import rules_irreg, rules_non_unique, rules_other, rules_unique


class Lemmatizer:
    def __init__(self, vocabulary: dict):
        self.vocabulary = vocabulary

    def _remove_diacritics(self, word: str):
        sub_tildes = [
            ("[áà]", "a"),
            ("[éè]", "e"),
            ("[íì]", "i"),
            ("[óò]", "o"),
            ("[úùü]", "u"),
        ]
        for t, o in sub_tildes:
            word = regex.sub(t, o, word)
        return word

    def to_masc(self, word: str):
        """
        Convert femenine to masculine
        """
        # Word already appears in vocabulary
        if word in self.vocabulary:
            return self.vocabulary[word]

        patterns = [
            (r"a$", "o"),
            (r"a$", "e"),
            (r"a$", ""),
            (r"e$", "a"),
            (r"na$", "e"),
            (r"esa$", ""),
            (r"esa$", "e"),
            (r"isa$", "a"),
            (r"ina$", "e"),
            (r"triz$", "tor"),
            (r"triz$", "dor"),
        ]
        for pattern, replacement in patterns:
            if regex.search(pattern, word):
                lem = regex.sub(pattern, replacement, word)
                if lem in self.vocabulary:
                    return lem
                lem = self._remove_diacritics(lem)
                if lem in self.vocabulary:
                    return lem
        return word

    def to_singular(self, word: str):
        """
        Convert from plural to singular
        """
        # Word already appears in vocabulary
        if word in self.vocabulary:
            return self.vocabulary[word]

        # Patrones de expresiones regulares para identificar y convertir plurales
        accented_vowels = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
        inv_accented_vowels = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}

        def replacement_aguda(match):
            vowel = match.group(1)
            rest_of_word = match.group(2)
            return accented_vowels[vowel] + rest_of_word

        def replacement_y(match):
            vowel = match.group(1)
            return inv_accented_vowels[vowel] + "y"

        def remove_last(match):
            return match.group(1)

        patterns = [
            (r"(ces)$", "z"),
            (r"(?<![aeiouáéíóúy])(es)$", "e"),
            (r"([aeiou])([ns]?)(es)$", replacement_aguda),
            (r"(.*)(es)$", remove_last),
            (r"([aeiouáéíóú])(s)$", remove_last),
            (r"([áé])(is)$", replacement_y),
        ]

        for pattern, replacement in patterns:
            if regex.search(pattern, word):
                lem = regex.sub(pattern, replacement, word)
                if lem in self.vocabulary:
                    return lem
                lem = self._remove_diacritics(lem)
                if lem in self.vocabulary:
                    return lem
        return word

    def lemmatize_verb(self, word: str):
        """
        Convert verb to infinitive given a vocabulary.
        """
        # Verbos con tilde en infinitivo
        verb_tilde = {
            "dahir": "dahír",
            "desleir": "desleír",
            "desoir": "desoír",
            "desvair": "desvaír",
            "embair": "embaír",
            "engreir": "engreír",
            "entreoir": "entreoír",
            "esleir": "esleír",
            "freir": "freír",
            "invehir": "invehír",
            "oir": "oír",
            "refreir": "refreír",
            "reir": "reír",
            "sofreir": "sofreír",
            "sonreir": "sonreír",
            "trasoir": "trasoír",
        }

        if (
            word.endswith(("ar", "er", "ir", "arse", "erse", "irse"))
            or word in verb_tilde.values()
        ) and (word in self.vocabulary):
            return word

        # CD/CI/reflexive
        termination = r"((?<!s)te|(?<!m)os|los|las|les|nos|la|le|me|lo)$"
        # Changes in irregular forms
        repl_stem = [
            ("", ""),
            ("(up)|(ep)", "ab"),
            ("ie", "e"),
            ("ie", "i"),
            ("ue", "o"),
            ("i", "e"),
            ("(ir)|(yer)", "er"),
            ("hue", "o"),
            ("ue", "u"),
            ("u", "o"),
            ("uj", "uc"),
            ("(ig)|(ij)", "ec"),
            ("aj", "a"),
        ]

        def _prop(suff: str, replace: str, term: str):
            proposal = regex.sub(suff, replace, term, flags=regex.IGNORECASE).strip()
            if proposal != term:
                proposal = self._remove_diacritics(proposal)
                # proposal = "".join(
                #     [unidecode(el) if el != "ñ" else el for el in proposal.lower()]
                # )
                proposal = verb_tilde.get(proposal, proposal)
                if proposal in self.vocabulary:
                    return proposal
            return None

        alt_word = None
        words = [word]
        if regex.search(termination, word):
            # Remove CD termination
            alt_word = regex.sub(
                r"(me|te|lo|la|nos|os|los|las)$", "", word, flags=regex.IGNORECASE
            )
            # Remove CI termination
            alt_word = regex.sub(
                r"(me|te|le|nos|os|les)$", "", alt_word, flags=regex.IGNORECASE
            )
            # Remove reflexive termination
            alt_word = regex.sub(
                r"(me|te|se|nos|os|se)$", "", alt_word, flags=regex.IGNORECASE
            )
            alt_word = "".join(
                [unidecode(el) if el != "ñ" else el for el in alt_word.lower()]
            )
            if len(alt_word):
                words.append(alt_word)

        for w in words:
            for suff, replace in rules_irreg:
                proposal = _prop(suff, replace, w)
                if proposal:
                    return proposal

            for suff, replace in rules_unique:
                proposal = _prop(suff, replace, w)
                if proposal:
                    return proposal

            for suff, replace in rules_non_unique:
                proposal = _prop(suff, replace, w)
                if proposal:
                    return proposal

            for repl, stem in repl_stem:
                term = regex.sub(
                    rf"(?<!{repl}.*){repl}(?!.*{repl})*",
                    rf"{stem}",
                    w,
                    flags=regex.IGNORECASE,
                )
                for suff, replace in rules_other:
                    proposal = _prop(suff, replace, term)
                    if proposal:
                        return proposal

        return word

    def lemmatize_spanish(self, word: str):
        """
        Lemmatize any word in Spanish.
        """
        if type(word) == spacy.tokens.token.Token:
            pos = word.pos_

            if pos in ["VERB", "AUX"]:
                # try verb
                lem = self.lemmatize_verb(word.text)
                return lem
            else:
                w_sing = self.to_singular(word.text)
                if pos in ["ADJ"]:
                    w_masc = self.to_masc(w_sing)
                    if w_masc in self.vocabulary:
                        return self.vocabulary[w_masc]
                    else:
                        w_masc = self.lemmatize_verb(w_masc)
                        return w_masc
                # pos in ["NOUN", "PROPN", "ADJ", "DET"]
                else:
                    if w_sing in self.vocabulary:
                        return self.vocabulary[w_sing]
                    else:
                        w_sing = self.lemmatize_verb(w_sing)
                        return w_sing

        else:
            # Check if the word is in the vocabulary
            if word in self.vocabulary:
                # print(word)
                return self.vocabulary[word]

            # If noun/adj
            w_sing = self.to_singular(word)
            # print(w_sing)
            w_masc = self.to_masc(w_sing)
            # print(w_masc)
            if w_masc in self.vocabulary:
                return w_masc

            # If verb
            lem = self.lemmatize_verb(word)
            if lem in self.vocabulary:
                return lem
            else:
                lem = self.lemmatize_verb(w_masc)
                if lem in self.vocabulary:
                    return lem

        return word
