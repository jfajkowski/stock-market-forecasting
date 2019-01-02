# stock-market-forecasting

# PLAN
1. Zamiana na klasy. Różne okresy agregacji danych (dzień, 3 dni, tydzień).
2. Czyszczenie korpusu:
    1. Znaleźć i wykorzystać słownik skrótów do czyszczenia korpusu.
    2. https://github.com/savoirfairelinux/num2words wykorzystać tę bibliotekę do zamiany wartości liczbowych na słowa.
    3. Wykorzystanie słownika symboli walut do zamiany ich na słowa.
3. Reprezentacja wektorowa danych:
    1. Wektoryzacja Doc2Vec (./scripts/features/vectorize_doc2vec.py), GloVe (./scripts/features/vectorize_glove.py).
    2. Lemmatyzacja, stemming (./scripts/experiments/reduce_lexicon.py).
    3. Rozszerzenie wektora na wejściu sieci o topiki zakodowane one-hotem 
    (https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21) oraz o wartości
    indeksu giełdowego w dniach poprzednich.
4. Modele:
    1. Regresja logistyczna.
    2. SVM.
    3. Sieć neuronowa.
5. Ewaluacja:
    1. Accuracy.
    2. Uśredniona precyzja i recall (dla każdej klasy policzyć i uśrednić).
    3. Macierz pomyłek.