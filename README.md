# stock-market-forecasting

# PLAN
Klepnąć mocka przyjmującego linijkę nagłówków porozdzielanych spacjami, a zwracać będzie [0.8, 0.2, 0.1] <- p-stwa klas.

1. ++ Zamiana na klasy. Różne okresy agregacji danych (dzień, 3 dni, tydzień).
2. Czyszczenie korpusu:
    1. ++ (Faja) Wykorzystać słownik skrótów do czyszczenia korpusu (https://public.oed.com/how-to-use-the-oed/abbreviations).
    2. -- (Janusz) https://github.com/savoirfairelinux/num2words wykorzystać tę bibliotekę do zamiany wartości liczbowych na słowa.
    3. -- (Tomek) Wykorzystanie słownika symboli walut do zamiany ich na słowa.
3. Reprezentacja wektorowa danych:
    1. ++ Wektoryzacja Doc2Vec (./scripts/features/vectorize_doc2vec.py), GloVe (./scripts/features/vectorize_glove.py).
    2. +- Lemmatyzacja, stemming (./scripts/experiments/reduce_lexicon.py).
    3. -- (Faja) Rozszerzenie wektora na wejściu sieci o topiki zakodowane one-hotem 
    (https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21) oraz o wartości
    indeksu giełdowego w dniach poprzednich.
4. Modele:
    1. +- (Janusz - multiclass) Regresja logistyczna.
    2. +- (Tomek - multiclass) SVM.
    3. ++ Sieć neuronowa.
5. ++ (Faja) Ewaluacja:
    1. Accuracy.
    2. Uśredniona precyzja i recall (dla każdej klasy policzyć i uśrednić).
    3. Macierz pomyłek.