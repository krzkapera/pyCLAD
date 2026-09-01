# Continual-MEGA w pyCLAD — notatki implementacyjne

Dziennik decyzji, znalezionych błędów i odstępstw od referencji.

Referencje:
- paper: *Continual-MEGA: A Large-scale Benchmark for Generalizable Continual Anomaly Detection* (Neurocomputing 700, 2026)
- kod: https://github.com/Continual-Mega/Continual-MEGA-Baseline
- dane: https://huggingface.co/datasets/Continual-Mega/Continual-MEGA-Benchmark

Model bazowy (ADCT / CLIP) jest celowo poza zakresem — implementujemy sam benchmark.

---

## 1. Błędy i niespójności znalezione w referencji

### 1.1 `eval_continual.py` zapisuje same zera

```python
results_image = np.full((num_tasks, num_tasks), 0)   # dtype int64
...
results_image[args.task_id, i] = img_auc_mean        # 0.85 -> 0
```

Macierz wyników jest tworzona jako `int64`, więc przypisanie AUROC/AP obcina każdą wartość do zera i taki
CSV trafia do `calculate_metrics.py`. `train_continual.py` używa w tym samym miejscu `np.nan` (float), więc
ścieżka treningowa jest poprawna — błąd dotyczy wyłącznie samodzielnej ewaluacji z checkpointów.

Nie odtwarzamy tego zachowania.

### 1.2 `cls_name` w próbce nie zgadza się z kluczem klasy

W `meta_files/*.json` klucz słownika jest prefiksowany nazwą zbioru (`mvtec_leather`, `visa_pcb4`,
`btad_03`, `mpdd_connector`), ale pole `cls_name` wewnątrz próbki dla części zbiorów jest gołe
(`leather`, `pcb4`, `03`, `connector`), a dla części prefiksowane (`continual_ad_*`, `real_iad_*`,
`viaduct_*`). Tożsamość klasy bierzemy wyłącznie z klucza słownika; pole `cls_name` jest ignorowane.

### 1.3 Brak meta dla zero-shot na VisA

Paper raportuje zero-shot na MVTec-AD **i** VisA (Tab. 2 i 3), ale repozytorium zawiera tylko
`meta_files/meta_mvtec.json`. Zbiór testowy VisA budujemy z indeksu katalogu przez istniejący
`VisABenchmarkReader` (split `split_csv/1cls.csv`) — patrz §3.7.

### 1.4 `meta_mvtec.json` ma inny katalog bazowy niż pliki strumienia

Ścieżki w plikach scenariuszowych są względne wobec katalogu z wszystkimi zbiorami
(`continual_ad/...`, `mvtec_anomaly_detection/...`), natomiast w `meta_mvtec.json` są względne wobec
samego katalogu MVTec (`bottle/test/...`) — zgodnie z `eval_zero.sh`, które podaje
`--data_root data/mvtec_anomaly_detection`. Obsługujemy oba korzenie osobno.

### 1.5 Kolejność klas nie jest zagnieżdżona między rozmiarami zadań

Dla tego samego scenariusza strumienie 5-, 10- i 30-klasowe mają **różną** kolejność klas (sprawdzone:
żadna para nie jest permutacją blokową drugiej). Nie da się ich wyprowadzić z jednej listy — pliki meta
są jedynym źródłem prawdy.

### 1.6 Splity treningowe nie są spójne między scenariuszami

Dla 30 klas wspólnych dla base scenariusza 1 i 2 **żadna** nie ma tego samego zestawu 20 obrazów
treningowych. W obrębie jednego scenariusza train i test są rozłączne, ale obraz treningowy z S1 bywa
obrazem testowym w S2. Wniosek: ContinualAD nie ma kanonicznego splitu — patrz §4.1.

### 1.7 Tabela A nie sumuje się do własnych wartości zbiorczych

Liczby per klasa w Tab. A są dla **każdej** klasy o 10 niższe od rzeczywistych — zarówno dla obrazów
normalnych, jak i anomalnych (Apple 490/502 w tabeli vs 500/512 w danych, Ruler 277/490 vs 287/500,
Energy-bar 329/542 vs 339/552, i tak dla wszystkich 30 klas). Wygląda to na liczności *zbioru testowego*,
czyli po odjęciu 10 normalnych i 10 anomalnych obrazów treningowych.

Kolumny tabeli nie sumują się więc do wartości podanych w jej własnym tekście: 14 355 vs deklarowane
14 655 obrazów normalnych (różnica to dokładnie 30 klas × 10). Rzeczywiste sumy z plików meta to
14 655 normalnych i 15 826 anomalnych — pierwsza zgadza się z deklaracją, druga jest o 1 mniejsza niż
podane 15 827.

Zweryfikowane na Heliosie: dla 10 pobranych klas zawartość dysku pokrywa się z plikami meta co do
obrazu (0 ścieżek z meta nieobecnych na dysku, 0 obrazów na dysku nieujętych w meta).

### 1.8 Nieznormalizowane nazwy defektów i dwie konwencje nazw masek w ContinualAD

Katalogi anomalii zawierają warianty i literówki: `missing part`, `missing_part`, `misisng part`,
`crack`, `crack1`, `crack2`. Maski występują w dwóch konwencjach — `<stem>.png` oraz `mask_<stem>.png` —
a katalog urządzenia bywa pominięty (ok. 3 000 z 59 000 obrazów anomalnych ma ścieżkę
`anomaly/<defekt>/<plik>` zamiast `anomaly/<defekt>/<urządzenie>/<plik>`). Reader obsługuje oba warianty;
nazw defektów nie normalizujemy — `defect_type` przechowuje nazwę katalogu bez zmian.

W archiwach na HuggingFace w katalogach anomalii siedzą też pliki `.DS_Store` (np.
`Candy/anomaly/crack/.DS_Store`). Reader filtruje po rozszerzeniu, więc ich nie widzi, ale skanowanie
katalogu „wszystko co jest plikiem" zliczyłoby je jako obrazy.

---

## 2. Błędy znalezione w pyCLAD (nienaprawione)

### 2.1 Literówka w kluczu `tran_concepts_no`

`ConceptsDataset.additional_info()` zwraca `{"tran_concepts_no": ...}` zamiast `train_concepts_no`.
Klucz trafia do `output.json`, więc poprawka zepsułaby parsowanie istniejących wyników. Zostawiamy,
zgłaszamy osobno.

---

## 3. Odstępstwa wymuszone przez architekturę pyCLAD

### 3.1 Koncepty treningowe to grupy, testowe to klasy

Referencja utrzymuje dwa poziomy granulacji: strumień przychodzi grupami (base, task_1 … task_N), ale
metryka to **średnia po klasach** wewnątrz grupy, liczona z osobnym `DataLoader`em na klasę.

`ConceptsDataset` nie wymaga zgodności list train i test, więc odwzorowujemy to wprost:
`train_concepts` = grupy, `test_concepts` = pojedyncze klasy. Mapowanie klasa → grupa wystawia
`ContinualMegaDataset.group_by_concept()`, a `GroupedConceptMetricCallback` składa z niego macierz
grupową, uśredniając w grupie.

To jednocześnie rozwiązuje problem pamięci: mediana klasy testowej ma 250 obrazów, największa
(`real_iad_mint`) 5 285, więc szczyt to ~2,4 GB map anomalii zamiast ~52 GB dla całej grupy base.

**Konsekwencja:** w tej konfiguracji nie wolno użyć zwykłego `ConceptMetricCallback` ani
`VisionPixelConceptMetricCallback` — obie budują macierz kwadratową indeksowaną nazwami nauczonych
konceptów i przy 145 konceptach testowych vs 7 treningowych rzucą `KeyError`.

### 3.2 Leniwe budowanie konceptów

Zbiór testowy scenariusza 1 to 193 258 obrazów. Zmaterializowany na raz to ~65 GB obrazów i ~22 GB
masek, przy czym `build_concepts_dataset_from_samples` ładował maski niezależnie od `data_mode`.
`LazyVisionConceptList` buduje koncept dopiero przy iteracji i nie trzyma go, więc w pamięci żyje jeden
koncept naraz. Kosztem jest ponowne dekodowanie obrazów i masek w każdym etapie — pomijalne wobec
1,35 mln przebiegów modelu.

### 3.3 Pełna macierz zamiast dolnej trójkątnej

Referencja liczy tylko `M[i][j]` dla `j ≤ i`; pyCLAD ocenia wszystkie koncepty po każdym etapie.
Zmierzone dla scenariusza 1 / 10 klas: 1 352 806 vs 1 117 708 inferencji, czyli **17%** różnicy — grupa
base to 60% zbioru testowego i tak jest liczona w każdym etapie. Za tę oszczędność trzeba by wpuścić
`NaN` do macierzy i przerobić `ContinualAverage` oraz `ForwardTransfer`, więc zostawiamy pełną macierz.
Górny trójkąt to wyniki zero-shot na klasach jeszcze nienauczonych, więc `ForwardTransfer` działa gratis.

`AverageAccuracy` i `ForgettingMeasureStrict` liczone są wyłącznie na kwadratowej podmacierzy grup
uczonych — grupy zero-shot trafiają do osobnej sekcji `held_out_groups`.

### 3.4 Nadzorowany trening jako równoległy kontrakt

`Model.fit(data)` i `Strategy.learn(data)` nie przenoszą etykiet ani masek, a benchmark trenuje na
10 normalnych + 10 anomalnych obrazach z maskami pikselowymi. Rozszerzenie istniejących sygnatur
wywróciłoby `der.learn`, `agem.learn`, `mste.learn` (brak `**kwargs`) oraz wszystkie 8 implementacji
`fit`, w tym modele spoza repozytorium.

Zamiast tego dokładamy równoległy kontrakt: `SupervisedStrategy.learn_concept(concept)` w rdzeniu
i `SupervisedVisionModel.fit_supervised(data, labels, masks)` w warstwie vision, spięte przez
`NaiveSupervisedStrategy`. Scenariusze rozgałęziają się po `isinstance`, więc istniejący kod działa bez
zmian. Strategia dostaje `Concept`, bo tylko ona wie, czy koncept niesie maski; model dostaje tablice.

### 3.5 `ImageLoadOptions` zamiast `validate_read_options`

Parametry ładowania obrazu urosły z trzech do pięciu (doszły `interpolation` i `apply_exif_transpose`)
i wędrują przez cztery warstwy. Zamiast przepychać pięć argumentów zebrane są w zamrożoną
`ImageLoadOptions`, która waliduje się sama w `__post_init__`. Funkcja `validate_read_options` znika,
a `materialize_samples` i `_load_image` przyjmują jeden obiekt zamiast listy parametrów.

Publiczne `read_vision_dataset` / `read_dataset` zachowują dotychczasowe argumenty.

### 3.6 Zero-shot bez osobnego callbacku

Koncepty MVTec-AD i VisA są dodane do `test_concepts`, ale nie do `train_concepts`. Scenariusz ocenia je
po każdym etapie bez żadnej zmiany kodu, a `GroupedConceptMetricCallback` rozpoznaje grupy, które nigdy
nie wystąpiły jako nauczone, i raportuje je w `held_out_groups`. Callback z referencją do strategii
okazał się niepotrzebny.

Zero-shot jest odrzucany dla scenariusza 1, bo MVTec-AD i VisA są tam częścią strumienia treningowego.

### 3.7 VisA zero-shot z indeksu katalogu

Wobec braku pliku meta (§1.3) zbiór testowy VisA powstaje z `VisABenchmarkReader` — wszystkie wiersze
`split_csv/1cls.csv` z `split == test`. Nazwy konceptów prefiksujemy (`visa_<klasa>`, `mvtec_<klasa>`),
zgodnie z konwencją plików strumienia; w scenariuszach 2 i 3 nie ma kolizji, bo oba zbiory są wyłączone
ze strumienia.

---

## 4. Decyzje projektowe

### 4.1 Split ContinualAD jako samodzielnego zbioru

ContinualAD nie ma kanonicznego splitu (§1.6). Domyślny w `ContinualADBenchmarkReader` odwzorowuje
benchmark: 10 normalnych + 10 anomalnych na klasę do treningu, reszta do testu. Wybór jest
deterministyczny — ziarno to `seed + crc32("<klasa>/<normal|anomaly>")` — więc nie zależy od kolejności
ani podzbioru klas. `train_anomaly_per_category=0` daje wariant normal-only.

### 4.2 EXIF

ContinualAD to zdjęcia z 10 telefonów, więc obrazy niosą tagi orientacji EXIF. Referencyjny
`dataset/continual.py` woła `ImageOps.exif_transpose` **tylko na obrazie**, nie na masce — maski są już
zapisane w orientacji po transpozycji. Odtwarzamy to dokładnie: `_load_image` transponuje, ładowanie
masek nie.

Włączenie tego globalnie po cichu zmieniłoby wyniki istniejących użytkowników, więc domyślnie
`apply_exif_transpose=False`, a `ContinualMegaDataset` i przykład dla ContinualAD włączają je jawnie.

### 4.3 Interpolacja

Referencja skaluje obrazy BICUBIC, maski NEAREST. pyCLAD używał BILINEAR dla obrazów (maski już były
NEAREST). Globalna wartość domyślna zostaje BILINEAR, `ContinualMegaDataset` ustawia BICUBIC.

### 4.4 `train_samples`

`ContinualMegaDataset(train_samples="all")` odtwarza benchmark (10+10 z maskami) i wymaga modelu
nadzorowanego. `train_samples="normal"` odfiltrowuje anomalie, żeby dało się uruchomić benchmark
z istniejącymi modelami pyCLAD (PaSTe, FastFlow). Bez tego przełącznika model one-class dostałby
anomalie oznaczone jako dane treningowe i uczyłby się ich jako normalności.

### 4.5 `nanmean` w makro-średniej

`PixelAveragePrecision` zwraca `NaN`, gdy klasa nie ma dodatnich pikseli. W plikach meta taka klasa nie
występuje (sprawdzone: 0 klas testowych z jedną etykietą, 0 anomalii bez maski), ale uśrednianie w grupie
używa `nanmean`, żeby pojedyncza zdegenerowana klasa nie wyzerowała całej grupy.

### 4.6 Brak konwertera do manifestu CSV

`ContinualMegaDataset` czyta pliki meta wprost. Konwersja do manifestu pyCLAD (`*_samples.csv`) miałaby
sens dopiero przy hostowaniu specyfikacji obok danych — odłożone razem z tematem pobierania dużych
plików.

---

## 5. Pomiary na Heliosie (Cyfronet)

Środowisko: partycja `plgrid-gpu-gh200`, NVIDIA GH200 120GB (aarch64), Python 3.11.5,
torch 2.11.0+cu128. Wszystkie dane, cache i wyniki w `$SCRATCH/continual-mega/`.

Uwaga techniczna: węzły logowania są x86_64, a węzły GPU aarch64, więc venv zbudowany na loginie tam
nie działa. Zadania wymagają `--export=NONE`, inaczej `MODULEPATH` odziedziczony z loginu każe
`module load Python` wybrać build x86_64.

### 5.1 Smoke test

Syntetyczny zbiór odwzorowujący layout ContinualAD (obie konwencje nazw masek, obie głębokości
katalogów) plus pliki meta dla scenariuszy 1–3 i zbiory zero-shot. 18 asercji, wszystkie przechodzą:
odkrywanie klas, rozmiary splitu few-shot, rozwiązywanie masek, determinizm splitu, etykiety i maski
na koncepcie treningowym, dyspozycja do `fit_supervised` vs `fit`, kolejność grup, wydzielenie grup
zero-shot oraz zgodność `AverageAccuracy` i `ForgettingMeasureStrict` z wartościami policzonymi ręcznie
z macierzy grupowej.

### 5.2 Przebieg na realnych danych

10 klas ContinualAD (18 GB) jako strumień 10 konceptów, FastFlow z backbonem resnet18, obrazy 256×256,
200 obrazów normalnych na klasę do treningu, 10 epok, strategia `NaiveStrategy`.

Image ROC-AUC — przekątna (wynik zaraz po nauczeniu klasy) kontra ostatni wiersz (po nauczeniu
wszystkich dziesięciu):

| klasa | po nauczeniu | na końcu |
|---|---|---|
| Apple | 0.771 | 0.263 |
| Candy | 0.885 | 0.608 |
| Capsule | 0.949 | 0.553 |
| Cup | 0.576 | 0.286 |
| Energy-bar | 0.819 | 0.355 |
| Eraser | 0.939 | 0.237 |
| Flash-drive | 0.937 | 0.305 |
| Food-container | 0.812 | 0.608 |
| Mouse | 0.762 | 0.614 |
| Ruler | 0.860 | 0.860 |

Czyli model uczy się każdej klasy poprawnie (przekątna 0.58–0.95), ale bez żadnego mechanizmu
przeciwdziałania zapominaniu wyniki na wcześniejszych klasach spadają poniżej losowego — dokładnie to,
co strategia `Naive` ma pokazywać.

Ten sam strumień ze strategią `CumulativeStrategy` (retrening na wszystkich danych widzianych do tej
pory) dla porównania:

| metryka | Naive | Cumulative |
|---|---|---|
| image ROC-AUC — ACC | 0.469 | **0.555** |
| image ROC-AUC — FM | 0.402 | **0.194** |
| image ROC-AUC — BWT | −0.080 | **−0.030** |
| image ROC-AUC — ContinualAverage | 0.523 | **0.649** |
| pixel ROC-AUC — ACC | 0.679 | **0.743** |
| pixel ROC-AUC — FM | 0.122 | **0.061** |
| pixel AP — ACC | 0.028 | 0.028 |

Cumulative zmniejsza zapominanie o połowę i podnosi ACC, czyli uporządkowanie strategii wychodzi
zgodnie z oczekiwaniem. Pixel AP pozostaje bardzo niskie w obu przypadkach — to zgadza się z główną
obserwacją paperu, że lokalizacja pikselowa jest najsłabszym punktem metod na tym benchmarku.

### 5.3 Walidacja krzyżowa callbacku grupowego

`GroupedConceptMetricCallback` z mapowaniem identycznościowym (klasa → własna grupa) daje wartości
identyczne z klasycznym `ConceptMetricCallback`: maksymalna różnica bezwzględna wynosi dokładnie 0.0
w obu przebiegach.

### 5.4 Gdzie idzie czas

Trening zajął 45 s (Naive) i 222 s (Cumulative), ewaluacja odpowiednio 3 027 s i 3 052 s — czyli 93–99%
czasu przebiegu. Materializacja zbioru (dekodowanie i skalowanie ~10 tys. JPEG-ów) zajęła dodatkowe
912 s i odbywa się zachłannie, bo samodzielny `ContinualADBenchmarkReader` idzie przez `read_dataset`,
a nie przez `LazyVisionConceptList`.

Potwierdza to założenie z §3.1 i §3.2: kosztem benchmarku jest ewaluacja, nie trening, więc granulacja
i sposób trzymania danych testowych w pamięci są ważniejsze niż cokolwiek po stronie treningu.
