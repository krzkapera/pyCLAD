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

### 1.8 Layout ContinualAD jest niejednorodny

Katalogi anomalii zawierają warianty i literówki nazw defektów: `missing part`, `missing_part`,
`misisng part`, `crack`, `crack1`, `crack2`. Maski występują w dwóch konwencjach — `<stem>.png` oraz
`mask_<stem>.png` — a katalog urządzenia bywa pominięty (ok. 3 000 z 59 000 obrazów anomalnych ma
ścieżkę `anomaly/<defekt>/<plik>` zamiast `anomaly/<defekt>/<urządzenie>/<plik>`). W archiwach na
HuggingFace w katalogach anomalii siedzą dodatkowo pliki `.DS_Store`.

Dla nas to bez znaczenia: ścieżki do obrazów i masek bierzemy wprost z plików meta, nie ze skanowania
katalogów. Notujemy, bo każdy reader oparty na przechodzeniu drzewa katalogów musi te warianty
obsłużyć.

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

`GroupedConceptMetricCallback` i jego wariant pikselowy są osobnymi klasami, niezależnymi od
`ConceptMetricCallback` i `VisionPixelConceptMetricCallback`: budują macierz **grup**, nie konceptów,
więc jeden koncept treningowy może obejmować wiele testowych. Wariant pikselowy dziedziczy po obrazowym
i nadpisuje wyłącznie odczyt wartości.

Rozważaliśmy wpięcie grupowania w istniejące klasy (haki na wartość i na kolumnę w
`ConceptMetricCallback`, wariant pikselowy przez MRO). Wychodziło o ~120 linii mniej i usuwało
duplikację, która jest w projekcie od wcześniej, ale wymagało przebudowy dwóch klas publicznych.
Wybraliśmy izolację kosztem powtórzenia: klasy z `main` zostają nietknięte.

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
uczonych — grupy zero-shot trafiają do osobnej sekcji `held_out_columns`.

### 3.4 Nadzorowany trening jako równoległy kontrakt

`Model.fit(data)` i `Strategy.learn(data)` nie przenoszą etykiet ani masek, a benchmark trenuje na
10 normalnych + 10 anomalnych obrazach z maskami pikselowymi. Rozszerzenie istniejących sygnatur
wywróciłoby `der.learn`, `agem.learn`, `mste.learn` (brak `**kwargs`) oraz wszystkie 8 implementacji
`fit`, w tym modele spoza repozytorium.

Rozwiązanie: **czwarty kontrakt strumienia**, dokładnie tak, jak projekt już rozwiązuje ten problem.
pyCLAD ma jedno ABC strategii na rodzaj strumienia (`ConceptAwareStrategy`, `ConceptIncrementalStrategy`,
`ConceptAgnosticStrategy`), każde z własną sygnaturą `learn`, i jeden scenariusz mówiący tym kontraktem.
Nadzór to kolejna oś tego samego podziału, więc dokładamy `SupervisedStrategy.learn(concept)` oraz
`SupervisedConceptIncrementalScenario`.

Klasa bazowa `Strategy` nie deklaruje `learn` w ogóle, więc nowa sygnatura z niczym nie koliduje.
`strategy.py` i istniejące scenariusze zostają nietknięte — zero różnic względem `main`. Ceną jest
skopiowana pętla `run()` (~25 linii), ale `concept_incremental.py` i `concept_aware.py` już dziś różnią
się między sobą tylko dwoma wywołaniami, więc czwarta kopia jest zgodna z konwencją projektu, a nie
nowym zapachem.
Po stronie modeli `SupervisedModel` (rdzeń) deklaruje `fit(data, labels)` i jest **rodzeństwem**
`Model`, nie jego podtypem — `fit(data)` i `fit(data, labels)` to różne kontrakty, a model nadzorowany
nie może wystąpić tam, gdzie oczekiwany jest nienadzorowany. Kosztem jest powtórzenie deklaracji
`predict`/`name`/`info`; alternatywą byłoby wydzielenie wspólnej bazy bez `fit`, czyli zmiana
istniejącego `Model`.

Maski są sygnałem pikselowym, więc nie ma ich w rdzeniu. `SupervisedVisionModel` dokłada je w warstwie
vision jako `fit(data, labels, masks=None)` i **jest** podtypem `SupervisedModel`, bo parametr
opcjonalny nie zawęża kontraktu.

`NaiveSupervisedStrategy` implementuje `SupervisedStrategy.learn(concept)` i woła
`fit(data, labels, masks)`. `SupervisionRequiredError` zostaje na jeden przypadek: koncept bez etykiet.
Wcześniejsza wersja musiała dodatkowo zaślepiać odziedziczone `learn(data)` — przy własnym ABC nie ma
czego zaślepiać.

Strategia dostaje `Concept`, bo tylko ona wie, czy koncept niesie maski; model dostaje tablice.

### 3.5 `ImageLoadOptions` zamiast `validate_read_options`

Parametry ładowania obrazu urosły z trzech do pięciu (doszły `interpolation` i `apply_exif_transpose`)
i wędrują przez cztery warstwy. Zamiast przepychać pięć argumentów zebrane są w zamrożoną
`ImageLoadOptions`, która waliduje się sama w `__post_init__`. Funkcja `validate_read_options` znika,
a `materialize_samples` i `_load_image` przyjmują jeden obiekt zamiast listy parametrów.

Publiczne `read_vision_dataset` / `read_dataset` zachowują dotychczasowe argumenty.

### 3.6 Zero-shot bez osobnego callbacku

Koncepty MVTec-AD i VisA są dodane do `test_concepts`, ale nie do `train_concepts`. Scenariusz ocenia je
po każdym etapie bez żadnej zmiany kodu, a `GroupedConceptMetricCallback` rozpoznaje grupy, które nigdy
nie wystąpiły jako nauczone, i raportuje je w `held_out_columns`. Callback z referencją do strategii
okazał się niepotrzebny.

Zero-shot jest odrzucany dla scenariusza 1, bo MVTec-AD i VisA są tam częścią strumienia treningowego.

### 3.7 VisA zero-shot z indeksu katalogu

Wobec braku pliku meta (§1.3) zbiór testowy VisA powstaje z `VisABenchmarkReader` — wszystkie wiersze
`split_csv/1cls.csv` z `split == test`. Nazwy konceptów prefiksujemy (`visa_<klasa>`, `mvtec_<klasa>`),
zgodnie z konwencją plików strumienia; w scenariuszach 2 i 3 nie ma kolizji, bo oba zbiory są wyłączone
ze strumienia.

---

## 4. Decyzje projektowe

### 4.1 Zakres: tylko benchmark, bez samodzielnego ContinualAD

ContinualAD jest udostępniany jako osobny zbiór, ale nie ma kanonicznego splitu train/test (§1.6), więc
użycie go poza benchmarkiem wymagałoby wymyślenia własnego podziału. Zrezygnowaliśmy z tego: jedynym
wejściem jest `ContinualMegaBenchmarkReader`, który bierze ścieżki z plików meta. Zbudowany wcześniej
`ContinualADBenchmarkReader` (split few-shot 10+10 seedowany po `crc32`) został usunięty.

Przy okazji: w `pyclad/vision/data/benchmarks/` słowo „benchmark" oznacza „znany publiczny zbiór o
znanym layoucie na dysku" (`MVTecBenchmarkReader`, `VisABenchmarkReader`, …), a nie protokół ewaluacji.
`ContinualMegaBenchmarkReader` jest jedyną klasą w tym module, która czyta benchmark w sensie paperu —
protokół plus dane — i dlatego świadomie nie dziedziczy po `VisionBenchmarkReader`, którego kontrakt
(`index_samples` z limitami per kategoria) opisuje czytanie zbioru danych, nie scenariusza.

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

---

## 6. Baseline ADCT — ustalenia z lektury referencji

Port modelu bazowego (`pyclad/vision/models/adct/`) odtwarza referencję, a nie „poprawną" wersję CLIP.
Trzy odstępstwa referencji od standardowego użycia CLIP są istotne dla wyników i muszą być powtórzone,
bo checkpointy zostały wytrenowane właśnie tak.

### 6.1 Backbone używa `nn.GELU` zamiast QuickGELU

`CLIP/clip.py::create_model` buduje model przez `CLIP(**model_cfg, cast_dtype=cast_dtype)` — bez
`quick_gelu`, a `_build_vision_tower` ma `quick_gelu: bool = False`, więc aktywacją jest `nn.GELU`.
Wagi pochodzą z `load_openai_model`, który buduje pomocniczy model z `quick_gelu=True`, ale ten model
służy wyłącznie do wyciągnięcia `state_dict()` i jest odrzucany. Komentarz w ich własnym kodzie
(`model.py:82`) mówi wprost, że modele OpenAI trenowano z QuickGELU.

Odtwarzamy to: `build_clip_backbone` tworzy `open_clip.create_model("ViT-L-14-336")` (czyli `nn.GELU`,
bo konfiguracja o tej nazwie nie ustawia `quick_gelu`) i wgrywa do niego wagi OpenAI.

### 6.2 Enkoder tekstu działa bez maski przyczynowej

`CoOp.py::TextEncoder.forward` woła `self.transformer(x)`, a ich `Transformer.forward` ma
`attn_mask: Optional[torch.Tensor] = None`. Standardowy CLIP przekazuje tu maskę przyczynową. Nasz
`ClipTextEncoder` powtarza wywołanie bez maski.

### 6.3 Obrazy nie są normalizowane statystykami CLIP

`dataset/continual.py` w ścieżce ewaluacji robi tylko `convert("RGB")`, `exif_transpose`,
`Resize(336, BICUBIC)` i `ToTensor()`. `create_model` ustawia `model.visual.image_mean/image_std`, ale
nikt ich nie używa. Do modelu wchodzą wartości z zakresu [0, 1]. Nasz `Adct._to_tensor` dzieli przez 255
i nie normalizuje.

### 6.4 Brak `albumentations` w `requirements.txt`

`dataset/continual.py` importuje `albumentations` i `albumentations.pytorch.ToTensorV2`, a
`requirements.txt` ich nie wymienia — instalacja z pliku nie wystarcza do uruchomienia repozytorium.

### 6.5 Podpis Tabeli 2 wskazuje zły scenariusz

Tabela 2 ma podpis „Experimental results on Scenario 3", identyczny jak Tabela 3, ale jej kolumny to
58-5 (12 zadań), 58-10 (6 zadań) i 58-30 (2 zadania), co odpowiada 60 nowym klasom, czyli scenariuszowi
2 (scenariusz 3 ma 30 nowych klas, więc 6/3/1 zadań — i to są kolumny Tabeli 3). Tekst artykułu
potwierdza: „we refer to the quantitative results from Scenarios 2 and 3, presented in Table 2 and
Table 3". Podpis Tabeli 2 jest błędny.

---

## 7. Pozyskanie danych

Benchmark wymaga pięciu zbiorów. Meta referencji odwołują się do nich prefiksami
`continual_ad`, `Real-IAD-512`, `VIADUCT`, `BTAD`, `MPDD` — to definiuje docelowy układ katalogów pod
`--data_root`.

### 7.1 Real-IAD wymaga wariantu 512 i zgody na licencję

Repozytorium `Real-IAD/Real-IAD` na HuggingFace ma `gated: auto` (CC BY-NC-SA 4.0) — bez
uwierzytelnienia `resolve` zwraca „Access to dataset Real-IAD/Real-IAD is restricted". Potrzebna jest
akceptacja warunków na stronie zbioru i token z `canReadGatedRepos`.

Całe repozytorium to 622 GiB w czterech wariantach rozdzielczości, ale meta wskazują wyłącznie
`Real-IAD-512`, więc wystarczy `realiad_512/*.zip` — **14,17 GiB** w 30 archiwach. `realiad_raw`
(507 GiB), `realiad_1024` (54 GiB) i `realiad_256` (4 GiB) są zbędne.

Archiwa mają wewnątrz prefiks `<klasa>/`, a meta oczekują `Real-IAD-512/images/<klasa>/...`, więc
rozpakowanie musi celować w podkatalog `images`, nie w korzeń zbioru.

### 7.2 Nazwy plików w Real-IAD mają niespójny separator

Część klas używa podwójnego podkreślenia po nazwie klasy (`audiojack__0001_NG_BX_C1_...jpg`), część
pojedynczego (`toy_brick_0258_OK_C5_...jpg`). Meta odwzorowują to wiernie, więc dla nas jest to
nieistotne, ale każdy kod składający ścieżki z nazwy klasy i identyfikatora próbki się na tym wywróci.

### 7.3 VIADUCT: 9 z 49 archiwów bez `Content-Length`

Repozytorium fordatis (`handle/fordatis/363.2`) udostępnia jedno archiwum ZIP na klasę, bez
uwierzytelnienia. Dla 40 z 49 `HEAD` zwraca `Content-Length` (razem 13,21 GiB), dla pozostałych 9
odpowiedź jest chunkowana i rozmiaru nie podaje — weryfikacja kompletności pobrania musi w tych
przypadkach opierać się na odczycie centralnego katalogu ZIP, nie na porównaniu rozmiaru.

Nazwy archiwów odpowiadają dokładnie nazwom katalogów klas w meta (włącznie ze spacjami i numerycznym
prefiksem, np. `11 ring cable lug`), a prefiks wewnątrz archiwum to `<klasa>/`.

### 7.4 BTAD ma inny prefiks w archiwum niż w meta

`btad.zip` rozpakowuje się do `BTech_Dataset_transformed/`, a meta oczekują `BTAD/`. Katalog trzeba
przenieść lub przemianować po rozpakowaniu.

### 7.5 MPDD nie ma źródła nadającego się do skryptu

README oryginalnego zbioru wskazuje folder SharePoint uczelni (`vutbr-my.sharepoint.com`), którego nie
da się pobrać bezwarunkowym żądaniem HTTP, a repozytorium `stepanje/MPDD` nie ma wydań z danymi. MPDD
trzeba dostarczyć ręcznie.

### 7.6 MVTec-AD również wymaga obejścia

Link z `datasets_download_link.txt` prowadzi na stronę MVTeca, która publicznie wystawia wyłącznie
archiwum z kodem ewaluacji; sam zbiór jest za formularzem, a historyczny bezpośredni odsyłacz do
mydrive zwraca 404. Użyliśmy kopii `ProgrammerGnome/MVTecAD` na HuggingFace, która trzyma dosłowne
archiwum `mvtec_anomaly_detection.tar.xz` o rozmiarze zgodnym z oryginałem. Zawartość zweryfikowana
przeciwko meta: `scenario1_base` i `meta_mvtec` rozwiązują się bez braków.

VisA pobiera się natomiast wprost z oryginalnego źródła (`amazon-visual-anomaly.s3.us-west-2`,
bez uwierzytelnienia).

### 7.7 Scenariusz 1 wymaga siedmiu zbiorów, nie pięciu

Scenariusze 2 i 3 składają się z ContinualAD, Real-IAD, VIADUCT, MPDD i BTAD, ale scenariusz 1 dokłada
MVTec-AD i VisA — trenuje na nich, dlatego nie ma w nim ewaluacji zero-shot. Rozliczenie klas zrobione
tylko na metach scenariusza 2 przeoczy te dwa zbiory.

### 7.8 Stan po pobraniu

Wszystkie meta rozwiązują się bez braków (`img_path` i `mask_path` sprawdzone plik po pliku):

| meta | obrazy | maski |
| --- | --- | --- |
| `scenario1_base` | 117 275 | 45 151 |
| `scenario1_{5,10,30}classes_tasks` | 78 883 | 28 740 |
| `scenario2_base` | 96 253 | 36 552 |
| `scenario2_30classes_tasks` | 95 748 | 34 881 |
| `scenario3_base` | 108 078 | 37 710 |
| `scenario3_30classes_tasks` | 53 442 | 17 897 |
| `meta_mvtec` (zero-shot) | 5 354 | 1 258 |

Zajętość na dysku: ContinualAD 66 GB, VIADUCT 20 GB, Real-IAD-512 15 GB, BTAD 5,6 GB, MVTec-AD 5,0 GB,
VisA 1,9 GB, MPDD 1,8 GB — razem 115 GB.
