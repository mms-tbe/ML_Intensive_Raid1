# Предсказание оттока клиентов телеком-оператора (Telco Churn)

Этот проект направлен на разработку модели машинного обучения, которая помогает маркетинговой команде Компании заранее находить
клиентов с высоким риском оттока (`Churn`) для того, чтобы заблаговременно обзванивать **до** их фактического ухода/отписки от услуг оператора и предложить им перейти на среднесрочные контракты (от 1 до 2 лет) на более выгодных условиях.

## Бизнес-контекст и почему это влияет на метрику?

Стоимость **пропущенного оттока клиентов** (false negative — не позвонили тому клиенту, который по факту ушёл/отписался от услуг оператора)
примерно в **5 раз выше** стоимости лишнего звонка клиенту, довольному услугами оператора (false positive).
Эта асимметрия определяет весь дизайн Проекта, поэтому мы оптимизируем **recall** по классу `churn`
и осознанно жертвуем частью точности.

## Почему метрика Accuracy бессмысленна? 

Классы несбалансированы: **73.5% non-churn / 26.5% churn**. Модель, всегда
предсказывающая мажоритарный класс, получит **accuracy ≈ 73.5%**, не поймав
**ни одного** churn клиента. Поэтому accuracy здесь бессмысленна.

Это иллюстрирует канонический пример из книги Chip Huyen, *Designing ML Systems* (гл. 4):
две модели для задачи CANCER/NORMAL имеют одинаковую accuracy 0.9, но одна ловит
10/100 раковых случаев заболевания, а другая - 90/100. Одинаковая accuracy, принципиально разная
полезность. Поэтому метрики выбраны по позитивному классу:

- **Основная метрика отбора и тюнинга — PR-AUC (`average_precision`).** При сильном
  дисбалансе precision-recall-кривая информативнее ROC (Case Davis & Goadrich из книги Chip Huyen, *Designing ML Systems* гл. 4),
  т.к. ROC «льстит» за счёт большого числа данных true negatives.
- **recall (churn)** — ключевая бизнес-метрика (из-за 5× стоимости FN).
- **ROC-AUC, precision, F1** — репортятся для контекста.

## Данные

IBM Telco Customer Churn (~7043 строки, 21 колонка). Файл лежит в
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Ловушки данных (обработаны)

1. **`TotalCharges` грузится как `object`** — в 11 строках стоит пустая строка `" "`
   (у клиентов с `tenure=0`). Исправляется/обходится путём создания `pd.to_numeric(errors='coerce')` → `NaN`
   и импутацией медианой **внутри Pipeline**.
2. **Подсчёт активных сервисов (`n_services`)** — колонки сервисов содержат не только
   `Yes/No`: `InternetService ∈ {DSL, Fiber optic, No}`, аддоны используют
   `No internet service`, `MultipleLines` — `No phone service`. Сервис считается
   активным, если значение **не входит** в `{No, No internet service, No phone service}`.
   Наивный счёт `== "Yes"` молча обнулил бы `InternetService`, поэтому ловушка обойдена.

## Инженерные признаки (3 шт., все — внутри Pipeline)

| Признак | Как считается | Зачем |
|---|---|---|
| `tenure_bucket` | бины `tenure`: `0–12 / 13–24 / 25–48 / 49+` | риск оттока резко падает с «возрастом» клиента; бины ловят нелинейность |
| `charges_per_month_of_tenure` | `TotalCharges / max(tenure, 1)` | прокси средней удельной платы; защита от деления на 0 |
| `n_services` | число активных сервисов из 9 колонок (с обходом ловушки выше) | вовлечённость клиента в экосистему — сильный антипредиктор оттока |

Все признаки строятся **построчно** кастомными трансформерами (`scripts/preprocessing.py`),
поэтому воспроизводятся идентично на инференсе и **не могут "протечь"**.

## Архитектура и защита от утечки данных

Единый `sklearn.Pipeline`:

```
NumericCoercer → FeatureEngineer → ColumnTransformer(
    num: SimpleImputer(median) → StandardScaler
    cat: OneHotEncoder(handle_unknown='ignore')
) → Classifier
```

**Весь фиттинг** (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) происходит
**внутри Pipeline, только на train-фолде**. Препроцессинг никогда не «видит» тест —
это исключает самую частую утечку. Здоровые CV-скоры (ROC-AUC ≈ 0.82–0.85, а не 0.99)
подтверждают отсутствие утечки.

## Baselines и сравнение моделей

Сначала — baselines (Huyen гл. 6: «метрика без baseline ничего не значит»):

| Baseline | recall (churn) | ROC-AUC |
|---|---|---|
| `DummyClassifier(most_frequent)` (zero-rule) | 0.00 | 0.500 |
| `DummyClassifier(stratified)` (random) | 0.278 | 0.507 |

Все три модели уверенно бьют оба baseline по recall и ROC-AUC.

Сравнение по 5-fold StratifiedKFold (mean ± std, класс churn):

| Модель | recall | precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LogisticRegression (`class_weight='balanced'`) | 0.797 ± 0.035 | 0.519 ± 0.017 | 0.628 ± 0.021 | 0.846 ± 0.011 | 0.662 ± 0.012 |
| RandomForest (`class_weight='balanced'`) | 0.470 ± 0.023 | 0.650 ± 0.036 | 0.545 ± 0.026 | 0.824 ± 0.011 | 0.619 ± 0.028 |
| **GradientBoosting** | 0.522 ± 0.025 | 0.662 ± 0.036 | 0.583 ± 0.026 | **0.848 ± 0.011** | **0.667 ± 0.022** |

> **Замечание по дисбалансу:** `class_weight='balanced'` (= cost-sensitive learning,
> Huyen гл. 4) применён к LogReg и RandomForest. `GradientBoostingClassifier` в
> scikit-learn **не имеет параметра `class_weight`**, поэтому дисбаланс у GB закрывается
> выбором порога и метрикой отбора (PR-AUC).

**Лучшая модель — GradientBoosting** (наивысший PR-AUC). Лёгкий тюнинг
`GridSearchCV` (`scoring='average_precision'`) по `n_estimators`, `learning_rate`,
`max_depth` дал: `learning_rate=0.05, max_depth=3, n_estimators=100`
(CV PR-AUC = 0.670).

## Калибровка вероятностей

Так как порог применяется к `predict_proba`, важна **калибровка** вероятностей
(Huyen гл. 6): число 0.7 должно означать «в 70% случаев клиент действительно уходит».
Мы сравнили исходный GB и его обёртку `CalibratedClassifierCV` (Platt/sigmoid, внутри
Pipeline) по **Brier score на OOF**:

| Вариант | Brier (OOF, меньше — лучше) |
|---|---|
| Без калибровки | 0.13385 |
| **С калибровкой (sigmoid)** | **0.13343** |

GB и так калиброван неплохо, поэтому выигрыш **"маргинальный"**, но калиброванная версия
чуть точнее и выбрана как финальная. Калибровочные кривые — `results/plots/calibration.png`.

## Порог решения

Порог — это бизнес-решение, а не 0.5 по умолчанию. Он выбран **только на
out-of-fold вероятностях train** (`cross_val_predict`) по правилу «recall ≥ 0.80 при
максимальной precision», затем **заморожен** в `results/threshold.json` и применён к
тесту **один раз**.

**Замороженный порог = 0.259.**

## Результат на held-out тесте (скорился один раз)

| Метрика (churn) | Значение | Гейт |
|---|---|---|
| **Recall** | **0.797** | ≥ 0.75 ✅ |
| **Precision** | **0.518** | ≥ 0.45 ✅ |
| F1 | 0.628 | — |
| ROC-AUC | 0.846 | — |
| PR-AUC | 0.661 | — |

Confusion matrix (тест, 1409 клиентов, порог 0.259):

| | Pred: No | Pred: Yes |
|---|---|---|
| **True: No** | 758 | 277 |
| **True: Yes** | 76 | 298 |

Бизнес-перевод и разбор по срезам — в [`results/results.md`](results/results.md).

## Как запустить
**на Windows** 
```bash
pip install -r requirements.txt
python scripts/train.py     # обучение, выбор порога, сохранение артефактов в results/
python scripts/predict.py   # предсказание на data/new_customers.csv -> results/predictions.csv
python -m pytest -v         # тесты (трансформеры, порог, инференс)
```
**на macOS**
```bash
python3 -m pip install -r requirements.txt
python3 scripts/train.py     # обучение, выбор порога, сохранение артефактов в results/
python3 scripts/predict.py   # предсказание на data/new_customers.csv -> results/predictions.csv
python3 -m pytest -v         # тесты (трансформеры, порог, инференс)
```

## Структура проекта

```
data/        исходный датасет + new_customers.csv (5 строк без Churn)
notebook/    EDA.ipynb
scripts/     preprocessing.py (трансформеры+фабрика), train.py, predict.py
results/     churn_pipeline.pkl, threshold.json, metrics.json, predictions.csv,
             results.md, plots/
tests/       pytest: test_preprocessing, test_threshold, test_predict
docs/        дизайн-спека и план реализации
```

## Артефакты

- `results/churn_pipeline.pkl` — обученный Pipeline (joblib).
- `results/threshold.json` — замороженный порог.
- `results/metrics.json` — все CV/тест-метрики, baselines, Brier (калибровка), срезы, бизнес-сводка.
- `results/plots/` — распределение churn, гистограммы, PR-кривая, confusion matrix, калибровочные кривые.
