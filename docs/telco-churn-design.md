# Дизайн-спека: предсказание оттока клиентов телеком-оператора (Telco Churn)

- **Источник требований:** `RAID_W6_Telco_Churn.md`
- **Теоретическая опора:** Chip Huyen, *Designing Machine-Learning Systems* — гл. 4 «Class Imbalance → Using the right evaluation metrics», гл. 6 «Model Development and Offline Evaluation» (Baselines, Evaluation Methods).

---

## 1. Контекст и бизнес-цель

Junior ML-инженер строит модель предсказания оттока (`Churn`) для маркетинговой команды, чтобы обзванивать клиентов из группы риска **до** ухода. Ключевое бизнес-условие:

> Стоимость **false negative** (пропустили реального churn'ера) ≈ **5× выше** стоимости **false positive** (позвонили довольному клиенту).

Эта асимметрия определяет всё: выбор метрики, веса классов и порог принятия решения. Accuracy запрещена как основная метрика (обоснование — §3).

## 2. Данные

- Датасет **IBM Telco Customer Churn** (~7043 строки, 21 колонка). Скачивается автоматически с прямой ссылки GitHub в `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, путь — относительный от корня репозитория.
- Целевая переменная `Churn`: `Yes/No → 1/0`. Позитивный класс = `1` (churn).
- Баланс классов ≈ **73% non-churn / 27% churn** — умеренный дисбаланс.

### Ловушки данных (обязательны к обработке)

1. **`TotalCharges` грузится как `object`** из-за пустых строк `" "`. Обработка: `pd.to_numeric(df['TotalCharges'], errors='coerce')` → `NaN`, далее импутация `SimpleImputer(strategy='median')` **внутри Pipeline**.
2. **Ловушка `n_services`**: колонки сервисов содержат не только `Yes/No`. `InternetService ∈ {DSL, Fiber optic, No}`; аддоны используют `No internet service` как третье значение; `MultipleLines` использует `No phone service`. Сервис считается **активным**, если его значение **не входит** в `{No, No internet service, No phone service}`. Наивный счёт `== "Yes"` молча обнуляет `InternetService` — это и есть ловушка.
3. **Дисбаланс**: accuracy бессмысленна (мажоритарный baseline ≈ 73%). Планируем вокруг дисбаланса с первого дня.

## 3. Метрика и её обоснование (на основе книги)

### 3.1 Почему не accuracy
Chip Huyen, *Designing Machine-Learning Systems* (гл. 4, «Using the right evaluation metrics»): accuracy и error rate трактуют классы одинаково, поэтому метрика доминируется мажоритарным классом. Канонический пример CANCER A/B (стр. 125–126): обе модели имеют accuracy 0.9, но модель A ловит 10/100 раковых, B — 90/100. Это прямой аналог нашей 5× асимметрии FN. → В README цитируем этот пример как обоснование отказа от accuracy и привязываемся к ~73% baseline.

### 3.2 Набор метрик
По позитивному классу (`churn=1`, `pos_label=1`): **recall, precision, F1, ROC-AUC, average_precision (PR-AUC)**.

- **Основная метрика отбора модели и тюнинга — `average_precision` (PR-AUC).** Обоснование (гл. 4, стр. 127, Davis & Goadrich): при сильном дисбалансе PR-кривая информативнее ROC, т.к. ROC «льстит» за счёт большого числа true negatives. ROC-AUC всё равно репортим (требование задания).
- **recall** — ключевая бизнес-метрика (из-за 5× FN), целевые гейты см. §8.

### 3.3 Baselines (гл. 6)
«Метрика без baseline ничего не значит». Реализуем два `DummyClassifier`:
- `strategy='most_frequent'` = **zero rule baseline** (всегда мажоритарный класс).
- `strategy='stratified'` = **random baseline по распределению меток**.

Каждая обучаемая модель обязана побить оба baseline по **recall** и **ROC-AUC**. В README цитируем предостережение Хьюен: модель с F1=0.90 может «с тем же успехом гадать случайно», если позитивный класс редкий.

### 3.4 Cost-sensitive learning (гл. 4, стр. 130–131)
`class_weight='balanced'` = «class-balanced loss»: вес класса обратно пропорционален его частоте (≈ 73/27 ≈ 2.7). Бизнес-стоимость FN=5×FP кодируется явной cost-matrix → `class_weight={0:1, 1:5}`. **Вес класса трактуем как гиперпараметр** (`'balanced'` vs `{0:1,1:5}`) и выбираем по CV.

### 3.5 ⚠️ Ограничение корректности: GradientBoosting без `class_weight`
Подтверждено в окружении (sklearn 1.5.2): `GradientBoostingClassifier` **не имеет параметра `class_weight`**. → Для GB дисбаланс закрывается **порогом и метрикой отбора (`average_precision`)**, а не весами. `class_weight` применяем только к `LogisticRegression` и `RandomForestClassifier`. Это предотвращает ошибку на этапе `GridSearchCV`.

## 4. Архитектура

### 4.1 Структура репозитория
```
Raid/
  README.md                 # оценивается;
  requirements.txt
  .gitignore
  RAID_W6_Telco_Churn.md     # исходное задание
  docs/telco-churn-design.md
  data/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
    new_customers.csv                        # 5 строк из raw, без Churn
  notebook/
    EDA.ipynb               # не оценивается, но обязателен по содержанию
  scripts/
    preprocessing.py        # кастомные трансформеры + фабрика пайплайна
    train.py                # split → baseline → CV → тюнинг → порог → save
    predict.py              # load pkl → предсказание на новых клиентах
  results/
    plots/                  # churn-распределение, гистограммы, PR-кривая, confusion matrix
    predictions.csv
    churn_pipeline.pkl
    threshold.json          # замороженный порог (детерминизм predict.py)
    metrics.json            # CV + test метрики (для воспроизводимости README)
    results.md              # перевод в бизнес-язык;
  tests/
    test_preprocessing.py
    test_threshold.py
    test_predict.py
```
Дополнения к структуре из задания: `tests/` (TDD), `results/threshold.json` и `results/metrics.json` (детерминизм и воспроизводимость), `docs/`. `churn_pipeline.pkl` остаётся sklearn-Pipeline, как требует задание.

### 4.2 Единый Pipeline (без утечки)
```
Pipeline([
  ('coerce',   NumericCoercer(['TotalCharges'])),        # " " → NaN, к float
  ('engineer', FeatureEngineer()),                       # +tenure_bucket, +charges_per_month_of_tenure, +n_services
  ('prep', ColumnTransformer([
      ('num', Pipeline([SimpleImputer('median'), StandardScaler()]), NUM_COLS),
      ('cat', OneHotEncoder(handle_unknown='ignore'),                CAT_COLS),
  ])),
  ('model', <classifier>),
])
```
- **NUM_COLS:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `charges_per_month_of_tenure`, `n_services`.
- **CAT_COLS:** все исходные `object`-категории (`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`) + инженерная `tenure_bucket`.
- `customerID` дропается в скриптах **до** подачи в Pipeline (идентификатор, не фича).
- Весь фиттинг (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) — только внутри Pipeline, на train-фолде. Фит препроцессинга на полном датасете до сплита = утечка (главный способ завалить проект).

## 5. Компоненты

### 5.1 `scripts/preprocessing.py`
Все классы определены здесь, чтобы быть импортируемыми при `joblib.load` (распиклинг). И `train.py`, и `predict.py` импортируют их из `scripts.preprocessing`.

- **`NumericCoercer(BaseEstimator, TransformerMixin)`** — параметр `cols`. `transform`: `pd.to_numeric(errors='coerce')` для указанных колонок; возвращает DataFrame. Stateless (`fit` → `self`).
- **`FeatureEngineer(BaseEstimator, TransformerMixin)`** — добавляет 3 фичи, возвращает DataFrame. Stateless. Логика построчная → утечка невозможна по построению:
  - `tenure_bucket` = `pd.cut(tenure, bins=[-1,12,24,48,inf], labels=['0-12','13-24','25-48','49+'])` (категориальная).
  - `charges_per_month_of_tenure` = `TotalCharges / max(tenure, 1)` (числовая; защита от деления на 0).
  - `n_services` = число активных сервисов по 9 колонкам, активный = значение ∉ `{No, No internet service, No phone service}` (числовая).
- **`make_pipeline(model, num_cols, cat_cols)`** — фабрика, собирающая Pipeline из §4.2 с переданным классификатором.
- **`select_threshold(y_true, y_proba, min_recall=0.80)`** — по PR-кривой возвращает порог с recall ≥ `min_recall` и максимальной precision. Используется в `train.py`, тестируется в `tests/test_threshold.py`.

### 5.2 `scripts/train.py`
CLI-скрипт, запускается `python scripts/train.py` из корня. Шаги — см. §6.

### 5.3 `scripts/predict.py`
CLI-скрипт, `python scripts/predict.py`. Контракт — см. §7.

## 6. Поток обучения (`train.py`)

1. Загрузить CSV; `y = Churn.map({'Yes':1,'No':0})`; `X = df.drop(columns=['Churn'])`; отложить и дропнуть `customerID`.
2. **Stratified train/test split**: `test_size=0.2`, `random_state=42`, `stratify=y`. Тест трогаем ровно один раз — в конце.
3. **Baselines** на train (5-fold): `DummyClassifier('most_frequent')`, `DummyClassifier('stratified')` — фиксируем их recall/ROC-AUC, цитируем ~73% accuracy.
4. **Сравнение 3 классификаторов** (меняется только шаг `model`): `LogisticRegression(class_weight=..., max_iter=1000)`, `RandomForestClassifier(class_weight=...)`, `GradientBoostingClassifier()`. **5-fold StratifiedKFold**, отчёт **mean ± std** по recall, precision, F1, ROC-AUC, average_precision. Таблица → README.
5. **Выбор лучшей** модели по `average_precision` + recall (с оглядкой на ROC-AUC).
6. **Лёгкий тюнинг только лучшей** модели: `GridSearchCV`/`RandomizedSearchCV`, ≥2 гиперпараметра через `'model__param'`, `scoring='average_precision'` (явно; не accuracy). Для LogReg/RF в сетку входит и `class_weight` (`'balanced'` vs `{0:1,1:5}`).
7. **Порог**: `cross_val_predict(best_pipe, X_train, y_train, cv=StratifiedKFold(5), method='predict_proba')` → OOF-вероятности → `select_threshold(..., min_recall=0.80)` → **заморозка** в `results/threshold.json`. Сохранить PR-кривую в `results/plots/`.
8. **Рефит** лучшего пайплайна на полном train.
9. **Оценка на тесте (один раз)**: применить замороженный порог → confusion matrix как **подписанный DataFrame** (True/Predicted) → recall, precision, F1, ROC-AUC. Проверить гейты §8.
10. **Сохранить**: `joblib.dump(best_pipe, 'results/churn_pipeline.pkl')`, `threshold.json`, `metrics.json`, графики. Сгенерировать/обновить таблицы для README и `results.md`.

> Дисциплина (совет задания): сначала довести train→test→predict на дефолтной LogReg, потом тюнить. CV-скор ~0.99 = утечка, искать до тюнинга.

## 7. Поток предсказания (`predict.py`)

Контракт: `python scripts/predict.py` из корня после `pip install -r requirements.txt`.

1. Импорт трансформеров из `scripts.preprocessing` (иначе распиклинг упадёт). Скрипт добавляет корень репо в `sys.path` для устойчивости импорта.
2. Загрузить `results/churn_pipeline.pkl` + `results/threshold.json`.
3. Прочитать `data/new_customers.csv` (та же схема, без `Churn`).
4. Отложить `customerID`, дропнуть его → `proba = pipe.predict_proba(X)[:,1]` → `churn_pred = (proba >= threshold).astype(int)`.
5. Приклеить `customerID` обратно по индексу → `results/predictions.csv` с колонками `customerID, churn_pred, churn_proba`.

`data/new_customers.csv` — 5 строк из raw-данных с убранной колонкой `Churn` (генерируется на этапе подготовки данных).

## 8. Гейты приёмки (на тесте, при замороженном пороге)

- **Test recall (churn) ≥ 0.75** — записать в README.
- **Test precision (churn) ≥ 0.45** — записать в README.
- Все модели бьют оба baseline по recall и ROC-AUC.
- Порог выбран **только** на OOF/валидации, не на тесте (иначе — утечка, провал).
- `python scripts/predict.py` отрабатывает с нуля и пишет корректный `predictions.csv`.

## 9. EDA (`notebook/EDA.ipynb`, не оценивается, но обязателен)

`df.dtypes / df.shape / df.head()` + заметка о mistyped-колонках (`TotalCharges`); bar-chart `Churn`; гистограммы числовых; анализ пропусков (явные `NaN` + неявные `" "` в `TotalCharges`); кросс-таблицы `Churn` × {`Contract`, `PaymentMethod`, `InternetService`}; 2–3 письменных инсайта (например, «month-to-month churn ≈ 43% против ≈ 3% у two-year»).

## 10. Тестирование (TDD)

- **`test_preprocessing.py`**: `NumericCoercer` (`" "`→`NaN`, числа сохраняются, тип float); `FeatureEngineer.n_services` — корректный счёт активных, включая ловушку (`DSL`/`Fiber optic`=active, `No`/`No internet service`/`No phone service`=inactive); границы `tenure_bucket`; `charges_per_month_of_tenure` при `tenure=0` (деление на `max(tenure,1)`); стабильность `transform` независимо от `fit` (нет утечки).
- **`test_threshold.py`**: `select_threshold` на синтетике возвращает порог с recall ≥ 0.80 и максимальной при этом precision; монотонность поведения.
- **`test_predict.py`**: smoke end-to-end — фабрика собирает Pipeline, фит на мини-выборке, `predict_proba` работает; схема вывода (`customerID, churn_pred, churn_proba`); реаттач `customerID` по индексу; sanity-проверка против утечки (CV-скор на shuffled-таргете близок к случайному, не ~0.99).

## 11. Дополнения из книги, вшитые в дизайн

1. **PR-AUC (`average_precision`) — основная метрика** отбора/тюнинга (§3.2). ROC-AUC репортим.
2. **GB без `class_weight`** — отдельная ветка обработки дисбаланса (§3.5). ✅ подтверждено в окружении.
3. **Калибровка вероятностей** (гл. 6, стр. 183–184): раз режем по `predict_proba`, калибровка важна; LogReg калибрована «из коробки», RF/GB — хуже. **Опционально** (если победит RF/GB и останется время): `CalibratedClassifierCV` + калибровочная кривая в `results/plots/`.
4. **Slice-based evaluation** (гл. 6, стр. 185–187, пример Хьюен — снова churn): в `results.md` мини-разбор recall по срезам (`Contract`: month-to-month vs two-year), защита от Simpson’s paradox.

Памятка из книги (гл. 4, стр. 128): **никогда не оценивать на ресэмплированных данных**. Мы используем `class_weight`, не SMOTE → риска нет. SMOTE — явно вне scope (если бы применяли — только внутри CV-пайплайна через imbalanced-learn).

## 12. Зависимости (`requirements.txt`)

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `jupyter`, `pytest`. Версии пинуем под окружение (pandas 2.0.3, numpy 1.26.4, sklearn 1.3.2, matplotlib 3.7, joblib 1.3, jupyter 1.0, pytest 8.0)
