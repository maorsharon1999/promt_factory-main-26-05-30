# מדריך טכני מלא — תהליך EDA וניקוי נתונים
## פרויקט זיהוי סימפטומי PTSD בסלנג צבאי עברי
### HIT — שבוע 9, Interim Report

---

## תוכן עניינים

1. [תמונת על — Pipeline](#1-תמונת-על--pipeline)
2. [קובץ הנתונים — dataset1240.json](#2-קובץ-הנתונים--dataset1240json)
3. [Pillar 1 — quality_judge.py](#3-pillar-1--quality_judgepy)
4. [Pillar 2 — eda.py](#4-pillar-2--edapy)
5. [Pillar 3 — split.py](#5-pillar-3--splitpy)
6. [Pillar 4 — report.py](#6-pillar-4--reportpy)
7. [מודלים וספריות](#7-מודלים-וספריות)
8. [זרימת קבצים](#8-זרימת-קבצים)
9. [קריטריוני איכות — Failure Codes](#9-קריטריוני-איכות--failure-codes)
10. [שאלות נפוצות בהגנה](#10-שאלות-נפוצות-בהגנה)
11. [הגבלות וסיכונים](#11-הגבלות-וסיכונים)

---

## 1. תמונת על — Pipeline

### מה הפרויקט עושה?

המטרה היא לבנות מערכת שמזהה **סימפטומי PTSD** בהודעות טקסט בעברית — בסלנג צבאי. הנתונים שלנו הם **סינתטיים** (נוצרו על ידי LLM), ולכן לפני כל אימון של מודל סיווג, חייבים לעבור 4 שלבים:

```
[dataset1240.json]
      |
      v
[quality_judge.py]  <-- מסנן דגימות לא טבעיות / עם דליפות קליניות
      |
      v
[dataset1240.clean.json]  +  [quality_metrics.json]  +  [judge_cache.json]  +  [judge_feedback.json]
      |
      v
[eda.py]  <-- ניתוח סטטיסטי ויצירת גרפים
      |
      v
[visuals/01_label_marginals.png ... 08_slang_top20.png]  +  [eda_tables.json]
      |
      v
[split.py]  <-- חלוקה אסטרטגית לסט אימון ובדיקה
      |
      v
[train_dataset.json]  +  [test_dataset.json]  +  [split_manifest.json]
      |
      v
[report.py]  <-- יצירת מסמכי הדוח
      |
      v
[slide3_summary.md]  +  [README_eda.md]
```

### הסדר בפועל להרצה:
```bash
python quality_judge.py   # Pillar 1
python eda.py             # Pillar 2
python split.py           # Pillar 3
python report.py          # Pillar 4
```

> **מה צריך לדעת לשאלות:** כל שלב מייצר קבצי פלט שמשמשים שלבים עוקבים. הפייפליין הוא חד-כיווני וחייב להתבצע בסדר הנ"ל.

---

## 2. קובץ הנתונים — dataset1240.json

### מבנה הקובץ

`dataset1240.json` הוא מערך (array) של 100 רשומות JSON. כל רשומה מייצגת הודעה סינתטית אחת:

```json
{
  "id": "syn_0000",
  "text": "מאז שהיינו שם בפנים אני מתעצבן מכל שטות...",
  "labels": ["anger_irritability", "functional_impairment", "guilt_shame"],
  "example_type": "positive_clear",
  "platform": "diary",
  "explicitness": "behavioral",
  "severity": "strong",
  "slang_used": [],
  "synthetic": true
}
```

### תיאור כל שדה

| שדה | סוג | ערכים אפשריים | הסבר |
|-----|-----|---------------|------|
| `id` | string | `"syn_0000"` ... `"syn_XXXX"` | מזהה ייחודי לדגימה |
| `text` | string | טקסט עברי חופשי | ההודעה הסינתטית עצמה |
| `labels` | list[string] | ראה טבלת labels | תגיות סימפטום — יכולות להיות ריקות (`[]`) |
| `example_type` | string | `positive_clear`, `implicit`, `hard_negative`, `ambiguous` | מידת הבהירות של הסימפטום |
| `platform` | string | `whatsapp`, `tweet`, `reddit`, `diary` | הפלטפורמה שמדמה את ההודעה |
| `explicitness` | string | `explicit`, `implicit`, `behavioral` | איך הסימפטום מוצג — ישיר/רמוז/התנהגותי |
| `severity` | string | `mild`, `medium`, `strong` | עוצמת הסימפטום |
| `slang_used` | list[string] | מילים בסלנג צבאי | המילים הסלנגיות שהופיעו בטקסט |
| `synthetic` | bool | `true` | כל הנתונים סינתטיים |

### 8 תגיות הסימפטום (Labels)

| תגית (Label) | משמעות | תדירות בדאטאסט |
|-------------|---------|----------------|
| `sleep_disturbance` | הפרעות שינה — אינסומניה, סיוטים, ערות ממושכת | 25 (הנפוץ ביותר) |
| `hypervigilance` | דריכות יתר — סריקת סביבה, בהלה מרעשים | 18 |
| `anger_irritability` | כעס ועצבנות — פרצי כעס, סף נמוך לתסכול | 16 |
| `guilt_shame` | אשמה ובושה — האשמה עצמית, survivor guilt | 12 |
| `intrusive_memories` | זיכרונות פולשניים — פלשבקים, מחשבות בלתי רצוניות | 12 |
| `functional_impairment` | פגיעה בתפקוד — נטישת תחביבים, קשיי עבודה | 10 |
| `emotional_numbing` | קהות רגשית — ניתוק, חוסר תגובה רגשית | 6 (הנדיר ביותר) |
| `avoidance` | הימנעות — מניעת מקומות, מצבים, מדיה | 6 (הנדיר ביותר) |

### Hard-Negatives — `labels=[]`

25 מתוך 100 הדגימות הן **hard-negatives** (`example_type = "hard_negative"`). הן מכילות `"labels": []` — כלומר, אין בהן סימפטום. אלה הודעות שנראות דומות לסימפטום אבל אינן כזה (למשל: "עייף מהעבודה" ללא הקשר טראומטי). הן חיוניות לאימון מסווג שיכול **להבחין** בין סימפטום לבין הודעה רגילה.

> **מה צריך לדעת לשאלות:** hard-negatives לא מוסרות, אלא מטופלות כ"מחלקה 9" (`__NEG__`) בשלב ה-split.

---

## 3. Pillar 1 — quality_judge.py

### מטרה

להפעיל **LLM-as-a-Judge** — כלומר, להשתמש במודל שפה (`llama3`) כדי לבדוק האם כל דגימה עומדת בקריטריוני איכות לפני שהיא נכנסת לאימון.

הגישה מבוססת על **G-Eval framework** (Wang et al., 2023) — מסגרת להערכת LLM-generated text על ידי LLM אחר, תוך כפיית Chain-of-Thought (CoT) לפני ההחלטה.

---

### `JudgeVerdict` — dataclass

```python
@dataclass
class JudgeVerdict:
    verdict: Literal["ACCEPT", "REJECT"]
    failure_codes: list[str]
    reasoning: str
```

- `verdict`: ההחלטה הסופית — `"ACCEPT"` (הדגימה עוברת) או `"REJECT"` (נדחית).
- `failure_codes`: רשימת קודי הכשל שנמצאו (ראה [סעיף 9](#9-קריטריוני-איכות--failure-codes)).
- `reasoning`: הסבר בעברית/אנגלית של ה-LLM בשלבי ה-CoT.

---

### `QualityFilterMetrics` — מחלקת מדדים

```python
@dataclass
class QualityFilterMetrics:
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    per_code_counts: dict[str, int]
    per_label_rejected: dict[str, int]
    per_platform_rejected: dict[str, int]
```

**מה נשמר בקובץ `quality_metrics.json`:**
```json
{
  "total_generated": 100,
  "total_accepted": 85,
  "total_rejected": 15,
  "rejection_rate": 0.15,
  "per_code_counts": {"UNNATURAL_ROBOTIC": 8, "TRANSLATIONESE": 4, ...},
  "per_label_rejection_rate": {"sleep_disturbance": 0.04, ...},
  "per_platform_rejection_rate": {"whatsapp": 0.05, ...}
}
```

`rejection_rate` מחושב כ: `total_rejected / total_generated`.

---

### `JudgeOllamaClient` — לקוח ה-LLM לשיפוט

```python
class JudgeOllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="llama3", timeout=300):
        ...
    def _chat(self, system: str, user: str) -> str:
        ...
    def generate(self, user_prompt: str) -> str:
        ...  # retry x3
```

**למה `/api/chat` ולא `/api/generate`?**

`/api/generate` שולח פרומפט גולמי — `llama3` מזהה את העברית בפרומפט ומגיב בעברית, ולא ב-JSON אנגלי כנדרש. `/api/chat` מאפשר להפריד בין:
- `"role": "system"` — הגדרת תפקיד השופט (באנגלית)
- `"role": "user"` — הדגימה לשיפוט

עם system role, `llama3` מבין שצריך לענות ב-JSON אנגלי.

**למה `temperature=0.1`?**

טמפרטורה נמוכה = פלט דטרמיניסטי ועקבי. שופט שמחליט שונה על אותה דגימה בכל הרצה הוא לא שופט אמין. 0.1 מבטיח עקביות תוך מתן גמישות קטנה.

**למה ה-retry מחזיר ACCEPT אחרי 3 כישלונות?**

כדי שהפייפליין לא יקרוס. עדיף לקבל דגימה ספק-באיכות מאשר להפסיד את כל ה-run. הדגימות האלה יסומנו בלוג (`SKIP — LLM timeout after 3 retries`).

---

### `_JUDGE_SYSTEM` ו-`_JUDGE_PROMPT_TEMPLATE` — פרומפט השופט

**System role:**
```
"You are a clinical-linguistics auditor. You evaluate synthetic Hebrew text
for quality issues. You ALWAYS respond in English. You ALWAYS output ONLY
a valid JSON object..."
```

**8 שלבי G-Eval (Chain-of-Thought) בפרומפט:**

| שלב | מה נבדק |
|-----|---------|
| STEP 1 — RESTATE | ציטוט מילולי של הטקסט |
| STEP 2 — REGISTER CHECK | האם הסגנון מתאים לפלטפורמה (diary/tweet/whatsapp/reddit)? |
| STEP 3 — CLINICAL LEAK | האם הטקסט מזכיר מפורשות "PTSD", "טראומה", "הפרעת דחק" וכו'? |
| STEP 4 — TRANSLATIONESE | האם יש קלקים מאנגלית ("להחזיק את זה ביחד")? |
| STEP 5 — ROBOTIC CHECK | האם כל 4 תנאי הרובוטיות מתקיימים ביחד? |
| STEP 6 — LABEL CHECK | האם כל label מוצדק על ידי תוכן הטקסט? |
| STEP 7 — SLANG CHECK | האם כל סלנג בשדה `slang_used` נכון להקשר? |
| STEP 8 — VERDICT | ACCEPT / REJECT |

**פלט נדרש (JSON בלבד):**
```json
{"reasoning": "...", "failure_codes": [], "verdict": "ACCEPT"}
```

---

### מנגנון ה-Cache (SHA1)

```python
def _text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
```

- כל דגימה מזוהה על פי **SHA1 hash** של הטקסט שלה.
- לפני כל קריאה ל-LLM, בודקים אם ה-hash כבר ב-`judge_cache.json`.
- אם כן — משתמשים בתוצאה השמורה (בלי לקרוא ל-Ollama).
- Cache נשמר לדיסק **אחרי כל רשומה** — כך שאם הפייפליין קורס, ניתן להמשיך ממש מהמקום שנעצר.

**למה SHA1 ולא MD5?** אין סיבה קריפטוגרפית — SHA1 מספיק מהיר ונמנע מ-collisions בהיקף של 100-1500 רשומות.

---

### `_parse_verdict` — פירוש תשובת ה-LLM

```python
def _parse_verdict(raw: str) -> JudgeVerdict:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    ...
```

- מסיר markdown fences (` ```json `) אם ה-LLM הוסיף.
- מחפש את ה-JSON object הראשון בטקסט (regex `\{[\s\S]*\}`).
- אם לא נמצא JSON — מחזיר `REJECT` עם קוד `UNNATURAL_ROBOTIC` ו-`[PARSE ERROR]`.
- **סניטציה:** רק קודי כשל שמופיעים בסט `FAILURE_CODES` עוברים — שאר הקודים שה-LLM ממציא מסוננים.

---

### `compute_feedback` — לולאת משוב

```python
def compute_feedback(metrics: QualityFilterMetrics, current_temperature: float) -> dict:
```

מחשב המלצה דטרמיניסטית לכיוון `temperature` בפרומפט-פקטורי:

| תנאי | פעולה מומלצת |
|------|-------------|
| `rejection_rate > 35%` + קוד דומיננטי = `UNNATURAL_ROBOTIC` | `temperature += 0.1` (עד max 1.0) — הוסף מגוון |
| קוד דומיננטי = `TRANSLATIONESE` | `temperature -= 0.05` + inject הוראה "avoid English calques" |
| קוד דומיננטי = `CLINICAL_LEAK` | חזק את `_format_constraints` בפרומפט-פקטורי — לא לשנות temperature |

**חשוב:** `compute_feedback` **לא** משנה את הפרומפט-פקטורי בפועל. הוא רק מייצר JSON עם המלצה (`judge_feedback.json`). השינוי יבוצע ידנית.

---

### קבצי פלט של Pillar 1

| קובץ | תוכן |
|------|------|
| `dataset1240.clean.json` | כל הרשומות שקיבלו `verdict=ACCEPT`, עם שדות `judge_verdict` ו-`judge_codes` מוסיפים |
| `quality_metrics.json` | סטטיסטיקות סינון מלאות |
| `judge_cache.json` | SHA1 → verdict cache לריצות חוזרות |
| `judge_feedback.json` | המלצת temperature/prompt |

> **מה צריך לדעת לשאלות:** G-Eval בחר בהחלטה בינארית (ACCEPT/REJECT) ולא בציון מספרי כי ציון מספרי דורש calibration — קשה לדעת אם 7/10 הוא "מספיק טוב". REJECT עם קוד ספציפי הוא actionable ומאפשר לולאת משוב מדויקת.

---

## 4. Pillar 2 — eda.py

### מטרה

ניתוח סטטיסטי מלא של הנתונים לאחר הסינון, כולל יצירת 8 גרפים ב-`visuals/`.

---

### `load_dataset` — טעינה ובינאריזציה

```python
def load_dataset(path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    with open(path, encoding="utf-8-sig") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["text"].apply(lambda t: len(str(t)))
    df["label_count"] = df["labels"].apply(len)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])
    classes = list(mlb.classes_)
    return df, Y, classes
```

**`MultiLabelBinarizer`** (מ-`sklearn.preprocessing`):
- הופך רשימת labels לוקטור בינארי.
- לדוגמה: `["sleep_disturbance", "anger_irritability"]` → `[1,0,0,0,1,0,0,0]`.
- `Y` היא מטריצה בצורת `(100, 8)` — שורה לכל רשומה, עמודה לכל label.
- `classes` = רשימה מסודרת של שמות ה-labels.

**עמודות חישוביות שנוספות:**
- `word_count`: מספר מילים ב-`text` (split על רווח).
- `char_count`: מספר תווים ב-`text`.
- `label_count`: מספר labels ברשומה (0 עבור hard-negatives).

---

### `save_fig` — שמירת גרף

```python
def save_fig(fig, name):
    VISUALS_DIR.mkdir(exist_ok=True)
    fig.savefig(VISUALS_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

- **300 DPI** — רזולוציה גבוהה לפרסום/הצגה.
- **`bbox_inches="tight"`** — חותך שוליים מיותרים כדי שהגרף לא ייחתך.
- **`plt.close(fig)`** — שחרור זיכרון חשוב כשיוצרים 8 גרפים ברצף.

---

### RTL Pipeline — רינדור עברית ב-matplotlib

```python
import arabic_reshaper
from bidi.algorithm import get_display as bidi_display

def rtl(text: str) -> str:
    return bidi_display(arabic_reshaper.reshape(text))
```

**הבעיה:** matplotlib מרנדר טקסט משמאל-לימין (LTR). עברית היא ימין-לשמאל (RTL). בלי טיפול, אותיות עבריות יופיעו הפוכות או לא מסודרות.

**הפתרון:**
1. `arabic_reshaper.reshape(text)` — מחבר אותיות עבריות/ערביות שנמצאות בצורות מבודדות.
2. `bidi_display(...)` — מחיל את אלגוריתם Unicode BiDi (Bidirectional Algorithm) ששם את הטקסט בסדר הנכון.

**חשוב:** RTL מוחל **רק על תוויות צירים (tick labels) ומקרא (legend)** — לא על הנתונים עצמם.

**Fallback chain לגופן:**
```python
"font.sans-serif": ["Noto Sans Hebrew", "David CLM", "Arial", "DejaVu Sans"]
```
matplotlib ינסה כל גופן ברצף עד שימצא אחד שתומך בעברית.

---

### 8 הגרפים — פירוט

#### `01_label_marginals.png` — תדירות שולית של labels

**מה מוצג:** bar chart אופקי — כל label והתדירות שלו במדגם.
**חישוב:** `Y.sum(axis=0)` — סכום כל עמודה במטריצה הבינארית.
**מה לחפש:** labels עם n≤6 (`emotional_numbing`, `avoidance`) — אלה ה-sparse labels שיגרמו לבעיות בסינון.

#### `02_label_cooccurrence.png` — מטריצת שיתוף-הופעה

**מה מוצג:** heatmap 8×8.
**חישוב:** `Y.T @ Y` — מכפלת מטריצות מייצרת ספירת כמה פעמים כל זוג labels מופיע יחד.
**מה לחפש:** זוגות חזקים (כמו `sleep_disturbance`+`hypervigilance`) שעשויים לדמות ביחד תמונה קלינית אמיתית.

#### `03_cardinality.png` — התפלגות מספר labels לרשומה

**מה מוצג:** היסטוגרמה של `label_count`.
**חישוב:** `Counter(df["label_count"])`.
**מה לחפש:** האם רוב הרשומות הן single-label (1 label) או multi-label. 0 = hard-negatives.

#### `04_length_by_platform.png` — אורך טקסט לפי פלטפורמה

**מה מוצג:** violin plot — `word_count` ו-`char_count` לפי `platform`.
**חישוב:** seaborn `violinplot` — מציג את צורת ההתפלגות, לא רק ממוצע.
**sanity check:** אם `median_words < 8` — מדפיס אזהרה (ייתכן collapse לטקסטים קצרים מדי).

#### `05_platform_x_type.png` — פלטפורמה לפי סוג דגימה

**מה מוצג:** stacked bar chart.
**חישוב:** `pd.crosstab(df["platform"], df["example_type"])`.
**מה לחפש:** האם hard-negatives מפוזרות שווה בין פלטפורמות, או ריכוז בפלטפורמה אחת.

#### `06_severity_x_label.png` — חומרה לפי label

**מה מוצג:** heatmap — שורות = `mild/medium/strong`, עמודות = labels.
**חישוב:** לכל `severity`, סכום `Y[mask]` — כמה מהרשומות עם severity זו מכילות כל label.
**מה לחפש:** האם `sleep_disturbance` מופיע יותר בחומרה strong מאשר mild (צפוי).

#### `07_explicitness_x_label.png` — מפורשות לפי label

**מה מוצג:** heatmap — שורות = `explicit/implicit/behavioral`.
**חישוב:** כנ"ל עם `df["explicitness"]`.
**מה לחפש:** `hypervigilance` צפוי להיות יותר behavioral, `guilt_shame` יותר explicit.

#### `08_slang_top20.png` — 20 מילות הסלנג הנפוצות

**מה מוצג:** bar chart אופקי של 20 הסלנג הכי נפוצים.
**חישוב:** `Counter(all_slang).most_common(20)` — כאשר `all_slang` הוא רשימה שטוחה של כל מילות הסלנג מכל הרשומות.
**coverage:** `% רשומות עם לפחות מילת סלנג אחת` — מודפס לקונסול.

---

### `build_eda_tables` — טבלאות מספריות

```python
def build_eda_tables(df, Y, classes) -> dict:
```

מייצר `eda_tables.json` עם:
- `n_records`: מספר רשומות
- `label_marginals`: ספירה לכל label
- `platform_distribution`: התפלגות פלטפורמות
- `text_length_stats`: mean/std/min/max של word_count ו-char_count
- `label_cardinality_distribution`: {0: 25, 1: 45, 2: 20, ...}
- `mean_label_cardinality`: ממוצע מספר labels לרשומה
- `slang_coverage_rate`: אחוז רשומות עם סלנג

> **מה צריך לדעת לשאלות:** violin plot עדיף על boxplot כשרוצים לראות את צורת ההתפלגות (bimodal? skewed?) ולא רק quartiles.

---

## 5. Pillar 3 — split.py

### מטרה

לחלק את הנתונים הנקיים ל-**75% אימון** ו-**25% בדיקה** בצורה שמבטיחה ייצוג של **כל label** בשני הסטים.

---

### למה `sklearn.train_test_split` לא עובד?

```python
# זה ייכשל:
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.25, stratify=df["labels"])
```

**הסיבה:** `stratify=` מקבל רק וקטור חד-ממדי. עבור multi-label, נצטרך להפוך את ה-labels ל"class powerset" (כל **שילוב** של labels = class נפרד). עם N=100 ו-8 labels, יש עשרות שילובים ייחודיים — רבים מהם מופיעים רק פעם אחת. sklearn ידחה אותם עם `ValueError`.

---

### הפתרון: `_NEG_LABEL = "__NEG__"`

```python
_NEG_LABEL = "__NEG__"

augmented_labels = [
    record["labels"] if record["labels"] else [_NEG_LABEL]
    for record in dataset
]
```

hard-negatives (labels ריקות) מקבלות label סינתטי `"__NEG__"` **רק לצורך הסטרטיפיקציה**. אחרי החלוקה, שדה ה-labels בקבצי הפלט חוזר ל-`[]` הרגיל.

**למה זה נחוץ?** בלי `__NEG__`, הפונקציה לא יודעת לאזן את 25 ה-hard-negatives בין train ו-test. עם `__NEG__`, היא מבינה שיש "מחלקה 9" שגם אותה צריך לחלק 75/25.

---

### Iterative Stratification — האלגוריתם

**מקור:** Sechidis, Tsoumakas, Vlahavas — "On the Stratification of Multi-Label Data" (ECML-PKDD 2011).

**ספרייה:** `skmultilearn.model_selection.iterative_train_test_split` (v0.2.0)

**איך עובד האלגוריתם (בשפה פשוטה):**

1. **מיון labels** לפי תדירות עולה — מתחילים מהנדיר ביותר (`avoidance`, n=6).
2. **לכל דגימה**, מחשבים לאיזה fold (train/test) היא הכי "נחוצה" — הfold שרחוק יותר מהיעד היחסי שלו עבור ה-label הנדיר הנוכחי.
3. **מקצים את הדגימה** לאותו fold.
4. **חוזרים** עם ה-label הבא עד שכולם הוקצו.

**נוסחה מתמטית:**

בכל שלב בוחרים:
```
ℓ* = argmin_ℓ  remaining_examples(ℓ)   # הLabel הנדיר ביותר שנשאר
f* = argmax_f  D_{f, ℓ*}               # הFold שהכי "צמא" ל-ℓ*
```
כאשר `D_{f, ℓ}` = מספר הדגימות שה-fold עדיין צריך עם label `ℓ`.

**הבטחה מתמטית:** לכל label `ℓ`, הסטייה המוחלטת בין train share הרצוי לבין המושג ≤ 1 דגימה. כלומר עבור `avoidance` (n=6, יעד test=1.5): **מובטח** שיהיו 1 או 2 דגימות ב-test, לא 0.

---

### Sanity Gates — שלושה שערי בטיחות

```python
def _run_sanity_gates(Y_train, Y_test, classes):
```

**Gate 1:** כל label מופיע **לפחות פעם אחת** ב-test.
```python
absent = [classes[i] for i in range(len(classes)) if Y_test[:,i].sum() == 0]
```

**Gate 2:** לכל label, `train_share ∈ [0.65, 0.85]`:
```python
share = train_sums[i] / (train_sums[i] + test_sums[i])
```

**Gate 3:** JS divergence בין התפלגות label ב-train לבין test < 0.05:

**נוסחת Jensen-Shannon Divergence:**
```
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```
כאשר `M = 0.5*(P+Q)` וה-+1e-12 הוא Laplace smoothing למניעת log(0).

```python
def _js_divergence(p, q):
    p = p + 1e-12  # smoothing
    q = q + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5*np.sum(p*np.log(p/m)) + 0.5*np.sum(q*np.log(q/m))
```

**תוצאה בפועל:** JS divergence = **0.0039** (< 0.05 ✓) — הסטים מאוזנים מאד.

---

### קבצי פלט של Pillar 3

| קובץ | תוכן |
|------|------|
| `train_dataset.json` | 76 רשומות (75% מהנקיות) |
| `test_dataset.json` | 24 רשומות (25% מהנקיות) |
| `split_manifest.json` | seed, ratios, JS divergence, גרסת scikit-multilearn |

> **מה צריך לדעת לשאלות:** 75/25 ולא 80/20 כי N=100 קטן — 20 רשומות test נותנות מספיק ייצוג לכל 8 ה-labels. אם היה N=1240, היינו יכולים ללכת על 80/20.

---

## 6. Pillar 4 — report.py

### מטרה

לאסוף את כל תוצאות Pillars 1–3 ולייצר מסמכי דוח סופיים.

---

### `_SLIDE3_TEMPLATE` — תבנית שקף 3

```python
_SLIDE3_TEMPLATE = """
### Dataset & EDA — Week 9 Interim
- Source: Synthetic Hebrew military-slang utterances...
- Size: {N_raw} raw -> {N_clean} after G-Eval filter ({pct_kept:.1f}% kept).
- Label space: {n_labels} labels (mean cardinality {mean_card:.2f})
...
"""
```

הפונקציה `render_slide3` טוענת את 3 קבצי ה-JSON, מחשבת את הערכים הנדרשים, ומזינה אותם לתבנית. לא אמורים להיות `{...}` פתוחים בפלט הסופי.

**ערכים שמחושבים:**
- `N_raw`, `N_clean`, `pct_kept` — מ-`quality_metrics.json`
- `rarest_label`, `top_label` — מ-`eda_tables.json["label_marginals"]`
- `js`, `n_train`, `n_test` — מ-`split_manifest.json`

---

### `_VISUAL_CAPTIONS` ו-`README_eda.md`

```python
_VISUAL_CAPTIONS = {
    "01_label_marginals.png": "Marginal frequency of each PTSD symptom label...",
    ...
}
```

`render_readme` יוצר `README_eda.md` עם טבלת markdown שמקשרת כל קובץ PNG לתיאור.

---

### `verify_charts` — בדיקת שלמות גרפים

```python
def verify_charts() -> list[str]:
    missing = []
    for fname in _VISUAL_CAPTIONS:
        p = VISUALS_DIR / fname
        if not p.exists():
            missing.append(fname)
        elif p.stat().st_size < 20_000:  # < 20KB
            missing.append(f"{fname} (suspiciously small)")
    return missing
```

בודק שכל 8 PNG קיימים **וגדולים מ-20KB** — גרף קטן מדי מרמז על שגיאה בשמירה.

---

### קבצי פלט של Pillar 4

| קובץ | תוכן |
|------|------|
| `slide3_summary.md` | טקסט מוכן לשקף 3 של הצגה |
| `README_eda.md` | תיעוד של כל הגרפים עם קישורים |

> **מה צריך לדעת לשאלות:** `report.py --no-eda` מדלג על הרצת `eda.py` מחדש ומשתמש בגרפים קיימים — שימושי כשרוצים לשנות רק את טקסט הדוח.

---

## 7. מודלים וספריות

### מודל הבסיס — llama3

| פרמטר | ערך |
|--------|-----|
| שם | `llama3` |
| מפיץ | Meta AI (דרך Ollama) |
| גודל | ~4.7 GB (quantized) |
| שרת | `http://localhost:11434` (Ollama) |
| שימוש | G-Eval judge בלבד — לא מאמן, לא מסווג |
| endpoint | `/api/chat` (לא `/api/generate`) |
| temperature | `0.1` (דטרמיניסטי) |

### ספריות Python

| ספרייה | גרסה | שימוש |
|--------|------|-------|
| `scikit-multilearn` | 0.2.0 | `iterative_train_test_split` |
| `scikit-learn` | latest | `MultiLabelBinarizer`, `preprocessing` |
| `pandas` | latest | DataFrame + crosstab |
| `numpy` | latest | מטריצת labels Y, JS divergence |
| `matplotlib` | latest | יצירת כל הגרפים, `rcParams` |
| `seaborn` | latest | `heatmap`, `violinplot`, `barplot` |
| `arabic-reshaper` | latest | תיקון צורת אותיות עבריות |
| `python-bidi` | latest | אלגוריתם BiDi לסדר RTL |
| `hashlib` | stdlib | SHA1 cache |
| `importlib` | stdlib | טעינה דינמית של `prompt_factory1240 (1).py` |

### ירושה מ-`prompt_factory1240 (1).py`

הקובץ `prompt_factory1240 (1).py` הוא ה-**Read-Only parent** של הפרויקט. `quality_judge.py` טוען אותו דינמית:

```python
_spec = spec_from_file_location("prompt_factory1240", _FACTORY_FILE)
_factory_mod = module_from_spec(_spec)
sys.modules["prompt_factory1240"] = _factory_mod  # חייב לרשום לפני exec!
_spec.loader.exec_module(_factory_mod)
```

**הסיבה לרישום ב-`sys.modules` לפני `exec_module`:** `@dataclass` decorator מחפש את המודול ב-`sys.modules` כדי לפתור את `__module__`. בלי הרישום — `AttributeError: 'NoneType'`.

**Variable names שנשמרים מהפקטורי (verbatim):**

| שם | מה זה |
|----|-------|
| `LLMProvider` | Enum: `OLLAMA`, `OPENROUTER`, `OPENAI` |
| `LLMConfig` | dataclass: הגדרות LLM |
| `LLMClient` | ABC: ממשק LLM |
| `ResilienceLLMClient` | LLM עם fallback chain |
| `create_llm_client` | factory function |
| `OllamaClient` | לקוח Ollama (`/api/generate`) |
| `MockLLMClient` | לקוח לבדיקות (מחזיר עברית קבועה) |

---

## 8. זרימת קבצים

```
INPUT
├── dataset1240.json         (100 רשומות — נתוני גלם)
└── prompt_factory1240 (1).py  (read-only — LLM abstraction)

PILLAR 1 (quality_judge.py)
├── IN:  dataset1240.json
├── OUT: dataset1240.clean.json    (רשומות ACCEPT בלבד + שדות judge_*)
├── OUT: quality_metrics.json      (סטטיסטיקות סינון)
├── OUT: judge_cache.json          (SHA1 → verdict cache)
└── OUT: judge_feedback.json       (המלצת temperature)

PILLAR 2 (eda.py)
├── IN:  dataset1240.clean.json (fallback: dataset1240.json)
├── OUT: visuals/01_label_marginals.png
├── OUT: visuals/02_label_cooccurrence.png
├── OUT: visuals/03_cardinality.png
├── OUT: visuals/04_length_by_platform.png
├── OUT: visuals/05_platform_x_type.png
├── OUT: visuals/06_severity_x_label.png
├── OUT: visuals/07_explicitness_x_label.png
├── OUT: visuals/08_slang_top20.png
└── OUT: eda_tables.json

PILLAR 3 (split.py)
├── IN:  dataset1240.clean.json
├── OUT: train_dataset.json       (76 רשומות, ~75%)
├── OUT: test_dataset.json        (24 רשומות, ~25%)
└── OUT: split_manifest.json      (seed, ratios, JS divergence, library version)

PILLAR 4 (report.py)
├── IN:  quality_metrics.json
├── IN:  split_manifest.json
├── IN:  eda_tables.json
├── IN:  visuals/*.png  (verify_charts בודק קיום + גודל)
├── OUT: slide3_summary.md
└── OUT: README_eda.md
```

---

## 9. קריטריוני איכות — Failure Codes

ה-LLM judge יכול להשתמש רק ב-5 קודים מוגדרים מראש (closed taxonomy):

| קוד | משמעות | דוגמה שתגרום לקוד |
|-----|---------|-------------------|
| `UNNATURAL_ROBOTIC` | טקסט תבניתי מדי, לא טבעי | פותח תמיד ב"מאז ש...", פחות מ-8 מילים, ללא סלנג, severity != mild |
| `TRANSLATIONESE` | קלקים מאנגלית, ניסוח לא-טבעי | "להחזיק את זה ביחד", "לקחת את זה יום אחד בכל פעם" |
| `CLINICAL_LEAK` | אזכור מפורש של אבחנה | "יש לי PTSD", "הפרעת דחק", "טראומה" — הנתונים חייבים **לרמוז** ולא **לקרוא בשם** |
| `LABEL_MISMATCH` | הטקסט לא תומך ב-label שהוכרז | label = `hypervigilance` אבל הטקסט לא מדבר על דריכות/סריקה |
| `SLANG_MISUSE` | שימוש לא-נכון בסלנג צבאי | "בסיס" בהקשר של בסיס ידע ולא בסיס צבאי |

**לוגיקת הספירה בפרסינג:**
```python
codes = [c for c in raw_codes if c in FAILURE_CODES]  # סניטציה
if codes and verdict == "ACCEPT":
    verdict = "REJECT"  # override — codes מנצחים
if not codes and verdict == "REJECT":
    codes = ["UNNATURAL_ROBOTIC"]  # ברירת מחדל
```

> **מה צריך לדעת לשאלות:** closed taxonomy (5 קודים בלבד) מונע מה-LLM להמציא קודים שרירותיים. הסניטציה ב-`_parse_verdict` מוחקת כל קוד שאינו בסט המוגדר.

---

## 10. שאלות נפוצות בהגנה

### 1. למה בחרת G-Eval ולא ציון מספרי (Likert scale)?

**תשובה:** ציון מספרי (למשל 1–10) דורש calibration — קשה לדעת האם 7 "מספיק טוב" בלי threshold שרירותי. בנוסף, ציון מספרי מייצר בעיית aggregation: 7 בגלל שנה half-translationese ו-7 בגלל שהוא קצת קצר — שתיהן ייראו אותו הדבר. קוד בינארי עם failure code ספציפי הוא actionable: יודעים בדיוק למה הדגימה נדחתה ואיך לתקן את הפרומפט-פקטורי.

---

### 2. למה Iterative Stratification ולא `stratify=` הרגיל של sklearn?

**תשובה:** `sklearn.train_test_split(stratify=y)` מקבל וקטור חד-ממדי. עבור multi-label, היינו צריכים ליצור "powerset classes" — כל שילוב ייחודי של labels הופך ל-class. עם N=100 ו-8 labels, יש >30 שילובים ייחודיים, רבים מהם עם דגימה בודדת. sklearn ידחה עם ValueError. Iterative Stratification מטפל ב-multi-label natively.

---

### 3. למה 75/25 ולא 80/20?

**תשובה:** N=100 (לאחר סינון). עם 20/20 split (80/20), ה-test set היה מכיל ~20 רשומות עם סיכוי גבוה ש-`emotional_numbing` ו-`avoidance` (n=6 כ"א) יופיעו פעם אחת בלבד בtest. 25 רשומות test מגדילים את הסבירות לייצוג של 2 דגימות לפחות לכל sparse label, כפי שמראה split_manifest.json.

---

### 4. למה `temperature=0.1` לשופט?

**תשובה:** השופט צריך להיות **עקבי ודטרמיניסטי** — אם הוא מחליט ACCEPT/REJECT שונה על אותה דגימה בכל ריצה, הcache חסר משמעות והתוצאות אינן reproduced. טמפרטורה נמוכה = sampling גריד = פלט דטרמיניסטי. 0.1 (ולא 0.0) כי llama3 לפעמים מסרב לענות בtempˌ=0 ומחזיר שגיאה.

---

### 5. מה זה JS Divergence ולמה הסף הוא 0.05?

**תשובה:** Jensen-Shannon Divergence מודד את הדמיון בין שתי התפלגויות הסתברות (label marginals של train ו-test). הסף 0.05 שרירותי אבל מקובל בספרות: מתחתיו, ההתפלגויות "דומות מספיק" שמסווג שמאמן על train יוכל להכליל על test. בפרויקט שלנו JS=0.0039 — הרבה מתחת לסף.

---

### 6. למה hard-negatives חשובות?

**תשובה:** מסווג שרואה רק דגימות חיוביות (עם סימפטומים) ילמד "הכל הוא סימפטום". hard-negatives מלמדות אותו לדחות הודעות שנראות דומות אבל אינן טראומטיות. בסלנג צבאי זה קריטי — "עייף מהאימון" ≠ `sleep_disturbance`.

---

### 7. למה השתמשת ב-`/api/chat` ולא ב-`/api/generate`?

**תשובה:** `llama3` עם `/api/generate` ורב-לשוניות — כשהפרומפט מכיל עברית, המודל מזהה את השפה הדומיננטית ומגיב בעברית. `/api/chat` עם `system` role מאפשר להגדיר את ה-persona של השופט **לפני** שהמודל רואה את הטקסט העברי, ולכן הוא שומר על פלט JSON באנגלית.

---

### 8. מה קורה אם Ollama timeout?

**תשובה:** `JudgeOllamaClient.generate` מנסה 3 פעמים עם 5/10/15 שניות המתנה בין הניסיונות. אחרי 3 כישלונות — מחזיר ACCEPT עם reasoning `"SKIP — LLM timeout after 3 retries"`. הדגימה עוברת לנתונים הנקיים. זה conservative — עדיף לכלול דגימה ספק-טובה מאשר להפסיד את כל הריצה. Cache שומר מה שנעשה אחרי כל רשומה.

---

### 9. מה זה `MultiLabelBinarizer` ולמה צריך אותו?

**תשובה:** `sklearn.preprocessing.MultiLabelBinarizer` הופך רשימת lists של labels למטריצה בינארית. זה נחוץ כי:
- אלגוריתמי ML עובדים עם מספרים, לא strings.
- `iterative_train_test_split` מצפה למטריצה Y בצורת (n_samples, n_classes).
- חישובי co-occurrence (`Y.T @ Y`) אפשריים רק על מטריצה נומרית.

---

### 10. למה הנתונים סינתטיים ולא אמיתיים?

**תשובה:** נתונים קליניים אמיתיים על PTSD מצריכים אישורי IRB (ועדת אתיקה), הגנת פרטיות, ואין מאגר ציבורי בעברית עם אנוטציה multi-label. `prompt_factory1240` יוצר נתונים מבוקרים — ניתן לשלוט ב-platform, severity, explicitness ו-slang level. החיסרון: הנתונים לא מייצגים את ה-variance האמיתי של דיבור אנשים שחוו טראומה. זו מגבלה מרכזית שנדון בה בפרק הבא.

---

## 11. הגבלות וסיכונים

### א. N קטן

הדאטאסט מכיל 100 רשומות (מתוכן ~75 לאחר סינון). זה קטן מדי לאימון מסווג רב-מחלקה אמין. `emotional_numbing` ו-`avoidance` עם 6 דגימות כ"א = ייצוג **לא מספיק** לכל label. **תוכנית:** הגדלה ל-1240 נקיות על ידי over-generation של 1500 ועיבוד דרך קו הסינון.

### ב. מקור סינתטי בלבד

כל הנתונים נוצרו על ידי LLM. `prompt_factory1240` מגביל את ה-variance — ייתכן שכל הטקסטים של `diary` נראים "דומים" כי נוצרו על ידי אותו מודל עם אותו פרומפט. מסווג שיאמן על נתונים אלה עלול לא להכליל על ביטויים ייחודיים שאנשים אמיתיים משתמשים בהם.

### ג. שופט יחיד (Single LLM Judge)

`llama3` הוא השופט היחיד. בעולם אידיאלי, משתמשים בכמה שופטים ובודקים את ה-agreement ביניהם (Inter-Annotator Agreement, IAA). שופט יחיד עלול לפספס `TRANSLATIONESE` עדין בגלל מגבלות הבנת עברית של llama3.

### ד. חולשת עברית של llama3

`llama3` אומן בעיקר על אנגלית. הבנתו של עברית צבאית/סלנגית מוגבלת. ייתכן שהוא מחזיר `ACCEPT` על דגימות עם `SLANG_MISUSE` עדין מכיוון שהוא לא מכיר את ההקשר הצבאי הישראלי.

### ה. אין IAA (Inter-Annotator Agreement)

תגיות ה-labels ב-`dataset1240.json` נוצרו על ידי הפרומפט-פקטורי ולא נבדקו על ידי מוערכים אנושיים. אין Kappa score או Fleiss' Kappa שמאשרים שהתגיות עקביות.

### ו. SHA1 Cache — לא מוצפן

`judge_cache.json` מכיל את הטקסטים (via SHA1) והפסיקות. הוא לא מוצפן ולא מאובטח. אם הדאטאסט יכיל מידע רגיש בעתיד — יש צורך בהצפנה.

---

*מסמך זה מתאר את ה-Pipeline כפי שמומש בקובצי המקור הנוכחיים.*
*עודכן לאחרונה: שבוע 9, HIT NLP Course 2026.*
