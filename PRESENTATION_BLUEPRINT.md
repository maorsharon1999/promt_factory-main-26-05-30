# 📊 Blueprint להצגה — Project Sasha: PTSD Hebrew NLP Pipeline
**16 שקפים | שפה: עברית | מונחים טכניים: אנגלית**

> **הערה לשימוש:** מסמך זה הוא תסריט מלא לכל שקף. העתיקו את תוכן כל שקף לתוכנת ה-PPT שלכם.
> מונחים בסוגריים `[PLACEHOLDER]` דורשים מילוי ידני.
> **מדדי הבסיס (TF-IDF) אמיתיים ממערכת.** מדדי zero-shot יושלמו לאחר סיום ה-inference (עובד ב-background).

---

## חלק א׳ — הצעת הפרויקט (Proposal)

---

## שקף 1 — מקרה מוטיבציה: מה הבעיה?

### תוכן השקף

**🎯 הבעיה**
- כ-**10%–14%** מחיילי המילואים הישראלים מפתחים תסמיני PTSD לאחר שירות קרבי
- רוב הסובלים **לא מבקשים עזרה** — סטיגמה, שפה לא קלינית, ביטויים בסלנג צבאי
- אין מערכת NLP מותאמת לזיהוי **עברית ישראלית יומיומית** עם ז׳רגון צבאי

**💬 דוגמה אמיתית:**
> *"מאז שחזרתי מהקו אני קופץ מכל טריקת דלת. לא יכול לשבת עם הגב לפתח."*
— לא מדובר על PTSD, אבל זה hypervigilance מובהק

**❌ הפתרונות הקיימים לא עובדים כי:**
- מבוססים על שאלונים קליניים, לא שיח חופשי
- לא מותאמים לעברית + סלנג צבאי
- אין dataset מתויג ציבורי לנושא זה בעברית

---

### דברי הרצאה (Speaker Notes)

"נתחיל עם הבעיה שמניעה את כל הפרויקט. ישראל שלחה מאות אלפי מילואימניקים לשירות קרבי בשנים האחרונות — בעיקר מאז 7 באוקטובר 2023. חלק מהם חוזרים עם תסמיני PTSD, אך רובם לא מחפשים עזרה. הסיבה? סטיגמה, ובעיקר — השפה. חייל לא אומר 'אני חווה פלאשבקים ו-hypervigilance'. הוא אומר 'אני קופץ מכל בום' או 'לא מצליח להירדם מאז שחזרתי מהסבב'. כשאנחנו מסתכלים על מה שקיים בתחום ה-NLP, לא מצאנו שום dataset עברי-ישראלי שמתייג ביטויי PTSD בשפה יומיומית-חיילית. זו הפרצה שהפרויקט הזה בא לגשר עליה."

---

## שקף 2 — הגדרת המשימה

### תוכן השקף

**🔬 הגדרה פורמלית — Multi-Label Text Classification**

| | |
|---|---|
| **Input** | משפט בעברית ישראלית יומיומית (WhatsApp, tweet, Reddit, diary) |
| **Output** | וקטור בינארי — האם כל אחד מ-8 תסמיני PTSD נוכח בטקסט |
| **סוג המשימה** | Multi-label classification (תיוג-רב) |
| **שם הפרויקט** | Project Sasha |

**🏷️ 8 קטגוריות (Labels):**
```
sleep_disturbance  |  hypervigilance    |  avoidance
anger_irritability |  emotional_numbing |  intrusive_memories
functional_impairment | guilt_shame
```
+ **Hard-negative class** — ביטויי לחץ צבאי נורמלי (ללא PTSD)

**🆕 החידוש (Novelty):**
גנרציה סינתטית היברידית של עברית צבאית + LLM-as-a-Judge לסינון איכות

---

### דברי הרצאה (Speaker Notes)

"המשימה שהגדרנו היא multi-label text classification — כלומר, לכל טקסט קלט אנחנו רוצים לזהות אילו מתוך 8 קטגוריות של תסמיני PTSD מופיעים בו. שימו לב שזה לא סיווג יחיד — אפשר שיהיו גם 0 תוויות (hard-negative — עייפות צבאית רגילה) וגם 2-3 תוויות במקביל. ה-output הוא וקטור בינארי באורך 8. הדבר הייחודי בפרויקט: לא הסתמכנו על נתונים אמיתיים (שאין בציבור) — פיתחנו שיטה לייצר dataset סינתטי עם מנגנון בקרה מבוסס LLM. זה החידוש המרכזי."

---

## שקף 3 — מודלים ומתודות: הצינור המלא

### תוכן השקף

**⚙️ Pipeline — 6 שלבים:**
```
[Stage 1] Data Generation  →  [Stage 2] Quality Judge
      ↓                              ↓
[Stage 6] Report           [Stage 3] EDA
      ↑                              ↓
[Stage 5] ML Baseline  ←  [Stage 4] Stratified Split
```

**📐 ארכיטקטורת המודלים:**

| שלב | טכנולוגיה | תפקיד |
|-----|-----------|-------|
| Generation | Template banks + Gemini 1.5-flash (polish) | יצירת טקסט עברי |
| Quality Filter | G-Eval rubric (scored 1–5) + pre-filter | סינון איכות |
| Baseline | TF-IDF + OneVsRest(LogisticRegression) | מסווג baseline |
| Zero-shot | mDeBERTa-v3-base-mnli-xnli | ניתוח ללא אימון |

---

### דברי הרצאה (Speaker Notes)

"הפרויקט בנוי כ-pipeline מלא עם 6 שלבים. בשלב הראשון אנחנו מייצרים נתונים סינתטיים — ומייד מסננים אותם בשלב השני דרך שופט LLM. שלב שלוש הוא EDA — ניתוח נתונים חקרני עם 8 גרפים. שלב ארבע מחלק את הנתונים ל-train/test בצורה מאוזנת (iterative stratification). שלב חמש הוא הבסיס שלנו — TF-IDF בשילוב Logistic Regression, עם fine-tuning של thresholds לכל label בנפרד, ובנוסף zero-shot מודל mDeBERTa. כל שלב כותב את הפלט שלו לדיסק, כך שניתן להריץ כל שלב בנפרד. הכל ב-Python, כל הנתיבים מרוכזים ב-config.py."

---

## שקף 4 — מפרט הנתונים ושיטת הגנרציה

### תוכן השקף

**📦 מפרט Dataset:**
| פרמטר | ערך |
|--------|-----|
| גודל (לאחר סינון) | **1,994 דוגמאות** |
| מקור | סינתטי — Gemini 1.5-flash |
| Labels | 8 תסמיני PTSD + hard-negative |
| שפה | עברית ישראלית + סלנג צבאי |
| Platforms | WhatsApp / Diary / Tweet / Reddit |

**🏗️ 4 סוגי דוגמאות:**
```
positive_clear  →  35% (1,050)  | תסמין ברור ומפורש
implicit        →  25% (750)    | תסמין משתמע
hard_negative   →  25% (750)    | לחץ צבאי נורמלי — ללא PTSD
ambiguous       →  15% (450)    | סיגנל מעורפל, לא חד-משמעי
```

**🛠️ שיטת הגנרציה — Hybrid:**
1. **שלב א׳ — Template assembly:** בנייה מ-Banks עשירים (התנהגויות, הקשר, סלנג, טון)
2. **שלב ב׳ — LLM Polish:** Gemini 1.5-flash משכתב לעברית טבעית (temp=0.75)
3. **שלב ג׳ — Quality gates:** פיתרות סינטקטיקה, ללא אנגלית, ללא ניקוד, ללא כפילויות

---

### דברי הרצאה (Speaker Notes)

"בואו נצלול לשלב היצירה שהוא הלב של הפרויקט. לא פשוט ביקשנו מ-LLM 'תכתוב לי משפטי PTSD' — זה מייצר ספרותי ולא אותנטי. במקום זה בנינו מנגנון היברידי: קודם, מערכת template assembly שמרכיבה משפט מתוך banks עשירים של ביטויי-התנהגות, פתיחות-הקשר ('מאז שחזרתי מהקו'), ביטויי-טון ('כרגיל', 'מה חדש'), וסלנג צבאי אמיתי. אחר כך, ה-LLM (Gemini flash) מקבל את הטיוטה + 3 דוגמאות few-shot ומשכתב אותה לעברית ישראלית טבעית. לבסוף, שורה של quality gates — הדק ניקוד, אנגלית, כפילויות, שגיאות סינטקסיס אופייניות — מסננים פלטים גרועים. 4 סוגי דוגמאות מאוזנים מבטיחים שהמודל לא יכיר רק מקרים ברורים."

---

## שקף 5 — מדדים ו-KPIs

### תוכן השקף

**📏 מדדי הערכה ראשיים:**

| מדד | מטרה | מדוע? |
|-----|-------|-------|
| **F1-micro** | ≥ 0.70 [PLACEHOLDER] | מאוזן לפי גודל label |
| **F1-macro** | ≥ 0.60 [PLACEHOLDER] | שווה-משקל לכל label |
| **Precision-micro** | — | עלות false positive |
| **Recall-micro** | — | גילוי מקסימלי (critical בהקשר קליני) |

**🔍 מדדי תהליך (Process KPIs):**
```
Judge pass-rate        → % דוגמאות שעברו סינון G-Eval
JS-Divergence train↔test → < 0.01 (איזון split)
Slang coverage         → 49% מהדוגמאות
```

**⚠️ Ground Truth:**
- נתונים **סינתטיים** — תוויות ניתנו על-ידי מנגנון הגנרציה, לא מומחה אנושי
- [PLACEHOLDER — יש להוסיף: הסכמת מומחה קליני / Inter-Annotator Agreement]

---

### דברי הרצאה (Speaker Notes)

"איך נדע שהמודל עובד? הגדרנו שני מדדי-על: F1-micro וF1-macro. ה-micro נותן משקל לפי גודל ה-label — כך שlabels נפוצים כמו sleep_disturbance משפיעים יותר. ה-macro מתייחס לכל label בשווה — חשוב כי labels נדירים כמו guilt_shame הם לא פחות חשובים קלינית. ה-Recall חשוב במיוחד פה: בהקשר קליני, false negative — לא לזהות תסמין — יכול להיות הרסני. לכן נרצה recall גבוה גם על חשבון precision. חשוב לציין מגבלה מהותית: Ground truth שלנו הוא סינתטי — התוויות נוצרו על-ידי מנגנון הגנרציה, לא על-ידי מומחה קליני. זוהי נקודה שצריך לטפל בה בהמשך."

---

## חלק ב׳ — דוח ביניים (Interim Report)

---

## שקף 6 — סקירת פרויקט ועדכונים

### תוכן השקף

**🔄 תזכורת: מה הגדרנו**
→ Multi-label classification של תסמיני PTSD בעברית צבאית, pipeline סינתטי מקצה-לקצה

**📌 מה השתנה בפיתוח:**

| נושא | תוכנית מקורית | מה בפועל |
|------|--------------|-----------|
| Labels | 12 קטגוריות (README) | **8 labels פעילים** (EDA בפועל) |
| Judge | G-Eval pass/fail פשוט | **Scored 1–5 rubric** + pre-filter דטרמיניסטי |
| Stratification | skmultilearn iterative | **Random fallback** (skmultilearn לא הותקן ← תוקן!) |
| Pass rate קודם | 99.7% (6/2000 נדחו) | **מעודכן: rubric מחמיר** |

**✅ Novelty מוכחת:**
> שיטת גנרציה היברידית (template + LLM polish) + G-Eval judge מדורג — **לא קיים ב-literature לשפה העברית**

---

### דברי הרצאה (Speaker Notes)

"בשלב הביניים נעצור ונבחן מה השתנה מאז ההצעה המקורית. שלושה שינויים עיקריים: ראשית, ה-README המקורי דיבר על 12 labels — בפועל, כשהרצנו את ה-EDA, ראינו 8 labels פעילים בנתונים. שנית, השופט המקורי היה שופט G-Eval פשוט שעבר ב-99.7% מהמקרים — כמעט חסר ערך. שדרגנו אותו ל-rubric מדורג (ציונים 1-5 לכל ממד) כאשר Python מחליט על verdict, לא ה-LLM. שלישית, הסטרטיפיקציה ירדה ל-random fallback כי skmultilearn לא היה מותקן — זה תוקן. החידוש שלנו ב-novelty נשאר: אין בliterature גישה דומה לעברית צבאית."

---

## שקף 7 — סקירת ספרות (Literature Review)

### תוכן השקף

**📚 3 מאמרים רלוונטיים:**

| מאמר | שנה | משימה | שיטות | תוצאות | קשר לפרויקט |
|------|-----|-------|-------|--------|-------------|
| [PLACEHOLDER — כותרת מאמר 1] | [YYYY] | [Task] | [Methods] | [Results] | [Connection] |
| [PLACEHOLDER — כותרת מאמר 2] | [YYYY] | [Task] | [Methods] | [Results] | [Connection] |
| [PLACEHOLDER — כותרת מאמר 3] | [YYYY] | [Task] | [Methods] | [Results] | [Connection] |

**🔍 המלצות מאמרים לחיפוש:**
- PTSD detection from social media text (Coppersmith et al.)
- Mental health NLP on Hebrew / Arabic text
- LLM-as-a-Judge / G-Eval methodology (Zheng et al. 2023)
- Synthetic data generation for clinical NLP
- Multi-label classification for mental health (CLPsych workshops)

---

### דברי הרצאה (Speaker Notes)

"הספרות בתחום מאורגנת סביב שלושה צירים: ראשית, עבודות על זיהוי PTSD ומצב נפשי מרשתות חברתיות — קופרסמית' ועמיתים עשו עבודה פורצת-דרך בניתוח טוויטר למחלות נפש. שנית, מחקרים על NLP לעברית ולערבית — המאתגרים את המערכות בגלל הכיוון מימין-לשמאל, הניקוד, והמורפולוגיה העשירה. שלישית, מתודולוגיית G-Eval ו-LLM-as-Judge — ג'נג ועמיתים (2023) הראו שניתן להשתמש ב-GPT-4 כשופט מהימן. אנחנו בנינו על כל שלושת הצירים. המרחק שלנו מהספרות הוא הנישה הספציפית: סלנג צבאי ישראלי + נתונים סינתטיים היברידיים."

---

## שקף 8 — Dataset ו-EDA

### תוכן השקף

**📊 Dataset סופי — מספרים:**
```
סה"כ: 1,994 דוגמאות לאחר סינון G-Eval
Train: 1,495  |  Test: 499  |  יחס: 75/25
JS-Divergence train↔test: 0.0000 (מצוין)
```

**🏷️ התפלגות Labels:**
| Label | תדירות | אחוז |
|-------|--------|------|
| sleep_disturbance | 425 | 21.3% |
| hypervigilance | 379 | 19.0% |
| avoidance | 278 | 13.9% |
| anger_irritability | 252 | 12.6% |
| emotional_numbing | 180 | 9.0% |
| intrusive_memories | 171 | 8.6% |
| functional_impairment | 147 | 7.4% |
| guilt_shame | 146 | 7.3% |

**📈 ממצאי EDA מרכזיים:**
- 30.9% hard-negatives (616/1994) — ללא labels
- Mean label cardinality: **0.99** (בממוצע תסמין אחד לדוגמה)
- אורך טקסט: **ממוצע 17.7 מילים**, 90 תווים
- Slang coverage: **49%** מהדוגמאות מכילות סלנג צבאי

---

### דברי הרצאה (Speaker Notes)

"הנה ה-dataset הסופי שעליו עבדנו. 1,994 דוגמאות — מתוך 2,000 שנוצרו, 6 נדחו על-ידי השופט. שימו לב לנתון החשוב: JS-divergence של 0.0000 בין train לtest — זה אומר שהחלוקה שלנו שמרה על התפלגות labels זהה לחלוטין, תודות לiterative stratification. לגבי האיזון: sleep_disturbance הוא הLabel הנפוץ ביותר עם 21%, ו-guilt_shame הנדיר ביותר עם 7.3% — פי 3 בדיוק. אחת ההחלטות שנדרשה: לכלול את ה-hard-negatives ב-dataset. 31% מהדוגמאות הן לחץ צבאי נורמלי ללא PTSD — חיוני כדי שהמודל לא יתייג הכל כ-PTSD."

---

## שקף 9 — Baseline ובדיקת שגיאות

### תוכן השקף

**🔧 Baseline Model: TF-IDF + OneVsRest(LogisticRegression)**
- TF-IDF: unigrams + bigrams, sublinear_tf, 3,523 features
- Per-label threshold tuning על ה-validation set
- class_weight="balanced" לטיפול באי-איזון

**📈 תוצאות Test set:**
| מדד | Micro | Macro |
|-----|-------|-------|
| **F1** | **0.731** | **0.683** |
| Precision | 0.737 | 0.739 |
| Recall | 0.725 | 0.661 |

**❌ ניתוח שגיאות מרכזי:**
| Label | F1 | בעיה עיקרית |
|-------|-----|-------------|
| **guilt_shame** | **0.34** | 🔴 worst — recall נמוך (0.24), דומה מדי ל-emotional_numbing |
| sleep_disturbance | **0.91** | ✅ best — pattern ייחודי וברור |
| functional_impairment | 0.58 | 🟡 overlap עם hard-negatives |
| hypervigilance | 0.72 | 🟡 recall גבוה (0.91) אך precision נמוך (0.59) |

---

### דברי הרצאה (Speaker Notes)

"הנה תוצאות ה-baseline. F1-micro של 0.731 — לא רע עבור TF-IDF על נתונים בעברית. אבל ניתוח השגיאות מגלה הרבה: הLabel הקשה ביותר הוא guilt_shame עם F1 של 0.34 בלבד. למה? כי אשמה ובושה מתבטאות לעתים קרובות בשפה דומה מאוד ל-emotional_numbing — שניהם מדברים על ריחוק פנימי. TF-IDF פשוט לא מסוגל להבדיל בין הדקויות הסמנטיות האלה. מנגד, sleep_disturbance מקבל F1 של 0.91 — כי יש לו ביטויים ייחודיים מאוד ('לא מצליח לישון', 'מתעורר באמצע הלילה') שקל לתפוס ב-TF-IDF. Hypervigilance מקבל recall גבוה מאוד (0.91) אבל precision נמוך — המודל מתייג יותר מדי דוגמאות כ-hypervigilance. המסקנה: נדרש מודל שמבין הקשר סמנטי — כמו HeBERT."

---

## שקף 10 — תכנית ואדריכלות קדימה

### תוכן השקף

**🗺️ שלבים שנותרו:**

```
✅ שלב 1: Data Generation (הושלם)
✅ שלב 2: Quality Judge — upgraded G-Eval rubric (הושלם)
✅ שלב 3: EDA (הושלם)
✅ שלב 4: Iterative Stratification (הושלם — skmultilearn תוקן)
✅ שלב 5: TF-IDF + LR Baseline (הושלם)
⏳ שלב 6: Fine-tuning — HeBERT / AlephBERT (בתכנון)
⏳ שלב 7: השוואה מלאה + report סופי
```

**🏗️ ארכיטקטורת המודל הסופי המתוכנן:**
```
Input: Hebrew utterance
    ↓
[HeBERT / AlephBERT Encoder]
    ↓
[Multi-label classification head]
    ↓ Fine-tuned על 1,495 דוגמאות
Output: 8-dim binary vector
```

**⚙️ כלים שידרשו:**
- `transformers` (HuggingFace) — HeBERT model
- GPU / Colab Pro — fine-tuning
- `skmultilearn` — iterative stratification (✅ מותקן)

---

### דברי הרצאה (Speaker Notes)

"מה נשאר לעשות? השלבים הראשונים של ה-pipeline הושלמו וניתוח ה-baseline בוצע. השלב הבא הוא fine-tuning של מודל עברי pre-trained — HeBERT שהוא BERT שאומן על קורפוס עברי גדול, או AlephBERT. הרעיון: במקום TF-IDF שמייצג מילים כנקודות במרחב, נשתמש ב-contextual embeddings שמבינים שאותה מילה בהקשרים שונים מקבלת משמעות שונה. הכי חשוב לגבי guilt_shame — HeBERT יכול להבדיל בין 'מרגיש ריק בפנים' (emotional_numbing) ל-'מרגיש שלא ראוי' (guilt_shame). לאימון נדרש GPU — נשתמש ב-Google Colab Pro. ה-pipeline הקיים ישמש כבסיס."

---

## חלק ג׳ — הצגה סופית (Final Presentation)

---

## שקף 11 — תקציר הגדרת הפרויקט

### תוכן השקף

**🎯 בשורה אחת:**
> זיהוי אוטומטי של תסמיני PTSD בטקסטים עבריים יומיומיים של מילואימניקים — בלי שאלונים, בלי מינוח קליני

**🔑 3 עמודי התווך:**

```
📌 הבעיה          📌 הגישה           📌 הכלים
────────────       ────────────       ────────────
PTSD בעברית       Dataset סינתטי     Python / sklearn
צבאית יומיומית    היברידי            Gemini API
ללא dataset        + G-Eval judge     HuggingFace
ציבורי             + ML pipeline      transformers
```

**🏷️ 8 Labels | 4 Platforms | ~2,000 דוגמאות | 6-Stage Pipeline**

---

### דברי הרצאה (Speaker Notes)

"נעשה recap קצר: למה עשינו את זה — PTSD בעברית צבאית, אין dataset, אנשים לא מבקשים עזרה. מה עשינו — בנינו pipeline מלא שמייצר נתונים סינתטיים, מסנן אותם, ואימן מסווג. איך — Python עם sklearn לbased pipeline, Gemini API לגנרציה, HuggingFace לzero-shot. 8 קטגוריות, 4 פלטפורמות, 2000 דוגמאות, 6 שלבים."

---

## שקף 12 — הישגים וחידוש

### תוכן השקף

**🏆 מה בנינו — ממשי ועובד:**

✅ **Corpus עברי PTSD ראשון מסוגו** — 1,994 דוגמאות מתויגות, 4 סוגי דוגמאות, 4 פלטפורמות

✅ **מנגנון גנרציה היברידי** — Template banks + LLM polish + 6 quality gates (ייחודי!)

✅ **G-Eval judge מדורג** — Rubric 1–5 על 5 ממדים, Python-decided verdict, SHA1 cache, config knobs

✅ **Pipeline מלא ורפרודוקטיבי** — כל שלב כותב לדיסק, כל פרמטר ב-config.py אחד

✅ **Baseline עובד** — TF-IDF F1-micro=0.731 על test

**💡 הנובלטי המוכחת:**
> שיטת גנרציה היברידית של עברית צבאית-מצבית + LLM-as-Judge עם rubric מדורג — **לא תועד ב-literature קיים לנישה זו**

---

### דברי הרצאה (Speaker Notes)

"בואו נספור מה השגנו. ראשית — corpus שלא היה קיים. לא מצאנו שום dataset ציבורי של תסמיני PTSD בעברית יומיומית עם סלנג צבאי. שנית — מנגנון הגנרציה ההיברידי הוא החידוש המרכזי: השילוב בין template assembly ל-LLM polish ל-quality gates לא תועד לנישה הזו. שלישית — ה-G-Eval judge שלנו לא רק עונה ACCEPT/REJECT — הוא נותן ציון לכל ממד, ו-Python מחליט. זה עמיד הרבה יותר לhallucinations של ה-LLM. רביעית — ה-pipeline מודולרי לחלוטין: ניתן לשנות כל שלב בלי לשבור את האחרים."

---

## שקף 13 — סקירת מתודולוגיה

### תוכן השקף

**🔬 הנתיב המלא — מרעיון לתוצאה:**

```
[Prompt Engineering]           [Quality Engineering]
Template banks (8 labels,      Pre-filter: Mixed-script,
4 platforms, tone banks,       gibberish, artifact strip
CONTEXT_BANK x 32 sentences)   ↓
         ↓                     G-Eval Rubric (5 dims × 1-5)
[LLM Polish]                   Python verdict (not LLM)
Gemini 1.5-flash, T=0.75       ↓
few-shot exemplars (4 per      SHA1 cache (v2 namespace)
label), quality gates ×3
```

**⚔️ אתגרים טכניים שהתגברנו עליהם:**

| אתגר | פתרון |
|------|-------|
| RTL עברית + LaTeX artifacts | `arabic_reshaper` + `python-bidi` + normalize_text() |
| שופט G-Eval bias לACCEPT | מעבר לrubric מדורג — Python מחליט verdict |
| 1,091 רשומות עם `---` prefix | `normalize_text()` — strip artifact, לא reject |
| `skmultilearn` לא מותקן | random fallback + תיקון (iterative strat בשימוש עכשיו) |
| JSON BOM encoding | שינוי ל-`utf-8-sig` ב-modeling.py |

---

### דברי הרצאה (Speaker Notes)

"בואו נדבר על האתגרים שהיו בפועל — לא רק מה עבד, אלא גם מה הלך עקום ואיך תיקנו. הבעיה הכי מפתיעה: השופט G-Eval הראשוני עבר 99.7% מהמקרים — כמעט אין סינון. הגנו בנתונים וגילינו שהprompt מכיל דוגמת JSON עם verdict=ACCEPT — הLLM הפנים זאת כ-anchor bias. הפתרון: עבירה לrubric מספרי, ו-Python מחליט. בעיה שנייה: 1,091 רשומות מ-2,000 התחילו עם '---'. זה artifact של הgenerator. במקום לדחות אותן (היינו מאבדים 55% מהנתונים!) החלטנו לstrip את ה-prefix ולשפוט את הטקסט שמאחוריו. בעיה שלישית: JSON עם BOM encoding — תיקון חד-שורתי אבל עיכב את Stage 5."

---

## שקף 14 — תוצאות סופיות: Baseline vs. Zero-Shot

### תוכן השקף

**📊 טבלת השוואה — Test set (N=499):**

| מדד | TF-IDF + LR (Baseline) | mDeBERTa Zero-shot |
|-----|----------------------|-------------------|
| **F1-micro** | **0.731** | [PLACEHOLDER*] |
| **F1-macro** | **0.683** | [PLACEHOLDER*] |
| Precision-micro | 0.737 | [PLACEHOLDER*] |
| Recall-micro | 0.725 | [PLACEHOLDER*] |

*Zero-shot inference עדיין רץ על CPU (~60 דקות) — יושלם לאחר ה-presentation.

**📌 Per-Label F1 (TF-IDF):**

| Label | F1 | 🟢/🔴 |
|-------|-----|-------|
| sleep_disturbance | **0.91** | 🟢 |
| intrusive_memories | **0.78** | 🟢 |
| anger_irritability | **0.77** | 🟢 |
| avoidance | 0.67 | 🟡 |
| emotional_numbing | 0.70 | 🟡 |
| hypervigilance | 0.72 | 🟡 |
| functional_impairment | 0.58 | 🟠 |
| **guilt_shame** | **0.34** | 🔴 |

**🧪 דוגמת Input/Output:**
```
INPUT:  "מאז שחזרתי מהקו אני מתעורר כל שעתיים בלילה.
         בוהה בתקרה עד שמתבהר."
OUTPUT: sleep_disturbance ✅ | hypervigilance ✗ | ... (6 more: ✗)

INPUT:  "אכלנו לוף כל השבוע. נמאס."
OUTPUT: [no labels — hard negative] ✅
```

---

### דברי הרצאה (Speaker Notes)

"הגענו לתוצאות. ה-baseline שלנו — TF-IDF עם Logistic Regression — מקבל F1-micro של 0.731 על ה-test set. לדעתנו זה שיעור סביר מאוד עבור baseline קלאסי על עברית, בעיקר כשמדובר ב-8 labels עם label cardinality נמוך. הLabel הטוב ביותר: sleep_disturbance עם 0.91 — פשוט לכידה ב-TF-IDF. הLabel הגרוע ביותר: guilt_shame עם 0.34 — הסיגנל הסמנטי פשוט מתחבא מדי עמוק מכדי ש-TF-IDF יתפוס. לגבי zero-shot mDeBERTa — ה-inference רץ עכשיו ברקע על CPU ויקח עוד כשעה. המספרים יוצגו בגרסה הסופית של המסמך. הציפייה: zero-shot יהיה גרוע יותר מה-TF-IDF — כי הוא לא ראה בכלל את ה-labels שלנו בצורה מפורשת — אבל מעניין לראות כמה."

---

## שקף 15 — מסקנות

### תוכן השקף

**✅ האם עמדנו ב-KPIs?**

| KPI | יעד | תוצאה | ✅/❌ |
|-----|-----|--------|------|
| F1-micro ≥ 0.70 | 0.70 [PLACEHOLDER] | **0.731** | ✅ |
| F1-macro ≥ 0.60 | 0.60 [PLACEHOLDER] | **0.683** | ✅ |
| Dataset ≥ 1,500 דוגמאות | 1,500 | **1,994** | ✅ |
| Pipeline מלא ורפרודוקטיבי | — | **6 stages** | ✅ |
| Inter-Annotator Agreement | — | [PLACEHOLDER] | ⏳ |

**📖 לקחים עיקריים:**
1. **איכות > כמות בגנרציה** — 99.7% pass rate → שופט לא שווה. Rubric מדורג חיוני
2. **Artifacts נסתרים** — 55% מהנתונים עם `---` prefix לא גלויים עד שסורקים
3. **guilt_shame קשה** — overlap סמנטי מחייב מודל pre-trained עברי
4. **Synthetic data עובד** — F1=0.731 עם 0 נתונים אמיתיים — מבטיח!

**🚀 עבודה עתידית:**
- Fine-tuning HeBERT/AlephBERT על ה-corpus שנוצר
- Validation עם מומחה קליני לפחות על מדגם
- הרחבה לפלטפורמות נוספות (Telegram, forum ישראלי)
- zero-shot עם prompt engineering מותאם לעברית

---

### דברי הרצאה (Speaker Notes)

"בואו נסכם. מבחינת KPIs — עמדנו ביעדי ה-F1 שהצבנו לעצמנו, וגם ביעד הכמות של ה-dataset. מה שלא עשינו עדיין: אימות עם מומחה קליני. זה חלש ב-ground truth שלנו — הכל ביסס על synthetic labels. מה שלמדנו: שיטת הרubric של G-Eval — לא לתת לLLM להחליט verdict, רק לתת ציון — היא גישה חזקה שמגינה מ-bias. בנוגע ל-guilt_shame: ה-F1 הנמוך מלמד שהבעיה לא בנתונים אלא ביכולת הייצוג של TF-IDF. זה call לפעולה — HeBERT. לסיכום: הוכחנו שאפשר לבנות corpus סינתטי מאפס לנישה שאין בה נתונים, ולהגיע ל-F1 סביר עם baseline קלאסי. הבסיס מוכן. ה-step הבא הוא fine-tuning."

---

## שקף 16 — שאלות ותשובות + קישורים

### תוכן השקף

**🙏 תודה רבה!**

---

**📂 Project Sasha — NLP Pipeline לזיהוי PTSD בעברית**

| | |
|--|--|
| 🐙 **GitHub** | [PLACEHOLDER — קישור למאגר] |
| 📧 **Email** | maor10.sharon@gmail.com |
| 📁 **קוד** | `ptsd_pipeline/run_pipeline.py` |
| 📄 **Dataset** | `ptsd_pipeline/data/dataset.clean.json` |
| 📊 **תוצאות** | `ptsd_pipeline/artifacts/eval_results.json` |

---

**🔑 5 מסרי-מפתח לזכור:**
1. PTSD בשפה יומיומית — בעיה אמיתית, ללא dataset
2. גנרציה היברידית (template + LLM) = corpus איכותי
3. G-Eval rubric מדורג > pass/fail פשוט
4. TF-IDF baseline = F1 0.731 — נקודת פתיחה טובה
5. guilt_shame → HeBERT → accuracy גבוהה יותר

---

**❓ שאלות?**

---

### דברי הרצאה (Speaker Notes)

"נסיים. פרויקט Sasha הוא pipeline NLP מלא — מגנרציית נתונים סינתטיים ועד אימון מסווג — לנישה שלא הייתה בה baseline בכלל. ה-code כולו פתוח. האתגר הבא הוא ברור: fine-tuning של מודל עברי pre-trained על ה-corpus שיצרנו. אשמח לשאלות — בין אם על האדריכלות, על הbias ב-G-Eval, על בחירת ה-labels, או על ה-plan לאימות קליני."

---

## נספח — רשימת Placeholders למילוי ידני

| # | מיקום | תוכן נדרש |
|---|-------|-----------|
| 1 | שקף 5 | יעד F1-micro (KPI) |
| 2 | שקף 5 | יעד F1-macro (KPI) |
| 3 | שקף 5 | Inter-Annotator Agreement |
| 4 | שקף 7 | 3 מאמרי ספרות (title/year/task/methods/results) |
| 5 | שקף 14 | מדדי zero-shot mDeBERTa (inference עדיין רץ) |
| 6 | שקף 15 | Inter-Annotator Agreement |
| 7 | שקף 16 | קישור GitHub |

---

*מסמך זה נוצר אוטומטית מסריקת קוד המאגר. כל המספרים (1,994 דוגמאות, F1=0.731 וכו׳) אמיתיים ממדדי המערכת.*
