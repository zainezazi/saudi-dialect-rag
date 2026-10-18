"""Saudi dialect -> Modern Standard Arabic query expansion.

Customer questions are usually written in dialect (Najdi/Hijazi/Gulf),
while help-centre content is written in MSA. A small, auditable lexicon
bridges the most common gaps. Keys are written in normalized form
(see normalize.normalize). This list is hand-written for the demo and is
deliberately small; extend it per domain.
"""
from .normalize import normalize

DIALECT_TO_MSA: dict[str, str] = {
    # intent / question words
    "ابي": "اريد", "ابغي": "اريد", "ابغا": "اريد", "ودي": "اريد", "بغيت": "اريد",
    "وش": "ماذا", "ايش": "ماذا", "شلون": "كيف", "ليش": "لماذا", "وين": "اين",
    "بكم": "سعر", "كم": "عدد", "عشان": "بسبب", "الحين": "الان",
    "اقدر": "يمكن", "اسوي": "افعل",
    # telecom / service vocabulary
    "النت": "الانترنت", "نت": "الانترنت", "ثقيل": "بطيء بطيئه", "مره": "جدا",
    "يشتغل": "يعمل تشغيل", "ضاعت": "فقدان المفقوده", "ضاع": "فقدان المفقوده",
    "فلوس": "مبلغ", "زياده": "غير صحيح", "اسدد": "دفع سداد", "سددت": "دفع",
    "دفعت": "السداد", "احول": "تحويل", "حولت": "تحويل", "الفايبر": "الالياف البصريه",
    "فايبر": "الالياف البصريه", "حينا": "حي", "رسايل": "رسائل", "اعلانات": "دعائيه",
    "ارقي": "الترقيه", "اكبر": "اعلي", "انقطع": "موقوف ايقاف", "قطعوا": "موقوف ايقاف",
    "اطلع": "اشتراك", "اكلم": "تواصل", "جوالي": "جهاز الاجهزه", "افعلها": "تفعيل",
    "افعل": "تفعيل", "بسافر": "السفر", "برا": "الدولي التجوال", "القي": "ساعات",
    "مفتوح": "تعمل", "جيجا": "جيجابايت", "شريحه": "الشريحه", "الكترونيه": "الالكترونيه",
}

_NORMALIZED = {normalize(k): normalize(v) for k, v in DIALECT_TO_MSA.items()}


def expand_query(query: str) -> str:
    """Append MSA equivalents for any dialect words found in the query."""
    tokens = normalize(query).split()
    extra = [_NORMALIZED[t] for t in tokens if t in _NORMALIZED]
    # Also match after stripping a leading conjunction waw ("وابي").
    extra += [_NORMALIZED[t[1:]] for t in tokens if t.startswith("و") and t[1:] in _NORMALIZED]
    return " ".join(tokens + extra)
